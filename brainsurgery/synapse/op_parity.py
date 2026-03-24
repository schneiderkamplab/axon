from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import torch
from omegaconf import OmegaConf
from torch import nn
from torch.utils._python_dispatch import TorchDispatchMode
from transformers import AutoModelForCausalLM

from .axon import lower_axon_program_to_synapse_spec, parse_axon_program_from_path
from .axon_test import (
    _extract_logits,
    _load_state_dict,
    _load_tokenizer,
    _looks_like_tokenizer_dir,
    _resolve_device,
    _resolve_safetensors_paths,
)
from .runtime import SynapseProgramModel

_CANONICAL_DTYPES = ("float32", "bfloat16", "float16")


def _resolve_dtype_name(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    raise ValueError(f"Unsupported dtype: {name}")


def _normalize_dtypes(dtypes: Sequence[str] | None) -> list[str]:
    if dtypes is None:
        return list(_CANONICAL_DTYPES)
    normalized: list[str] = []
    seen: set[str] = set()
    for item in dtypes:
        key = str(item).strip().lower()
        if key not in _CANONICAL_DTYPES:
            raise ValueError(f"Unsupported dtype in sweep: {item!r}")
        if key not in seen:
            seen.add(key)
            normalized.append(key)
    if not normalized:
        raise ValueError("At least one dtype is required")
    return normalized


def _normalize_texts(text: str | Sequence[str]) -> list[str]:
    if isinstance(text, str):
        return [text]
    out = [str(item) for item in text]
    return out if out else ["The future of AI is", "Hello world"]


def _first_tensor(value: Any) -> torch.Tensor | None:
    if torch.is_tensor(value):
        return value
    if isinstance(value, (list, tuple)):
        for item in value:
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
        return None
    if isinstance(value, dict):
        for key in ("logits", "hidden_states", "last_hidden_state"):
            if key in value:
                tensor = _first_tensor(value[key])
                if tensor is not None:
                    return tensor
        for item in value.values():
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
        return None
    return None


def _hf_kind(module: nn.Module) -> str | None:
    if isinstance(module, nn.Linear):
        return "linear"
    if isinstance(module, nn.LayerNorm):
        return "layernorm"
    if isinstance(module, nn.Embedding):
        return "embedding"
    name = module.__class__.__name__
    if name == "Conv1D":
        return "linear"
    if "RMSNorm" in name:
        return "rmsnorm"
    if "Attention" in name or name.endswith("SdpaAttention"):
        return "attention"
    return None


def _synapse_kind(op_name: str) -> str | None:
    if op_name in {"linear", "layernorm", "rmsnorm", "add", "embedding"}:
        return op_name
    return None


def _is_attention_output_projection(weight_path: str) -> bool:
    normalized = weight_path.replace("/", ".")
    suffixes = (
        ".self_attn.o_proj.weight",
        ".attn.c_proj.weight",
        ".attention.wo.weight",
        ".self_attn.out_proj.weight",
    )
    return any(normalized.endswith(suffix) for suffix in suffixes)


class _TracingSynapseProgramModel(SynapseProgramModel):
    def __init__(
        self,
        spec: dict[str, Any],
        state_dict: dict[str, torch.Tensor],
        *,
        trace_kinds: set[str],
    ) -> None:
        super().__init__(spec=spec, state_dict=state_dict)
        self._trace_kinds = trace_kinds
        self.trace: dict[str, list[torch.Tensor]] = defaultdict(list)
        self.trace_meta: dict[str, list[dict[str, Any]]] = defaultdict(list)

    def _append_trace(
        self,
        kind: str,
        tensor: torch.Tensor,
        *,
        meta: dict[str, Any],
    ) -> None:
        self.trace[kind].append(tensor.detach().float().cpu())
        self.trace_meta[kind].append(meta)

    def _execute_op(
        self,
        op: str,
        node_spec: dict[str, Any],
        env: dict[str, Any],
        *,
        node_path: str,
        scope: str,
        symbols: dict[str, int | float | bool],
    ) -> None:
        super()._execute_op(op, node_spec, env, node_path=node_path, scope=scope, symbols=symbols)
        kind = _synapse_kind(op)
        if kind is None or kind not in self._trace_kinds:
            kind = None
        bind = node_spec.get("_bind")
        bind_names = [bind] if isinstance(bind, str) else (bind if isinstance(bind, list) else [])
        weight_path: str | None = None
        if op == "linear":
            try:
                maybe_path = self._infer_param_path(
                    node_spec, node_path=node_path, param_name="weight"
                )
                if isinstance(maybe_path, str):
                    weight_path = maybe_path
            except Exception:
                weight_path = None
        if kind is not None:
            for name in bind_names:
                if not isinstance(name, str):
                    continue
                value = env.get(name)
                tensor = _first_tensor(value)
                if tensor is None:
                    continue
                meta = {"node_path": node_path, "op": op, "bind": name}
                if weight_path is not None:
                    meta["weight_path"] = weight_path
                self._append_trace(kind, tensor, meta=meta)
        if op == "linear" and "attention" in self._trace_kinds:
            if isinstance(weight_path, str) and _is_attention_output_projection(weight_path):
                for name in bind_names:
                    if not isinstance(name, str):
                        continue
                    value = env.get(name)
                    tensor = _first_tensor(value)
                    if tensor is None:
                        continue
                    self._append_trace(
                        "attention",
                        tensor,
                        meta={
                            "node_path": node_path,
                            "op": op,
                            "bind": name,
                            "weight_path": weight_path,
                            "attention_via": "linear_out_proj",
                        },
                    )


class _AddTraceMode(TorchDispatchMode):
    def __init__(self) -> None:
        super().__init__()
        self.outputs: list[torch.Tensor] = []
        self.meta: list[dict[str, Any]] = []

    def __torch_dispatch__(
        self,
        func: Any,
        types: tuple[type, ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        del types
        call_kwargs = kwargs or {}
        out = func(*args, **call_kwargs)
        try:
            packet = getattr(func, "overloadpacket", None)
            if packet is torch.ops.aten.add:
                if len(args) >= 2 and torch.is_tensor(args[0]) and torch.is_tensor(args[1]):
                    if args[0].is_floating_point() and args[1].is_floating_point():
                        tensor = _first_tensor(out)
                        if tensor is not None and tensor.is_floating_point():
                            self.outputs.append(tensor.detach().float().cpu())
                            self.meta.append(
                                {
                                    "func": str(func),
                                    "lhs_shape": list(args[0].shape),
                                    "rhs_shape": list(args[1].shape),
                                }
                            )
        except Exception:
            pass
        return out


def _capture_hf_trace(
    model: nn.Module, trace_kinds: set[str]
) -> tuple[dict[str, list[torch.Tensor]], dict[str, list[dict[str, Any]]], list[Any]]:
    trace: dict[str, list[torch.Tensor]] = defaultdict(list)
    trace_meta: dict[str, list[dict[str, Any]]] = defaultdict(list)
    hooks: list[Any] = []
    for module_name, module in model.named_modules():
        kind = _hf_kind(module)
        if kind is None or kind not in trace_kinds:
            continue
        kind_name = kind

        def _hook(
            _module: nn.Module,
            _inp: Any,
            out: Any,
            *,
            _kind: str = kind_name,
            _module_name: str = module_name,
        ) -> None:
            tensor = _first_tensor(out)
            if tensor is None:
                return
            trace[_kind].append(tensor.detach().float().cpu())
            trace_meta[_kind].append(
                {
                    "module_name": _module_name,
                    "module_type": _module.__class__.__name__,
                }
            )

        hooks.append(module.register_forward_hook(_hook))
    return trace, trace_meta, hooks


def _shape_key(tensor: torch.Tensor) -> tuple[int, ...]:
    return tuple(int(x) for x in tensor.shape)


def _ordered_float16_like_bits(x: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
    q = x.to(dtype=dtype)
    bits = q.view(torch.int16).to(torch.int32)
    sign = (bits >> 15) & 1
    return torch.where(sign.bool(), (~bits) & 0xFFFF, bits | 0x8000)


def _ulp_distance(a: torch.Tensor, b: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
    oa = _ordered_float16_like_bits(a, dtype=dtype)
    ob = _ordered_float16_like_bits(b, dtype=dtype)
    return (oa - ob).abs()


def _pair_stats(
    hf_tensors: list[torch.Tensor],
    syn_tensors: list[torch.Tensor],
    *,
    hf_meta: list[dict[str, Any]] | None = None,
    syn_meta: list[dict[str, Any]] | None = None,
    dtype_name: str | None = None,
) -> dict[str, Any]:
    pairs = min(len(hf_tensors), len(syn_tensors))
    matched = 0
    skipped_shape = 0
    numel = 0
    sum_abs = 0.0
    max_abs = 0.0
    sum_rel = 0.0
    max_rel = 0.0
    ulp_values: list[torch.Tensor] = []
    offenders: list[dict[str, Any]] = []
    hf_meta = hf_meta or []
    syn_meta = syn_meta or []
    hf_by_shape: dict[tuple[int, ...], list[tuple[int, torch.Tensor]]] = defaultdict(list)
    syn_by_shape: dict[tuple[int, ...], list[tuple[int, torch.Tensor]]] = defaultdict(list)
    for idx, tensor in enumerate(hf_tensors):
        hf_by_shape[_shape_key(tensor)].append((idx, tensor))
    for idx, tensor in enumerate(syn_tensors):
        syn_by_shape[_shape_key(tensor)].append((idx, tensor))
    all_shapes = set(hf_by_shape.keys()) | set(syn_by_shape.keys())
    for shape in sorted(all_shapes):
        hf_list = hf_by_shape.get(shape, [])
        syn_list = syn_by_shape.get(shape, [])
        local_pairs = min(len(hf_list), len(syn_list))
        skipped_shape += abs(len(hf_list) - len(syn_list))
        for pair_idx in range(local_pairs):
            hf_idx, hf_t = hf_list[pair_idx]
            syn_idx, syn_t = syn_list[pair_idx]
            diff = (syn_t - hf_t).abs()
            local_mean = float(diff.mean()) if diff.numel() else 0.0
            local_max = float(diff.max()) if diff.numel() else 0.0
            offenders.append(
                {
                    "index": pair_idx,
                    "hf_index": hf_idx,
                    "syn_index": syn_idx,
                    "shape": list(shape),
                    "mean_abs": local_mean,
                    "max_abs": local_max,
                    "hf_meta": hf_meta[hf_idx] if hf_idx < len(hf_meta) else None,
                    "syn_meta": syn_meta[syn_idx] if syn_idx < len(syn_meta) else None,
                }
            )
            matched += 1
            n = int(diff.numel())
            numel += n
            sum_abs += float(diff.sum())
            denom = torch.maximum(torch.maximum(hf_t.abs(), syn_t.abs()), torch.tensor(1.0e-12))
            rel = diff / denom
            sum_rel += float(rel.sum())
            local_rel_max = float(rel.max()) if rel.numel() else 0.0
            if local_rel_max > max_rel:
                max_rel = local_rel_max
            if local_max > max_abs:
                max_abs = local_max
            if dtype_name == "bfloat16":
                ulp_values.append(_ulp_distance(hf_t, syn_t, dtype=torch.bfloat16).reshape(-1))
            elif dtype_name == "float16":
                ulp_values.append(_ulp_distance(hf_t, syn_t, dtype=torch.float16).reshape(-1))
    offenders.sort(key=lambda item: item["mean_abs"], reverse=True)
    shape_hf_counts = {str(list(shape)): len(items) for shape, items in hf_by_shape.items()}
    shape_syn_counts = {str(list(shape)): len(items) for shape, items in syn_by_shape.items()}
    mean_ulp: float | None = None
    max_ulp: float | None = None
    p95_ulp: float | None = None
    p99_ulp: float | None = None
    if ulp_values:
        all_ulp = torch.cat(ulp_values).to(torch.float32)
        mean_ulp = float(all_ulp.mean())
        max_ulp = float(all_ulp.max())
        p95_ulp = float(torch.quantile(all_ulp, 0.95))
        p99_ulp = float(torch.quantile(all_ulp, 0.99))
    return {
        "hf_count": len(hf_tensors),
        "syn_count": len(syn_tensors),
        "paired": pairs,
        "matched": matched,
        "skipped_shape": skipped_shape,
        "mean_abs": (sum_abs / numel) if numel > 0 else None,
        "max_abs": max_abs if matched > 0 else None,
        "mean_rel": (sum_rel / numel) if numel > 0 else None,
        "max_rel": max_rel if matched > 0 else None,
        "mean_ulp": mean_ulp,
        "max_ulp": max_ulp,
        "p95_ulp": p95_ulp,
        "p99_ulp": p99_ulp,
        "top_offenders": offenders[:10],
        "shape_hf_counts": shape_hf_counts,
        "shape_syn_counts": shape_syn_counts,
    }


def _build_inputs(
    *,
    prompts: list[str],
    tokenizer_source: str,
    tokenizer_fallback: str | None,
    device: torch.device,
) -> tuple[Any, torch.Tensor, torch.Tensor | None]:
    tokenizer_obj = _load_tokenizer(tokenizer_source, fallback_repo_id=tokenizer_fallback)
    if len(prompts) > 1:
        tokenizer_obj.padding_side = "left"
        if tokenizer_obj.pad_token_id is None:
            if tokenizer_obj.eos_token_id is None:
                raise ValueError(
                    "Tokenizer has no pad token and no eos token; cannot batch prompts"
                )
            tokenizer_obj.pad_token = tokenizer_obj.eos_token
    inputs = tokenizer_obj(prompts, return_tensors="pt", padding=(len(prompts) > 1)).to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")
    return tokenizer_obj, input_ids, attention_mask


def _forward_kwargs(
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    for_hf: bool,
    model_input_names: set[str],
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"input_ids": input_ids}
    if attention_mask is None:
        return kwargs
    if for_hf:
        kwargs["attention_mask"] = attention_mask
        pos_ids = attention_mask.to(torch.long).cumsum(dim=-1) - 1
        pos_ids = pos_ids.masked_fill(attention_mask == 0, 1)
        kwargs["position_ids"] = pos_ids
        return kwargs
    if "attn_mask" in model_input_names:
        kwargs["attn_mask"] = attention_mask
    if "attention_mask" in model_input_names:
        kwargs["attention_mask"] = attention_mask
    return kwargs


def _run_single_dtype(
    *,
    dtype_name: str,
    dtype: torch.dtype,
    resolved_hf_model_dir: Path,
    lowered_spec: dict[str, Any],
    state_dict_paths: list[Path],
    tokenizer_source: str,
    tokenizer_fallback: str | None,
    prompts: list[str],
    device: torch.device,
) -> dict[str, Any]:
    trace_kinds = {"linear", "layernorm", "rmsnorm", "attention", "embedding", "add"}
    model_input_names = set(lowered_spec.get("model", {}).get("inputs", {}).keys())
    tokenizer_obj, input_ids, attention_mask = _build_inputs(
        prompts=prompts,
        tokenizer_source=tokenizer_source,
        tokenizer_fallback=tokenizer_fallback,
        device=device,
    )
    del tokenizer_obj
    hf_kwargs = _forward_kwargs(
        input_ids=input_ids,
        attention_mask=attention_mask,
        for_hf=True,
        model_input_names=model_input_names,
    )
    syn_kwargs = _forward_kwargs(
        input_ids=input_ids,
        attention_mask=attention_mask,
        for_hf=False,
        model_input_names=model_input_names,
    )

    hf_trace: dict[str, list[torch.Tensor]] = defaultdict(list)
    hf_trace_meta: dict[str, list[dict[str, Any]]] = defaultdict(list)
    syn_trace: dict[str, list[torch.Tensor]] = defaultdict(list)
    syn_trace_meta: dict[str, list[dict[str, Any]]] = defaultdict(list)
    hf_error: str | None = None
    syn_error: str | None = None
    logits_summary: dict[str, Any] | None = None
    try:
        hf_model: Any = AutoModelForCausalLM.from_pretrained(
            str(resolved_hf_model_dir), local_files_only=True, dtype=dtype
        )
        hf = hf_model.to(device)
        hf.eval()
        hooks: list[Any] = []
        add_mode = _AddTraceMode()
        try:
            hf_trace, hf_trace_meta, hooks = _capture_hf_trace(hf, trace_kinds=trace_kinds)
            with add_mode, torch.no_grad():
                hf_logits = _extract_logits(hf(**hf_kwargs, use_cache=False)).detach().float().cpu()
            hf_trace["add"] = list(add_mode.outputs)
            hf_trace_meta["add"] = list(add_mode.meta)
        finally:
            for hook in hooks:
                hook.remove()
            del hf
    except Exception as exc:  # noqa: BLE001
        hf_error = f"{type(exc).__name__}: {exc}"
        hf_logits = None

    try:
        state_dict = _load_state_dict(state_dict_paths, device=device, dtype=dtype)
        syn = _TracingSynapseProgramModel(lowered_spec, state_dict, trace_kinds=trace_kinds).to(
            device
        )
        syn.eval()
        with torch.no_grad():
            syn_logits = _extract_logits(syn(**syn_kwargs)).detach().float().cpu()
        syn_trace = syn.trace
        syn_trace_meta = syn.trace_meta
        del syn
        del state_dict
    except Exception as exc:  # noqa: BLE001
        syn_error = f"{type(exc).__name__}: {exc}"
        syn_logits = None

    if hf_logits is not None and syn_logits is not None:
        if hf_logits.shape == syn_logits.shape:
            diff = (syn_logits - hf_logits).abs()
            logits_summary = {
                "mean_abs": float(diff.mean()),
                "max_abs": float(diff.max()),
                "last_max_abs": float(diff[:, -1, :].max()),
                "top1_eq": bool(
                    (syn_logits[:, -1, :].argmax(-1) == hf_logits[:, -1, :].argmax(-1)).all()
                ),
            }
        else:
            logits_summary = {
                "shape_mismatch": {
                    "hf": list(hf_logits.shape),
                    "synapse": list(syn_logits.shape),
                }
            }

    by_op: dict[str, Any] = {}
    for kind in sorted(trace_kinds):
        by_op[kind] = _pair_stats(
            hf_trace.get(kind, []),
            syn_trace.get(kind, []),
            hf_meta=hf_trace_meta.get(kind, []),
            syn_meta=syn_trace_meta.get(kind, []),
            dtype_name=dtype_name,
        )

    return {
        "dtype": dtype_name,
        "hf_error": hf_error,
        "synapse_error": syn_error,
        "logits": logits_summary,
        "by_op": by_op,
    }


def run_axon_op_parity(
    *,
    axon_file: Path,
    weights: Path,
    hf_model_dir: Path | None = None,
    tokenizer: str | None = None,
    text: str | Sequence[str] = ("The future of AI is", "Hello world"),
    device: str = "cpu",
    dtypes: Sequence[str] | None = None,
    output_json: Path | None = None,
) -> dict[str, Any]:
    resolved_device = _resolve_device(device)
    resolved_dtypes = _normalize_dtypes(dtypes)
    prompts = _normalize_texts(text)
    axon_path = axon_file.resolve()
    weights_path = weights.resolve()
    if not axon_path.exists():
        raise FileNotFoundError(f"Axon file not found: {axon_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights path not found: {weights_path}")
    state_dict_paths = _resolve_safetensors_paths(weights_path)
    default_hf_dir = weights_path if weights_path.is_dir() else state_dict_paths[0].parent
    resolved_hf_model_dir = (hf_model_dir or default_hf_dir).resolve()
    tokenizer_source = tokenizer or str(resolved_hf_model_dir)
    if tokenizer is None:
        candidate_old = resolved_hf_model_dir.with_name(f"{resolved_hf_model_dir.name}.old")
        if not _looks_like_tokenizer_dir(resolved_hf_model_dir) and _looks_like_tokenizer_dir(
            candidate_old
        ):
            tokenizer_source = str(candidate_old)
    tokenizer_fallback = resolved_hf_model_dir.name if tokenizer is None else None

    with TemporaryDirectory(prefix="axon_op_parity_") as _tmp_dir:
        modules = parse_axon_program_from_path(axon_path)
        lowered_spec = lower_axon_program_to_synapse_spec(modules)
        loaded = OmegaConf.create(lowered_spec)
        loaded_dict = OmegaConf.to_container(loaded, resolve=True)
        if not isinstance(loaded_dict, dict):
            raise ValueError("Lowered synapse spec did not produce a mapping")
        final_spec: dict[str, Any] = {str(k): v for k, v in loaded_dict.items()}

        results: list[dict[str, Any]] = []
        for dtype_name in resolved_dtypes:
            dtype = _resolve_dtype_name(dtype_name)
            print(f"[op-parity] running dtype={dtype_name}")
            result = _run_single_dtype(
                dtype_name=dtype_name,
                dtype=dtype,
                resolved_hf_model_dir=resolved_hf_model_dir,
                lowered_spec=final_spec,
                state_dict_paths=state_dict_paths,
                tokenizer_source=tokenizer_source,
                tokenizer_fallback=tokenizer_fallback,
                prompts=prompts,
                device=resolved_device,
            )
            results.append(result)
            logits = result.get("logits")
            if isinstance(logits, dict) and "mean_abs" in logits:
                print(
                    f"[op-parity] dtype={dtype_name} logits mean/max/last/top1="
                    f"{logits['mean_abs']:.6f}/{logits['max_abs']:.6f}/{logits['last_max_abs']:.6f}/{logits['top1_eq']}"
                )
            if result.get("hf_error") or result.get("synapse_error"):
                print(
                    f"[op-parity] dtype={dtype_name} hf_error={result.get('hf_error')} synapse_error={result.get('synapse_error')}"
                )

    payload = {
        "axon_file": str(axon_path),
        "weights": str(weights_path),
        "hf_model_dir": str(resolved_hf_model_dir),
        "tokenizer_source": tokenizer_source,
        "device": str(resolved_device),
        "prompts": prompts,
        "results": results,
    }
    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[op-parity] wrote report: {output_json}")
    return payload


__all__ = ["run_axon_op_parity"]
