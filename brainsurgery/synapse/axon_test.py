from __future__ import annotations

import gc
import importlib.util
import json
import time
from collections.abc import Sequence
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import safetensors
import torch
from mltiming import timing
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer

from .axon import lower_axon_program_to_synapse_spec, parse_axon_program
from .codegen import emit_model_code_from_synapse_spec


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("Requested CUDA device, but CUDA is unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("Requested MPS device, but MPS is unavailable")
    return device


def _resolve_dtype(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    raise ValueError(f"Unsupported dtype: {name}")


def _cleanup(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if device.type == "mps":
        torch.mps.empty_cache()


def _extract_logits(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, dict):
        logits = output.get("logits")
        if not torch.is_tensor(logits):
            raise ValueError("Expected tensor logits in dict output")
        return logits
    if isinstance(output, tuple) and output and torch.is_tensor(output[0]):
        return output[0]
    raise ValueError(
        f"Unsupported model output type for logits extraction: {type(output).__name__}"
    )


def _load_state_dict(
    paths: list[Path],
    *,
    device: torch.device,
    dtype: torch.dtype,
    strip_model_prefix: bool,
) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for path in paths:
        st = safetensors.safe_open(str(path), framework="pt")
        for key in st.keys():
            mapped = key
            if strip_model_prefix and mapped.startswith("model."):
                mapped = mapped[len("model.") :]
            if mapped in out:
                raise ValueError(f"Duplicate tensor key while reading safetensors shards: {mapped}")
            tensor = st.get_tensor(key)
            if tensor.is_floating_point():
                tensor = tensor.to(device=device, dtype=dtype)
            else:
                tensor = tensor.to(device=device)
            out[mapped] = tensor
    return out


def _resolve_safetensors_paths(weights: Path) -> list[Path]:
    if weights.is_file():
        if weights.suffix != ".safetensors":
            raise ValueError(f"Expected a .safetensors file, got: {weights}")
        return [weights]

    if not weights.is_dir():
        raise FileNotFoundError(f"Weights path not found: {weights}")

    index_path = weights / "model.safetensors.index.json"
    if index_path.exists():
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = payload.get("weight_map")
        if not isinstance(weight_map, dict):
            raise ValueError(f"Invalid safetensors index (missing weight_map): {index_path}")
        shard_names = sorted({str(name) for name in weight_map.values()})
        paths = [weights / name for name in shard_names]
        missing = [str(path) for path in paths if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing safetensors shard(s) from index: {missing}")
        return paths

    preferred = weights / "model.safetensors"
    if preferred.exists():
        return [preferred]

    candidates = sorted(weights.glob("*.safetensors"))
    if len(candidates) == 1:
        return [candidates[0]]
    if not candidates:
        raise FileNotFoundError(f"No .safetensors files found in directory: {weights}")
    if all(path.name.startswith("model-") and "-of-" in path.name for path in candidates):
        return candidates
    raise ValueError(
        f"Multiple .safetensors files found in {weights}; pass an explicit .safetensors file path."
    )


def _load_tokenizer(tokenizer_source: str, *, fallback_repo_id: str | None = None) -> Any:
    candidate = Path(tokenizer_source).expanduser()
    if candidate.exists():
        source = str(candidate.resolve())
        try:
            return AutoTokenizer.from_pretrained(source, local_files_only=True)
        except Exception:
            return AutoTokenizer.from_pretrained(source, local_files_only=True, use_fast=False)

    try:
        return AutoTokenizer.from_pretrained(tokenizer_source, local_files_only=False)
    except Exception:
        if fallback_repo_id and fallback_repo_id != tokenizer_source:
            try:
                return AutoTokenizer.from_pretrained(fallback_repo_id, local_files_only=False)
            except Exception:
                pass
        return AutoTokenizer.from_pretrained(
            tokenizer_source, local_files_only=False, use_fast=False
        )


def _looks_like_tokenizer_dir(path: Path) -> bool:
    return (
        (path / "tokenizer.json").exists()
        or (path / "tokenizer.model").exists()
        or ((path / "vocab.json").exists() and (path / "merges.txt").exists())
    )


def _load_generated_class(py_path: Path, class_name: str) -> type[Any]:
    module_name = f"_axon_generated_{int(time.time() * 1e9)}"
    spec = importlib.util.spec_from_file_location(module_name, py_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import generated module: {py_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model_cls = getattr(module, class_name, None)
    if model_cls is None:
        raise RuntimeError(f"Generated class {class_name!r} not found in {py_path}")
    return model_cls


def _time_generate(label: str, fn: Any) -> tuple[Any, float]:
    t0 = time.perf_counter()
    with timing(message=label):
        out = fn()
    dt = time.perf_counter() - t0
    return out, dt


def _normalize_texts(text: str | Sequence[str]) -> list[str]:
    if isinstance(text, str):
        return [text]
    out: list[str] = []
    for item in text:
        if not isinstance(item, str):
            raise ValueError("All prompts passed via --text must be strings")
        out.append(item)
    if not out:
        return ["The future of AI is"]
    return out


def run_axon_test(
    *,
    axon_file: Path,
    weights: Path,
    device: str = "cpu",
    text: str | Sequence[str] = "The future of AI is",
    max_len: int = 32,
    hf_model_dir: Path | None = None,
    tokenizer: str | None = None,
    class_name: str = "AxonGeneratedModel",
    main_module: str | None = None,
    dtype: str = "float32",
    strip_model_prefix: bool = False,
) -> dict[str, Any]:
    resolved_device = _resolve_device(device)
    resolved_dtype = _resolve_dtype(dtype)

    axon_file = axon_file.resolve()
    weights_path = weights.resolve()
    if not axon_file.exists():
        raise FileNotFoundError(f"Axon file not found: {axon_file}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights path not found: {weights_path}")

    safetensors_files = _resolve_safetensors_paths(weights_path)
    default_hf_dir = weights_path if weights_path.is_dir() else safetensors_files[0].parent
    resolved_hf_model_dir = (hf_model_dir or default_hf_dir).resolve()
    tokenizer_source = tokenizer or str(resolved_hf_model_dir)
    if tokenizer is None:
        candidate_old = resolved_hf_model_dir.with_name(f"{resolved_hf_model_dir.name}.old")
        if not _looks_like_tokenizer_dir(resolved_hf_model_dir) and _looks_like_tokenizer_dir(
            candidate_old
        ):
            tokenizer_source = str(candidate_old)
    tokenizer_fallback = resolved_hf_model_dir.name if tokenizer is None else None
    prompts = _normalize_texts(text)

    with TemporaryDirectory(prefix="axon_benchmark_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        synapse_yaml_path = tmp_path / "lowered_synapse.yaml"
        generated_py_path = tmp_path / "generated_model.py"

        axon_source = axon_file.read_text(encoding="utf-8")
        modules = parse_axon_program(axon_source)
        synapse_spec = lower_axon_program_to_synapse_spec(modules, main_module=main_module)

        synapse_yaml_path.write_text(
            OmegaConf.to_yaml(synapse_spec, resolve=True), encoding="utf-8"
        )
        loaded = OmegaConf.load(synapse_yaml_path)
        loaded_dict = OmegaConf.to_container(loaded, resolve=True)
        if not isinstance(loaded_dict, dict):
            raise ValueError("Lowered synapse YAML did not produce a mapping")
        lowered_spec: dict[str, Any] = {str(key): value for key, value in loaded_dict.items()}

        code = emit_model_code_from_synapse_spec(lowered_spec, class_name=class_name)
        generated_py_path.write_text(code, encoding="utf-8")

        model_cls = _load_generated_class(generated_py_path, class_name)

        tokenizer_obj = _load_tokenizer(tokenizer_source, fallback_repo_id=tokenizer_fallback)
        hf_model: Any = AutoModelForCausalLM.from_pretrained(
            str(resolved_hf_model_dir), local_files_only=True, dtype=resolved_dtype
        )
        hf = hf_model.to(resolved_device).eval()
        if hasattr(hf, "generation_config"):
            hf.generation_config.do_sample = False
            hf.generation_config.top_p = None
            hf.generation_config.top_k = None

        if len(prompts) > 1:
            tokenizer_obj.padding_side = "left"
            if tokenizer_obj.pad_token_id is None:
                if tokenizer_obj.eos_token_id is None:
                    raise ValueError(
                        "Tokenizer has no pad token and no eos token; cannot batch prompts with padding."
                    )
                tokenizer_obj.pad_token = tokenizer_obj.eos_token
        inputs = tokenizer_obj(
            prompts,
            return_tensors="pt",
            padding=(len(prompts) > 1),
        ).to(resolved_device)
        input_ids = inputs["input_ids"]
        model_inputs = lowered_spec.get("model", {}).get("inputs", {})
        model_input_names = (
            set(model_inputs.keys()) if isinstance(model_inputs, dict) else {"input_ids"}
        )
        syn_mask_key = (
            "attn_mask"
            if "attn_mask" in model_input_names
            else ("attention_mask" if "attention_mask" in model_input_names else None)
        )
        hf_inputs: dict[str, Any] = {"input_ids": input_ids}
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            hf_inputs["attention_mask"] = attention_mask
        use_mask_for_syn = bool(attention_mask is not None)
        if use_mask_for_syn and len(prompts) == 1:
            # Single unpadded prompt does not need an explicit mask on the Synapse side.
            # Keeping it unset avoids extra per-step mask materialization in KV decode.
            use_mask_for_syn = bool((attention_mask == 0).any())

        def _run_hf_generate(model: Any = hf) -> torch.Tensor:
            return model.generate(
                **hf_inputs,
                max_new_tokens=max(1, max_len - int(input_ids.shape[1])),
                eos_token_id=tokenizer_obj.eos_token_id,
                pad_token_id=tokenizer_obj.eos_token_id,
            )

        hf_gen, hf_time = _time_generate("HF", _run_hf_generate)
        hf_forward_inputs = dict(hf_inputs)
        if attention_mask is not None:
            # Align forward-logit comparison with decoder generation semantics under padding.
            pos_ids = attention_mask.to(torch.long).cumsum(dim=-1) - 1
            pos_ids = pos_ids.masked_fill(attention_mask == 0, 1)
            hf_forward_inputs["position_ids"] = pos_ids
        with torch.no_grad():
            hf_logits = hf(**hf_forward_inputs, use_cache=False).logits

        del hf
        _cleanup(resolved_device)

        state_dict = _load_state_dict(
            safetensors_files,
            device=resolved_device,
            dtype=resolved_dtype,
            strip_model_prefix=strip_model_prefix,
        )
        syn = model_cls.from_state_dict(state_dict).to(resolved_device).eval()

        def _run_syn_generate(model: Any = syn) -> torch.Tensor:
            generate_kwargs: dict[str, Any] = {
                "eos_token_id": tokenizer_obj.eos_token_id,
                "max_len": max_len,
            }
            if use_mask_for_syn and attention_mask is not None:
                if syn_mask_key == "attn_mask":
                    generate_kwargs["attn_mask"] = attention_mask
                elif syn_mask_key == "attention_mask":
                    generate_kwargs["attention_mask"] = attention_mask
            return model.generate(input_ids, **generate_kwargs)

        syn_gen, syn_time = _time_generate("AxonDerived", _run_syn_generate)
        syn_inputs: dict[str, Any] = {"input_ids": input_ids}
        if use_mask_for_syn and attention_mask is not None and syn_mask_key is not None:
            syn_inputs[syn_mask_key] = attention_mask
        with torch.no_grad():
            syn_logits = _extract_logits(syn(**syn_inputs))

        gen_hf = int(hf_gen.shape[1] - input_ids.shape[1])
        gen_syn = int(syn_gen.shape[1] - input_ids.shape[1])

        if syn_logits.device != hf_logits.device:
            syn_logits = syn_logits.to(hf_logits.device)
        diff = (syn_logits.float() - hf_logits.float()).abs()
        mean_diff = float(diff.mean())
        max_diff = float(diff.max())
        last_max_diff = float(diff[:, -1, :].max())
        top1_eq = bool((syn_logits[:, -1, :].argmax(-1) == hf_logits[:, -1, :].argmax(-1)).all())

        masked_mean_diff: float | None = None
        masked_max_diff: float | None = None
        masked_last_max_diff: float | None = None
        masked_top1_eq: bool | None = None
        if attention_mask is not None:
            mask_bool = attention_mask.to(torch.bool)
            valid = mask_bool.unsqueeze(-1).expand_as(diff)
            valid_count = int(valid.sum().item())
            if valid_count > 0:
                valid_diff = diff[valid]
                masked_mean_diff = float(valid_diff.mean())
                masked_max_diff = float(valid_diff.max())
            else:
                masked_mean_diff = 0.0
                masked_max_diff = 0.0

            attn_bool = attention_mask.to(torch.bool)
            rev_last = torch.argmax(attn_bool.flip(dims=[1]).to(torch.long), dim=1)
            lengths = (attn_bool.shape[1] - 1) - rev_last
            any_valid = attn_bool.any(dim=1)
            lengths = torch.where(lengths >= 0, lengths, torch.zeros_like(lengths))
            lengths = torch.where(any_valid, lengths, torch.zeros_like(lengths))
            b_idx = torch.arange(attention_mask.shape[0], device=attention_mask.device)
            syn_last = syn_logits[b_idx, lengths]
            hf_last = hf_logits[b_idx, lengths]
            masked_last_max_diff = float((syn_last.float() - hf_last.float()).abs().max())
            masked_top1_eq = bool((syn_last.argmax(-1) == hf_last.argmax(-1)).all())

        if len(safetensors_files) == 1:
            safetensors_desc = str(safetensors_files[0])
        else:
            safetensors_desc = (
                f"{len(safetensors_files)} shards (first: {safetensors_files[0].name})"
            )

        print(f"Axon file:      {axon_file}")
        print(f"Safetensors:    {safetensors_desc}")
        print(f"Weights input:  {weights_path}")
        print(f"HF model dir:   {resolved_hf_model_dir}")
        print(f"Tokenizer:      {tokenizer_source}")
        print(f"Device:         {resolved_device}")
        print(f"Prompts:        {len(prompts)}")
        print()
        print(
            f"HF:             {hf_time:.4f}s total, {gen_hf / max(hf_time, 1e-9):.2f} tok/s, generated={gen_hf}"
        )
        print(
            f"Axon-derived:   {syn_time:.4f}s total, {gen_syn / max(syn_time, 1e-9):.2f} tok/s, generated={gen_syn}"
        )
        print(f"Speed ratio (Axon/HF): {syn_time / max(hf_time, 1e-9):.3f}x")
        print()
        for idx, prompt in enumerate(prompts):
            print(f"Prompt[{idx}]: {prompt!r}")
            print("HF completion:")
            print(tokenizer_obj.decode(hf_gen[idx].tolist(), skip_special_tokens=True))
            print("Axon-derived completion:")
            print(tokenizer_obj.decode(syn_gen[idx].tolist(), skip_special_tokens=True))
            print()
        print(
            "Logits diff (raw) | mean/max/last_max/top1_eq:",
            mean_diff,
            max_diff,
            last_max_diff,
            top1_eq,
        )
        if attention_mask is not None:
            print(
                "Logits diff (masked) | mean/max/last_max/top1_eq:",
                masked_mean_diff,
                masked_max_diff,
                masked_last_max_diff,
                masked_top1_eq,
            )

        result = {
            "hf_time": hf_time,
            "axon_time": syn_time,
            "speed_ratio_axon_over_hf": syn_time / max(hf_time, 1.0e-9),
            "mean_diff": mean_diff,
            "max_diff": max_diff,
            "last_max_diff": last_max_diff,
            "top1_eq": top1_eq,
            "masked_mean_diff": masked_mean_diff,
            "masked_max_diff": masked_max_diff,
            "masked_last_max_diff": masked_last_max_diff,
            "masked_top1_eq": masked_top1_eq,
            "prompts": prompts,
            "generated_hf": hf_gen,
            "generated_axon": syn_gen,
        }

        del syn
        del state_dict
        _cleanup(resolved_device)
        return result


__all__ = ["run_axon_test"]
