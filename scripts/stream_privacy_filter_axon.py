#!/usr/bin/env python3
"""Stream privacy-filter predictions: Axon compiled (MLX backend).

Compiles the privacy-filter .axon model through the full Axon pipeline,
loads real checkpoint weights, runs a forward pass on sample text with PII,
and streams BIOES label predictions to the terminal token-by-token.

Designed for side-by-side screen recording comparison with the vanilla
HuggingFace counterpart (stream_privacy_filter_hf.py).

Usage:
    python scripts/stream_privacy_filter_axon.py [OPTIONS]

Options:
    --text TEXT        Input text to classify (default: built-in PII sample)
    --model-dir DIR    Local model directory with safetensors + config + tokenizer
    --no-optimize      Disable graph optimization (SDPA, intrinsics)
    --seq-len N        Pad/truncate input to N tokens (default: 0 = no pad)
    --stream-delay S   Delay between tokens in seconds (default: 0.015)
    --no-color         Disable ANSI colour output
    --no-stream        Print all tokens at once (no streaming delay)
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

MODEL_REPO_ID = "openai/privacy-filter"
AXON_FILE = REPO_ROOT / "synapse" / "models" / "privacy-filter" / "generic-privacy-filter.axon"

DEFAULT_TEXT = (
    "Hi, my name is John Smith. You can reach me at john.smith@example.com "
    "or call me at +1-555-123-4567. I work at Acme Corp and my birthday is "
    "January 15, 1990. My bank account is 1234567890."
)

# ── ANSI colours ───────────────────────────────────────────────────────────

_RESET = "\033[0m"
_DIM = "\033[2m"
_BOLD = "\033[1m"

_ENTITY_PALETTE = [
    "\033[31m", "\033[32m", "\033[33m", "\033[34m",
    "\033[35m", "\033[36m", "\033[91m", "\033[92m",
    "\033[93m", "\033[94m", "\033[95m", "\033[96m",
]


def _entity_color(entity_type: str) -> str:
    if not entity_type:
        return _DIM
    h = sum(ord(c) for c in entity_type)
    return _ENTITY_PALETTE[h % len(_ENTITY_PALETTE)]


def _parse_label(label: str) -> tuple[str, str]:
    if label == "O" or label == "":
        return "O", ""
    parts = label.split("-", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return label, ""


# ── Model directory resolution ─────────────────────────────────────────────

def _resolve_model_dir(model_dir: str | None) -> Path:
    if model_dir:
        p = Path(model_dir).resolve()
        if p.is_dir():
            return p
        raise FileNotFoundError(f"Model directory not found: {p}")
    local = REPO_ROOT / "models" / "privacy-filter"
    if local.is_dir() and any(local.glob("*.safetensors")):
        return local
    raise FileNotFoundError(
        f"Model weights not found at {local}. "
        f"Download with: huggingface-cli download {MODEL_REPO_ID} --local-dir {local}"
    )


def _load_config(model_dir: Path) -> dict:
    config_path = model_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            return json.load(f)
    return {}


def _convert_hf_to_opf_state_dict(model_dir: Path) -> dict:
    """Convert HF checkpoint keys to opf layout expected by the .axon model.

    HF naming:  model.embed_tokens.weight, model.layers.N.self_attn.{q,k,v,o}_proj.*
                model.layers.N.mlp.experts.{gate_up,down}_proj*, model.norm.weight, score.weight
    opf naming: embedding.weight, block.N.attn.{qkv,out,sinks,norm}.*, block.N.mlp.{gate,swiglu,out,norm}.*,
                norm.scale, unembedding.weight
    """
    import mlx.core as mx
    import torch
    from safetensors import safe_open

    raw = {}
    for path in sorted(model_dir.glob("*.safetensors")):
        with safe_open(str(path), framework="pt") as f:
            for key in f.keys():
                raw[str(key)] = f.get_tensor(key)

    out: dict[str, mx.array] = {}

    def _to_mlx(t: torch.Tensor) -> mx.array:
        if t.dtype == torch.bfloat16:
            return mx.array(t.float().numpy()).astype(mx.bfloat16)
        return mx.array(t.numpy())

    out["embedding.weight"] = _to_mlx(raw["model.embed_tokens.weight"])
    out["norm.scale"] = _to_mlx(raw["model.norm.weight"])
    out["unembedding.weight"] = _to_mlx(raw["score.weight"])

    num_layers = 8
    for n in range(num_layers):
        p_hf = f"model.layers.{n}"
        p_opf = f"block.{n}"

        out[f"{p_opf}.attn.norm.scale"] = _to_mlx(raw[f"{p_hf}.input_layernorm.weight"])
        out[f"{p_opf}.mlp.norm.scale"] = _to_mlx(raw[f"{p_hf}.post_attention_layernorm.weight"])

        q_w = raw[f"{p_hf}.self_attn.q_proj.weight"]
        k_w = raw[f"{p_hf}.self_attn.k_proj.weight"]
        v_w = raw[f"{p_hf}.self_attn.v_proj.weight"]
        out[f"{p_opf}.attn.qkv.weight"] = _to_mlx(torch.cat([q_w, k_w, v_w], dim=0))

        q_b = raw[f"{p_hf}.self_attn.q_proj.bias"]
        k_b = raw[f"{p_hf}.self_attn.k_proj.bias"]
        v_b = raw[f"{p_hf}.self_attn.v_proj.bias"]
        out[f"{p_opf}.attn.qkv.bias"] = _to_mlx(torch.cat([q_b, k_b, v_b], dim=0))

        out[f"{p_opf}.attn.out.weight"] = _to_mlx(raw[f"{p_hf}.self_attn.o_proj.weight"])
        out[f"{p_opf}.attn.out.bias"] = _to_mlx(raw[f"{p_hf}.self_attn.o_proj.bias"])
        out[f"{p_opf}.attn.sinks"] = _to_mlx(raw[f"{p_hf}.self_attn.sinks"])

        out[f"{p_opf}.mlp.gate.weight"] = _to_mlx(raw[f"{p_hf}.mlp.router.weight"])
        out[f"{p_opf}.mlp.gate.bias"] = _to_mlx(raw[f"{p_hf}.mlp.router.bias"])

        out[f"{p_opf}.mlp.swiglu.weight"] = _to_mlx(raw[f"{p_hf}.mlp.experts.gate_up_proj"])
        out[f"{p_opf}.mlp.swiglu.bias"] = _to_mlx(raw[f"{p_hf}.mlp.experts.gate_up_proj_bias"])
        out[f"{p_opf}.mlp.out.weight"] = _to_mlx(raw[f"{p_hf}.mlp.experts.down_proj"])
        out[f"{p_opf}.mlp.out.bias"] = _to_mlx(raw[f"{p_hf}.mlp.experts.down_proj_bias"])

    return out


def _load_id2label(model_dir: Path) -> dict[int, str]:
    config_path = model_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            cfg = json.load(f)
        id2label = cfg.get("id2label")
        if id2label:
            return {int(k): v for k, v in id2label.items()}
        num_labels = cfg.get("num_labels", 33)
        return {i: f"LABEL_{i}" for i in range(num_labels)}
    return {i: f"LABEL_{i}" for i in range(33)}


# ── Axon compilation ────────────────────────────────────────────────────────

def _compile_and_load(model_dir: Path, optimize: bool):
    from synapse.axon import (
        resolve_axon_program_from_path,
        normalize_closed_axon_file,
        elaborate_closed_axon_file,
        flatten_closed_axon_file,
        typecheck2_flat_axon_file,
        lower_axon_program_to_graph_ir,
        optimize_graph_program,
        GraphOptimizeConfig,
    )
    from synapse.axon.codegen2_mlx.core import emit_model_code_from_graph_ir
    from server.loading import MLX_INTRINSICS_SPEC

    model_config = _load_config(model_dir)

    print("  Compiling .axon -> graph IR -> MLX code...")
    report = resolve_axon_program_from_path(AXON_FILE, strict=False)
    program = report.ast
    program = normalize_closed_axon_file(program)
    program = elaborate_closed_axon_file(program)
    program = flatten_closed_axon_file(program)
    program = typecheck2_flat_axon_file(program)
    graph = lower_axon_program_to_graph_ir(program)
    if optimize:
        # Named full serving grant (mirrors server.loading.MLX_INTRINSICS_SPEC
        # via import below): a bare "codegen2-mlx" no longer covers
        # __mlx_rope, which is opt-in-only because its emitted caches and
        # decode-offset machinery are serving-specific.
        graph = optimize_graph_program(
            graph, config=GraphOptimizeConfig(backend_intrinsics=MLX_INTRINSICS_SPEC)
        )
    code = emit_model_code_from_graph_ir(graph, model_config=model_config)

    out_path = Path("/tmp") / "pf_axon_stream_model.py"
    out_path.write_text(code)

    spec = importlib.util.spec_from_file_location("pf_axon_model", str(out_path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["pf_axon_model"] = mod
    spec.loader.exec_module(mod)

    model_cls = getattr(mod, "AxonMlxModel", None) or getattr(mod, "AxonGeneratedModel", None)
    if model_cls is None:
        raise RuntimeError("Generated model class not found")

    # Patch nn.Module.__setattr__ bug (same as bench_mlx_privacy_filter.py)
    _orig = model_cls.__setattr__
    def _patched(self, key, value):
        try:
            _orig(self, key, value)
        except AttributeError:
            object.__setattr__(self, key, value)
    model_cls.__setattr__ = _patched

    print("  Loading weights from safetensors...")
    state_dict = _convert_hf_to_opf_state_dict(model_dir)
    model = model_cls(state_dict, config=model_config, target_dtype="bfloat16")
    model.eval()

    return model


# ── Tokenizer ───────────────────────────────────────────────────────────────

def _load_tokenizer(model_dir: Path):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True, trust_remote_code=True)


# ── Forward pass ────────────────────────────────────────────────────────────

def _tokenize(tokenizer, text: str, seq_len: int):
    import mlx.core as mx

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    ids = inputs["input_ids"][0].tolist()
    attn_mask = [1] * len(ids)
    if seq_len > 0:
        if len(ids) < seq_len:
            pad_id = tokenizer.pad_token_id
            if pad_id is None:
                pad_id = tokenizer.eos_token_id
            if pad_id is None:
                pad_id = 0
            ids = ids + [pad_id] * (seq_len - len(ids))
            attn_mask = attn_mask + [0] * (seq_len - len(attn_mask))
        else:
            ids = ids[:seq_len]
            attn_mask = attn_mask[:seq_len]
    input_ids = mx.array([ids], dtype=mx.int32)
    attention_mask = mx.array([attn_mask], dtype=mx.int32)
    return input_ids, attention_mask, ids


def _run_forward(model, tokenizer, text: str, seq_len: int):
    import mlx.core as mx

    input_ids, attention_mask, raw_ids = _tokenize(tokenizer, text, seq_len)

    t0 = time.perf_counter()
    out = model(input_ids, attn_mask=attention_mask)
    mx.eval(out)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    logits = out if not isinstance(out, dict) else out.get("logits", out)
    if isinstance(logits, (list, tuple)):
        logits = logits[0]
    preds = mx.argmax(logits, axis=-1)[0].astype(mx.int32).tolist()

    tokens = tokenizer.convert_ids_to_tokens(raw_ids)

    return tokens, preds, elapsed_ms, input_ids


# ── Streaming display ───────────────────────────────────────────────────────

def _stream_results(
    tokens: list[str],
    preds: list[int],
    id2label: dict,
    elapsed_ms: float,
    *,
    stream_delay: float,
    use_color: bool,
    use_stream: bool,
):
    n = len(tokens)
    entity_types_seen: dict[str, str] = {}

    print(f"\n{_BOLD}Predictions{_RESET} ({n} tokens):\n")

    for i, (tok, pred_id) in enumerate(zip(tokens, preds)):
        label = id2label.get(pred_id, f"LABEL_{pred_id}")
        tag, etype = _parse_label(label)

        if use_color:
            if tag == "O":
                tok_str = f"{_DIM}{tok}{_RESET}"
                label_str = f"{_DIM}O{_RESET}"
            else:
                if etype not in entity_types_seen:
                    entity_types_seen[etype] = _entity_color(etype)
                color = entity_types_seen[etype]
                tok_str = f"{color}{tok}{_RESET}"
                label_str = f"{color}{label}{_RESET}"
        else:
            tok_str = tok
            label_str = label

        print(f"  {tok_str:<20s}  {label_str}")

        if use_stream and stream_delay > 0:
            time.sleep(stream_delay)

    print(f"\n{'─' * 60}")
    tps = n / (elapsed_ms / 1000) if elapsed_ms > 0 else float("inf")
    print(f"  Forward: {elapsed_ms:.2f} ms | {n} tokens | {tps:.0f} tok/s")
    print(f"{'═' * 60}")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--text", default=DEFAULT_TEXT, help="Input text to classify")
    parser.add_argument("--model-dir", default=None, help="Local model directory with safetensors + config")
    parser.add_argument("--no-optimize", action="store_true", help="Disable graph optimization")
    parser.add_argument("--seq-len", type=int, default=0, help="Pad/truncate input to N tokens")
    parser.add_argument("--stream-delay", type=float, default=0.0, help="Artificial delay between tokens (s)")
    parser.add_argument("--no-color", action="store_true", help="Disable ANSI colours")
    parser.add_argument("--no-stream", action="store_true", help="Print all at once")
    args = parser.parse_args()

    use_color = not args.no_color and sys.stdout.isatty()
    optimize = not args.no_optimize

    model_dir = _resolve_model_dir(args.model_dir)

    print(f"{'═' * 60}")
    print(f"  {_BOLD}Privacy Filter — Axon compiled (MLX backend){_RESET}")
    print(f"{'═' * 60}")
    print(f"\nInput:\n  \"{args.text}\"\n")

    model = _compile_and_load(model_dir, optimize=optimize)
    print(f"  Model loaded (optimize={'on' if optimize else 'off'})")

    id2label = _load_id2label(model_dir)
    print(f"  Labels: {len(id2label)} ({list(id2label.values())[:8]}...)\n")

    tokenizer = _load_tokenizer(model_dir)

    print("Computing forward pass...")
    tokens, preds, elapsed_ms, input_ids = _run_forward(model, tokenizer, args.text, args.seq_len)
    print(f"  Done in {elapsed_ms:.2f} ms\n")

    _stream_results(
        tokens, preds, id2label, elapsed_ms,
        stream_delay=args.stream_delay,
        use_color=use_color,
        use_stream=not args.no_stream,
    )


if __name__ == "__main__":
    main()
