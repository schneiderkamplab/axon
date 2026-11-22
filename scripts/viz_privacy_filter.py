#!/usr/bin/env python3
"""Visualize privacy-filter: HF vs Axon speed + prediction parity.

Both backends get warmup.  Timed warm passes.  Animated race shows speed.
Token-by-token label comparison shows prediction parity.

Usage:
    python scripts/viz_privacy_filter.py [OPTIONS]

Options:
    --text TEXT        Input text (default: built-in PII sample)
    --seq-len N        Pad/truncate to N tokens (default: 512)
    --model-dir DIR    Local model directory
    --no-optimize      Disable Axon graph optimization (default: on for correctness)
    --no-color         Disable ANSI colour
    --replay-speed S   Replay animation speed multiplier (default: 1.0)
    --show-all         Show all tokens (default: first 60)
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

# ── ANSI ───────────────────────────────────────────────────────────────────

_RESET = "\033[0m"
_DIM = "\033[2m"
_BOLD = "\033[1m"
_RED = "\033[31m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"

_BAR = "█"
_LIGHT = "░"

_SPINNER = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]


# ── Model dir / config ──────────────────────────────────────────────────────

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
        f"Download: python -c \"from huggingface_hub import snapshot_download; "
        f"snapshot_download('{MODEL_REPO_ID}', local_dir='{local}')\""
    )


def _load_config(model_dir: Path) -> dict:
    p = model_dir / "config.json"
    if p.exists():
        return json.loads(p.read_text())
    return {}


def _load_id2label(model_dir: Path) -> dict[int, str]:
    p = model_dir / "config.json"
    if p.exists():
        cfg = json.loads(p.read_text())
        id2label = cfg.get("id2label")
        if id2label:
            return {int(k): v for k, v in id2label.items()}
    return {i: f"LABEL_{i}" for i in range(33)}


# ── HF→opf key conversion ──────────────────────────────────────────────────

def _convert_hf_to_opf(model_dir: Path) -> dict:
    import mlx.core as mx
    import torch
    from safetensors import safe_open

    raw = {}
    for path in sorted(model_dir.glob("*.safetensors")):
        with safe_open(str(path), framework="pt") as f:
            for key in f.keys():
                raw[str(key)] = f.get_tensor(key)

    def _to_mlx(t):
        if t.dtype == torch.bfloat16:
            return mx.array(t.float().numpy()).astype(mx.bfloat16)
        return mx.array(t.numpy())

    out: dict[str, mx.array] = {}
    out["embedding.weight"] = _to_mlx(raw["model.embed_tokens.weight"])
    out["norm.scale"] = _to_mlx(raw["model.norm.weight"])
    out["unembedding.weight"] = _to_mlx(raw["score.weight"])
    for n in range(8):
        hf = f"model.layers.{n}"
        opf = f"block.{n}"
        out[f"{opf}.attn.norm.scale"] = _to_mlx(raw[f"{hf}.input_layernorm.weight"])
        out[f"{opf}.mlp.norm.scale"] = _to_mlx(raw[f"{hf}.post_attention_layernorm.weight"])
        out[f"{opf}.attn.qkv.weight"] = _to_mlx(torch.cat([
            raw[f"{hf}.self_attn.q_proj.weight"],
            raw[f"{hf}.self_attn.k_proj.weight"],
            raw[f"{hf}.self_attn.v_proj.weight"],
        ], dim=0))
        out[f"{opf}.attn.qkv.bias"] = _to_mlx(torch.cat([
            raw[f"{hf}.self_attn.q_proj.bias"],
            raw[f"{hf}.self_attn.k_proj.bias"],
            raw[f"{hf}.self_attn.v_proj.bias"],
        ], dim=0))
        out[f"{opf}.attn.out.weight"] = _to_mlx(raw[f"{hf}.self_attn.o_proj.weight"])
        out[f"{opf}.attn.out.bias"] = _to_mlx(raw[f"{hf}.self_attn.o_proj.bias"])
        out[f"{opf}.attn.sinks"] = _to_mlx(raw[f"{hf}.self_attn.sinks"])
        out[f"{opf}.mlp.gate.weight"] = _to_mlx(raw[f"{hf}.mlp.router.weight"])
        out[f"{opf}.mlp.gate.bias"] = _to_mlx(raw[f"{hf}.mlp.router.bias"])
        out[f"{opf}.mlp.swiglu.weight"] = _to_mlx(raw[f"{hf}.mlp.experts.gate_up_proj"])
        out[f"{opf}.mlp.swiglu.bias"] = _to_mlx(raw[f"{hf}.mlp.experts.gate_up_proj_bias"])
        out[f"{opf}.mlp.out.weight"] = _to_mlx(raw[f"{hf}.mlp.experts.down_proj"])
        out[f"{opf}.mlp.out.bias"] = _to_mlx(raw[f"{hf}.mlp.experts.down_proj_bias"])
    return out


# ── Tokenizer ──────────────────────────────────────────────────────────────

def _tokenize(tokenizer, text: str, seq_len: int):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    ids = inputs["input_ids"][0].tolist()
    attn_mask = [1] * len(ids)
    if seq_len > 0:
        if len(ids) < seq_len:
            pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
            ids = ids + [pad_id] * (seq_len - len(ids))
            attn_mask = attn_mask + [0] * (seq_len - len(attn_mask))
        else:
            ids = ids[:seq_len]
            attn_mask = attn_mask[:seq_len]
    tokens = tokenizer.convert_ids_to_tokens(ids)
    return ids, attn_mask, tokens


# ── HF backend ─────────────────────────────────────────────────────────────

def _load_hf(model_dir: Path, device: str):
    import torch
    from transformers import AutoConfig, AutoModelForTokenClassification

    config = AutoConfig.from_pretrained(str(model_dir), trust_remote_code=True)
    dev = torch.device(device)
    model = AutoModelForTokenClassification.from_pretrained(
        str(model_dir), local_files_only=True, torch_dtype=torch.bfloat16,
        config=config, trust_remote_code=True,
    ).to(dev).eval()
    return model, dev


def _hf_fwd(model, ids, attn_mask, dev):
    import torch
    iids = torch.tensor([ids], dtype=torch.long, device=dev)
    mask = torch.tensor([attn_mask], dtype=torch.long, device=dev)
    with torch.no_grad():
        out = model(input_ids=iids, attention_mask=mask)
    if dev.type == "mps": torch.mps.synchronize()
    elif dev.type == "cuda": torch.cuda.synchronize()
    return out.logits.argmax(dim=-1)[0].cpu().tolist()


# ── Axon backend ───────────────────────────────────────────────────────────

def _load_axon(model_dir: Path, optimize: bool):
    import mlx.core as mx
    from synapse.axon import (
        resolve_axon_program_from_path, normalize_closed_axon_file,
        elaborate_closed_axon_file, flatten_closed_axon_file,
        typecheck2_flat_axon_file, lower_axon_program_to_graph_ir,
        optimize_graph_program, GraphOptimizeConfig,
    )
    from synapse.axon.codegen2_mlx.core import emit_model_code_from_graph_ir
    from server.loading import MLX_INTRINSICS_SPEC

    cfg = _load_config(model_dir)
    report = resolve_axon_program_from_path(AXON_FILE, strict=False)
    prog = report.ast
    prog = normalize_closed_axon_file(prog)
    prog = elaborate_closed_axon_file(prog)
    prog = flatten_closed_axon_file(prog)
    prog = typecheck2_flat_axon_file(prog)
    graph = lower_axon_program_to_graph_ir(prog)
    if optimize:
        # Named full serving grant (see server.loading.MLX_INTRINSICS_SPEC):
        # bare "codegen2-mlx" no longer grants __mlx_rope (opt-in-only,
        # serving-specific state).
        graph = optimize_graph_program(graph, config=GraphOptimizeConfig(backend_intrinsics=MLX_INTRINSICS_SPEC))
    code = emit_model_code_from_graph_ir(graph, model_config=cfg)

    out_path = Path("/tmp") / "pf_viz_model.py"
    out_path.write_text(code)
    spec = importlib.util.spec_from_file_location("pf_viz_model", str(out_path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["pf_viz_model"] = mod
    spec.loader.exec_module(mod)

    cls = getattr(mod, "AxonMlxModel", None) or getattr(mod, "AxonGeneratedModel", None)
    _orig = cls.__setattr__
    def _patched(self, k, v):
        try: _orig(self, k, v)
        except AttributeError: object.__setattr__(self, k, v)
    cls.__setattr__ = _patched

    sd = _convert_hf_to_opf(model_dir)
    model = cls(sd, config=cfg, target_dtype="bfloat16")
    model.eval()
    return model


def _axon_fwd(model, ids, attn_mask):
    import mlx.core as mx
    iids = mx.array([ids], dtype=mx.int32)
    mask = mx.array([attn_mask], dtype=mx.int32)
    out = model(iids, attn_mask=mask)
    mx.eval(out)
    logits = out if not isinstance(out, dict) else out.get("logits", out)
    if isinstance(logits, (list, tuple)):
        logits = logits[0]
    return mx.argmax(logits, axis=-1)[0].astype(mx.int32).tolist()


# ── Animated race ──────────────────────────────────────────────────────────

def _animate_race(hf_ms: float, axon_ms: float, use_color: bool, replay_speed: float):
    hf_c = _RED if use_color else ""
    axon_c = _GREEN if use_color else ""
    r = _RESET if use_color else ""
    dim = _DIM if use_color else ""
    bold = _BOLD if use_color else ""
    bar_w = 40

    print(f"\n{'═' * 64}")
    print(f"  {bold}Speed replay{r}")
    print(f"{'═' * 64}\n")

    max_ms = max(hf_ms, axon_ms)
    total_s = max_ms / 1000 * replay_speed
    frames = max(1, int(total_s / 0.03))
    hf_ratio = hf_ms / max_ms
    axon_ratio = axon_ms / max_ms

    for frame in range(frames + 1):
        p = frame / frames
        elapsed_s = p * total_s
        hf_fill = min(p / hf_ratio, 1.0) if hf_ratio > 0 else 1.0
        axon_fill = min(p / axon_ratio, 1.0) if axon_ratio > 0 else 1.0
        hf_b = int(hf_fill * bar_w)
        axon_b = int(axon_fill * bar_w)

        lines = [
            f"  {'HF (torch)':<14s}  {hf_c}{_BAR * hf_b}{_LIGHT * (bar_w - hf_b)}{r}  {'done' if hf_fill >= 1.0 else dim+'running...'+r}",
            f"  {'Axon (MLX)':<14s}  {axon_c}{_BAR * axon_b}{_LIGHT * (bar_w - axon_b)}{r}  {'done' if axon_fill >= 1.0 else dim+'running...'+r}",
            f"  {dim}t = {elapsed_s:.2f}s{r}",
        ]
        for _ in lines:
            sys.stdout.write("\033[F\033[K")
        for line in lines:
            sys.stdout.write(line + "\n")
        sys.stdout.flush()
        if frame < frames:
            time.sleep(0.03)

    print(f"\n  {bold}HF: {hf_ms:.0f}ms  |  Axon: {axon_ms:.0f}ms  |  {hf_ms/axon_ms:.1f}x faster{r}")
    print(f"{'═' * 64}")


# ── Token comparison ───────────────────────────────────────────────────────

def _print_comparison(tokens, hf_preds, axon_preds, id2label, use_color, show_all=False):
    n = len(tokens)
    mismatches = sum(1 for h, a in zip(hf_preds, axon_preds) if h != a)
    match = n - mismatches

    bold = _BOLD if use_color else ""
    dim = _DIM if use_color else ""
    r = _RESET if use_color else ""
    red = _RED if use_color else ""
    green = _GREEN if use_color else ""
    yellow = _YELLOW if use_color else ""

    print(f"\n{'═' * 64}")
    print(f"  {bold}Prediction parity — HF vs Axon{r}")
    print(f"{'═' * 64}\n")

    if mismatches == 0:
        print(f"  {green}{bold}All {n} tokens match ✓{r}\n")
    else:
        print(f"  {green}{match}{r}/{n} match  |  {red}{mismatches}{r} differ\n")

    show = n if show_all else min(n, 60)

    print(f"  {'#':>3s}  {'Token':<18s}  {'HF':<22s}  {'Axon':<22s}  Match")
    print(f"  {'─' * 70}")

    for i in range(show):
        tok = tokens[i]
        hf_label = id2label.get(hf_preds[i], f"LABEL_{hf_preds[i]}")
        axon_label = id2label.get(axon_preds[i], f"LABEL_{axon_preds[i]}")
        is_match = hf_preds[i] == axon_preds[i]

        if use_color:
            tok_str = f"{dim}{tok}{r}" if hf_label == "O" else f"{bold}{tok}{r}"
            hf_str = f"{dim}{hf_label}{r}" if hf_label == "O" else f"{yellow}{hf_label}{r}"
            axon_str = f"{dim}{axon_label}{r}" if axon_label == "O" else f"{yellow}{axon_label}{r}"
            match_str = f"{green}✓{r}" if is_match else f"{red}✗{r}"
        else:
            tok_str = tok
            hf_str = hf_label
            axon_str = axon_label
            match_str = "✓" if is_match else "✗"

        print(f"  {i:>3d}  {tok_str:<18s}  {hf_str:<22s}  {axon_str:<22s}  {match_str}")

    if n > show:
        rest_mismatch = sum(1 for h, a in zip(hf_preds[show:], axon_preds[show:]) if h != a)
        print(f"  {dim}... ({n - show} more, {rest_mismatch} differ){r}")

    print(f"\n  {bold}{'EXACT MATCH' if mismatches == 0 else f'{match}/{n} tokens match'}{r}")
    print(f"{'═' * 64}")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--no-optimize", action="store_true", default=True)
    parser.add_argument("--no-color", action="store_true")
    parser.add_argument("--replay-speed", type=float, default=1.0)
    parser.add_argument("--show-all", action="store_true")
    args = parser.parse_args()

    use_color = not args.no_color and sys.stdout.isatty()
    bold = _BOLD if use_color else ""
    r = _RESET if use_color else ""
    dim = _DIM if use_color else ""

    import torch
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

    model_dir = _resolve_model_dir(args.model_dir)
    id2label = _load_id2label(model_dir)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True, trust_remote_code=True)
    ids, attn_mask, tokens = _tokenize(tokenizer, args.text, args.seq_len)
    n = len(ids)

    print(f"{'═' * 64}")
    print(f"  {bold}Privacy Filter — HF vs Axon{r}")
    print(f"{'═' * 64}")
    print(f"  Tokens: {n}  |  Device: {device}  |  Optimize: {'off' if args.no_optimize else 'on'}")
    print()

    # ── Load + warmup ──
    print(f"{bold}Loading + warmup...{r}\n")

    print("  HF (transformers)...")
    hf_model, hf_dev = _load_hf(model_dir, device)
    _hf_fwd(hf_model, ids, attn_mask, hf_dev)
    print(f"  {dim}done{r}\n")

    print("  Axon (compile + load)...")
    axon_model = _load_axon(model_dir, not args.no_optimize)
    _axon_fwd(axon_model, ids, attn_mask)
    import mlx.core as mx; mx.clear_cache()
    print(f"  {dim}done{r}\n")

    # ── Timed runs ──
    print(f"{bold}Timed forward passes (warm):{r}\n")

    t0 = time.perf_counter()
    hf_preds = _hf_fwd(hf_model, ids, attn_mask, hf_dev)
    hf_ms = (time.perf_counter() - t0) * 1000
    c = _GREEN if use_color else ""
    print(f"  HF:    {c}{hf_ms:.0f} ms{r}")

    t0 = time.perf_counter()
    axon_preds = _axon_fwd(axon_model, ids, attn_mask)
    axon_ms = (time.perf_counter() - t0) * 1000
    print(f"  Axon:  {c}{axon_ms:.0f} ms{r}\n")

    # ── Animated race ──
    _animate_race(hf_ms, axon_ms, use_color, args.replay_speed)

    # ── Token comparison ──
    _print_comparison(tokens, hf_preds, axon_preds, id2label, use_color, args.show_all)

    # Cleanup
    del hf_model, axon_model
    import gc; gc.collect()
    try:
        torch.mps.empty_cache()
        mx.clear_cache()
    except Exception:
        pass


if __name__ == "__main__":
    main()
