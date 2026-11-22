#!/usr/bin/env python3
"""Stream privacy-filter predictions: vanilla HuggingFace transformers.

Loads the openai/privacy-filter model via transformers, runs a forward pass
on sample text with PII, and streams BIOES label predictions to the terminal
token-by-token.  Designed for side-by-side screen recording comparison with
the Axon-compiled counterpart (stream_privacy_filter_axon.py).

Usage:
    python scripts/stream_privacy_filter_hf.py [OPTIONS]

Options:
    --text TEXT        Input text to classify (default: built-in PII sample)
    --model-dir DIR    Local model directory (default: auto-download from HF)
    --device DEV       cpu | mps | cuda (default: mps if available, else cpu)
    --dtype DTYPE      float32 | bfloat16 (default: bfloat16)
    --seq-len N        Pad/truncate input to N tokens (default: 0 = no pad)
    --stream-delay S   Delay between tokens in seconds (default: 0.015)
    --no-color         Disable ANSI colour output
    --no-stream        Print all tokens at once (no streaming delay)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

MODEL_REPO_ID = "openai/privacy-filter"

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
    "\033[31m",  # red
    "\033[32m",  # green
    "\033[33m",  # yellow
    "\033[34m",  # blue
    "\033[35m",  # magenta
    "\033[36m",  # cyan
    "\033[91m",  # bright red
    "\033[92m",  # bright green
    "\033[93m",  # bright yellow
    "\033[94m",  # bright blue
    "\033[95m",  # bright magenta
    "\033[96m",  # bright cyan
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


# ── Model loading ──────────────────────────────────────────────────────────

def _resolve_model_dir(model_dir: str | None) -> Path | None:
    if model_dir:
        p = Path(model_dir).resolve()
        if p.is_dir():
            return p
        raise FileNotFoundError(f"Model directory not found: {p}")
    local = REPO_ROOT / "models" / "privacy-filter"
    if local.is_dir() and any(local.glob("*.safetensors")):
        return local
    return None


def _load_hf_model(model_dir: Path | None, device: str, dtype: str):
    import torch
    from transformers import AutoConfig, AutoTokenizer

    trust_remote_code = True
    src = str(model_dir) if model_dir else MODEL_REPO_ID

    config = AutoConfig.from_pretrained(src, trust_remote_code=trust_remote_code)

    torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float32
    torch_device = torch.device(device)

    id2label = getattr(config, "id2label", None)
    if id2label is None:
        num_labels = getattr(config, "num_labels", 33)
        id2label = {i: f"LABEL_{i}" for i in range(num_labels)}
    else:
        id2label = {int(k): v for k, v in id2label.items()}

    from transformers import AutoModelForTokenClassification, AutoModelForMaskedLM

    model = None
    for cls in (AutoModelForTokenClassification, AutoModelForMaskedLM):
        try:
            model = cls.from_pretrained(
                src,
                local_files_only=model_dir is not None,
                torch_dtype=torch_dtype,
                config=config,
                trust_remote_code=trust_remote_code,
            )
            break
        except Exception:
            continue
    if model is None:
        raise RuntimeError(f"Could not load model from {src}")

    model = model.to(torch_device).eval()

    tokenizer = AutoTokenizer.from_pretrained(
        src,
        local_files_only=model_dir is not None,
        trust_remote_code=trust_remote_code,
    )

    return model, tokenizer, id2label, torch_device, torch_dtype


# ── Forward pass ────────────────────────────────────────────────────────────

def _tokenize(tokenizer, text: str, seq_len: int):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")
    if seq_len > 0:
        ids = input_ids[0].tolist()
        mask = attention_mask[0].tolist() if attention_mask is not None else [1] * len(ids)
        if len(ids) < seq_len:
            pad_id = tokenizer.pad_token_id
            if pad_id is None:
                pad_id = tokenizer.eos_token_id
            if pad_id is None:
                pad_id = 0
            ids = ids + [pad_id] * (seq_len - len(ids))
            mask = mask + [0] * (seq_len - len(mask))
        else:
            ids = ids[:seq_len]
            mask = mask[:seq_len]
        input_ids = __import__("torch").tensor([ids], dtype=input_ids.dtype)
        attention_mask = __import__("torch").tensor([mask], dtype=__import__("torch").long)
    return input_ids, attention_mask


def _run_forward(model, tokenizer, text: str, device, dtype, seq_len: int):
    import torch

    input_ids, attention_mask = _tokenize(tokenizer, text, seq_len)
    input_ids = input_ids.to(device)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t0) * 1000

    logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
    preds = logits.argmax(dim=-1)[0].cpu().tolist()

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu().tolist())

    return tokens, preds, elapsed_ms


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
    parser.add_argument("--model-dir", default=None, help="Local model directory")
    parser.add_argument("--device", default=None, help="cpu | mps | cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    parser.add_argument("--seq-len", type=int, default=0, help="Pad/truncate input to N tokens")
    parser.add_argument("--stream-delay", type=float, default=0.0, help="Artificial delay between tokens (s)")
    parser.add_argument("--no-color", action="store_true", help="Disable ANSI colours")
    parser.add_argument("--no-stream", action="store_true", help="Print all at once")
    args = parser.parse_args()

    import torch
    if args.device is None:
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    else:
        device = args.device

    use_color = not args.no_color and sys.stdout.isatty()

    model_dir = _resolve_model_dir(args.model_dir)

    print(f"{'═' * 60}")
    print(f"  {_BOLD}Privacy Filter — Vanilla HuggingFace (transformers){_RESET}")
    print(f"{'═' * 60}")
    print(f"\nInput:\n  \"{args.text}\"\n")

    print(f"Loading model from {'(local) ' + str(model_dir) if model_dir else MODEL_REPO_ID}...")
    model, tokenizer, id2label, torch_device, torch_dtype = _load_hf_model(
        model_dir, device, args.dtype
    )
    print(f"  Model loaded on {torch_device} ({torch_dtype})")
    print(f"  Labels: {len(id2label)} ({list(id2label.values())[:8]}...)\n")

    print("Computing forward pass...")
    tokens, preds, elapsed_ms = _run_forward(model, tokenizer, args.text, torch_device, torch_dtype, args.seq_len)
    print(f"  Done in {elapsed_ms:.2f} ms\n")

    _stream_results(
        tokens, preds, id2label, elapsed_ms,
        stream_delay=args.stream_delay,
        use_color=use_color,
        use_stream=not args.no_stream,
    )


if __name__ == "__main__":
    main()
