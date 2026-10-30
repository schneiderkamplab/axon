import json
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from ettin_mlx.loading import load_model, read_checkpoint, snapshot

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "tests" / "data"

DEFAULT_MODELS = ["ettin-encoder-17m", "ettin-decoder-17m", "ettin-encoder-32m", "ettin-decoder-32m"]
LARGE_MODELS = [
    f"ettin-{kind}-{size}" for size in ("68m", "150m", "400m", "1b") for kind in ("encoder", "decoder")
]


def models(kind: str | None = None):
    """Default models plus the opt-in ``large`` ones, optionally filtered by encoder/decoder."""
    params = [pytest.param(m, id=m) for m in DEFAULT_MODELS]
    params += [pytest.param(m, id=m, marks=pytest.mark.large) for m in LARGE_MODELS]
    return [p for p in params if kind is None or f"-{kind}-" in p.values[0]]


_refs: dict[str, dict] = {}


def reference(name: str) -> dict[str, np.ndarray]:
    """HF reference outputs for ``name``; generated with tools/make_reference.py if missing."""
    if name not in _refs:
        path = DATA / f"{name}.npz"
        if not path.exists():
            subprocess.run([sys.executable, str(ROOT / "tools" / "make_reference.py"), "--models", name], check=True)
        with np.load(path) as f:
            _refs.clear()
            _refs[name] = {k: f[k] for k in f.files}
    return _refs[name]


_models: OrderedDict = OrderedDict()


def model(name: str, dtype: str = "float32", **kwargs):
    """Loaded MLX model; keeps only the two most recent to bound memory with the 1B checkpoints."""
    key = (name, dtype, tuple(sorted(kwargs.items())))
    if key not in _models:
        while len(_models) >= 2:
            _models.popitem(last=False)
        mx.clear_cache()
        _models[key] = load_model(name, dtype=dtype, **kwargs)
    _models.move_to_end(key)
    return _models[key]


_vocab: dict[str, tuple[np.ndarray, np.ndarray | None]] = {}


def vocab_projection(name: str) -> tuple[np.ndarray, np.ndarray | None]:
    """The checkpoint's own (W, b) of the vocab projection, read straight from the file."""
    if name not in _vocab:
        _vocab.clear()
        raw = read_checkpoint(snapshot(name))
        cfg = json.loads((snapshot(name) / "config.json").read_text())
        w_key = "model.embeddings.tok_embeddings.weight" if cfg.get("tie_word_embeddings", True) else "decoder.weight"
        bias = np.array(raw["decoder.bias"]) if "decoder.bias" in raw else None
        _vocab[name] = (np.array(raw[w_key]), bias)
    return _vocab[name]


def reference_logits(name: str, case: str, fp64: bool = False) -> np.ndarray:
    """HF logits rebuilt as ``head @ W.T + b`` (see tools/make_reference.py).

    ``fp64=True`` uses the float64 HF run: the exact logits of the fp32 weights.
    """
    w, b = vocab_projection(name)
    if fp64:
        w, b = w.astype(np.float64), None if b is None else b.astype(np.float64)
    out = reference(name)[f"{case}/head64" if fp64 else f"{case}/head"] @ w.T
    return out if b is None else out + b


def meta(name: str) -> dict:
    return json.loads(str(reference(name)["meta"]))


CASES = ["short", "medium", "long", "padded"]


def max_abs(ours: np.ndarray, ref: np.ndarray, mask: np.ndarray) -> float:
    return float(np.abs(ours - ref).max(-1)[mask.astype(bool)].max())


def compare(ours: np.ndarray, ref: np.ndarray, top1: np.ndarray, mask: np.ndarray) -> dict:
    """Parity numbers over real (non-pad) positions.

    ``max_rel`` is per position: ``max_v |ours - ref| / max_v |ref|``, i.e. the error
    relative to that position's logit scale (elementwise ratios blow up near zero).
    """
    valid = mask.astype(bool)
    per_pos = np.abs(ours - ref).max(-1)
    scale = np.abs(ref).max(-1)
    return {
        "max_abs": float(per_pos[valid].max()),
        "max_rel": float((per_pos / scale)[valid].max()),
        "top1_agreement": float((ours.argmax(-1) == top1)[valid].mean()),
        "positions": int(valid.sum()),
    }
