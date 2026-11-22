"""Materialize the privacy-filter checkpoint in the opf key layout.

``generic-privacy-filter.axon`` binds the gpt-oss-derived opf parameter paths
(``embedding.weight``, ``block.N.attn.qkv.weight``, ``norm.scale``,
``unembedding.weight``, ...) while the downloaded HF checkpoint ships raw HF
names (``model.embed_tokens.weight``, ``model.layers.N.self_attn.q_proj.*``,
``score.weight``) with separate q/k/v projections. The rename+concat mapping
lives in ``stream_privacy_filter_axon._convert_hf_to_opf_state_dict`` — the
parity-tested loader. This script applies it once and writes a self-contained
checkpoint dir (the ``bifrost_v05`` pattern: variant subdir under the model
family with its own config/tokenizer) so the generic
``server.loading.load_mlx_model`` / ``from_safetensors`` path consumes it
directly, with no model-specific code in the server.

Usage:
    python scripts/materialize_privacy_filter_opf.py [--model-dir DIR] [--out DIR]

Default output: ``<model-dir>/opf/``.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mlx.core as mx  # noqa: E402

import stream_privacy_filter_axon as SAX  # noqa: E402

_COPY_GLOBS = ("config.json", "tokenizer*.json", "chat_template*",
               "special_tokens_map.json", "viterbi_calibration.json")
_SKIP = ("model.safetensors", "model.sig")


def materialize(model_dir: Path, out: Path) -> int:
    state = SAX._convert_hf_to_opf_state_dict(model_dir)
    out.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(out / "model.safetensors"), state)
    copied = []
    for pattern in _COPY_GLOBS:
        for src in sorted(model_dir.glob(pattern)):
            if src.name in _SKIP or not src.is_file():
                continue
            shutil.copy2(src, out / src.name)
            copied.append(src.name)
    print(f"materialized {len(state)} tensors -> {out/'model.safetensors'}")
    print(f"copied {len(copied)} metadata files: {', '.join(copied)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-dir", default=None, help="HF checkpoint dir (default: resolved by the stream harness)")
    parser.add_argument("--out", default=None, help="output dir (default: <model-dir>/opf)")
    args = parser.parse_args()
    model_dir = SAX._resolve_model_dir(args.model_dir)
    out = Path(args.out) if args.out else model_dir / "opf"
    return materialize(model_dir, out)


if __name__ == "__main__":
    raise SystemExit(main())
