from __future__ import annotations

import contextlib
import io
from pathlib import Path

import pytest

from brainsurgery.synapse import run_axon_test


def _resolve_pairs(examples_dir: Path, models_dir: Path) -> list[tuple[Path, Path]]:
    model_dirs = sorted(path for path in models_dir.iterdir() if path.is_dir())
    model_by_name = {path.name: path for path in model_dirs}

    pairs: list[tuple[Path, Path]] = []
    for axon_path in sorted(examples_dir.glob("*.axon")):
        stem = axon_path.stem
        if stem == "flexolmo":
            continue
        model_dir = model_by_name.get(stem)
        if model_dir is None:
            parts = stem.split("_")
            for cut in range(len(parts) - 1, 0, -1):
                candidate = "_".join(parts[:cut])
                model_dir = model_by_name.get(candidate)
                if model_dir is not None:
                    break
        if model_dir is not None:
            pairs.append((axon_path, model_dir))
    return pairs


_REPO_ROOT = Path(__file__).resolve().parents[1]
_PAIRS = _resolve_pairs(_REPO_ROOT / "examples", _REPO_ROOT / "models")


@pytest.mark.parametrize(
    ("axon_path", "model_dir"),
    _PAIRS,
    ids=[f"{axon.name}__{model.name}" for axon, model in _PAIRS],
)
def test_axon_matrix_quality(axon_path: Path, model_dir: Path) -> None:
    pytest.importorskip("transformers")

    with contextlib.redirect_stdout(io.StringIO()):
        result = run_axon_test(
            axon_file=axon_path,
            weights=model_dir,
            hf_model_dir=model_dir,
            device="cpu",
            dtype="float32",
            text="The future of AI is",
            max_len=16,
        )

    assert result["speed_ratio_axon_over_hf"] < (2 if axon_path.stem == "gpt2" else 1.2)
    assert result["max_diff"] < 1.0e-3
    assert result["top1_eq"] is True
