from __future__ import annotations

import contextlib
import io
from pathlib import Path

import pytest

from brainsurgery.synapse import run_axon_test
from tests.model_downloads import MATRIX_AXON_TO_MODEL_DIR


def _matrix_pairs(repo_root: Path) -> list[tuple[Path, Path]]:
    examples_dir = repo_root / "examples"
    models_dir = repo_root / "models"
    return [
        (examples_dir / f"{axon_stem}.axon", models_dir / model_dir_name)
        for axon_stem, model_dir_name in sorted(MATRIX_AXON_TO_MODEL_DIR.items())
    ]


_REPO_ROOT = Path(__file__).resolve().parents[1]
_PAIRS = _matrix_pairs(_REPO_ROOT)
_SPEED_RATIO_LIMITS_BY_MODEL: dict[str, float] = {
    "gpt2": 2.0,
}


@pytest.mark.parametrize(
    ("axon_path", "model_dir"),
    _PAIRS,
    ids=[f"{axon.name}__{model.name}" for axon, model in _PAIRS],
)
def test_axon_matrix_quality(
    axon_path: Path, model_dir: Path, ensure_matrix_test_models: None
) -> None:
    pytest.importorskip("transformers")
    _ = ensure_matrix_test_models

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

    speed_limit = _SPEED_RATIO_LIMITS_BY_MODEL.get(axon_path.stem, 1.5)
    assert result["speed_ratio_axon_over_hf"] < speed_limit
    assert result["max_diff"] < 1.0e-3
    assert result["top1_eq"] is True
