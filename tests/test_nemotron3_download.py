from __future__ import annotations

from pathlib import Path


def test_nemotron3_fixture_downloads_model_dir(nemotron3_local_path: Path) -> None:
    assert nemotron3_local_path.name == "nemotron3"
    assert (nemotron3_local_path / "config.json").exists()
    assert (nemotron3_local_path / "model.safetensors").exists() or (
        nemotron3_local_path / "model.safetensors.index.json"
    ).exists()
