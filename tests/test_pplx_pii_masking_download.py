from __future__ import annotations

from pathlib import Path


def test_pplx_pii_masking_fixture_downloads_model_dir(pplx_pii_masking_local_path: Path) -> None:
    assert pplx_pii_masking_local_path.name == "pplx_pii_masking"
    assert (pplx_pii_masking_local_path / "config.json").exists()
    assert (pplx_pii_masking_local_path / "model.safetensors").exists()
