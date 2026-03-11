from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from brainsurgery.engine import create_state_dict_provider
from brainsurgery.transforms.copy import CopyTransform


_GPT2_MODEL_URL = "https://huggingface.co/openai-community/gpt2/resolve/main/model.safetensors"


def _ensure_gpt2_model_path() -> Path:
    if shutil.which("curl") is None:
        pytest.skip("curl not available")

    repo_root = Path(__file__).resolve().parents[1]
    model_path = repo_root / "models" / "gpt2" / "model.safetensors"
    model_path.parent.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        download = subprocess.run(
            [
                "curl",
                "-fL",
                "-o",
                str(model_path),
                _GPT2_MODEL_URL,
            ],
            cwd=repo_root,
            capture_output=True,
            text=True,
        )
        assert download.returncode == 0, (
            "failed to download GPT-2 model.safetensors\n"
            f"stdout:\n{download.stdout}\n"
            f"stderr:\n{download.stderr}"
        )

    return model_path


def test_download_gpt2_model_and_run_example_yaml() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    _ensure_gpt2_model_path()

    run = subprocess.run(
        [
            sys.executable,
            "-m",
            "brainsurgery.cli",
            "--log-level",
            "warning",
            "--no-summarize",
            str(repo_root / "examples" / "gpt2.yaml"),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    assert run.returncode == 0, (
        "brainsurgery examples/gpt2.yaml failed\n"
        f"stdout:\n{run.stdout}\n"
        f"stderr:\n{run.stderr}"
    )

    assert (repo_root / "models" / "test" / "model.safetensors.index.json").exists()


@pytest.mark.parametrize("provider_name", ["inmemory", "arena"])
def test_gpt2_copy_tracks_access_counts_for_real_providers(tmp_path: Path, provider_name: str) -> None:
    model_path = _ensure_gpt2_model_path()
    provider = create_state_dict_provider(
        provider=provider_name,
        model_paths={"loaded": model_path},
        max_io_workers=1,
        arena_root=tmp_path / "arena",
        arena_segment_size="1GB",
    )

    try:
        loaded = provider.get_state_dict("loaded")
        tensor_names = list(loaded.keys())
        assert tensor_names

        provider.get_or_create_alias_state_dict("copied")
        spec = CopyTransform().compile(
            {"from": r"loaded::(.*)", "to": r"copied::\1"},
            default_model=None,
        )
        result = CopyTransform().apply(spec, provider)
        copied = provider.get_state_dict("copied")

        assert result.count == len(tensor_names)
        assert len(copied) == len(tensor_names)

        for name in tensor_names:
            assert loaded.access_counts(name) == {"reads": 1, "writes": 1}
            assert copied.access_counts(name) == {"reads": 0, "writes": 1}
    finally:
        provider.close()
