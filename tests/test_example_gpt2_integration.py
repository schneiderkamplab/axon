from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys

import pytest


_GPT2_MODEL_URL = "https://huggingface.co/openai-community/gpt2/resolve/main/model.safetensors"


def test_download_gpt2_model_and_run_example_yaml() -> None:
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
