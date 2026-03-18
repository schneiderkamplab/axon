from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from brainsurgery.engine import (
    RuntimeFlagLifecycleScope,
    create_state_dict_provider,
    reset_runtime_flags_for_scope,
    set_runtime_flag,
)
from brainsurgery.engine.execution import _execute_transform_pairs
from brainsurgery.engine.plan import compile_plan
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


def _rewrite_gpt2_plan_for_checkpoint_path_ambiguity(plan: dict[str, object]) -> dict[str, object]:
    patched = dict(plan)
    model_file = "models/gpt2/model.safetensors"

    inputs = patched.get("inputs")
    if isinstance(inputs, list):
        patched["inputs"] = [model_file if item == "models/gpt2" else item for item in inputs]

    transforms = patched.get("transforms")
    if isinstance(transforms, list):
        updated_transforms: list[object] = []
        for item in transforms:
            if (
                isinstance(item, dict)
                and "load" in item
                and isinstance(item["load"], dict)
                and item["load"].get("path") == "models/gpt2"
            ):
                load_payload = dict(item["load"])
                load_payload["path"] = model_file
                updated_transforms.append({"load": load_payload})
            else:
                updated_transforms.append(item)
        patched["transforms"] = updated_transforms

    return patched


def test_download_gpt2_model_and_run_example_yaml(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    _ensure_gpt2_model_path()
    source_plan = OmegaConf.to_container(
        OmegaConf.load(repo_root / "examples" / "gpt2.yaml"), resolve=True
    )
    assert isinstance(source_plan, dict)
    patched_plan = _rewrite_gpt2_plan_for_checkpoint_path_ambiguity(source_plan)
    plan_path = tmp_path / "gpt2_integration.yaml"
    plan_path.write_text(OmegaConf.to_yaml(patched_plan, resolve=True), encoding="utf-8")

    run = subprocess.run(
        [
            sys.executable,
            "-m",
            "brainsurgery.cli",
            "--log-level",
            "warning",
            "--no-summarize",
            str(plan_path),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    assert run.returncode == 0, (
        f"brainsurgery examples/gpt2.yaml failed\nstdout:\n{run.stdout}\nstderr:\n{run.stderr}"
    )

    assert (repo_root / "models" / "test" / "model.safetensors.index.json").exists()


@pytest.mark.parametrize("provider_name", ["inmemory", "arena"])
def test_gpt2_copy_tracks_access_counts_for_real_providers(
    tmp_path: Path, provider_name: str
) -> None:
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


@pytest.mark.parametrize("provider_name", ["inmemory", "arena"])
def test_gpt2_dry_run_pipeline_preserves_loaded_state_dict(
    tmp_path: Path,
    provider_name: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    model_path = _ensure_gpt2_model_path()
    raw = {
        "inputs": [f"loaded::{model_path}"],
        "transforms": [
            {"set": {"dry-run": True, "verbose": True}},
            {"copy": {"from": "ln_f.weight", "to": "ln_f_copy.weight"}},
            {"move": {"from": "^h(.*)", "to": r"i\1"}},
            {"move": {"from": ["i", "*rest"], "to": ["h", "*rest"]}},
            {"assign": {"from": "ln_f_copy.weight", "to": "ln_f.weight"}},
            {"scale": {"from": "ln_f_copy.weight", "to": "ln_f_scaled_equal.weight", "by": 1.0}},
            {"assert": {"equal": {"left": "ln_f.weight", "right": "ln_f_scaled_equal.weight"}}},
            {"scale": {"from": "ln_f_copy.weight", "to": "ln_f_scaled_half.weight", "by": 0.5}},
            {
                "assert": {
                    "not": {"equal": {"left": "ln_f.weight", "right": "ln_f_scaled_half.weight"}}
                }
            },
            {"copy": {"from": "ln_f_copy.weight::[:8]", "to": "demo.x"}},
            {"copy": {"from": "ln_f_copy.weight::[:8]", "to": "demo.y"}},
            {"scale_": {"target": "demo.y", "by": 2.0}},
            {"copy": {"from": "demo.x", "to": "demo.sum"}},
            {"add": {"from_a": "demo.x", "from_b": "demo.x", "to": "demo.sum"}},
            {"fill": {"from": "demo.x", "to": "demo.filled", "mode": "constant", "value": 1.0}},
            {"clamp": {"from": "demo.filled", "to": "demo.clamped", "min": 2.0, "max": 5.0}},
            {"cast": {"from": "demo.clamped", "to": "demo.clamped_cast", "dtype": "float16"}},
            {"reshape": {"from": "demo.clamped", "to": "demo.clamped_2d", "shape": [2, 4]}},
            {"permute": {"from": "demo.clamped_2d", "to": "demo.clamped_perm", "order": [1, 0]}},
            {
                "matmul": {
                    "from_a": "demo.clamped_2d",
                    "from_b": "demo.clamped_perm",
                    "to": "demo.matmul",
                }
            },
            {"copy": {"from": "h.0.attn.c_proj.weight::[:4,:4]", "to": "demo.matrix.weight"}},
            {"phlora_": {"target": "demo.matrix.weight", "rank": 2}},
            {
                "phlora": {
                    "target": "demo.matrix.weight",
                    "target_a": "demo.matrix.a.weight",
                    "target_b": "demo.matrix.b.weight",
                    "rank": 2,
                }
            },
            {"delete": {"target": "ln_f_copy.weight"}},
        ],
    }
    plan = compile_plan(raw)
    provider = create_state_dict_provider(
        provider=provider_name,
        model_paths=plan.inputs,
        max_io_workers=1,
        arena_root=tmp_path / "arena",
        arena_segment_size="1GB",
    )

    probe_names = ["ln_f.weight", "wte.weight", "h.0.attn.c_proj.weight"]
    try:
        loaded = provider.get_state_dict("loaded")
        baseline_tensors = {name: loaded[name].clone() for name in probe_names}
        baseline_counts = {name: loaded.access_counts(name) for name in probe_names}
        baseline_keys = set(loaded.keys())

        reset_runtime_flags_for_scope(RuntimeFlagLifecycleScope.CLI_RUN)
        should_continue, executed = _execute_transform_pairs(
            zip(raw["transforms"], plan.transforms, strict=False),
            provider,
            interactive=False,
        )
        assert should_continue is True
        assert len(executed) == len(raw["transforms"])

        output = capsys.readouterr().out
        assert "dry-run copy: ln_f.weight -> ln_f_copy.weight" in output
        assert "dry-run phlora_" in output

        # Verify persistent provider state is unchanged after leaving dry-run mode.
        set_runtime_flag("dry_run", False)
        loaded = provider.get_state_dict("loaded")
        assert set(loaded.keys()) == baseline_keys
        assert all(not key.startswith("demo.") for key in loaded.keys())
        for name in probe_names:
            assert loaded.access_counts(name) == baseline_counts[name]
            assert torch.equal(loaded[name], baseline_tensors[name]), name
    finally:
        reset_runtime_flags_for_scope(RuntimeFlagLifecycleScope.CLI_RUN)
        provider.close()
