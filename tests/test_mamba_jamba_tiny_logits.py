from __future__ import annotations

from pathlib import Path

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from brainsurgery.synapse import (
    SynapseProgramModel,
    lower_axon_program_to_synapse_spec,
    parse_axon_program_from_path,
)


def _load_spec(path: Path) -> dict[str, object]:
    return lower_axon_program_to_synapse_spec(parse_axon_program_from_path(path))


def _check_tiny_logits(
    *,
    repo_root: Path,
    axon_file: str,
    model_dir: str,
    prompt: str,
    max_tol: float,
    mean_tol: float,
) -> None:
    safetensors = pytest.importorskip("safetensors")

    model_path = repo_root / "models" / model_dir
    if not model_path.is_dir():
        pytest.skip(f"missing model dir: {model_path}")
    weights_path = model_path / "model.safetensors"
    if not weights_path.exists():
        pytest.skip(f"missing weights file: {weights_path}")

    state_dict = safetensors.torch.load_file(str(weights_path))
    # Align runtime parameter precision with HF reference model loading dtype.
    state_dict = {
        name: (tensor.to(torch.float32) if tensor.is_floating_point() else tensor)
        for name, tensor in state_dict.items()
    }
    spec = _load_spec(repo_root / "examples" / axon_file)
    model = SynapseProgramModel.from_spec(spec, state_dict=state_dict).eval()

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    hf_model = AutoModelForCausalLM.from_pretrained(
        str(model_path), local_files_only=True, dtype=torch.float32
    ).eval()

    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        ours = model(inputs["input_ids"])
        if isinstance(ours, dict):
            ours = ours["logits"]
        ref = hf_model(**inputs, use_cache=False).logits

    diff = (ours.float() - ref.float()).abs()
    assert float(diff.max()) <= max_tol
    assert float(diff.mean()) <= mean_tol


def test_mamba_tiny_axon_aligns_hf_logits(repo_root: Path) -> None:
    _check_tiny_logits(
        repo_root=repo_root,
        axon_file="mamba.axon",
        model_dir="mamba_tiny_random",
        prompt="hello world",
        max_tol=1e-6,
        mean_tol=1e-7,
    )


def test_jamba_tiny_axon_aligns_hf_logits(repo_root: Path) -> None:
    _check_tiny_logits(
        repo_root=repo_root,
        axon_file="jamba.axon",
        model_dir="jamba_tiny_random",
        prompt="hello world",
        max_tol=1e-5,
        mean_tol=1e-6,
    )
