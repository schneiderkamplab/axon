from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
import torch


class SingleModelProvider:
    def __init__(self, state_dict: object, model: str = "model") -> None:
        self.state_dict = state_dict
        self._model = model

    def get_state_dict(self, model: str):
        assert model == self._model
        return self.state_dict


class MultiModelProvider:
    def __init__(self, state_dicts: dict[str, dict[str, torch.Tensor]]) -> None:
        self.state_dicts = state_dicts

    def get_state_dict(self, model: str):
        return self.state_dicts[model]


@pytest.fixture
def single_model_provider() -> Callable[[object, str], SingleModelProvider]:
    def _make(state_dict: object, model: str = "model") -> SingleModelProvider:
        return SingleModelProvider(state_dict=state_dict, model=model)

    return _make


@pytest.fixture
def multi_model_provider() -> Callable[[dict[str, dict[str, torch.Tensor]]], MultiModelProvider]:
    def _make(state_dicts: dict[str, dict[str, torch.Tensor]]) -> MultiModelProvider:
        return MultiModelProvider(state_dicts=state_dicts)

    return _make


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def gpt2_local_paths(repo_root: Path) -> tuple[Path, Path]:
    synapse_weights = repo_root / "models" / "gpt2" / "model.safetensors"
    hf_model_dir = repo_root / "models" / "gpt2.old"
    if not synapse_weights.exists():
        pytest.skip(f"missing local GPT-2 checkpoint: {synapse_weights}")
    if not hf_model_dir.exists():
        pytest.skip(f"missing local GPT-2 HF directory: {hf_model_dir}")
    return synapse_weights, hf_model_dir


@pytest.fixture(scope="session")
def gemma3_local_path(repo_root: Path) -> Path:
    hf_model_dir = repo_root / "models" / "gemma3"
    if not hf_model_dir.exists():
        pytest.skip(f"missing local Gemma3 HF directory: {hf_model_dir}")
    return hf_model_dir


@pytest.fixture(scope="session")
def olmoe_local_path(repo_root: Path) -> Path:
    hf_model_dir = repo_root / "models" / "olmoe_1b_7b_0924"
    if not hf_model_dir.exists():
        pytest.skip(f"missing local OLMoE HF directory: {hf_model_dir}")
    return hf_model_dir
