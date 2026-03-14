from __future__ import annotations

from collections.abc import Callable

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
