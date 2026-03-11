from __future__ import annotations

import pytest
import torch

from brainsurgery.engine.execution import execute_transform_pairs
from brainsurgery.engine.plan import compile_plan
from brainsurgery.engine import InMemoryStateDict
from brainsurgery.core import TransformError


class _Provider:
    def __init__(self, state_dicts: dict[str, InMemoryStateDict]) -> None:
        self._state_dicts = state_dicts

    def get_state_dict(self, model: str) -> InMemoryStateDict:
        return self._state_dicts[model]


def _make_state_dict(values: dict[str, torch.Tensor]) -> InMemoryStateDict:
    sd = InMemoryStateDict()
    for key, tensor in values.items():
        sd[key] = tensor
    return sd


def test_concat_two_tensors_into_new_tensor() -> None:
    raw = {
        "inputs": ["/tmp/model.safetensors"],
        "transforms": [
            {"concat": {"from": ["a", "b"], "to": "ab", "dim": 0}},
        ],
    }
    plan = compile_plan(raw)
    provider = _Provider(
        {
            "model": _make_state_dict(
                {
                    "a": torch.tensor([[1.0, 2.0]], dtype=torch.float32),
                    "b": torch.tensor([[3.0, 4.0]], dtype=torch.float32),
                }
            )
        }
    )

    should_continue, executed = execute_transform_pairs(
        zip(raw["transforms"], plan.transforms, strict=False),
        provider,
        interactive=False,
    )

    assert should_continue is True
    assert executed == raw["transforms"]
    out = provider.get_state_dict("model")["ab"]
    assert out.shape == (2, 2)
    assert torch.equal(out, torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32))


def test_concat_supports_sliced_sources() -> None:
    raw = {
        "inputs": ["/tmp/model.safetensors"],
        "transforms": [
            {"concat": {"from": ["x::[:, :2]", "x::[:, 2:]"], "to": "x_rebuilt", "dim": 1}},
        ],
    }
    plan = compile_plan(raw)
    provider = _Provider(
        {
            "model": _make_state_dict(
                {
                    "x": torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float32),
                }
            )
        }
    )

    should_continue, executed = execute_transform_pairs(
        zip(raw["transforms"], plan.transforms, strict=False),
        provider,
        interactive=False,
    )

    assert should_continue is True
    assert len(executed) == 1
    assert torch.equal(provider.get_state_dict("model")["x_rebuilt"], provider.get_state_dict("model")["x"])


def test_concat_rejects_source_pattern_matching_multiple_tensors() -> None:
    raw = {
        "inputs": ["/tmp/model.safetensors"],
        "transforms": [
            {"concat": {"from": ["block\\..*\\.weight", "x"], "to": "out", "dim": 0}},
        ],
    }
    plan = compile_plan(raw)
    provider = _Provider(
        {
            "model": _make_state_dict(
                {
                    "block.0.weight": torch.tensor([1.0], dtype=torch.float32),
                    "block.1.weight": torch.tensor([2.0], dtype=torch.float32),
                    "x": torch.tensor([3.0], dtype=torch.float32),
                }
            )
        }
    )

    with pytest.raises(TransformError, match="match exactly one tensor"):
        execute_transform_pairs(
            zip(raw["transforms"], plan.transforms, strict=False),
            provider,
            interactive=False,
        )


def test_concat_rejects_incompatible_shapes() -> None:
    raw = {
        "inputs": ["/tmp/model.safetensors"],
        "transforms": [
            {"concat": {"from": ["a", "b"], "to": "ab", "dim": 0}},
        ],
    }
    plan = compile_plan(raw)
    provider = _Provider(
        {
            "model": _make_state_dict(
                {
                    "a": torch.tensor([[1.0, 2.0]], dtype=torch.float32),
                    "b": torch.tensor([[3.0, 4.0, 5.0]], dtype=torch.float32),
                }
            )
        }
    )

    with pytest.raises(TransformError, match="shape mismatch"):
        execute_transform_pairs(
            zip(raw["transforms"], plan.transforms, strict=False),
            provider,
            interactive=False,
        )
