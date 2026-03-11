from __future__ import annotations

import pytest
import torch

from brainsurgery.engine.execution import execute_transform_pairs
from brainsurgery.engine.plan import compile_plan
from brainsurgery.providers import InMemoryStateDict
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


def test_split_and_concat_roundtrip() -> None:
    raw = {
        "inputs": ["/tmp/model.safetensors"],
        "transforms": [
            {"split": {"from": "x", "to": ["x0", "x1"], "sizes": [2, 2], "dim": 0}},
            {"concat": {"from": ["x0", "x1"], "to": "x_rebuilt", "dim": 0}},
        ],
    }
    plan = compile_plan(raw)
    provider = _Provider({"model": _make_state_dict({"x": torch.tensor([1.0, 2.0, 3.0, 4.0])})})
    should_continue, executed = execute_transform_pairs(
        zip(raw["transforms"], plan.transforms, strict=False),
        provider,
        interactive=False,
    )
    assert should_continue is True
    assert len(executed) == 2
    model_sd = provider.get_state_dict("model")
    assert torch.equal(model_sd["x_rebuilt"], model_sd["x"])


def test_matmul_creates_destination() -> None:
    raw = {
        "inputs": ["/tmp/model.safetensors"],
        "transforms": [
            {"matmul": {"from_a": "a", "from_b": "b", "to": "out"}},
        ],
    }
    plan = compile_plan(raw)
    provider = _Provider(
        {
            "model": _make_state_dict(
                {
                    "a": torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
                    "b": torch.tensor([[10.0], [20.0]], dtype=torch.float32),
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
    assert torch.equal(
        provider.get_state_dict("model")["out"],
        torch.tensor([[50.0], [110.0]], dtype=torch.float32),
    )


def test_permute_reshape_and_reshape_in_place() -> None:
    raw = {
        "inputs": ["/tmp/model.safetensors"],
        "transforms": [
            {"permute": {"from": "x", "to": "xp", "order": [1, 0]}},
            {"reshape": {"from": "xp", "to": "flat", "shape": [6]}},
            {"reshape_": {"target": "x", "shape": [3, 2]}},
        ],
    }
    plan = compile_plan(raw)
    provider = _Provider(
        {
            "model": _make_state_dict(
                {
                    "x": torch.arange(6, dtype=torch.float32).reshape(2, 3),
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
    assert len(executed) == 3
    model_sd = provider.get_state_dict("model")
    assert model_sd["xp"].shape == (3, 2)
    assert model_sd["flat"].shape == (6,)
    assert model_sd["x"].shape == (3, 2)


def test_fill_modes_and_clamp_variants() -> None:
    raw = {
        "inputs": ["/tmp/model.safetensors"],
        "transforms": [
            {"fill": {"from": "x", "to": "const", "mode": "constant", "value": 3}},
            {"fill": {"from": "x", "to": "vector", "mode": "tensor", "values": [1.0, 2.0]}},
            {"fill": {"from": "x", "to": "rand_a", "mode": "rand", "seed": 123}},
            {"fill": {"from": "x", "to": "rand_b", "mode": "rand", "seed": 123}},
            {"clamp": {"from": "const", "to": "const_clamped", "min": 0.0, "max": 2.0}},
            {"fill_": {"target": "x", "mode": "tensor", "values": [9.0, -9.0]}},
            {"clamp_": {"target": "x", "min": -1.0, "max": 1.0}},
        ],
    }
    plan = compile_plan(raw)
    provider = _Provider({"model": _make_state_dict({"x": torch.tensor([0.0, 0.0], dtype=torch.float32)})})
    should_continue, executed = execute_transform_pairs(
        zip(raw["transforms"], plan.transforms, strict=False),
        provider,
        interactive=False,
    )
    assert should_continue is True
    assert len(executed) == len(raw["transforms"])
    model_sd = provider.get_state_dict("model")
    assert torch.equal(model_sd["const"], torch.tensor([3.0, 3.0], dtype=torch.float32))
    assert torch.equal(model_sd["vector"], torch.tensor([1.0, 2.0], dtype=torch.float32))
    assert torch.equal(model_sd["rand_a"], model_sd["rand_b"])
    assert torch.equal(model_sd["const_clamped"], torch.tensor([2.0, 2.0], dtype=torch.float32))
    assert torch.equal(model_sd["x"], torch.tensor([1.0, -1.0], dtype=torch.float32))


def test_split_rejects_size_mismatch() -> None:
    raw = {
        "inputs": ["/tmp/model.safetensors"],
        "transforms": [
            {"split": {"from": "x", "to": ["a", "b"], "sizes": [1, 1], "dim": 0}},
        ],
    }
    plan = compile_plan(raw)
    provider = _Provider({"model": _make_state_dict({"x": torch.tensor([0.0, 1.0, 2.0])})})
    with pytest.raises(TransformError, match="must sum to source size"):
        execute_transform_pairs(
            zip(raw["transforms"], plan.transforms, strict=False),
            provider,
            interactive=False,
        )
