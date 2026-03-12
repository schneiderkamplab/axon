from importlib import import_module

import pytest
import torch

from brainsurgery.engine import InMemoryStateDict
from brainsurgery.core import TransformError

_module = import_module("brainsurgery.transforms.init_tensors")
globals().update(
    {name: getattr(_module, name) for name in dir(_module) if not name.startswith("_")}
)


class _Provider:
    def __init__(self) -> None:
        self._state_dict = InMemoryStateDict()

    def get_state_dict(self, model: str):
        assert model == "m"
        return self._state_dict


def test_zeroes_creates_new_tensor_with_shape() -> None:
    provider = _Provider()
    spec = ZeroesTransform().compile(
        {"target": "x", "shape": [2, 3]},
        default_model="m",
    )
    result = ZeroesTransform().apply(spec, provider)
    assert result.count == 1
    assert provider._state_dict["x"].shape == (2, 3)
    assert torch.equal(provider._state_dict["x"], torch.zeros((2, 3), dtype=torch.float32))


def test_ones_creates_new_tensor_with_shape() -> None:
    provider = _Provider()
    spec = OnesTransform().compile(
        {"target": "x", "shape": [4]},
        default_model="m",
    )
    OnesTransform().apply(spec, provider)
    assert torch.equal(provider._state_dict["x"], torch.ones((4,), dtype=torch.float32))


def test_rand_seed_is_reproducible() -> None:
    provider_a = _Provider()
    provider_b = _Provider()
    spec = RandTransform().compile(
        {"target": "x", "shape": [5], "seed": 123},
        default_model="m",
    )
    RandTransform().apply(spec, provider_a)
    RandTransform().apply(spec, provider_b)
    assert torch.equal(provider_a._state_dict["x"], provider_b._state_dict["x"])


def test_rand_normal_rejects_non_positive_std() -> None:
    with pytest.raises(TransformError, match="rand.std must be > 0"):
        RandTransform().compile(
            {"target": "x", "shape": [3], "distribution": "normal", "std": 0},
            default_model="m",
        )


def test_shape_create_requires_missing_destination() -> None:
    provider = _Provider()
    provider._state_dict["x"] = torch.zeros((1,), dtype=torch.float32)
    spec = ZeroesTransform().compile(
        {"target": "x", "shape": [1]},
        default_model="m",
    )
    with pytest.raises(TransformError, match="destination already exists"):
        ZeroesTransform().apply(spec, provider)


@pytest.mark.parametrize("bad_shape", [None, [1, "x"], [2, 0]])
def test_shape_parsing_rejects_invalid_shapes(bad_shape: object) -> None:
    with pytest.raises(TransformError, match="shape must be a non-empty list of positive integers"):
        ZeroesTransform().compile({"target": "x", "shape": bad_shape}, default_model="m")


def test_target_must_be_literal_name_for_shape_create() -> None:
    with pytest.raises(TransformError, match="target must resolve to a single tensor name"):
        ZeroesTransform().compile({"target": ["x"], "shape": [1]}, default_model="m")


def test_rand_normal_distribution_and_completion_candidates() -> None:
    provider = _Provider()
    spec = RandTransform().compile(
        {"target": "x", "shape": [3], "distribution": "normal", "mean": 1.5, "std": 0.25},
        default_model="m",
    )
    RandTransform().apply(spec, provider)
    assert provider._state_dict["x"].shape == (3,)
    assert RandTransform().completion_value_candidates("distribution", "n", []) == ["normal"]
    assert RandTransform().completion_value_candidates("other", "", []) is None


def test_rand_rejects_invalid_distribution_and_seed_type() -> None:
    with pytest.raises(TransformError, match="distribution must be one of"):
        RandTransform().compile(
            {"target": "x", "shape": [2], "distribution": "gaussian"},
            default_model="m",
        )
    with pytest.raises(TransformError, match="seed must be an integer"):
        RandTransform().compile(
            {"target": "x", "shape": [2], "seed": 1.5},
            default_model="m",
        )


def test_rand_uniform_options_and_bounds() -> None:
    spec = RandTransform().compile(
        {"target": "x", "shape": [2], "distribution": "uniform", "low": -1, "high": 2},
        default_model="m",
    )
    assert spec.low == -1.0
    assert spec.high == 2.0
    with pytest.raises(TransformError, match="requires low < high"):
        RandTransform().compile(
            {"target": "x", "shape": [2], "distribution": "uniform", "low": 2, "high": 2},
            default_model="m",
        )
