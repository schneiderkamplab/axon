from __future__ import annotations

import pytest
import torch

from brainsurgery.transform import TransformError
from brainsurgery.transforms.scale import ScaleTransform, ScaleTransformError


class DictProvider:
    def __init__(self, state_dict: dict[str, torch.Tensor]) -> None:
        self.state_dict = state_dict

    def get_state_dict(self, model: str):
        assert model == "model"
        return self.state_dict


def test_scale_compile_rejects_non_numeric_factor() -> None:
    with pytest.raises(TransformError, match="scale.by must be numeric"):
        ScaleTransform().compile({"from": "x", "to": "y", "by": "nan?!"}, default_model="model")


def test_scale_compile_accepts_numeric_string_factor() -> None:
    spec = ScaleTransform().compile({"from": "x", "to": "y", "by": "2.5"}, default_model="model")
    assert spec.factor == 2.5


def test_scale_compile_rejects_sliced_destination() -> None:
    with pytest.raises(ScaleTransformError, match="destination must not be sliced"):
        ScaleTransform().compile(
            {"from": "x", "to": "y::[:]", "by": 1.0},
            default_model="model",
        )


def test_scale_apply_creates_scaled_tensor_from_slice() -> None:
    provider = DictProvider(
        {"x": torch.tensor([1.0, 2.0, 3.0, 4.0]), "z": torch.tensor([0.0])}
    )

    spec = ScaleTransform().compile(
        {"from": "x::[1:3]", "to": "y", "by": 10.0},
        default_model="model",
    )

    ScaleTransform().apply(spec, provider)

    assert provider.state_dict["x"].tolist() == [1.0, 2.0, 3.0, 4.0]
    assert provider.state_dict["y"].tolist() == [20.0, 30.0]


def test_scale_apply_rejects_existing_destination() -> None:
    provider = DictProvider(
        {
            "x": torch.tensor([1.0, 2.0]),
            "y": torch.tensor([0.0, 0.0]),
        }
    )

    spec = ScaleTransform().compile(
        {"from": "x", "to": "y", "by": 2.0},
        default_model="model",
    )

    with pytest.raises(TransformError, match="destination already exists"):
        ScaleTransform().apply(spec, provider)
