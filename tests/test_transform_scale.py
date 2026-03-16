from __future__ import annotations

from collections.abc import Callable

import pytest
import torch

from brainsurgery.core import TransformError
from brainsurgery.transforms.scale import ScaleTransform


def test_scale_compile_rejects_non_numeric_factor() -> None:
    with pytest.raises(TransformError, match="scale.by must be numeric"):
        ScaleTransform().compile({"from": "x", "to": "y", "by": "nan?!"}, default_model="model")


def test_scale_compile_accepts_numeric_string_factor() -> None:
    spec = ScaleTransform().compile({"from": "x", "to": "y", "by": "2.5"}, default_model="model")
    assert spec.factor == 2.5


def test_scale_compile_rejects_sliced_destination() -> None:
    with pytest.raises(TransformError, match="destination must not be sliced"):
        ScaleTransform().compile(
            {"from": "x", "to": "y::[:]", "by": 1.0},
            default_model="model",
        )


def test_scale_apply_creates_scaled_tensor_from_slice(
    single_model_provider: Callable[[dict[str, torch.Tensor], str], object],
) -> None:
    provider = single_model_provider(
        {"x": torch.tensor([1.0, 2.0, 3.0, 4.0]), "z": torch.tensor([0.0])}
    )

    spec = ScaleTransform().compile(
        {"from": "x::[1:3]", "to": "y", "by": 10.0},
        default_model="model",
    )

    ScaleTransform().apply(spec, provider)

    assert provider.state_dict["x"].tolist() == [1.0, 2.0, 3.0, 4.0]
    assert provider.state_dict["y"].tolist() == [20.0, 30.0]


def test_scale_apply_rejects_existing_destination(
    single_model_provider: Callable[[dict[str, torch.Tensor], str], object],
) -> None:
    provider = single_model_provider(
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
