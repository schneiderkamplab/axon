from __future__ import annotations

from collections.abc import Callable

import pytest
import torch

from brainsurgery.core import TransformError
from brainsurgery.transforms.cast import CastTransform

def test_cast_compile_rejects_unknown_dtype() -> None:
    with pytest.raises(TransformError, match="cast.dtype"):
        CastTransform().compile(
            {"from": "x", "to": "y", "dtype": "not_a_dtype"},
            default_model="model",
        )

def test_cast_compile_requires_dtype_key() -> None:
    with pytest.raises(TransformError, match="cast.dtype is required"):
        CastTransform().compile({"from": "x", "to": "y"}, default_model="model")

def test_cast_compile_rejects_sliced_destination() -> None:
    with pytest.raises(TransformError, match="destination must not be sliced"):
        CastTransform().compile(
            {"from": "x", "to": "y::[:]", "dtype": "float16"},
            default_model="model",
        )

def test_cast_apply_creates_new_tensor_with_new_dtype(
    single_model_provider: Callable[[dict[str, torch.Tensor], str], object]
) -> None:
    provider = single_model_provider({"x": torch.ones((2,), dtype=torch.float32)})

    spec = CastTransform().compile(
        {"from": "x", "to": "y", "dtype": "float16"},
        default_model="model",
    )

    CastTransform().apply(spec, provider)

    assert provider.state_dict["x"].dtype == torch.float32
    assert provider.state_dict["y"].dtype == torch.float16

def test_cast_apply_honors_source_slice(
    single_model_provider: Callable[[dict[str, torch.Tensor], str], object]
) -> None:
    provider = single_model_provider(
        {"x": torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)}
    )

    spec = CastTransform().compile(
        {"from": "x::[1:3]", "to": "y", "dtype": "float16"},
        default_model="model",
    )

    CastTransform().apply(spec, provider)

    assert provider.state_dict["y"].tolist() == [2.0, 3.0]
    assert provider.state_dict["y"].dtype == torch.float16
