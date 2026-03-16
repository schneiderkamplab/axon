from collections.abc import Callable
from importlib import import_module

import torch

from brainsurgery.core import BinaryMappingSpec, TensorRef, TransformError
from brainsurgery.engine.state_dicts import _InMemoryStateDict

_module = import_module("brainsurgery.transforms.assign")
globals().update(
    {name: getattr(_module, name) for name in dir(_module) if not name.startswith("_")}
)


def test_assign_dtype_compatibility(single_model_provider: Callable[[object, str], object]) -> None:
    state_dict = _InMemoryStateDict()
    state_dict["src"] = torch.ones((2, 2), dtype=torch.float32)
    state_dict["dst"] = torch.ones((2, 2), dtype=torch.float16)
    provider = single_model_provider(state_dict)

    spec = BinaryMappingSpec(
        from_ref=TensorRef(model="model", expr="src"),
        to_ref=TensorRef(model="model", expr="dst"),
    )

    try:
        AssignTransform().apply_mapping(spec, "src", "dst", provider)
    except TransformError as exc:
        assert "dtype mismatch" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected dtype mismatch error")


def test_assign_shape_compatibility(single_model_provider: Callable[[object, str], object]) -> None:
    state_dict = _InMemoryStateDict()
    state_dict["src"] = torch.ones((2, 2), dtype=torch.float32)
    state_dict["dst"] = torch.ones((3, 2), dtype=torch.float32)
    provider = single_model_provider(state_dict)

    spec = BinaryMappingSpec(
        from_ref=TensorRef(model="model", expr="src"),
        to_ref=TensorRef(model="model", expr="dst"),
    )

    try:
        AssignTransform().apply_mapping(spec, "src", "dst", provider)
    except TransformError as exc:
        assert "shape mismatch" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected shape mismatch error")


def test_assign_successful_copy(single_model_provider: Callable[[object, str], object]) -> None:
    state_dict = _InMemoryStateDict()
    state_dict["src"] = torch.tensor([1.0, 2.0], dtype=torch.float32)
    state_dict["dst"] = torch.tensor([0.0, 0.0], dtype=torch.float32)
    provider = single_model_provider(state_dict)
    spec = BinaryMappingSpec(
        from_ref=TensorRef(model="model", expr="src"),
        to_ref=TensorRef(model="model", expr="dst"),
    )
    AssignTransform().apply_mapping(spec, "src", "dst", provider)
    assert torch.equal(provider.state_dict["dst"], provider.state_dict["src"])
