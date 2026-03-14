from importlib import import_module
from collections.abc import Callable

import torch

from brainsurgery.engine.state_dicts import _InMemoryStateDict
_module = import_module("brainsurgery.transforms.assign")
globals().update({name: getattr(_module, name) for name in dir(_module) if not name.startswith("_")})

def test_assign_dtype_compatibility(
    single_model_provider: Callable[[object, str], object]
) -> None:
    state_dict = _InMemoryStateDict()
    state_dict["src"] = torch.ones((2, 2), dtype=torch.float32)
    state_dict["dst"] = torch.ones((2, 2), dtype=torch.float16)
    provider = single_model_provider(state_dict)

    item = ResolvedMapping(
        src_model="model",
        src_name="src",
        src_slice=None,
        dst_model="model",
        dst_name="dst",
        dst_slice=None,
    )

    try:
        AssignTransform().apply_mapping(item, provider)
    except TransformError as exc:
        assert "dtype mismatch" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected dtype mismatch error")

def test_assign_shape_compatibility(
    single_model_provider: Callable[[object, str], object]
) -> None:
    state_dict = _InMemoryStateDict()
    state_dict["src"] = torch.ones((2, 2), dtype=torch.float32)
    state_dict["dst"] = torch.ones((3, 2), dtype=torch.float32)
    provider = single_model_provider(state_dict)

    item = ResolvedMapping(
        src_model="model",
        src_name="src",
        src_slice=None,
        dst_model="model",
        dst_name="dst",
        dst_slice=None,
    )

    try:
        AssignTransform().apply_mapping(item, provider)
    except TransformError as exc:
        assert "shape mismatch" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected shape mismatch error")

def test_assign_successful_copy(
    single_model_provider: Callable[[object, str], object]
) -> None:
    state_dict = _InMemoryStateDict()
    state_dict["src"] = torch.tensor([1.0, 2.0], dtype=torch.float32)
    state_dict["dst"] = torch.tensor([0.0, 0.0], dtype=torch.float32)
    provider = single_model_provider(state_dict)
    item = ResolvedMapping(
        src_model="model",
        src_name="src",
        src_slice=None,
        dst_model="model",
        dst_name="dst",
        dst_slice=None,
    )
    AssignTransform().apply_mapping(item, provider)
    assert torch.equal(provider.state_dict["dst"], provider.state_dict["src"])
