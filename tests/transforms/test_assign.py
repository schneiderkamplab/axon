from collections.abc import Callable
from importlib import import_module

import torch

from brainsurgery.core import BinaryMappingSpec, TensorRef, TransformError
from brainsurgery.engine.state_dicts import _InMemoryStateDict

_module = import_module("brainsurgery.transforms.assign")
AssignTransform = _module.AssignTransform


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


def test_assign_regex_capture_substitution(
    single_model_provider: Callable[[object, str], object],
) -> None:
    state_dict = _InMemoryStateDict()
    state_dict["block.0.weight"] = torch.tensor([1.0, 2.0], dtype=torch.float32)
    state_dict["block.1.weight"] = torch.tensor([3.0, 4.0], dtype=torch.float32)
    state_dict["copy.0.weight"] = torch.zeros(2, dtype=torch.float32)
    state_dict["copy.1.weight"] = torch.zeros(2, dtype=torch.float32)
    provider = single_model_provider(state_dict)

    transform = AssignTransform()
    spec = transform.compile(
        {"from": r"block\.(\d+)\.weight", "to": r"copy.\1.weight"},
        default_model="model",
    )
    result = transform.apply(spec, provider)
    assert result.count == 2
    assert torch.equal(state_dict["copy.0.weight"], state_dict["block.0.weight"])
    assert torch.equal(state_dict["copy.1.weight"], state_dict["block.1.weight"])


def test_assign_structured_capture_substitution(
    single_model_provider: Callable[[object, str], object],
) -> None:
    state_dict = _InMemoryStateDict()
    state_dict["block.0.weight"] = torch.tensor([1.0, 2.0], dtype=torch.float32)
    state_dict["block.1.weight"] = torch.tensor([3.0, 4.0], dtype=torch.float32)
    state_dict["copy.0.weight"] = torch.zeros(2, dtype=torch.float32)
    state_dict["copy.1.weight"] = torch.zeros(2, dtype=torch.float32)
    provider = single_model_provider(state_dict)

    transform = AssignTransform()
    spec = transform.compile(
        {
            "from": ["block", "$i", "weight"],
            "to": ["copy", "${i}", "weight"],
        },
        default_model="model",
    )
    result = transform.apply(spec, provider)
    assert result.count == 2
    assert torch.equal(state_dict["copy.0.weight"], state_dict["block.0.weight"])
    assert torch.equal(state_dict["copy.1.weight"], state_dict["block.1.weight"])


def test_assign_capture_substitution_requires_existing_destinations(
    single_model_provider: Callable[[object, str], object],
) -> None:
    state_dict = _InMemoryStateDict()
    state_dict["block.0.weight"] = torch.tensor([1.0, 2.0], dtype=torch.float32)
    provider = single_model_provider(state_dict)

    transform = AssignTransform()
    spec = transform.compile(
        {"from": r"block\.(\d+)\.weight", "to": r"copy.\1.weight"},
        default_model="model",
    )
    try:
        transform.apply(spec, provider)
    except TransformError as exc:
        assert "destination missing" in str(exc)
        assert "model::copy.0.weight" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected destination missing error")
