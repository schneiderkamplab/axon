from importlib import import_module

import torch

from brainsurgery.core import TensorRef, TernaryMappingSpec
from brainsurgery.engine.state_dicts import _InMemoryStateDict

_module = import_module("brainsurgery.transforms.multiply")
MultiplyTransform = _module.MultiplyTransform
TransformError = _module.TransformError


def test_multiply_apply_success() -> None:
    class _Provider:
        def __init__(self) -> None:
            self._state_dict = _InMemoryStateDict()
            self._state_dict["a"] = torch.tensor([2.0, 3.0])
            self._state_dict["b"] = torch.tensor([4.0, 5.0])
            self._state_dict["dst"] = torch.tensor([0.0, 0.0])

        def get_state_dict(self, model: str):
            assert model == "m"
            return self._state_dict

    provider = _Provider()
    item = ("a", "b", "dst")
    spec = TernaryMappingSpec(
        from_a_ref=TensorRef(model="m", expr="a"),
        from_b_ref=TensorRef(model="m", expr="b"),
        to_ref=TensorRef(model="m", expr="dst"),
    )
    MultiplyTransform().apply_item(spec, item, provider)
    assert provider._state_dict["dst"].tolist() == [8.0, 15.0]


def test_multiply_compile_slices_allowed() -> None:
    spec = MultiplyTransform().compile(
        {"from_a": "a::[:2]", "from_b": "b::[:2]", "to": "c::[:2]"},
        default_model="m",
    )
    assert spec.from_a_ref.slice_spec == "[:2]"
    assert spec.from_b_ref.slice_spec == "[:2]"
    assert spec.to_ref.slice_spec == "[:2]"


def test_multiply_shape_mismatch() -> None:
    class _Provider:
        def __init__(self) -> None:
            self._state_dict = _InMemoryStateDict()
            self._state_dict["a"] = torch.tensor([1.0, 2.0])
            self._state_dict["b"] = torch.tensor([2.0])
            self._state_dict["dst"] = torch.tensor([0.0, 0.0])

        def get_state_dict(self, model: str):
            assert model == "m"
            return self._state_dict

    item = ("a", "b", "dst")
    try:
        spec = TernaryMappingSpec(
            from_a_ref=TensorRef(model="m", expr="a"),
            from_b_ref=TensorRef(model="m", expr="b"),
            to_ref=TensorRef(model="m", expr="dst"),
        )
        MultiplyTransform().apply_item(spec, item, _Provider())
    except TransformError as exc:
        assert "shape mismatch" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected shape mismatch")
