from importlib import import_module

import torch

from brainsurgery.core import TensorRef
from brainsurgery.providers import InMemoryStateDict
from brainsurgery.core import TernaryMappingSpec

_module = import_module("brainsurgery.transforms.multiply")
globals().update({name: getattr(_module, name) for name in dir(_module) if not name.startswith("_")})


def test_multiply_apply_success() -> None:
    class _Provider:
        def __init__(self) -> None:
            self._state_dict = InMemoryStateDict()
            self._state_dict["a"] = torch.tensor([2.0, 3.0])
            self._state_dict["b"] = torch.tensor([4.0, 5.0])
            self._state_dict["dst"] = torch.tensor([0.0, 0.0])

        def get_state_dict(self, model: str):
            assert model == "m"
            return self._state_dict

    provider = _Provider()
    item = ResolvedTernaryMapping(
        src_a_model="m",
        src_a_name="a",
        src_a_slice=None,
        src_b_model="m",
        src_b_name="b",
        src_b_slice=None,
        dst_model="m",
        dst_name="dst",
        dst_slice=None,
    )
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
            self._state_dict = InMemoryStateDict()
            self._state_dict["a"] = torch.tensor([1.0, 2.0])
            self._state_dict["b"] = torch.tensor([2.0])
            self._state_dict["dst"] = torch.tensor([0.0, 0.0])

        def get_state_dict(self, model: str):
            assert model == "m"
            return self._state_dict

    item = ResolvedTernaryMapping(
        src_a_model="m",
        src_a_name="a",
        src_a_slice=None,
        src_b_model="m",
        src_b_name="b",
        src_b_slice=None,
        dst_model="m",
        dst_name="dst",
        dst_slice=None,
    )
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
