from importlib import import_module

import torch

from brainsurgery.core import TensorRef

from brainsurgery.core import BinaryMappingSpec

from brainsurgery.engine.state_dicts import _InMemoryStateDict
_module = import_module("brainsurgery.transforms.add")
globals().update({name: getattr(_module, name) for name in dir(_module) if not name.startswith("_")})

def test_add_in_place_apply_success() -> None:
    class _Provider:
        def __init__(self) -> None:
            self._state_dict = _InMemoryStateDict()
            self._state_dict["src"] = torch.tensor([1.0, 2.0], dtype=torch.float32)
            self._state_dict["dst"] = torch.tensor([3.0, 4.0], dtype=torch.float32)

        def get_state_dict(self, model: str):
            assert model == "model"
            return self._state_dict

    provider = _Provider()
    item = ("src", "dst")
    spec = BinaryMappingSpec(
        from_ref=TensorRef(model="model", expr="src"),
        to_ref=TensorRef(model="model", expr="dst"),
    )
    AddInPlaceTransform().apply_item(spec, item, provider)
    assert provider._state_dict["dst"].tolist() == [4.0, 6.0]

def test_add_in_place_shape_mismatch() -> None:
    class _Provider:
        def __init__(self) -> None:
            self._state_dict = _InMemoryStateDict()
            self._state_dict["src"] = torch.tensor([1.0, 2.0], dtype=torch.float32)
            self._state_dict["dst"] = torch.tensor([3.0], dtype=torch.float32)

        def get_state_dict(self, model: str):
            assert model == "model"
            return self._state_dict

    item = ("src", "dst")
    try:
        spec = BinaryMappingSpec(
            from_ref=TensorRef(model="model", expr="src"),
            to_ref=TensorRef(model="model", expr="dst"),
        )
        AddInPlaceTransform().apply_item(spec, item, _Provider())
    except TransformError as exc:
        assert "shape mismatch" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected shape mismatch")

def test_add_in_place_compile_accepts_slices() -> None:
    spec = AddInPlaceTransform().compile(
        {"from": "a::[:2]", "to": "b::[:2]"},
        default_model="model",
    )
    assert spec.from_ref.slice_spec == "[:2]"
    assert spec.to_ref.slice_spec == "[:2]"
