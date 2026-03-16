from importlib import import_module

import torch

from brainsurgery.core import TensorRef, TernaryMappingSpec
from brainsurgery.engine.state_dicts import _InMemoryStateDict

_module = import_module("brainsurgery.transforms.add")
globals().update(
    {name: getattr(_module, name) for name in dir(_module) if not name.startswith("_")}
)


def test_add_apply_success() -> None:
    class _Provider:
        def __init__(self) -> None:
            self._state_dict = _InMemoryStateDict()
            self._state_dict["a"] = torch.tensor([1.0, 2.0])
            self._state_dict["b"] = torch.tensor([3.0, 4.0])
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
    AddTransform().apply_item(spec, item, provider)
    assert provider._state_dict["dst"].tolist() == [4.0, 6.0]


def test_add_compile_requires_from_b() -> None:
    try:
        AddTransform().compile({"from_a": "a", "to": "b"}, default_model="m")
    except TransformError as exc:
        assert "from_b" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected required-key validation error")


def test_add_dtype_mismatch() -> None:
    class _Provider:
        def __init__(self) -> None:
            self._state_dict = _InMemoryStateDict()
            self._state_dict["a"] = torch.tensor([1.0, 2.0], dtype=torch.float32)
            self._state_dict["b"] = torch.tensor([3.0, 4.0], dtype=torch.float16)
            self._state_dict["dst"] = torch.tensor([0.0, 0.0], dtype=torch.float32)

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
        AddTransform().apply_item(spec, item, _Provider())
    except TransformError as exc:
        assert "dtype mismatch" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected dtype mismatch")
