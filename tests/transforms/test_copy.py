from importlib import import_module

import torch

from brainsurgery.core import BinaryMappingSpec, TensorRef
from brainsurgery.engine import reset_runtime_flags, set_runtime_flag

_module = import_module("brainsurgery.transforms.copy")
CopyTransform = _module.CopyTransform
TransformError = _module.TransformError


def test_copy_compile_rejects_sliced_destination() -> None:
    try:
        CopyTransform().compile({"from": "a", "to": "b::[:]"}, default_model="model")
    except TransformError as exc:
        assert "destination must not be sliced" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected sliced destination error")


def test_copy_compile_accepts_sliced_source() -> None:
    spec = CopyTransform().compile({"from": "a::[:1]", "to": "b"}, default_model="model")
    assert spec.from_ref.slice_spec == "[:1]"
    assert spec.to_ref.slice_spec is None


def test_copy_apply_clones_tensor() -> None:
    class _Provider:
        def __init__(self) -> None:
            self._state_dict = {"src": torch.tensor([1.0, 2.0])}

        def get_state_dict(self, model: str):
            assert model == "model"
            return self._state_dict

    provider = _Provider()
    item = ("src", "dst")
    spec = BinaryMappingSpec(
        from_ref=TensorRef(model="model", expr="src"),
        to_ref=TensorRef(model="model", expr="dst"),
    )
    CopyTransform().apply_item(spec, item, provider)
    assert torch.equal(provider._state_dict["src"], provider._state_dict["dst"])
    assert provider._state_dict["src"] is not provider._state_dict["dst"]


def test_copy_apply_emits_verbose_activity_line(capsys) -> None:
    class _Provider:
        def __init__(self) -> None:
            self._state_dict = {"h.0.attn.bias": torch.tensor([1.0])}

        def get_state_dict(self, model: str):
            assert model == "model"
            return self._state_dict

    provider = _Provider()
    item = ("h.0.attn.bias", "i.0.attn.bias")
    spec = BinaryMappingSpec(
        from_ref=TensorRef(model="model", expr="h.0.attn.bias"),
        to_ref=TensorRef(model="model", expr="i.0.attn.bias"),
    )

    reset_runtime_flags()

    set_runtime_flag("verbose", True)
    CopyTransform().apply_item(spec, item, provider)
    assert "copy: h.0.attn.bias -> i.0.attn.bias" in capsys.readouterr().out

    provider._state_dict["i.0.attn.bias"] = provider._state_dict.pop("i.0.attn.bias")
    set_runtime_flag("verbose", False)
    item = ("h.0.attn.bias", "j.0.attn.bias")
    spec = BinaryMappingSpec(
        from_ref=TensorRef(model="model", expr="h.0.attn.bias"),
        to_ref=TensorRef(model="model", expr="j.0.attn.bias"),
    )
    CopyTransform().apply_item(spec, item, provider)
    assert capsys.readouterr().out == ""

    reset_runtime_flags()
