from importlib import import_module

import torch

from brainsurgery.core import TensorRef
from brainsurgery.core import BinaryMappingSpec
from brainsurgery.engine import reset_runtime_flags, set_runtime_flag

_module = import_module("brainsurgery.transforms.copy")
globals().update({name: getattr(_module, name) for name in dir(_module) if not name.startswith("_")})


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
    item = ResolvedMapping(
        src_model="model",
        src_name="src",
        src_slice=None,
        dst_model="model",
        dst_name="dst",
        dst_slice=None,
    )
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
    item = ResolvedMapping(
        src_model="model",
        src_name="h.0.attn.bias",
        src_slice=None,
        dst_model="model",
        dst_name="i.0.attn.bias",
        dst_slice=None,
    )
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
    item = ResolvedMapping(
        src_model="model",
        src_name="h.0.attn.bias",
        src_slice=None,
        dst_model="model",
        dst_name="j.0.attn.bias",
        dst_slice=None,
    )
    spec = BinaryMappingSpec(
        from_ref=TensorRef(model="model", expr="h.0.attn.bias"),
        to_ref=TensorRef(model="model", expr="j.0.attn.bias"),
    )
    CopyTransform().apply_item(spec, item, provider)
    assert capsys.readouterr().out == ""

    reset_runtime_flags()
