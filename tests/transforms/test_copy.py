from importlib import import_module

_module = import_module("brainsurgery.transforms.copy")
globals().update({name: getattr(_module, name) for name in dir(_module) if not name.startswith("_")})


def test_copy_compile_rejects_sliced_destination() -> None:
    try:
        CopyTransform().compile({"from": "a", "to": "b::[:]"}, default_model="model")
    except CopyTransformError as exc:
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
    CopyTransform().apply_mapping(item, provider)
    assert torch.equal(provider._state_dict["src"], provider._state_dict["dst"])
    assert provider._state_dict["src"] is not provider._state_dict["dst"]
