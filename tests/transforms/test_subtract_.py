from importlib import import_module

_module = import_module("brainsurgery.transforms.subtract_")
globals().update({name: getattr(_module, name) for name in dir(_module) if not name.startswith("_")})


def test_subtract_in_place_apply_success() -> None:
    class _Provider:
        def __init__(self) -> None:
            self._state_dict = {
                "src": torch.tensor([1.0, 2.0], dtype=torch.float32),
                "dst": torch.tensor([5.0, 7.0], dtype=torch.float32),
            }

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
    SubtractInPlaceTransform().apply_mapping(item, provider)
    assert provider._state_dict["dst"].tolist() == [4.0, 5.0]


def test_subtract_in_place_dtype_mismatch() -> None:
    class _Provider:
        def __init__(self) -> None:
            self._state_dict = {
                "src": torch.tensor([1.0, 2.0], dtype=torch.float16),
                "dst": torch.tensor([5.0, 7.0], dtype=torch.float32),
            }

        def get_state_dict(self, model: str):
            assert model == "model"
            return self._state_dict

    item = ResolvedMapping(
        src_model="model",
        src_name="src",
        src_slice=None,
        dst_model="model",
        dst_name="dst",
        dst_slice=None,
    )
    try:
        SubtractInPlaceTransform().apply_mapping(item, _Provider())
    except SubtractInPlaceTransformError as exc:
        assert "dtype mismatch" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected dtype mismatch")


def test_subtract_in_place_compile_accepts_slices() -> None:
    spec = SubtractInPlaceTransform().compile(
        {"from": "a::[:2]", "to": "b::[:2]"},
        default_model="model",
    )
    assert spec.from_ref.slice_spec == "[:2]"
    assert spec.to_ref.slice_spec == "[:2]"
