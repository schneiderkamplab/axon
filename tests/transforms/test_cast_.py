from importlib import import_module

_module = import_module("brainsurgery.transforms.cast")
globals().update({name: getattr(_module, name) for name in dir(_module) if not name.startswith("_")})

def test_cast_in_place_compile_rejects_unknown_dtype() -> None:
    try:
        CastInPlaceTransform().compile({"target": "x", "to": "not_a_dtype"}, default_model="model")
    except TransformError as exc:
        assert "cast_.to" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected cast_ dtype parse error")

def test_cast_in_place_compile_requires_to_key() -> None:
    try:
        CastInPlaceTransform().compile({"target": "x"}, default_model="model")
    except TransformError as exc:
        assert "cast_.to is required" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected missing to key error")

def test_cast_in_place_apply_changes_dtype() -> None:
    class _Provider:
        def __init__(self) -> None:
            self._state_dict = {"x": torch.ones((2,), dtype=torch.float32)}

        def get_state_dict(self, model: str):
            assert model == "model"
            return self._state_dict

    provider = _Provider()
    spec = CastInPlaceSpec(target_ref=TensorRef(model="model", expr="x"), dtype=torch.float16)
    CastInPlaceTransform().apply_to_target(spec, "x", provider)
    assert provider._state_dict["x"].dtype == torch.float16
