from importlib import import_module

_module = import_module("brainsurgery.transforms.load")
globals().update({name: getattr(_module, name) for name in dir(_module) if not name.startswith("_")})


def test_load_compile_defaults_alias_to_model_without_context() -> None:
    spec = LoadTransform().compile({"path": "/tmp/x.safetensors"}, default_model=None)
    assert spec.alias == "model"
    assert spec.tensor_name is None


def test_load_compile_to_conflict_raises() -> None:
    try:
        LoadTransform().compile(
            {"path": "/tmp/t.pt", "alias": "a", "to": "b::x"},
            default_model=None,
        )
    except LoadTransformError as exc:
        assert "conflicts" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected alias conflict error")


def test_load_rejects_non_auto_format_for_state_dict() -> None:
    try:
        LoadTransform().compile(
            {"path": "/tmp/x.safetensors", "format": "torch"},
            default_model="model",
        )
    except LoadTransformError as exc:
        assert "only supported for tensor loads" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected load.format validation error")
