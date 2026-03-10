from importlib import import_module

_module = import_module("brainsurgery.transforms.save")
globals().update({name: getattr(_module, name) for name in dir(_module) if not name.startswith("_")})


def test_save_compile_defaults_to_default_model() -> None:
    spec = SaveTransform().compile({"path": "/tmp/x.safetensors"}, default_model="model")
    assert spec.alias == "model"
    assert spec.format is None
    assert spec.tensor_name is None
    assert spec.shard_size is None


def test_save_compile_rejects_tensor_format_for_state_dict() -> None:
    try:
        SaveTransform().compile(
            {"path": "/tmp/x.safetensors", "format": "numpy"},
            default_model="model",
        )
    except SaveTransformError as exc:
        assert "state_dict save" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected save.format validation error")


def test_save_compile_rejects_alias_conflict() -> None:
    try:
        SaveTransform().compile(
            {"path": "/tmp/x.safetensors", "alias": "a", "target": "b::x"},
            default_model=None,
        )
    except SaveTransformError as exc:
        assert "conflicts" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected alias conflict error")


def test_save_compile_rejects_shard_for_tensor_save() -> None:
    try:
        SaveTransform().compile(
            {"path": "/tmp/x.safetensors", "target": "model::x", "shard": "1MB"},
            default_model="model",
        )
    except SaveTransformError as exc:
        assert "state_dict save" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected save.shard validation error")
