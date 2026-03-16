from importlib import import_module

from brainsurgery.engine.state_dicts import _InMemoryStateDict

_module = import_module("brainsurgery.transforms.save")
globals().update(
    {name: getattr(_module, name) for name in dir(_module) if not name.startswith("_")}
)
from pathlib import Path

import pytest
import torch

from brainsurgery.core import TensorRef


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


def test_save_compile_additional_validation_paths() -> None:
    spec = SaveTransform().compile("/tmp/x.safetensors", default_model="model")
    assert spec.path.name == "x.safetensors"

    with pytest.raises(SaveTransformError, match="save.alias must be a non-empty string"):
        SaveTransform().compile({"path": "/tmp/x.safetensors", "alias": ""}, default_model=None)
    with pytest.raises(SaveTransformError, match="save.format must be a non-empty string"):
        SaveTransform().compile({"path": "/tmp/x.safetensors", "format": ""}, default_model=None)
    with pytest.raises(SaveTransformError, match="save.target must be a non-empty string"):
        SaveTransform().compile({"path": "/tmp/x.safetensors", "target": ""}, default_model="model")
    with pytest.raises(SaveTransformError, match="must not be sliced"):
        SaveTransform().compile(
            {"path": "/tmp/x.safetensors", "target": "model::x::[:1]"}, default_model="model"
        )
    original_parse_model_expr = _module.parse_model_expr
    try:
        _module.parse_model_expr = lambda raw, default_model=None: TensorRef(
            model="model", expr=["x"], slice_spec=None
        )
        with pytest.raises(SaveTransformError, match="single tensor name"):
            SaveTransform().compile(
                {"path": "/tmp/x.safetensors", "target": "model::x"}, default_model="model"
            )
    finally:
        _module.parse_model_expr = original_parse_model_expr
    with pytest.raises(SaveTransformError, match="tensor save must be one of"):
        SaveTransform().compile(
            {"path": "/tmp/x.safetensors", "target": "model::x", "format": "bad"},
            default_model="model",
        )
    dcp_spec = SaveTransform().compile(
        {"path": "/tmp/dcp_out", "format": "dcp"}, default_model="model"
    )
    assert dcp_spec.format == "dcp"
    with pytest.raises(SaveTransformError, match="only supported for safetensors state_dict save"):
        SaveTransform().compile(
            {"path": "/tmp/dcp_out", "format": "dcp", "shard": "1MB"},
            default_model="model",
        )


def test_save_apply_and_helper_error_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Provider:
        def __init__(self) -> None:
            self.sd = _InMemoryStateDict()
            self.sd["x"] = torch.ones(1)

        def get_state_dict(self, model: str) -> _InMemoryStateDict:
            del model
            return self.sd

    provider = _Provider()
    state_spec = SaveSpec(
        path=Path("/tmp/out.safetensors"),
        alias="model",
        tensor_name=None,
        format=None,
        shard_size=None,
    )
    monkeypatch.setattr(
        _module,
        "persist_state_dict",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("write failed")),
    )
    with pytest.raises(SaveTransformError, match="write failed"):
        SaveTransform().apply(state_spec, provider)  # type: ignore[arg-type]

    missing_spec = SaveSpec(
        path=Path("/tmp/out.safetensors"),
        alias="model",
        tensor_name="missing",
        format=None,
        shard_size=None,
    )
    with pytest.raises(SaveTransformError, match="save target missing"):
        SaveTransform().apply(missing_spec, provider)  # type: ignore[arg-type]

    tensor_spec = SaveSpec(
        path=Path("/tmp/out.npy"), alias="model", tensor_name="x", format="numpy", shard_size=None
    )
    monkeypatch.setattr(
        _module,
        "save_tensor_to_path",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("tensor failed")),
    )
    with pytest.raises(SaveTransformError, match="tensor failed"):
        SaveTransform().apply(tensor_spec, provider)  # type: ignore[arg-type]

    assert _module._resolve_max_io_workers(type("_P", (), {"max_io_workers": 0})()) == 1

    with pytest.raises(SaveTransformError, match="save.shard must be a non-empty string"):
        _module._parse_save_shard(1)
    monkeypatch.setattr(
        _module,
        "parse_shard_size",
        lambda raw: (_ for _ in ()).throw(RuntimeError("output.shard invalid")),
    )
    with pytest.raises(SaveTransformError, match="save.shard invalid"):
        _module._parse_save_shard("bad")
