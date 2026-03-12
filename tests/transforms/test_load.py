from importlib import import_module
from pathlib import Path

_module = import_module("brainsurgery.transforms.load")
globals().update({name: getattr(_module, name) for name in dir(_module) if not name.startswith("_")})
import pytest
import torch
from brainsurgery.engine import InMemoryStateDict, InMemoryStateDictProvider
from brainsurgery.core import TensorRef


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


def test_load_compile_additional_validation_paths() -> None:
    spec = LoadTransform().compile("/tmp/x.safetensors", default_model=None)
    assert spec.path.name == "x.safetensors"

    with pytest.raises(LoadTransformError, match="load.alias must be a non-empty string"):
        LoadTransform().compile({"path": "/tmp/x.safetensors", "alias": ""}, default_model=None)
    with pytest.raises(LoadTransformError, match="load.format must be a non-empty string"):
        LoadTransform().compile({"path": "/tmp/x.safetensors", "format": ""}, default_model=None)
    with pytest.raises(LoadTransformError, match="one of: auto, torch, safetensors, numpy"):
        LoadTransform().compile({"path": "/tmp/x.safetensors", "format": "weird"}, default_model=None)
    with pytest.raises(LoadTransformError, match="load.to must be a non-empty string"):
        LoadTransform().compile({"path": "/tmp/x.safetensors", "to": ""}, default_model=None)
    with pytest.raises(LoadTransformError, match="must not be sliced"):
        LoadTransform().compile({"path": "/tmp/x.safetensors", "to": "a::x::[:1]"}, default_model=None)
    original_parse_model_expr = _module.parse_model_expr
    try:
        _module.parse_model_expr = lambda raw, default_model=None: TensorRef(model="a", expr=["x"], slice_spec=None)
        with pytest.raises(LoadTransformError, match="single tensor name"):
            LoadTransform().compile({"path": "/tmp/x.safetensors", "to": "a::x"}, default_model=None)
    finally:
        _module.parse_model_expr = original_parse_model_expr


def test_load_apply_additional_error_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    state_spec = LoadSpec(path=Path("/tmp/x.safetensors"), alias="a", tensor_name=None, format="auto")

    class _NoBaseProvider:
        pass

    with pytest.raises(LoadTransformError, match="supports creating new aliases"):
        LoadTransform().apply(state_spec, _NoBaseProvider())  # type: ignore[arg-type]

    provider = InMemoryStateDictProvider({}, max_io_workers=1)
    monkeypatch.setattr(provider, "load_alias_from_path", lambda alias, path: (_ for _ in ()).throw(ProviderError("model alias already exists")))
    with pytest.raises(LoadTransformError, match="load alias already exists"):
        LoadTransform().apply(state_spec, provider)

    monkeypatch.setattr(provider, "load_alias_from_path", lambda alias, path: (_ for _ in ()).throw(RuntimeError("bad checkpoint")))
    with pytest.raises(LoadTransformError, match="bad checkpoint"):
        LoadTransform().apply(state_spec, provider)

    tensor_spec = LoadSpec(path=Path("/tmp/t.npy"), alias="a", tensor_name="x", format="numpy")
    monkeypatch.setattr(_module, "load_tensor_from_path", lambda path, format: (_ for _ in ()).throw(RuntimeError("bad tensor")))
    with pytest.raises(LoadTransformError, match="bad tensor"):
        LoadTransform().apply(tensor_spec, provider)

    sd = InMemoryStateDict()
    sd["x"] = torch.ones(1)
    monkeypatch.setattr(_module, "load_tensor_from_path", lambda path, format: torch.zeros(1))
    monkeypatch.setattr(_module, "get_or_create_alias_state_dict", lambda *args, **kwargs: sd)
    with pytest.raises(LoadTransformError, match="destination already exists"):
        LoadTransform().apply(tensor_spec, provider)
