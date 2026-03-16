from importlib import import_module

_module = import_module("brainsurgery.transforms.exit")
ExitSpec = _module.ExitSpec
ExitTransform = _module.ExitTransform
ExitTransformError = _module.ExitTransformError
TransformControl = _module.TransformControl


def test_exit_compile_rejects_payload() -> None:
    try:
        ExitTransform().compile({"unexpected": 1}, default_model=None)
    except ExitTransformError as exc:
        assert "does not take any payload" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected exit payload error")


def test_exit_compile_accepts_none_and_empty_mapping() -> None:
    t = ExitTransform()
    assert isinstance(t.compile(None, default_model=None), ExitSpec)
    assert isinstance(t.compile({}, default_model=None), ExitSpec)


def test_exit_apply_returns_exit_control() -> None:
    result = ExitTransform().apply(ExitSpec(), provider=None)  # type: ignore[arg-type]
    assert result.control == TransformControl.EXIT
    assert result.count == 0
