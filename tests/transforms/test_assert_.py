from importlib import import_module

_module = import_module("brainsurgery.transforms.assert_")
globals().update({name: getattr(_module, name) for name in dir(_module) if not name.startswith("_")})


def test_assert_compile_rejects_unknown_op() -> None:
    try:
        AssertTransform().compile({"unknown": {}}, default_model="model")
    except AssertTransformError as exc:
        assert "unknown assert op" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected unknown assert op error")


def test_assert_compile_builds_spec() -> None:
    spec = AssertTransform().compile({"exists": "x"}, default_model="model")
    assert isinstance(spec, AssertSpec)
    assert spec.collect_models() == {"model"}


def test_assert_infer_output_model_requires_single_model() -> None:
    spec = AssertTransform().compile(
        {"equal": {"left": "a::x", "right": "b::x"}},
        default_model=None,
    )
    try:
        AssertTransform().infer_output_model(spec)
    except AssertTransformError as exc:
        assert "exactly one model" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected single-model inference error")
