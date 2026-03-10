from importlib import import_module

_module = import_module("brainsurgery.transforms.dump")
globals().update({name: getattr(_module, name) for name in dir(_module) if not name.startswith("_")})


def test_dump_compile_rejects_unknown_format() -> None:
    try:
        DumpTransform().compile({"target": "x", "format": "xml"}, default_model="model")
    except DumpTransformError as exc:
        assert "dump.format must be one of" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected dump.format validation error")


def test_dump_insert_tree_rejects_invalid_node_shape() -> None:
    try:
        insert_into_tree([], ["x"], 1)  # type: ignore[arg-type]
    except DumpTransformError as exc:
        assert "invalid tree structure" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected invalid tree structure error")


def test_dump_compile_defaults() -> None:
    spec = DumpTransform().compile({}, default_model="model")
    assert spec.target_ref.expr == ".*"
    assert spec.format == "compact"
    assert spec.verbosity == "shape"


def test_dump_apply_to_target_is_not_used() -> None:
    try:
        DumpTransform().apply_to_target(
            DumpSpec(target_ref=TensorRef(model="model", expr="x"), format="tree", verbosity="shape"),
            "x",
            provider=None,  # type: ignore[arg-type]
        )
    except AssertionError as exc:
        assert "overrides apply()" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected apply_to_target guard assertion")
