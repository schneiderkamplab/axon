from importlib import import_module

import pytest

_module = import_module("brainsurgery.transforms.dump")
DumpSpec = _module.DumpSpec
DumpTransform = _module.DumpTransform
DumpTransformError = _module.DumpTransformError
TensorRef = _module.TensorRef
insert_into_tree = _module.insert_into_tree


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
    assert spec.target_ref is None
    assert spec.dump_all_models is True
    assert spec.default_model_hint == "model"
    assert spec.format == "compact"
    assert spec.verbosity == "shape"
    assert spec.collect_models() == {"model"}

    spec_no_default = DumpTransform().compile({}, default_model=None)
    assert spec_no_default.collect_models() == set()


def test_dump_apply_to_target_is_not_used() -> None:
    try:
        DumpTransform().apply_to_target(
            DumpSpec(
                target_ref=TensorRef(model="model", expr="x"), format="tree", verbosity="shape"
            ),
            "x",
            provider=None,  # type: ignore[arg-type]
        )
    except AssertionError as exc:
        assert "overrides apply()" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected apply_to_target guard assertion")


def test_dump_infer_output_model_rejected() -> None:
    try:
        DumpTransform()._infer_output_model(object())
    except DumpTransformError as exc:
        assert "does not infer an output model" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected _infer_output_model validation error")


def test_dump_apply_rejects_missing_target_when_not_dumping_all_models() -> None:
    spec = DumpTransform().build_spec(
        target_ref=None,
        payload={"format": "compact", "verbosity": "shape"},
        dump_all_models=False,
    )
    assert spec.collect_models() == set()
    with pytest.raises(DumpTransformError, match="dump target missing"):
        DumpTransform().apply(spec, provider=None)  # type: ignore[arg-type]
