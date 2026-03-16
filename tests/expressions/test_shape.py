from importlib import import_module

import torch

_module = import_module("brainsurgery.expressions.shape")
ShapeExpr = _module.ShapeExpr
TensorRef = _module.TensorRef
TransformError = _module.TransformError
compile_shape_expr = _module.compile_shape_expr


def test_shape_compile_rejects_non_integer_shape() -> None:
    try:
        compile_shape_expr({"of": "x", "is": [1, "2"]}, default_model="model")
    except TransformError as exc:
        assert "shape.is must be a list of integers" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected shape integer validation error")


def test_shape_evaluate_success() -> None:
    class _Provider:
        def get_state_dict(self, model: str):
            assert model == "model"
            return {"x": torch.ones((2, 3))}

    ShapeExpr(ref=TensorRef(model="model", expr="x"), is_value=(2, 3)).evaluate(_Provider())


def test_shape_evaluate_mismatch() -> None:
    class _Provider:
        def get_state_dict(self, model: str):
            assert model == "model"
            return {"x": torch.ones((2, 3))}

    try:
        ShapeExpr(ref=TensorRef(model="model", expr="x"), is_value=(3, 2)).evaluate(_Provider())
    except TransformError as exc:
        assert "has shape (2, 3), expected (3, 2)" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected shape mismatch")


def test_shape_evaluate_pattern_checks_all_matches() -> None:
    class _Provider:
        def get_state_dict(self, model: str):
            assert model == "model"
            return {"x0": torch.ones((2, 3)), "x1": torch.ones((2, 4))}

    try:
        ShapeExpr(ref=TensorRef(model="model", expr="x.*"), is_value=(2, 3)).evaluate(_Provider())
    except TransformError as exc:
        assert "model::x1" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected shape mismatch on one matched tensor")
