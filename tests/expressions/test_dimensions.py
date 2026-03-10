from importlib import import_module

_module = import_module("brainsurgery.expressions.dimensions")
globals().update({name: getattr(_module, name) for name in dir(_module) if not name.startswith("_")})


def test_dimensions_compile_rejects_non_int_is() -> None:
    try:
        compile_dimensions_expr({"of": "x", "is": "1"}, default_model="model")
    except AssertTransformError as exc:
        assert "dimensions.is must be an integer" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected dimensions.is integer validation error")


def test_dimensions_evaluate_success() -> None:
    class _Provider:
        def get_state_dict(self, model: str):
            assert model == "model"
            return {"x": torch.ones((2, 3, 4))}

    expr = DimensionsExpr(ref=TensorRef(model="model", expr="x"), is_value=3)
    expr.evaluate(_Provider())


def test_dimensions_evaluate_mismatch() -> None:
    class _Provider:
        def get_state_dict(self, model: str):
            assert model == "model"
            return {"x": torch.ones((2, 3))}

    expr = DimensionsExpr(ref=TensorRef(model="model", expr="x"), is_value=3)
    try:
        expr.evaluate(_Provider())
    except AssertTransformError as exc:
        assert "has 2 dims, expected 3" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected dimension mismatch error")


def test_dimensions_evaluate_pattern_checks_all_matches() -> None:
    class _Provider:
        def get_state_dict(self, model: str):
            assert model == "model"
            return {"x0": torch.ones((2, 3, 4)), "x1": torch.ones((2, 3))}

    expr = DimensionsExpr(ref=TensorRef(model="model", expr="x.*"), is_value=3)
    try:
        expr.evaluate(_Provider())
    except AssertTransformError as exc:
        assert "model::x1" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected dimensions mismatch on one matched tensor")
