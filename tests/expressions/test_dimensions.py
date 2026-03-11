from importlib import import_module

_module = import_module("brainsurgery.expressions.dimensions")
globals().update({name: getattr(_module, name) for name in dir(_module) if not name.startswith("_")})


def test_dimensions_compile_rejects_non_int_is() -> None:
    try:
        compile_dimensions_expr({"of": "x", "is": "1"}, default_model="model")
    except TransformError as exc:
        assert "dimensions.is must be a non-negative integer" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected dimensions.is integer validation error")


def test_dimensions_compile_accepts_inequalities() -> None:
    expr = compile_dimensions_expr({"of": "x", "ge": 2, "lt": 5}, default_model="model")
    assert expr.comparison.ge == 2
    assert expr.comparison.lt == 5


def test_dimensions_compile_rejects_contradictory_bounds() -> None:
    try:
        compile_dimensions_expr({"of": "x", "gt": 3, "le": 3}, default_model="model")
    except TransformError as exc:
        assert "contradictory bounds" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected contradictory dimensions bounds error")


def test_dimensions_evaluate_success() -> None:
    class _Provider:
        def get_state_dict(self, model: str):
            assert model == "model"
            return {"x": torch.ones((2, 3, 4))}

    expr = DimensionsExpr(
        ref=TensorRef(model="model", expr="x"),
        comparison=ScalarComparison(exact=None, ge=3, gt=None, le=None, lt=4),
    )
    expr.evaluate(_Provider())


def test_dimensions_evaluate_mismatch() -> None:
    class _Provider:
        def get_state_dict(self, model: str):
            assert model == "model"
            return {"x": torch.ones((2, 3))}

    expr = DimensionsExpr(
        ref=TensorRef(model="model", expr="x"),
        comparison=ScalarComparison(exact=None, ge=3, gt=None, le=None, lt=None),
    )
    try:
        expr.evaluate(_Provider())
    except TransformError as exc:
        assert "has 2 dims, expected >= 3" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected dimension mismatch error")


def test_dimensions_evaluate_pattern_checks_all_matches() -> None:
    class _Provider:
        def get_state_dict(self, model: str):
            assert model == "model"
            return {"x0": torch.ones((2, 3, 4)), "x1": torch.ones((2, 3))}

    expr = DimensionsExpr(
        ref=TensorRef(model="model", expr="x.*"),
        comparison=ScalarComparison(exact=3, ge=None, gt=None, le=None, lt=None),
    )
    try:
        expr.evaluate(_Provider())
    except TransformError as exc:
        assert "model::x1" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected dimensions mismatch on one matched tensor")
