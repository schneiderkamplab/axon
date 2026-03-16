from importlib import import_module

_module = import_module("brainsurgery.expressions.count")
CountExpr = _module.CountExpr
TensorRef = _module.TensorRef
TransformError = _module.TransformError
compile_count_expr = _module.compile_count_expr


def test_count_compile_rejects_non_int_is() -> None:
    try:
        compile_count_expr({"of": "x", "is": "1"}, default_model="model")
    except TransformError as exc:
        assert "count.is must be an integer" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected count.is integer validation error")


def test_count_evaluate_exact_match() -> None:
    class _Provider:
        def get_state_dict(self, model: str):
            assert model == "model"
            return {"a": object(), "b": object(), "c": object()}

    expr = CountExpr(ref=TensorRef(model="model", expr="[ab]"), is_value=2)
    expr.evaluate(_Provider())


def test_count_evaluate_mismatch_raises() -> None:
    class _Provider:
        def get_state_dict(self, model: str):
            assert model == "model"
            return {"a": object(), "b": object()}

    expr = CountExpr(ref=TensorRef(model="model", expr="a"), is_value=2)
    try:
        expr.evaluate(_Provider())
    except TransformError as exc:
        assert "matched 1 tensors, expected 2" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected count mismatch error")
