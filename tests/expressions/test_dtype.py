from importlib import import_module

_module = import_module("brainsurgery.expressions.dtype")
globals().update({name: getattr(_module, name) for name in dir(_module) if not name.startswith("_")})

def test_dtype_compile_rejects_empty_is() -> None:
    try:
        compile_dtype_expr({"of": "x", "is": ""}, default_model="model")
    except TransformError as exc:
        assert "dtype.is must be a non-empty string" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected dtype.is non-empty string validation error")

def test_dtype_evaluate_success() -> None:
    class _Provider:
        def get_state_dict(self, model: str):
            assert model == "model"
            return {"x": torch.ones((2,), dtype=torch.float32)}

    expr = DtypeExpr(ref=TensorRef(model="model", expr="x"), is_value=torch.float32)
    expr.evaluate(_Provider())

def test_dtype_evaluate_mismatch() -> None:
    class _Provider:
        def get_state_dict(self, model: str):
            assert model == "model"
            return {"x": torch.ones((2,), dtype=torch.float16)}

    expr = DtypeExpr(ref=TensorRef(model="model", expr="x"), is_value=torch.float32)
    try:
        expr.evaluate(_Provider())
    except TransformError as exc:
        assert "has dtype torch.float16, expected torch.float32" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected dtype mismatch")

def test_dtype_evaluate_pattern_checks_all_matches() -> None:
    class _Provider:
        def get_state_dict(self, model: str):
            assert model == "model"
            return {
                "x0": torch.ones((2,), dtype=torch.float32),
                "x1": torch.ones((2,), dtype=torch.float16),
            }

    expr = DtypeExpr(ref=TensorRef(model="model", expr="x.*"), is_value=torch.float32)
    try:
        expr.evaluate(_Provider())
    except TransformError as exc:
        assert "model::x1" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected dtype mismatch on one matched tensor")
