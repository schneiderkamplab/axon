from importlib import import_module

_module = import_module("brainsurgery.expressions.all")
AllExpr = _module.AllExpr
TransformError = _module.TransformError
compile_all_expr = _module.compile_all_expr


def test_all_compile_rejects_empty_list() -> None:
    try:
        compile_all_expr([], default_model="model")
    except TransformError as exc:
        assert "all must be a non-empty list" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected non-empty list validation error")


def test_all_evaluate_short_success_path() -> None:
    class _Expr:
        def __init__(self) -> None:
            self.called = False

        def evaluate(self, provider) -> None:
            del provider
            self.called = True

        def collect_models(self) -> set[str]:
            return {"model"}

    left = _Expr()
    right = _Expr()
    expr = AllExpr(exprs=[left, right])
    expr.evaluate(provider=None)  # type: ignore[arg-type]
    assert left.called and right.called
