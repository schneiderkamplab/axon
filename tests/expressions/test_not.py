from importlib import import_module

_module = import_module("brainsurgery.expressions.not")
NotExpr = _module.NotExpr
TransformError = _module.TransformError
compile_not_expr = _module.compile_not_expr


def test_not_compile_rejects_invalid_expr() -> None:
    try:
        compile_not_expr({"unknown": {}}, default_model="model")
    except TransformError as exc:
        assert "unknown assert op" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected unknown assert op error")


def test_not_evaluate_succeeds_when_inner_fails() -> None:
    class _FailExpr:
        def evaluate(self, provider) -> None:
            del provider
            raise TransformError("fail")

        def collect_models(self) -> set[str]:
            return {"model"}

    NotExpr(expr=_FailExpr()).evaluate(provider=None)  # type: ignore[arg-type]


def test_not_evaluate_fails_when_inner_succeeds() -> None:
    class _PassExpr:
        def evaluate(self, provider) -> None:
            del provider

        def collect_models(self) -> set[str]:
            return {"model"}

    try:
        NotExpr(expr=_PassExpr()).evaluate(provider=None)  # type: ignore[arg-type]
    except TransformError as exc:
        assert "inner assertion succeeded" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected not failure")
