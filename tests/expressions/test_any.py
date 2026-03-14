from importlib import import_module

_module = import_module("brainsurgery.expressions.any")
globals().update({name: getattr(_module, name) for name in dir(_module) if not name.startswith("_")})

def test_any_compile_rejects_empty_list() -> None:
    try:
        compile_any_expr([], default_model="model")
    except TransformError as exc:
        assert "any must be a non-empty list" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected non-empty list validation error")

def test_any_evaluate_succeeds_when_one_branch_passes() -> None:
    class _FailExpr:
        def evaluate(self, provider) -> None:
            del provider
            raise TransformError("fail")

        def collect_models(self) -> set[str]:
            return {"model"}

    class _PassExpr:
        def evaluate(self, provider) -> None:
            del provider

        def collect_models(self) -> set[str]:
            return {"model"}

    AnyExpr(exprs=[_FailExpr(), _PassExpr()]).evaluate(provider=None)  # type: ignore[arg-type]

def test_any_evaluate_reports_all_failures() -> None:
    class _FailExpr:
        def __init__(self, msg: str) -> None:
            self.msg = msg

        def evaluate(self, provider) -> None:
            del provider
            raise TransformError(self.msg)

        def collect_models(self) -> set[str]:
            return {"model"}

    try:
        AnyExpr(exprs=[_FailExpr("a"), _FailExpr("b")]).evaluate(provider=None)  # type: ignore[arg-type]
    except TransformError as exc:
        message = str(exc)
        assert "all alternatives failed" in message
        assert "- a" in message
        assert "- b" in message
    else:  # pragma: no cover
        raise AssertionError("expected aggregated failure")
