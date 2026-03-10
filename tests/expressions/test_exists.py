from importlib import import_module

_module = import_module("brainsurgery.expressions.exists")
globals().update({name: getattr(_module, name) for name in dir(_module) if not name.startswith("_")})


def test_exists_compile_rejects_empty_ref() -> None:
    try:
        compile_exists_expr("", default_model="model")
    except AssertTransformError as exc:
        assert "exists must be a non-empty string reference" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected non-empty reference validation error")


def test_exists_evaluate_success() -> None:
    class _Provider:
        def get_state_dict(self, model: str):
            assert model == "model"
            return {"abc": object()}

    ExistsExpr(ref=TensorRef(model="model", expr="a.*")).evaluate(_Provider())


def test_exists_evaluate_failure() -> None:
    class _Provider:
        def get_state_dict(self, model: str):
            assert model == "model"
            return {"abc": object()}

    try:
        ExistsExpr(ref=TensorRef(model="model", expr="z.*")).evaluate(_Provider())
    except AssertTransformError as exc:
        assert "matched zero tensors" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected exists failure")
