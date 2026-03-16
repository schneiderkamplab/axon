from importlib import import_module

import pytest

_module = import_module("brainsurgery.expressions.exists")
ExistsExpr = _module.ExistsExpr
TensorRef = _module.TensorRef
TransformError = _module.TransformError
compile_exists_expr = _module.compile_exists_expr


def test_exists_compile_rejects_empty_ref() -> None:
    try:
        compile_exists_expr("", default_model="model")
    except TransformError as exc:
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
    except TransformError as exc:
        assert "matched zero tensors" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected exists failure")


def test_exists_evaluate_zero_match_branch_with_monkeypatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_module, "resolve_matches", lambda *_args, **_kwargs: [])
    with pytest.raises(TransformError, match="matched zero tensors"):
        ExistsExpr(ref=TensorRef(model="model", expr="x")).evaluate(provider=object())  # type: ignore[arg-type]
