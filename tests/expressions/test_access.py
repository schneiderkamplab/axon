import pytest
import torch

import brainsurgery.expressions.access as access_module
from brainsurgery.core import TensorRef, TransformError
from brainsurgery.engine.state_dicts import _InMemoryStateDict
from brainsurgery.expressions.access import (
    ScalarComparison,
    TensorAccessExpr,
    compile_reads_expr,
    compile_writes_expr,
)


def test_reads_compile_requires_comparison() -> None:
    try:
        compile_reads_expr({"of": "x"}, default_model="model")
    except TransformError as exc:
        assert "must include at least one of" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected reads comparison validation error")


def test_writes_compile_rejects_contradictory_bounds() -> None:
    try:
        compile_writes_expr({"of": "x", "gt": 2, "lt": 2}, default_model="model")
    except TransformError as exc:
        assert "contradictory bounds" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected writes bounds validation error")


def test_reads_compile_accepts_generalized_comparators_and_aliases() -> None:
    expr = compile_reads_expr(
        {"of": "x", "ge": 1, "lt": 3, "at_least": 1, "at_most": 2},
        default_model="model",
    )
    assert expr.comparison.ge == 1
    assert expr.comparison.lt == 3
    assert expr.comparison.le == 2


def test_reads_evaluate_supports_ge() -> None:
    state_dict = _InMemoryStateDict()
    state_dict["a"] = torch.ones(1)
    state_dict["b"] = torch.ones(1)
    _ = state_dict["a"]
    _ = state_dict["b"]

    class _Provider:
        def get_state_dict(self, model: str):
            assert model == "model"
            return state_dict

    expr = TensorAccessExpr(
        ref=TensorRef(model="model", expr=".*"),
        field="reads",
        comparison=ScalarComparison(exact=None, ge=1, gt=None, le=None, lt=None),
    )
    expr.evaluate(_Provider())


def test_writes_evaluate_supports_lt() -> None:
    state_dict = _InMemoryStateDict()
    state_dict["a"] = torch.ones(1)

    class _Provider:
        def get_state_dict(self, model: str):
            assert model == "model"
            return state_dict

    expr = TensorAccessExpr(
        ref=TensorRef(model="model", expr="a"),
        field="writes",
        comparison=ScalarComparison(exact=None, ge=None, gt=None, le=None, lt=2),
    )
    expr.evaluate(_Provider())


def test_writes_evaluate_reports_mismatch() -> None:
    state_dict = _InMemoryStateDict()
    state_dict["a"] = torch.ones(1)

    class _Provider:
        def get_state_dict(self, model: str):
            assert model == "model"
            return state_dict

    expr = TensorAccessExpr(
        ref=TensorRef(model="model", expr="a"),
        field="writes",
        comparison=ScalarComparison(exact=0, ge=None, gt=None, le=None, lt=None),
    )
    try:
        expr.evaluate(_Provider())
    except TransformError as exc:
        assert "writes=1" in str(exc)
        assert "expected exactly 0" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected writes mismatch")


def test_reads_evaluate_rejects_uninstrumented_state_dict() -> None:
    class _Provider:
        def get_state_dict(self, model: str):
            assert model == "model"
            return {"a": torch.ones(1)}

    expr = TensorAccessExpr(
        ref=TensorRef(model="model", expr="a"),
        field="reads",
        comparison=ScalarComparison(exact=0, ge=None, gt=None, le=None, lt=None),
    )
    try:
        expr.evaluate(_Provider())
    except TransformError as exc:
        assert "instrumented state_dict backend" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected unsupported backend error")


def test_reads_evaluate_reports_zero_matches(monkeypatch: pytest.MonkeyPatch) -> None:
    state_dict = _InMemoryStateDict()
    state_dict["a"] = torch.ones(1)

    class _Provider:
        def get_state_dict(self, model: str):
            assert model == "model"
            return state_dict

    monkeypatch.setattr(access_module, "resolve_matches", lambda *_args, **_kwargs: [])
    expr = TensorAccessExpr(
        ref=TensorRef(model="model", expr="missing"),
        field="reads",
        comparison=ScalarComparison(exact=0, ge=None, gt=None, le=None, lt=None),
    )
    with pytest.raises(TransformError, match="matched zero tensors"):
        expr.evaluate(_Provider())


def test_tensor_access_collect_models() -> None:
    expr = TensorAccessExpr(
        ref=TensorRef(model="model", expr="x"),
        field="reads",
        comparison=ScalarComparison(exact=1, ge=None, gt=None, le=None, lt=None),
    )
    assert expr.collect_models() == {"model"}
