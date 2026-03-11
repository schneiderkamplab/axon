from __future__ import annotations

import pytest
import torch

from brainsurgery.core import AssertTransformError
from brainsurgery.expressions.iszero import IsZeroExpr, compile_iszero_expr
from brainsurgery.core import TensorRef


class DictProvider:
    def __init__(self, state_dict: dict[str, torch.Tensor]) -> None:
        self._state_dict = state_dict

    def get_state_dict(self, model: str):
        assert model == "model"
        return self._state_dict


def test_iszero_compile_rejects_invalid_ref_type() -> None:
    with pytest.raises(AssertTransformError, match="iszero must be a non-empty string reference"):
        compile_iszero_expr(123, default_model="model")


def test_iszero_evaluate_success() -> None:
    provider = DictProvider({"x": torch.zeros((2, 2))})
    IsZeroExpr(ref=TensorRef(model="model", expr="x")).evaluate(provider)


def test_iszero_evaluate_failure() -> None:
    provider = DictProvider({"x": torch.tensor([0.0, 1.0])})

    with pytest.raises(AssertTransformError, match="not all zeros"):
        IsZeroExpr(ref=TensorRef(model="model", expr="x")).evaluate(provider)


def test_iszero_evaluate_pattern_checks_all_matches() -> None:
    provider = DictProvider(
        {
            "x0": torch.zeros((2, 2)),
            "x1": torch.tensor([0.0, 1.0]),
        }
    )

    with pytest.raises(AssertTransformError, match="model::x1"):
        IsZeroExpr(ref=TensorRef(model="model", expr="x.*")).evaluate(provider)


def test_iszero_compile_rejects_negative_eps() -> None:
    with pytest.raises(AssertTransformError, match="iszero.eps must be a non-negative number"):
        compile_iszero_expr({"of": "x", "eps": -1e-3}, default_model="model")


def test_iszero_within_eps_succeeds() -> None:
    provider = DictProvider({"x": torch.tensor([0.0, 5e-4])})
    IsZeroExpr(ref=TensorRef(model="model", expr="x"), eps=1e-3).evaluate(provider)


def test_iszero_outside_eps_fails() -> None:
    provider = DictProvider({"x": torch.tensor([0.0, 5e-4])})

    with pytest.raises(AssertTransformError, match="within eps"):
        IsZeroExpr(ref=TensorRef(model="model", expr="x"), eps=1e-4).evaluate(provider)
