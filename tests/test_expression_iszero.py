from __future__ import annotations

from collections.abc import Callable

import pytest
import torch

from brainsurgery.core import TensorRef, TransformError
from brainsurgery.expressions.iszero import IsZeroExpr, compile_iszero_expr


def test_iszero_compile_rejects_invalid_ref_type() -> None:
    with pytest.raises(TransformError, match="iszero must be a non-empty string reference"):
        compile_iszero_expr(123, default_model="model")


def test_iszero_evaluate_success(
    single_model_provider: Callable[[dict[str, torch.Tensor], str], object],
) -> None:
    provider = single_model_provider({"x": torch.zeros((2, 2))})
    IsZeroExpr(ref=TensorRef(model="model", expr="x")).evaluate(provider)


def test_iszero_evaluate_failure(
    single_model_provider: Callable[[dict[str, torch.Tensor], str], object],
) -> None:
    provider = single_model_provider({"x": torch.tensor([0.0, 1.0])})

    with pytest.raises(TransformError, match="not all zeros"):
        IsZeroExpr(ref=TensorRef(model="model", expr="x")).evaluate(provider)


def test_iszero_evaluate_pattern_checks_all_matches(
    single_model_provider: Callable[[dict[str, torch.Tensor], str], object],
) -> None:
    provider = single_model_provider(
        {
            "x0": torch.zeros((2, 2)),
            "x1": torch.tensor([0.0, 1.0]),
        }
    )

    with pytest.raises(TransformError, match="model::x1"):
        IsZeroExpr(ref=TensorRef(model="model", expr="x.*")).evaluate(provider)


def test_iszero_compile_rejects_negative_eps() -> None:
    with pytest.raises(TransformError, match="iszero.eps must be a non-negative number"):
        compile_iszero_expr({"of": "x", "eps": -1e-3}, default_model="model")


def test_iszero_within_eps_succeeds(
    single_model_provider: Callable[[dict[str, torch.Tensor], str], object],
) -> None:
    provider = single_model_provider({"x": torch.tensor([0.0, 5e-4])})
    IsZeroExpr(ref=TensorRef(model="model", expr="x"), eps=1e-3).evaluate(provider)


def test_iszero_outside_eps_fails(
    single_model_provider: Callable[[dict[str, torch.Tensor], str], object],
) -> None:
    provider = single_model_provider({"x": torch.tensor([0.0, 5e-4])})

    with pytest.raises(TransformError, match="within eps"):
        IsZeroExpr(ref=TensorRef(model="model", expr="x"), eps=1e-4).evaluate(provider)
