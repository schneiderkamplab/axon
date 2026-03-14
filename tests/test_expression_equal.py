from __future__ import annotations

from collections.abc import Callable

import pytest
import torch

from brainsurgery.core import TransformError
from brainsurgery.expressions.equal import EqualExpr, compile_equal_expr
from brainsurgery.core import TensorRef

def test_equal_dtype_compatibility(
    multi_model_provider: Callable[[dict[str, dict[str, torch.Tensor]]], object]
) -> None:
    provider = multi_model_provider(
        {
            "model": {
                "left": torch.ones((2, 2), dtype=torch.float32),
                "right": torch.ones((2, 2), dtype=torch.float16),
            }
        }
    )

    expr = EqualExpr(
        left=TensorRef(model="model", expr="left"),
        right=TensorRef(model="model", expr="right"),
    )

    with pytest.raises(TransformError, match="dtype mismatch"):
        expr.evaluate(provider)

def test_equal_shape_compatibility(
    multi_model_provider: Callable[[dict[str, dict[str, torch.Tensor]]], object]
) -> None:
    provider = multi_model_provider(
        {
            "model": {
                "left": torch.ones((2, 2)),
                "right": torch.ones((3, 2)),
            }
        }
    )

    expr = EqualExpr(
        left=TensorRef(model="model", expr="left"),
        right=TensorRef(model="model", expr="right"),
    )

    with pytest.raises(TransformError, match="shape mismatch"):
        expr.evaluate(provider)

def test_equal_value_mismatch(
    multi_model_provider: Callable[[dict[str, dict[str, torch.Tensor]]], object]
) -> None:
    provider = multi_model_provider(
        {
            "model": {
                "left": torch.tensor([1.0, 2.0]),
                "right": torch.tensor([1.0, 3.0]),
            }
        }
    )

    expr = EqualExpr(
        left=TensorRef(model="model", expr="left"),
        right=TensorRef(model="model", expr="right"),
    )

    with pytest.raises(TransformError, match="!="):
        expr.evaluate(provider)

def test_equal_compile_rejects_negative_eps() -> None:
    with pytest.raises(TransformError, match="equal.eps must be a non-negative number"):
        compile_equal_expr({"left": "x", "right": "y", "eps": -1e-3}, default_model="model")

def test_equal_value_within_eps_succeeds(
    multi_model_provider: Callable[[dict[str, dict[str, torch.Tensor]]], object]
) -> None:
    provider = multi_model_provider(
        {
            "model": {
                "left": torch.tensor([1.0, 2.0]),
                "right": torch.tensor([1.0, 2.0005]),
            }
        }
    )

    expr = EqualExpr(
        left=TensorRef(model="model", expr="left"),
        right=TensorRef(model="model", expr="right"),
        eps=1e-3,
    )

    expr.evaluate(provider)

def test_equal_value_outside_eps_fails(
    multi_model_provider: Callable[[dict[str, dict[str, torch.Tensor]]], object]
) -> None:
    provider = multi_model_provider(
        {
            "model": {
                "left": torch.tensor([1.0, 2.0]),
                "right": torch.tensor([1.0, 2.0005]),
            }
        }
    )

    expr = EqualExpr(
        left=TensorRef(model="model", expr="left"),
        right=TensorRef(model="model", expr="right"),
        eps=1e-4,
    )

    with pytest.raises(TransformError, match="!="):
        expr.evaluate(provider)

def test_equal_regex_mapping_success(
    multi_model_provider: Callable[[dict[str, dict[str, torch.Tensor]]], object]
) -> None:
    provider = multi_model_provider(
        {
            "model": {
                "a": torch.tensor([1.0, 2.0]),
                "b": torch.tensor([3.0, 4.0]),
            },
            "orig": {
                "a": torch.tensor([1.0, 2.0]),
                "b": torch.tensor([3.0, 4.0]),
            },
        }
    )

    expr = EqualExpr(
        left=TensorRef(model="model", expr="(.+)"),
        right=TensorRef(model="orig", expr=r"\1"),
    )

    expr.evaluate(provider)

def test_equal_regex_mapping_missing_destination(
    multi_model_provider: Callable[[dict[str, dict[str, torch.Tensor]]], object]
) -> None:
    provider = multi_model_provider(
        {
            "model": {
                "a": torch.tensor([1.0, 2.0]),
                "b": torch.tensor([3.0, 4.0]),
            },
            "orig": {
                "a": torch.tensor([1.0, 2.0]),
            },
        }
    )

    expr = EqualExpr(
        left=TensorRef(model="model", expr="(.+)"),
        right=TensorRef(model="orig", expr=r"\1"),
    )

    with pytest.raises(TransformError, match="destination missing"):
        expr.evaluate(provider)

def test_equal_structured_mapping_success(
    multi_model_provider: Callable[[dict[str, dict[str, torch.Tensor]]], object]
) -> None:
    provider = multi_model_provider(
        {
            "model": {
                "block.0.weight": torch.tensor([1.0, 2.0]),
                "block.1.weight": torch.tensor([3.0, 4.0]),
                "backup.0.weight": torch.tensor([1.0, 2.0]),
                "backup.1.weight": torch.tensor([3.0, 4.0]),
            }
        }
    )

    expr = EqualExpr(
        left=TensorRef(model="model", expr=["block", "$i", "weight"]),
        right=TensorRef(model="model", expr=["backup", "${i}", "weight"]),
    )

    expr.evaluate(provider)
