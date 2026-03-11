from __future__ import annotations

import pytest
import torch

from brainsurgery.core import AssertTransformError
from brainsurgery.expressions.equal import EqualExpr, compile_equal_expr
from brainsurgery.core import TensorRef
from brainsurgery.core import TransformError


class DictProvider:
    def __init__(self, state_dicts: dict[str, dict[str, torch.Tensor]]) -> None:
        self._state_dicts = state_dicts

    def get_state_dict(self, model: str):
        return self._state_dicts[model]


def test_equal_dtype_compatibility() -> None:
    provider = DictProvider(
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


def test_equal_shape_compatibility() -> None:
    provider = DictProvider(
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


def test_equal_value_mismatch() -> None:
    provider = DictProvider(
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

    with pytest.raises(AssertTransformError, match="!="):
        expr.evaluate(provider)


def test_equal_compile_rejects_negative_eps() -> None:
    with pytest.raises(AssertTransformError, match="equal.eps must be a non-negative number"):
        compile_equal_expr({"left": "x", "right": "y", "eps": -1e-3}, default_model="model")


def test_equal_value_within_eps_succeeds() -> None:
    provider = DictProvider(
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


def test_equal_value_outside_eps_fails() -> None:
    provider = DictProvider(
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

    with pytest.raises(AssertTransformError, match="!="):
        expr.evaluate(provider)


def test_equal_regex_mapping_success() -> None:
    provider = DictProvider(
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


def test_equal_regex_mapping_missing_destination() -> None:
    provider = DictProvider(
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

    with pytest.raises(AssertTransformError, match="destination missing"):
        expr.evaluate(provider)


def test_equal_structured_mapping_success() -> None:
    provider = DictProvider(
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
