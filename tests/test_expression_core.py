from __future__ import annotations

import pytest

from brainsurgery.core import (
    TransformError,
    collect_expr_models,
    compile_assert_expr,
    compile_shape,
    compile_tensor_ref_expr,
    format_ref,
    get_assert_expr_help,
    get_assert_expr_names,
)
from brainsurgery.core import TensorRef

class _Expr:
    def __init__(self, model: str) -> None:
        self.model = model

    def collect_models(self) -> set[str]:
        return {self.model}

def test_assert_expression_registry_exposes_known_ops() -> None:
    names = get_assert_expr_names()
    assert "equal" in names
    assert get_assert_expr_help("equal").name == "equal"

def test_compile_assert_expr_and_helpers_validate_payloads() -> None:
    expr = compile_assert_expr({"exists": "base::weight"}, default_model=None)
    assert expr.collect_models() == {"base"}

    assert compile_tensor_ref_expr("base::weight::[:1]", None, "exists") == TensorRef(
        model="base",
        expr="weight",
        slice_spec="[:1]",
    )
    assert compile_shape([2, 3]) == (2, 3)
    assert format_ref(TensorRef(model="base", expr="weight")) == "base::weight"
    assert collect_expr_models([_Expr("a"), _Expr("b")]) == {"a", "b"}

    with pytest.raises(TransformError, match="single-key mapping"):
        compile_assert_expr({}, default_model=None)

    with pytest.raises(TransformError, match="must be a list of integers"):
        compile_shape([2, "3"])

def test_compile_tensor_ref_expr_rejects_invalid_payload() -> None:
    with pytest.raises(TransformError, match="non-empty string reference"):
        compile_tensor_ref_expr(1, "base", "exists")
