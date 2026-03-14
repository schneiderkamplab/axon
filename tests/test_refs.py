from __future__ import annotations

import pytest
import torch

from brainsurgery.core import (
    TensorRef,
    format_tensor_ref,
    must_model,
    parse_model_expr,
    parse_slice,
    select_tensor,
)
from brainsurgery.core import TransformError
from brainsurgery.core.specs.refs import _looks_like_slice, _validate_expr_kind

def test_parse_model_expr_supports_default_model_explicit_model_and_slices() -> None:
    assert parse_model_expr("weight", default_model="base") == TensorRef(
        model="base",
        expr="weight",
        slice_spec=None,
    )
    assert parse_model_expr("alt::weight") == TensorRef(model="alt", expr="weight", slice_spec=None)
    assert parse_model_expr("weight::[:2]", default_model="base") == TensorRef(
        model="base",
        expr="weight",
        slice_spec="[:2]",
    )
    assert parse_model_expr(["layer", "$idx"], default_model="base") == TensorRef(
        model="base",
        expr=["layer", "$idx"],
        slice_spec=None,
    )

def test_parse_model_expr_rejects_invalid_shapes() -> None:
    with pytest.raises(TransformError, match="missing model alias"):
        parse_model_expr("weight")

    with pytest.raises(TransformError, match="missing model alias"):
        parse_model_expr(["ok", ""])

    with pytest.raises(TransformError, match="non-empty list of non-empty strings"):
        parse_model_expr(["ok", ""], default_model="base")

    with pytest.raises(TransformError, match="invalid slice syntax"):
        parse_model_expr("base::weight::oops")

    with pytest.raises(TransformError, match="structured reference must be a non-empty list"):
        parse_model_expr([], default_model="base")

    with pytest.raises(TransformError, match="non-empty string or a non-empty list"):
        parse_model_expr(123, default_model="base")

    with pytest.raises(TransformError, match="missing model alias"):
        parse_model_expr("::weight::[:2]")

    with pytest.raises(TransformError, match="invalid reference syntax"):
        parse_model_expr("a::b::[:1]::extra")

def test_parse_slice_parses_indices_and_ranges() -> None:
    assert parse_slice("[:]") == (slice(None, None, None),)
    assert parse_slice("[1, 2:5, ::-1]") == (
        1,
        slice(2, 5, None),
        slice(None, None, -1),
    )

def test_parse_slice_rejects_empty_component() -> None:
    with pytest.raises(TransformError, match="empty slice component"):
        parse_slice("[1,,2]")

    with pytest.raises(TransformError, match="invalid slice syntax"):
        parse_slice("oops")

    assert parse_slice("[]") == tuple()

    with pytest.raises(TransformError, match="invalid slice component"):
        parse_slice("[1:2:3:4]")

    with pytest.raises(TransformError, match="invalid integer"):
        parse_slice("[x]")

def test_select_tensor_applies_slice_and_wraps_failures() -> None:
    tensor = torch.arange(12).reshape(3, 4)
    assert torch.equal(select_tensor(tensor, (slice(None), 1)), tensor[:, 1])

    with pytest.raises(TransformError, match="failed to apply slice"):
        select_tensor(tensor, (slice(None), 1, 2))

def test_validate_expr_kind_and_helpers() -> None:
    _validate_expr_kind(expr="layer.*", op_name="copy", role="source")
    _validate_expr_kind(expr=["layer", "$name"], op_name="copy", role="source")
    assert _looks_like_slice("[1:]") is True
    assert format_tensor_ref(TensorRef(model="base", expr="weight", slice_spec="[:2]")) == "base::weight::[:2]"
    assert format_tensor_ref(TensorRef(model="base", expr=["layer", "$i"], slice_spec=None)) == "base::['layer', '$i']"

    with pytest.raises(TransformError, match="invalid type"):
        _validate_expr_kind(expr=1, op_name="copy", role="source")

    with pytest.raises(TransformError, match="missing model alias"):
        must_model(TensorRef(model=None, expr="weight"))

    with pytest.raises(TransformError, match="regex must be non-empty"):
        _validate_expr_kind(expr="", op_name="copy", role="source")

    with pytest.raises(TransformError, match="pattern must be non-empty"):
        _validate_expr_kind(expr=[], op_name="copy", role="source")

    with pytest.raises(TransformError, match="list of non-empty strings"):
        _validate_expr_kind(expr=["ok", ""], op_name="copy", role="source")
