from __future__ import annotations

from hashlib import blake2s
from typing import Any

from ..axon.ast import AxonExprAscribe, AxonExprInt, AxonExprParen, DimExprBinary

# Channels-last 2D ops (Tensor[B,H,W,C]). A window output extent is
# floor((size + pad_before + pad_after - dilation * (kernel - 1) - 1) / stride) + 1.
# With static window geometry it is typed as that expression of the input extent (dim
# division floors), which stays exact when the input dim is substituted at a call site.
# Without static geometry the extent is an opaque dim named after its inputs, so that
# repeated inference of one call is stable.


def derived_spatial_dim(axis: str, size: Any, *geometry: Any) -> str:
    digest = blake2s(repr((size, geometry)).encode(), digest_size=5).hexdigest()
    return f"__sp{axis}_{digest}"


def dim_add(dim: Any, offset: int) -> Any:
    if offset == 0:
        return dim
    if type(dim) is int:
        return dim + offset
    if isinstance(dim, DimExprBinary) and dim.op in {"+", "-"} and type(dim.right) is int:
        inner = dim.right if dim.op == "+" else -dim.right
        return dim_add(dim.left, inner + offset)
    return DimExprBinary(op="+" if offset > 0 else "-", left=dim, right=abs(offset))


def dim_scale(dim: Any, factor: int) -> Any:
    if factor == 1:
        return dim
    if type(dim) is int:
        return dim * factor
    return DimExprBinary(op="*", left=dim, right=factor)


def dim_floor_div(dim: Any, divisor: int) -> Any:
    if divisor == 1:
        return dim
    if type(dim) is int:
        return dim // divisor
    return DimExprBinary(op="/", left=dim, right=divisor)


def int_arg(expr: Any, helpers: Any) -> int | None:
    current = expr
    while isinstance(current, AxonExprAscribe | AxonExprParen):
        current = current.expr if isinstance(current, AxonExprAscribe) else current.inner
    if isinstance(current, AxonExprInt):
        return int(current.value)
    token = helpers.expr_to_dim_token(current)
    seen: set[str] = set()
    while isinstance(token, str) and token not in seen:
        seen.add(token)
        resolved = helpers.resolve_name_expr(token)
        if resolved is None:
            break
        token = helpers.expr_to_dim_token(resolved)
    return int(token) if type(token) is int else None


def static_dim(dim: Any, helpers: Any) -> Any:
    """Resolve a dim that names a statically known integer; other dims pass through."""
    seen: set[str] = set()
    while isinstance(dim, str) and dim not in seen:
        seen.add(dim)
        resolved = helpers.resolve_name_expr(dim)
        if resolved is None:
            break
        value = int_arg(resolved, helpers)
        if value is not None:
            return value
        dim = helpers.expr_to_dim_token(resolved)
    return dim


def window_out_dim(
    size: Any,
    *,
    kernel: Any,
    stride: int | None,
    pad_before: int | None,
    pad_after: int | None,
    dilation: int | None,
    axis: str,
) -> Any:
    if all(type(v) is int for v in (kernel, stride, pad_before, pad_after, dilation)):
        span = pad_before + pad_after - dilation * (kernel - 1)
        if stride == 1:
            return dim_add(size, span)
        return dim_add(dim_floor_div(dim_add(size, span - 1), stride), 1)
    return derived_spatial_dim(axis, size, kernel, stride, pad_before, pad_after, dilation)


def transposed_out_dim(size: Any, *, kernel: Any, stride: int | None, pad: int | None, axis: str) -> Any:
    """(size - 1) * stride - 2 * pad + kernel."""
    if all(type(v) is int for v in (kernel, stride, pad)):
        return dim_add(dim_scale(size, stride), kernel - stride - 2 * pad)
    return derived_spatial_dim(axis, size, "transpose", kernel, stride, pad)
