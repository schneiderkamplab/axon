from __future__ import annotations

from typing import Any

from ._spatial2d_common import int_arg, window_out_dim

OP_NAME = "avg_pool2d"

# Channels-last input Tensor[B,H,W,C]; unpadded windows.
LOWERING_TYPE_SIGNATURE = {
    "args": (
        "Tensor[B,H,W,C]",
        "Int",
        "Int",
        "Int",
        "Int",
    ),
    "kwargs": {},
    "returns": "dynamic",
}
LOWERING_PARAM_NAMES = (
    "x",
    "kernel_h",
    "kernel_w",
    "stride_h",
    "stride_w",
)
PRIMITIVE_SEMANTICS = {
    "effect": "total_pure",
    "usage": "unrestricted",
}


def type_rule(
    *,
    arg_types: tuple[Any, ...],
    kwarg_types: dict[str, Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    helpers: Any,
) -> Any | None:
    del kwarg_types, kwargs
    if len(arg_types) != 5 or len(args) != 5:
        return None
    x_dims = helpers.type_dims(arg_types[0])
    if x_dims is None or len(x_dims) != 4:
        return None
    batch, height, width, channels = x_dims
    kernel_h, kernel_w, stride_h, stride_w = (
        int_arg(arg, helpers) for arg in args[1:5]
    )
    out_h = window_out_dim(
        height, kernel=kernel_h, stride=stride_h, pad_before=0, pad_after=0,
        dilation=1, axis="h",
    )
    out_w = window_out_dim(
        width, kernel=kernel_w, stride=stride_w, pad_before=0, pad_after=0,
        dilation=1, axis="w",
    )
    return helpers.type_tensor(dims=(batch, out_h, out_w, channels))


__all__ = [
    "LOWERING_PARAM_NAMES",
    "LOWERING_TYPE_SIGNATURE",
    "OP_NAME",
    "PRIMITIVE_SEMANTICS",
    "type_rule",
]
