from __future__ import annotations

from typing import Any

from ._spatial2d_common import int_arg, static_dim, window_out_dim

OP_NAME = "conv2d"

# Channels-last input Tensor[B,H,W,C]; weight in the checkpoint layout Tensor[O,C/groups,KH,KW].
LOWERING_TYPE_SIGNATURE = {
    "args": (
        "Tensor[B,H,W,C]",
        "Tensor[O,I,KH,KW]",
        "?Tensor[O]",
        "Int",
        "Int",
        "Int",
        "Int",
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
    "weight",
    "bias",
    "stride_h",
    "stride_w",
    "pad_top",
    "pad_bottom",
    "pad_left",
    "pad_right",
    "dilation",
    "groups",
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
    if len(arg_types) != 11 or len(args) != 11:
        return None
    x_dims = helpers.type_dims(arg_types[0])
    w_dims = helpers.type_dims(arg_types[1])
    if x_dims is None or w_dims is None or len(x_dims) != 4 or len(w_dims) != 4:
        return None
    batch, height, width, _ = x_dims
    out_channels, _, kernel_h, kernel_w = w_dims
    kernel_h = static_dim(kernel_h, helpers)
    kernel_w = static_dim(kernel_w, helpers)
    stride_h, stride_w, pad_top, pad_bottom, pad_left, pad_right, dilation = (
        int_arg(arg, helpers) for arg in args[3:10]
    )
    out_h = window_out_dim(
        height, kernel=kernel_h, stride=stride_h, pad_before=pad_top, pad_after=pad_bottom,
        dilation=dilation, axis="h",
    )
    out_w = window_out_dim(
        width, kernel=kernel_w, stride=stride_w, pad_before=pad_left, pad_after=pad_right,
        dilation=dilation, axis="w",
    )
    return helpers.type_tensor(dims=(batch, out_h, out_w, out_channels))


__all__ = [
    "LOWERING_PARAM_NAMES",
    "LOWERING_TYPE_SIGNATURE",
    "OP_NAME",
    "PRIMITIVE_SEMANTICS",
    "type_rule",
]
