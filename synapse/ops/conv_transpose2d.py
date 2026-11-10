from __future__ import annotations

from typing import Any

from ._spatial2d_common import int_arg, static_dim, transposed_out_dim

OP_NAME = "conv_transpose2d"

# Channels-last input Tensor[B,H,W,C]; weight in the checkpoint layout Tensor[C,O,KH,KW].
LOWERING_TYPE_SIGNATURE = {
    "args": (
        "Tensor[B,H,W,C]",
        "Tensor[C,O,KH,KW]",
        "?Tensor[O]",
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
    "pad_h",
    "pad_w",
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
    if len(arg_types) != 7 or len(args) != 7:
        return None
    x_dims = helpers.type_dims(arg_types[0])
    w_dims = helpers.type_dims(arg_types[1])
    if x_dims is None or w_dims is None or len(x_dims) != 4 or len(w_dims) != 4:
        return None
    batch, height, width, _ = x_dims
    _, out_channels, kernel_h, kernel_w = w_dims
    stride_h, stride_w, pad_h, pad_w = (int_arg(arg, helpers) for arg in args[3:7])
    out_h = transposed_out_dim(
        height, kernel=static_dim(kernel_h, helpers), stride=stride_h, pad=pad_h, axis="h"
    )
    out_w = transposed_out_dim(
        width, kernel=static_dim(kernel_w, helpers), stride=stride_w, pad=pad_w, axis="w"
    )
    return helpers.type_tensor(dims=(batch, out_h, out_w, out_channels))


__all__ = [
    "LOWERING_PARAM_NAMES",
    "LOWERING_TYPE_SIGNATURE",
    "OP_NAME",
    "PRIMITIVE_SEMANTICS",
    "type_rule",
]
