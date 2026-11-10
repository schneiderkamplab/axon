from __future__ import annotations

from typing import Any

OP_NAME = "upsample_nearest2d_like"

# Nearest-neighbour upsampling of channels-last x to the spatial size of ref, which must be
# an integer multiple of the size of x.
LOWERING_TYPE_SIGNATURE = {
    "args": ("Tensor[B,H,W,C]", "Tensor[B,Ho,Wo,D]"),
    "kwargs": {},
    "returns": "dynamic",
}
LOWERING_PARAM_NAMES = ("x", "ref")
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
    del kwarg_types, args, kwargs
    if len(arg_types) != 2:
        return None
    x_dims = helpers.type_dims(arg_types[0])
    ref_dims = helpers.type_dims(arg_types[1])
    if x_dims is None or ref_dims is None or len(x_dims) != 4 or len(ref_dims) != 4:
        return None
    return helpers.type_tensor(dims=(x_dims[0], ref_dims[1], ref_dims[2], x_dims[3]))


__all__ = [
    "LOWERING_PARAM_NAMES",
    "LOWERING_TYPE_SIGNATURE",
    "OP_NAME",
    "PRIMITIVE_SEMANTICS",
    "type_rule",
]
