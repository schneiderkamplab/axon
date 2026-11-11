from __future__ import annotations

from typing import Any

OP_NAME = "global_avg_pool2d"

# Channels-last input Tensor[B,H,W,C]; the mean over H and W, kept as Tensor[B,1,1,C].
LOWERING_TYPE_SIGNATURE = {
    "args": ("Tensor[B,H,W,C]",),
    "kwargs": {},
    "returns": "dynamic",
}
LOWERING_PARAM_NAMES = ("x",)
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
    if len(arg_types) != 1:
        return None
    x_dims = helpers.type_dims(arg_types[0])
    if x_dims is None or len(x_dims) != 4:
        return None
    return helpers.type_tensor(dims=(x_dims[0], 1, 1, x_dims[3]))


__all__ = [
    "LOWERING_PARAM_NAMES",
    "LOWERING_TYPE_SIGNATURE",
    "OP_NAME",
    "PRIMITIVE_SEMANTICS",
    "type_rule",
]
