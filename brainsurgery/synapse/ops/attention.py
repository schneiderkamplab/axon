from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional as F

OP_NAME = "attention"
LOWERING_ARITY = (3, 3)
LOWERING_ALLOWED_KWARGS: set[str] = {"scale", "mask", "causal"}
LOWERING_REQUIRED_KWARGS: set[str] = set()
LOWERING_KWARG_KINDS: dict[str, Any] = {"causal": "bool", "mask": "str", "scale": "number"}


def uses_node_path(emitter: Any, node_spec: dict[str, Any]) -> bool:
    del emitter, node_spec
    return False


def interpret(
    model: Any,
    node_spec: dict[str, Any],
    env: dict[str, Any],
    *,
    node_path: str,
    scope: str,
    symbols: dict[str, int],
) -> None:
    ins = node_spec.get("_args")
    if not isinstance(ins, list) or len(ins) != 3:
        raise ValueError("attention expects [q, k, v]")
    q = env[ins[0]]
    k_tensor = env[ins[1]]
    v = env[ins[2]]
    mask = None
    mask_name = node_spec.get("mask")
    if isinstance(mask_name, str):
        mask = env.get(mask_name)
    if bool(getattr(model, "_hf_align_mask_contract", False)):
        if torch.is_tensor(mask) and mask.is_floating_point() and mask.numel() > 0:
            mask_max = float(mask.max())
            mask_min = float(mask.min())
            mask_floor = float(torch.finfo(mask.dtype).min)
            if mask_max == 0.0 and mask_min <= (0.5 * mask_floor):
                # Convert additive mask (0 / -inf-like) to bool keep-mask.
                mask = mask == 0
    scale_expr = node_spec.get("scale")
    scale_value = None if scale_expr is None else float(model._eval_expr(scale_expr, env, symbols))
    is_causal_flag = bool(node_spec.get("causal", True)) and q.shape[2] > 1 and mask is None
    attn_out = F.scaled_dot_product_attention(
        q,
        k_tensor,
        v,
        attn_mask=mask,
        dropout_p=0.0,
        is_causal=is_causal_flag,
        scale=scale_value,
    )
    out_name = model._require_name(node_spec.get("_bind"), field="attention._bind")
    env[out_name] = attn_out
    return


def compile(
    emitter: Any,
    node_spec: dict[str, Any],
    env: dict[str, str],
    *,
    node_path_var: str,
    scope_var: str,
    indent: str,
) -> list[str]:
    lines: list[str] = []

    def assign_out_var(out_name: str) -> str:
        return emitter._assign_out_var(env, out_name)

    def infer_param(param_name: str) -> str:
        return emitter._infer_param_expr(node_spec, node_path_var, param_name)

    def read(name: str) -> str:
        return emitter._read_env_var(env, name)

    ins = node_spec.get("_args")
    if not isinstance(ins, list) or len(ins) != 3:
        raise ValueError("attention expects 3 inputs")
    q = read(str(ins[0]))
    k = read(str(ins[1]))
    v = read(str(ins[2]))
    out_name = str(node_spec.get("_bind"))
    out_var = assign_out_var(out_name)
    mask_name = node_spec.get("mask")
    mask_expr = "None"
    if isinstance(mask_name, str) and mask_name in env:
        mask_expr = env[mask_name]
    mask_for_sdpa = mask_expr
    if mask_expr != "None":
        normalized_mask = emitter._fresh("mask_for_sdpa")
        mask_max = emitter._fresh("mask_max")
        mask_min = emitter._fresh("mask_min")
        mask_floor = emitter._fresh("mask_floor")
        lines.append(f"{indent}{normalized_mask} = {mask_expr}")
        lines.append(f"{indent}if getattr(self, '_hf_align_mask_contract', False):")
        lines.append(
            f"{indent}    if torch.is_tensor({normalized_mask}) and {normalized_mask}.is_floating_point() and {normalized_mask}.numel() > 0:"
        )
        lines.append(f"{indent}        {mask_max} = float({normalized_mask}.max())")
        lines.append(f"{indent}        {mask_min} = float({normalized_mask}.min())")
        lines.append(
            f"{indent}        {mask_floor} = float(torch.finfo({normalized_mask}.dtype).min)"
        )
        lines.append(
            f"{indent}        if {mask_max} == 0.0 and {mask_min} <= (0.5 * {mask_floor}):"
        )
        lines.append(f"{indent}            {normalized_mask} = ({normalized_mask} == 0)")
        mask_for_sdpa = normalized_mask
    if bool(node_spec.get("causal", True)):
        is_causal = f"({q}.shape[-2] > 1 and {mask_for_sdpa} is None)"
    else:
        is_causal = "False"
    scale_value = node_spec.get("scale")
    scale_expr = "None" if scale_value is None else emitter._expr_code(scale_value, env)
    lines.append(
        f"{indent}{out_var} = F.scaled_dot_product_attention({q}, {k}, {v}, attn_mask={mask_for_sdpa}, dropout_p=0.0, is_causal={is_causal}, scale={scale_expr})"
    )
    return lines


__all__ = [
    "LOWERING_ARITY",
    "LOWERING_ALLOWED_KWARGS",
    "LOWERING_REQUIRED_KWARGS",
    "LOWERING_KWARG_KINDS",
    "OP_NAME",
    "interpret",
    "compile",
    "uses_node_path",
]
