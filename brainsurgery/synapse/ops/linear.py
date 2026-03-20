from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional as F

OP_NAME = "linear"


def _validate_linear_keys(node_spec: dict[str, Any]) -> None:
    if "weight_layout" in node_spec:
        raise ValueError("linear does not support weight_layout; use transpose=true/false")
    if "tie_weight" in node_spec:
        raise ValueError("linear does not support tie_weight; use linear@<path> or weight=<path>")
    if "share" in node_spec:
        raise ValueError("linear does not support share; use linear@<path> or weight=<path>")


def _resolve_transpose(node_spec: dict[str, Any]) -> bool:
    _validate_linear_keys(node_spec)
    transpose = node_spec.get("transpose", False)
    if isinstance(transpose, bool):
        return transpose
    raise ValueError("linear transpose must be boolean when provided")


def uses_node_path(emitter: Any, node_spec: dict[str, Any]) -> bool:
    del emitter
    has_bias = bool(node_spec["bias"]) if "bias" in node_spec else False
    explicit_weight = node_spec.get("weight")
    has_explicit_weight = isinstance(explicit_weight, str) and "." in explicit_weight
    if not has_bias and has_explicit_weight:
        return False
    return True


def interpret(
    model: Any,
    node_spec: dict[str, Any],
    env: dict[str, Any],
    *,
    node_path: str,
    scope: str,
    symbols: dict[str, int],
) -> None:
    del scope, symbols
    x = model._read_tensor_input(node_spec.get("_args"), env)
    linear_weight_path = model._infer_param_path(
        node_spec, node_path=node_path, param_name="weight"
    )
    weight = model._state[linear_weight_path]

    bias = None
    if node_spec.get("bias", False):
        bias_path = model._infer_param_path(node_spec, node_path=node_path, param_name="bias")
        bias = model._state.get(bias_path)

    transpose = _resolve_transpose(node_spec)
    out = model._require_name(node_spec.get("_bind"), field="linear._bind")
    if x.numel() == 0:
        out_dim = int(weight.shape[-1]) if transpose else int(weight.shape[0])
        env[out] = x.new_empty((*x.shape[:-1], out_dim))
        return
    if transpose:
        y = torch.matmul(x, weight)
        env[out] = y + bias if bias is not None else y
    else:
        env[out] = F.linear(x, weight, bias)


def compile(
    emitter: Any,
    node_spec: dict[str, Any],
    env: dict[str, str],
    *,
    node_path_var: str,
    scope_var: str,
    indent: str,
) -> list[str]:
    del scope_var
    lines: list[str] = []

    def assign_out_var(out_name: str) -> str:
        return emitter._assign_out_var(env, out_name)

    def infer_param(param_name: str) -> str:
        return emitter._infer_param_expr(node_spec, node_path_var, param_name)

    def read(name: str) -> str:
        return emitter._read_env_var(env, name)

    src = read(str(node_spec.get("_args")))
    out_name = str(node_spec.get("_bind"))
    out_var = assign_out_var(out_name)
    weight_expr = infer_param("weight")
    has_bias = bool(node_spec["bias"]) if "bias" in node_spec else False
    bias_expr = f"self._state.get({infer_param('bias')})" if has_bias else "None"
    transpose = _resolve_transpose(node_spec)
    out_dim_expr = (
        f"self._param({weight_expr}).shape[-1]"
        if transpose
        else f"self._param({weight_expr}).shape[0]"
    )

    lines.append(f"{indent}if {src}.numel() == 0:")
    lines.append(
        f"{indent}    {out_var} = {src}.new_empty((*{src}.shape[:-1], int({out_dim_expr})))"
    )
    lines.append(f"{indent}else:")
    if transpose:
        lines.append(f"{indent}    {out_var} = torch.matmul({src}, self._param({weight_expr}))")
        lines.append(f"{indent}    if {bias_expr} is not None:")
        lines.append(f"{indent}        {out_var} = {out_var} + {bias_expr}")
    else:
        lines.append(
            f"{indent}    {out_var} = F.linear({src}, self._param({weight_expr}), {bias_expr})"
        )

    return lines


__all__ = ["OP_NAME", "interpret", "compile", "uses_node_path"]
