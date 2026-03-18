from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional as F

OP_NAME = "linear"


def uses_node_path(emitter: Any, node_spec: dict[str, Any]) -> bool:
    del emitter
    tie = node_spec.get("tie_weight")
    has_bias = bool(node_spec["bias"]) if "bias" in node_spec else False
    explicit_weight = node_spec.get("weight")
    has_explicit_weight = isinstance(explicit_weight, str) and "." in explicit_weight
    if not has_bias and (isinstance(tie, str) or has_explicit_weight):
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
    x = model._read_tensor_input(node_spec.get("in"), env)
    linear_weight_path: str | None = node_spec.get("tie_weight")
    if not isinstance(linear_weight_path, str):
        linear_weight_path = model._infer_param_path(
            node_spec, node_path=node_path, param_name="weight"
        )
    weight = model._state[linear_weight_path]

    bias = None
    if node_spec.get("bias", False):
        bias_path = model._infer_param_path(node_spec, node_path=node_path, param_name="bias")
        bias = model._state.get(bias_path)

    weight_layout = str(node_spec.get("weight_layout", "oi"))
    out = model._require_name(node_spec.get("out"), field="linear.out")
    if weight_layout == "oi":
        env[out] = F.linear(x, weight, bias)
    elif weight_layout == "io":
        y = torch.matmul(x, weight)
        env[out] = y + bias if bias is not None else y
    else:
        raise ValueError(f"Unsupported linear weight_layout: {weight_layout}")


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

    src = read(str(node_spec.get("in")))
    out_name = str(node_spec.get("out"))
    out_var = assign_out_var(out_name)
    tie = node_spec.get("tie_weight")
    weight_expr = repr(tie) if isinstance(tie, str) else infer_param("weight")
    has_bias = bool(node_spec["bias"]) if "bias" in node_spec else False
    bias_expr = f"self._state.get({infer_param('bias')})" if has_bias else "None"

    weight_layout = str(node_spec.get("weight_layout", "oi"))
    if weight_layout == "oi":
        lines.append(
            f"{indent}{out_var} = F.linear({src}, self._param({weight_expr}), {bias_expr})"
        )
    elif weight_layout == "io":
        lines.append(f"{indent}{out_var} = torch.matmul({src}, self._param({weight_expr}))")
        lines.append(f"{indent}if {bias_expr} is not None:")
        lines.append(f"{indent}    {out_var} = {out_var} + {bias_expr}")
    else:
        raise ValueError(f"Unsupported linear weight_layout: {weight_layout}")

    return lines


__all__ = ["OP_NAME", "interpret", "compile", "uses_node_path"]
