from __future__ import annotations

from typing import Any

import torch

OP_NAME = "rmsnorm"


def uses_node_path(emitter: Any, node_spec: dict[str, Any]) -> bool:
    del emitter, node_spec
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
    x = model._read_tensor_input(node_spec.get("_args"), env)
    weight = model._state[
        model._infer_param_path(node_spec, node_path=node_path, param_name="weight")
    ]
    eps_value = float(model._eval_expr(node_spec.get("eps", 1e-6), env, symbols))
    cast_float = bool(node_spec.get("cast_float", False))
    unit_offset = bool(node_spec.get("unit_offset", False))
    x_norm_src = x.float() if cast_float else x
    w_src = weight.float() if cast_float else weight
    x_norm = x_norm_src * torch.rsqrt(
        torch.mean(x_norm_src * x_norm_src, dim=-1, keepdim=True) + eps_value
    )
    y = x_norm * ((1.0 + w_src) if unit_offset else w_src)
    out = model._require_name(node_spec.get("_bind"), field="rmsnorm._bind")
    env[out] = y.type_as(x) if cast_float else y
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

    src = read(str(node_spec.get("_args")))
    out_name = str(node_spec.get("_bind"))
    out_var = assign_out_var(out_name)
    eps = emitter._expr_code(node_spec.get("eps", 1e-6), env)
    tmp = emitter._fresh("xnorm")
    cast_float = bool(node_spec.get("cast_float", False))
    unit_offset = bool(node_spec.get("unit_offset", False))
    x_norm_src = f"{src}.float()" if cast_float else src
    w_src = (
        f"emitter._param({infer_param('weight')}).float()"
        if cast_float
        else f"emitter._param({infer_param('weight')})"
    )
    lines.append(
        f"{indent}{tmp} = {x_norm_src} * torch.rsqrt(torch.mean({x_norm_src} * {x_norm_src}, dim=-1, keepdim=True) + float({eps}))"
    )
    if unit_offset:
        lines.append(f"{indent}{out_var} = {tmp} * (1.0 + {w_src})")
    else:
        lines.append(f"{indent}{out_var} = {tmp} * {w_src}")
    if cast_float:
        lines.append(f"{indent}{out_var} = {out_var}.type_as({src})")
    return lines


__all__ = ["OP_NAME", "interpret", "compile", "uses_node_path"]
