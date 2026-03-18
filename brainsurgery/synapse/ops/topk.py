from __future__ import annotations

from typing import Any

import torch

OP_NAME = "topk"


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
    x = model._read_tensor_input(node_spec.get("in"), env)
    outs = node_spec.get("out")
    if not isinstance(outs, list) or len(outs) != 2:
        raise ValueError("topk expects out=[values,indices]")
    k = int(model._eval_expr(node_spec.get("k"), env, symbols))
    dim = int(model._eval_expr(node_spec.get("dim", -1), env, symbols))
    largest = bool(node_spec.get("largest", True))
    sorted_flag = bool(node_spec.get("sorted", True))
    values, indices = torch.topk(x, k, dim=dim, largest=largest, sorted=sorted_flag)
    env[outs[0]] = values
    env[outs[1]] = indices
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

    src = read(str(node_spec.get("in")))
    outs = node_spec.get("out")
    if not isinstance(outs, list) or len(outs) != 2:
        raise ValueError("topk expects out=[values,indices]")
    values_var = assign_out_var(str(outs[0]))
    indices_var = assign_out_var(str(outs[1]))
    k = emitter._expr_code(node_spec.get("k"), env)
    dim = emitter._expr_code(node_spec.get("dim", -1), env)
    largest = bool(node_spec.get("largest", True))
    sorted_flag = bool(node_spec.get("sorted", True))
    lines.append(
        f"{indent}{values_var}, {indices_var} = torch.topk({src}, int({k}), dim=int({dim}), largest={largest}, sorted={sorted_flag})"
    )
    return lines


__all__ = ["OP_NAME", "interpret", "compile", "uses_node_path"]
