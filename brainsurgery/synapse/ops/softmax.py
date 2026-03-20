from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional as F

OP_NAME = "softmax"


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
    x = model._read_tensor_input(node_spec.get("_args"), env)
    out = model._require_name(node_spec.get("_bind"), field="softmax._bind")
    dim = int(model._eval_expr(node_spec.get("dim", -1), env, symbols))
    dtype_name = node_spec.get("dtype")
    if dtype_name is None:
        env[out] = F.softmax(x, dim=dim)
    else:
        if not isinstance(dtype_name, str):
            raise ValueError("softmax dtype must be a string when provided")
        dtype_map: dict[str, torch.dtype] = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        if dtype_name not in dtype_map:
            raise ValueError(f"Unsupported softmax dtype: {dtype_name}")
        env[out] = F.softmax(x, dim=dim, dtype=dtype_map[dtype_name])
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
    dim = emitter._expr_code(node_spec.get("dim", -1), env)
    dtype_name = node_spec.get("dtype")
    if dtype_name is None:
        lines.append(f"{indent}{out_var} = F.softmax({src}, dim=int({dim}))")
    else:
        if not isinstance(dtype_name, str):
            raise ValueError("softmax dtype must be a string when provided")
        dtype_map: dict[str, str] = {
            "float32": "torch.float32",
            "float16": "torch.float16",
            "bfloat16": "torch.bfloat16",
        }
        if dtype_name not in dtype_map:
            raise ValueError(f"Unsupported softmax dtype: {dtype_name}")
        lines.append(
            f"{indent}{out_var} = F.softmax({src}, dim=int({dim}), dtype={dtype_map[dtype_name]})"
        )
    return lines


__all__ = ["OP_NAME", "interpret", "compile", "uses_node_path"]
