from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional as F

OP_NAME = "activation"


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
    kind = node_spec.get("kind", "gelu")
    out = model._require_name(node_spec.get("_bind"), field="activation._bind")
    if kind == "gelu_new" or kind == "gelu_pytorch_tanh":
        env[out] = 0.5 * x * (1.0 + torch.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)))
    elif kind == "gelu":
        env[out] = F.gelu(x)
    elif kind == "relu":
        env[out] = F.relu(x)
    elif kind == "silu":
        env[out] = F.silu(x)
    elif kind == "swiglu":
        env[out] = F.silu(x) * x
    else:
        raise ValueError(f"Unsupported activation kind: {kind}")
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
    kind = node_spec.get("kind", "gelu")
    out_var = assign_out_var(out_name)
    if kind in {"gelu_new", "gelu_pytorch_tanh"}:
        lines.append(
            f"{indent}{out_var} = 0.5 * {src} * (1.0 + torch.tanh(0.7978845608028654 * ({src} + 0.044715 * {src} * {src} * {src})))"
        )
    elif kind == "gelu":
        lines.append(f"{indent}{out_var} = F.gelu({src})")
    elif kind == "relu":
        lines.append(f"{indent}{out_var} = F.relu({src})")
    elif kind == "silu":
        lines.append(f"{indent}{out_var} = F.silu({src})")
    elif kind == "swiglu":
        lines.append(f"{indent}{out_var} = F.silu({src}) * {src}")
    else:
        raise ValueError(f"Unsupported activation kind: {kind}")
    return lines


__all__ = ["OP_NAME", "interpret", "compile", "uses_node_path"]
