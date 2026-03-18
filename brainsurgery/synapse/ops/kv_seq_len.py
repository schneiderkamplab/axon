from __future__ import annotations

from typing import Any

import torch

OP_NAME = "kv_seq_len"


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
    ref = node_spec.get("in")
    out_name = model._require_name(node_spec.get("out"), field="kv_seq_len.out")
    if not isinstance(ref, str):
        raise ValueError("kv_seq_len.in must be a string")
    value = env.get(ref)
    if value is None:
        env[out_name] = 0
        return
    if not isinstance(value, tuple) or len(value) < 1 or not torch.is_tensor(value[0]):
        raise ValueError("kv_seq_len expects kv tuple (k, v)")
    env[out_name] = int(value[0].shape[-2])
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

    ref = node_spec.get("in")
    if not isinstance(ref, str):
        raise ValueError("kv_seq_len.in must be string")
    out_name = str(node_spec.get("out"))
    out_var = assign_out_var(out_name)
    src = read(ref)
    lines.append(f"{indent}if {src} is None:")
    lines.append(f"{indent}    {out_var} = 0")
    lines.append(f"{indent}else:")
    lines.append(f"{indent}    {out_var} = int({src}[0].shape[-2])")
    return lines


__all__ = ["OP_NAME", "interpret", "compile", "uses_node_path"]
