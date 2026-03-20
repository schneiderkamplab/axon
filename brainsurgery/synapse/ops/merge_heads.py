from __future__ import annotations

from typing import Any

OP_NAME = "merge_heads"


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
    bsz, heads, seq_len, head_dim = x.shape
    merged = x.transpose(1, 2).contiguous().view(bsz, seq_len, heads * head_dim)
    out = model._require_name(node_spec.get("_bind"), field="merge_heads._bind")
    env[out] = merged
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
    lines.append(
        f"{indent}{out_var} = {src}.transpose(1, 2).contiguous().view({src}.shape[0], {src}.shape[2], {src}.shape[1] * {src}.shape[3])"
    )
    return lines


__all__ = ["OP_NAME", "interpret", "compile", "uses_node_path"]
