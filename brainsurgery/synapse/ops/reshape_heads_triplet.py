from __future__ import annotations

from typing import Any

OP_NAME = "reshape_heads_triplet"


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
    ins = node_spec.get("in")
    outs = node_spec.get("out")
    heads = int(model._eval_expr(node_spec.get("heads"), env, symbols))
    head_dim = int(model._eval_expr(node_spec.get("head_dim"), env, symbols))
    if not isinstance(ins, list) or not isinstance(outs, list) or len(ins) != 3 or len(outs) != 3:
        raise ValueError("reshape_heads_triplet requires 3 inputs and 3 outputs")
    for src_name, dst_name in zip(ins, outs, strict=True):
        src = env[src_name]
        bsz, seq_len, _ = src.shape
        reshaped = src.view(bsz, seq_len, heads, head_dim).transpose(1, 2)
        env[dst_name] = reshaped
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

    ins = node_spec.get("in")
    outs = node_spec.get("out")
    if not isinstance(ins, list) or not isinstance(outs, list) or len(ins) != 3 or len(outs) != 3:
        raise ValueError("reshape_heads_triplet requires 3 inputs and outputs")
    heads = emitter._expr_code(node_spec.get("heads"), env)
    head_dim = emitter._expr_code(node_spec.get("head_dim"), env)
    for src_name, out_name in zip(ins, outs, strict=True):
        src = read(str(src_name))
        out_var = assign_out_var(str(out_name))
        lines.append(
            f"{indent}{out_var} = {src}.view({src}.shape[0], {src}.shape[1], int({heads}), int({head_dim})).transpose(1, 2)"
        )
    return lines


__all__ = ["OP_NAME", "interpret", "compile", "uses_node_path"]
