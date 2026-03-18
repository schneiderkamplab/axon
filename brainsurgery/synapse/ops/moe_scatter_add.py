from __future__ import annotations

from typing import Any

OP_NAME = "moe_scatter_add"


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
    if not isinstance(ins, list) or len(ins) != 4:
        raise ValueError("moe_scatter_add expects in=[accum,token_idx,updates,scores]")
    accum = model._read_tensor_input(ins[0], env)
    token_idx = model._read_tensor_input(ins[1], env)
    updates = model._read_tensor_input(ins[2], env)
    scores = model._read_tensor_input(ins[3], env)
    out = model._require_name(node_spec.get("out"), field="moe_scatter_add.out")
    if token_idx.numel() == 0:
        env[out] = accum
        return
    accum_flat = accum.reshape(-1, accum.shape[-1])
    weighted = updates * scores.unsqueeze(-1).to(updates.dtype)
    accum_flat.index_add_(0, token_idx, weighted.to(accum_flat.dtype))
    env[out] = accum
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
    if not isinstance(ins, list) or len(ins) != 4:
        raise ValueError("moe_scatter_add expects in=[accum,token_idx,updates,scores]")
    accum = read(str(ins[0]))
    token_idx = read(str(ins[1]))
    updates = read(str(ins[2]))
    scores = read(str(ins[3]))
    out_name = str(node_spec.get("out"))
    out_var = assign_out_var(out_name)
    lines.append(f"{indent}{out_var} = {accum}")
    lines.append(f"{indent}if {token_idx}.numel() != 0:")
    lines.append(f"{indent}    _acc = {out_var}.reshape(-1, {out_var}.shape[-1])")
    lines.append(f"{indent}    _upd = {updates} * {scores}.unsqueeze(-1).to({updates}.dtype)")
    lines.append(f"{indent}    _acc.index_add_(0, {token_idx}, _upd.to(_acc.dtype))")
    return lines


__all__ = ["OP_NAME", "interpret", "compile", "uses_node_path"]
