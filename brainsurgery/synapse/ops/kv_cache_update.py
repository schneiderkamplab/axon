from __future__ import annotations

from typing import Any

import torch

OP_NAME = "kv_cache_update"


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
    if not isinstance(ins, list) or len(ins) != 3 or not isinstance(outs, list) or len(outs) != 3:
        raise ValueError("kv_cache_update expects in=[past,k,v], out=[k_all,v_all,present]")
    past = env.get(ins[0])
    k_new = env[ins[1]]
    v_new = env[ins[2]]
    if past is None:
        k_all = k_new
        v_all = v_new
    else:
        k_all = torch.cat([past[0], k_new], dim=-2)
        v_all = torch.cat([past[1], v_new], dim=-2)
    present = (k_all, v_all)
    env[outs[0]] = k_all
    env[outs[1]] = v_all
    env[outs[2]] = present
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
    if not isinstance(ins, list) or len(ins) != 3 or not isinstance(outs, list) or len(outs) != 3:
        raise ValueError("kv_cache_update expects in=[past,k,v], out=[k_all,v_all,present]")
    past = read(str(ins[0])) if str(ins[0]) in env else "None"
    k_new = read(str(ins[1]))
    v_new = read(str(ins[2]))
    k_all = assign_out_var(str(outs[0]))
    v_all = assign_out_var(str(outs[1]))
    present = assign_out_var(str(outs[2]))
    lines.append(f"{indent}if {past} is None:")
    lines.append(f"{indent}    {k_all} = {k_new}")
    lines.append(f"{indent}    {v_all} = {v_new}")
    lines.append(f"{indent}else:")
    lines.append(f"{indent}    {k_all} = torch.cat([{past}[0], {k_new}], dim=-2)")
    lines.append(f"{indent}    {v_all} = torch.cat([{past}[1], {v_new}], dim=-2)")
    lines.append(f"{indent}{present} = ({k_all}, {v_all})")
    return lines


__all__ = ["OP_NAME", "interpret", "compile", "uses_node_path"]
