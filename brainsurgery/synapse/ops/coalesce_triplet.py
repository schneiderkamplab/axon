from __future__ import annotations

from typing import Any

OP_NAME = "coalesce_triplet"


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
    if not isinstance(ins, list) or len(ins) != 4 or not isinstance(outs, list) or len(outs) != 2:
        raise ValueError("coalesce_triplet expects in=[k_all,v_all,k,v], out=[k_ctx,v_ctx]")
    k_ctx = env[ins[0]] if ins[0] in env and env[ins[0]] is not None else env[ins[2]]
    v_ctx = env[ins[1]] if ins[1] in env and env[ins[1]] is not None else env[ins[3]]
    env[outs[0]] = k_ctx
    env[outs[1]] = v_ctx
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
    if not isinstance(ins, list) or len(ins) != 4 or not isinstance(outs, list) or len(outs) != 2:
        raise ValueError("coalesce_triplet expects in=[k_all,v_all,k,v], out=[k_ctx,v_ctx]")
    k_all = read(str(ins[0])) if str(ins[0]) in env else "None"
    v_all = read(str(ins[1])) if str(ins[1]) in env else "None"
    k = read(str(ins[2]))
    v = read(str(ins[3]))
    k_ctx = assign_out_var(str(outs[0]))
    v_ctx = assign_out_var(str(outs[1]))
    if k_all != "None":
        lines.append(
            f"{indent}{k_ctx} = ({k_all} if ('{k_all}' in locals() and {k_all} is not None) else {k})"
        )
    else:
        lines.append(f"{indent}{k_ctx} = {k}")
    if v_all != "None":
        lines.append(
            f"{indent}{v_ctx} = ({v_all} if ('{v_all}' in locals() and {v_all} is not None) else {v})"
        )
    else:
        lines.append(f"{indent}{v_ctx} = {v}")
    return lines


__all__ = ["OP_NAME", "interpret", "compile", "uses_node_path"]
