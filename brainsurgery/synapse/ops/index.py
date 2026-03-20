from __future__ import annotations

from typing import Any

OP_NAME = "index"


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
    ins = node_spec.get("_args")
    if not isinstance(ins, list) or len(ins) != 2:
        raise ValueError("index expects [collection, index]")
    collection = env.get(ins[0])
    out_name = model._require_name(node_spec.get("_bind"), field="index._bind")
    if collection is None:
        env[out_name] = None
        return
    idx = (
        int(model._eval_expr(ins[1], env, symbols))
        if not isinstance(ins[1], str)
        else int(env[ins[1]])
    )
    env[out_name] = collection[idx]
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

    ins = node_spec.get("_args")
    if not isinstance(ins, list) or len(ins) != 2:
        raise ValueError("index expects [collection,index]")
    coll = read(str(ins[0]))
    idx_expr = emitter._expr_code(ins[1], env)
    out_name = str(node_spec.get("_bind"))
    out_var = assign_out_var(out_name)
    lines.append(f"{indent}{out_var} = None if {coll} is None else {coll}[int({idx_expr})]")
    return lines


__all__ = ["OP_NAME", "interpret", "compile", "uses_node_path"]
