from __future__ import annotations

from typing import Any

OP_NAME = "_ir_alias"


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
    del node_path, scope, symbols
    source = model._require_name(node_spec.get("in"), field="_ir_alias.in")
    out_name = model._require_name(node_spec.get("out"), field="_ir_alias.out")
    env[out_name] = env[source]
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
    del node_path_var, scope_var
    source = str(node_spec.get("in"))
    out_name = str(node_spec.get("out"))
    source_expr = emitter._read_env_var(env, source)
    out_var = emitter._assign_out_var(env, out_name)
    return [f"{indent}{out_var} = {source_expr}"]


__all__ = ["OP_NAME", "interpret", "compile", "uses_node_path"]
