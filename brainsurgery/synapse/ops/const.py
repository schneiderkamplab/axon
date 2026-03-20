from __future__ import annotations

from typing import Any

OP_NAME = "_ir_const"


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
    out_name = model._require_name(node_spec.get("_bind"), field="_ir_const._bind")
    env[out_name] = node_spec.get("value")
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
    out_name = str(node_spec.get("_bind"))
    out_var = emitter._assign_out_var(env, out_name)
    value_code = repr(node_spec.get("value"))
    return [f"{indent}{out_var} = {value_code}"]


__all__ = ["OP_NAME", "interpret", "compile", "uses_node_path"]
