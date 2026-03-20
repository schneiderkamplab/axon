from __future__ import annotations

from typing import Any

OP_NAME = "coalesce"


def uses_node_path(emitter: Any, node_spec: dict[str, Any]) -> bool:
    del emitter, node_spec
    return False


def _resolve_outputs(node_spec: dict[str, Any]) -> list[str]:
    outs = node_spec.get("_bind")
    if not isinstance(outs, list) or len(outs) == 0:
        raise ValueError("coalesce requires non-empty list out")
    return [str(name) for name in outs]


def _resolve_groups(node_spec: dict[str, Any], out_count: int) -> list[list[str]]:
    ins = node_spec.get("_args")
    if isinstance(ins, list) and all(isinstance(x, str) for x in ins):
        values = [str(x) for x in ins]
        if len(values) % out_count != 0:
            raise ValueError("coalesce input count must be divisible by output count")
        rounds = len(values) // out_count
        if rounds <= 0:
            raise ValueError("coalesce requires at least one candidate per output")
        return [values[idx::out_count] for idx in range(out_count)]
    raise ValueError("coalesce requires list in")


def interpret(
    model: Any,
    node_spec: dict[str, Any],
    env: dict[str, Any],
    *,
    node_path: str,
    scope: str,
    symbols: dict[str, int],
) -> None:
    del model, node_path, scope, symbols
    outs = _resolve_outputs(node_spec)
    groups = _resolve_groups(node_spec, len(outs))
    for out_name, candidates in zip(outs, groups, strict=True):
        value = None
        for candidate in candidates:
            if candidate not in env:
                raise ValueError(f"coalesce candidate {candidate!r} missing in env")
            if env[candidate] is not None:
                value = env[candidate]
                break
        env[out_name] = value


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
    lines: list[str] = []
    outs = _resolve_outputs(node_spec)
    groups = _resolve_groups(node_spec, len(outs))
    for out_name, candidates in zip(outs, groups, strict=True):
        out_var = emitter._assign_out_var(env, out_name)
        candidate_vars = [emitter._read_env_var(env, candidate) for candidate in candidates]
        first = candidate_vars[0]
        tail = candidate_vars[1:]
        expr = first
        for var in tail:
            expr = f"({expr} if {expr} is not None else {var})"
        lines.append(f"{indent}{out_var} = {expr}")
    return lines


__all__ = ["OP_NAME", "interpret", "compile", "uses_node_path"]
