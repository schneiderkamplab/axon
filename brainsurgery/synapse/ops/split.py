from __future__ import annotations

from typing import Any

OP_NAME = "split"


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
    del node_path, scope
    x = model._read_tensor_input(node_spec.get("_args"), env)
    outs = node_spec.get("_bind")
    if not isinstance(outs, list) or len(outs) == 0:
        raise ValueError("split requires non-empty list out")
    sizes = node_spec.get("sizes")
    if sizes is not None:
        if not isinstance(sizes, list) or len(sizes) != len(outs):
            raise ValueError("split sizes must be a list with same length as out")
        split_sizes = [int(model._eval_expr(size, env, symbols)) for size in sizes]
        chunks = x.split(split_sizes, dim=-1)
    else:
        parts = int(model._eval_expr(node_spec.get("parts", len(outs)), env, symbols))
        chunks = x.chunk(parts, dim=-1)
    for name, tensor in zip(outs, chunks, strict=True):
        env[str(name)] = tensor


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
    src = emitter._read_env_var(env, str(node_spec.get("_args")))
    outs = node_spec.get("_bind")
    if not isinstance(outs, list) or len(outs) == 0:
        raise ValueError("split requires non-empty list out")
    tmp = emitter._fresh("split")
    sizes = node_spec.get("sizes")
    if sizes is not None:
        if not isinstance(sizes, list) or len(sizes) != len(outs):
            raise ValueError("split sizes must be a list with same length as out")
        sizes_code = ", ".join(emitter._expr_code(size, env) for size in sizes)
        lines.append(f"{indent}{tmp} = torch.split({src}, [{sizes_code}], dim=-1)")
    else:
        parts = emitter._expr_code(node_spec.get("parts", len(outs)), env)
        lines.append(f"{indent}{tmp} = torch.chunk({src}, int({parts}), dim=-1)")
    for idx, out_name in enumerate(outs):
        out_var = emitter._assign_out_var(env, str(out_name))
        lines.append(f"{indent}{out_var} = {tmp}[{idx}]")
    return lines


__all__ = ["OP_NAME", "interpret", "compile", "uses_node_path"]
