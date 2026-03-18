from __future__ import annotations

from typing import Any

OP_NAME = "split_last_dim"


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
    x = model._read_tensor_input(node_spec.get("in"), env)
    sizes = node_spec.get("sizes")
    outs = node_spec.get("out")
    if not isinstance(sizes, list) or not isinstance(outs, list):
        raise ValueError("split_last_dim requires list sizes and out")
    split_sizes = [int(model._eval_expr(size, env, symbols)) for size in sizes]
    tensors = x.split(split_sizes, dim=-1)
    for name, tensor in zip(outs, tensors, strict=True):
        env[name] = tensor
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

    src = read(str(node_spec.get("in")))
    outs = node_spec.get("out")
    sizes = node_spec.get("sizes")
    if not isinstance(outs, list) or not isinstance(sizes, list):
        raise ValueError("split_last_dim requires list out and sizes")
    size_code = ", ".join([emitter._expr_code(s, env) for s in sizes])
    tmp = emitter._fresh("split")
    lines.append(f"{indent}{tmp} = torch.split({src}, [{size_code}], dim=-1)")
    for idx, out_name in enumerate(outs):
        out_var = assign_out_var(str(out_name))
        lines.append(f"{indent}{out_var} = {tmp}[{idx}]")
    return lines


__all__ = ["OP_NAME", "interpret", "compile", "uses_node_path"]
