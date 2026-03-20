from __future__ import annotations

from typing import Any

from torch.nn import functional as F

OP_NAME = "embedding"


def uses_node_path(emitter: Any, node_spec: dict[str, Any]) -> bool:
    del emitter, node_spec
    return True


def interpret(
    model: Any,
    node_spec: dict[str, Any],
    env: dict[str, Any],
    *,
    node_path: str,
    scope: str,
    symbols: dict[str, int],
) -> None:
    x = model._read_tensor_input(node_spec.get("_args"), env)
    weight_path = model._infer_param_path(node_spec, node_path=node_path, param_name="weight")
    weight = model._state[weight_path]
    out = model._require_name(node_spec.get("_bind"), field="embedding._bind")
    y = F.embedding(x, weight)
    if node_spec.get("scale") is not None:
        scale = float(model._eval_expr(node_spec.get("scale"), env, symbols))
        y = y * y.new_tensor(scale)
    env[out] = y
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

    src = read(str(node_spec.get("_args")))
    out_name = str(node_spec.get("_bind"))
    out_var = assign_out_var(out_name)
    scale_expr = node_spec.get("scale")
    if scale_expr is None:
        lines.append(
            f"{indent}{out_var} = F.embedding({src}, emitter._param({infer_param('weight')}))"
        )
    else:
        scale = emitter._expr_code(scale_expr, env)
        lines.append(
            f"{indent}{out_var} = F.embedding({src}, emitter._param({infer_param('weight')}))"
        )
        lines.append(
            f"{indent}{out_var} = {out_var} * torch.tensor(float({scale}), dtype={out_var}.dtype, device={out_var}.device)"
        )
    return lines


__all__ = ["OP_NAME", "interpret", "compile", "uses_node_path"]
