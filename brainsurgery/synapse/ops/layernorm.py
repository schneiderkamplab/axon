from __future__ import annotations

from typing import Any

from torch.nn import functional as F

OP_NAME = "layernorm"


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
    x = model._read_tensor_input(node_spec.get("in"), env)
    weight = model._state[
        model._infer_param_path(node_spec, node_path=node_path, param_name="weight")
    ]
    bias = model._state[model._infer_param_path(node_spec, node_path=node_path, param_name="bias")]
    eps_value = model._eval_expr(node_spec.get("eps", 1e-5), env, symbols)
    out = model._require_name(node_spec.get("out"), field="layernorm.out")
    env[out] = F.layer_norm(x, (x.shape[-1],), weight=weight, bias=bias, eps=float(eps_value))
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
    out_name = str(node_spec.get("out"))
    out_var = assign_out_var(out_name)
    eps = emitter._expr_code(node_spec.get("eps", 1e-5), env)
    w = f"emitter._param({infer_param('weight')})"
    b = f"emitter._param({infer_param('bias')})"
    lines.append(
        f"{indent}{out_var} = F.layer_norm({src}, ({src}.shape[-1],), weight={w}, bias={b}, eps=float({eps}))"
    )
    return lines


__all__ = ["OP_NAME", "interpret", "compile", "uses_node_path"]
