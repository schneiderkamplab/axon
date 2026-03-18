from __future__ import annotations

from typing import Any

import torch

OP_NAME = "arange_positions"


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
    seq_len = x.shape[1]
    out = model._require_name(node_spec.get("out"), field="arange_positions.out")
    env[out] = torch.arange(seq_len, device=x.device, dtype=torch.long).unsqueeze(0)
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
    past_var = env.get("past_key_values")
    if isinstance(past_var, str):
        offset = emitter._fresh("pos_offset")
        lines.append(
            f"{indent}{offset} = 0 if {past_var} is None else int({past_var}[0][0].shape[-2])"
        )
        lines.append(
            f"{indent}{out_var} = torch.arange({offset}, {offset} + {src}.shape[1], device={src}.device, dtype=torch.long).unsqueeze(0)"
        )
    else:
        lines.append(
            f"{indent}{out_var} = torch.arange({src}.shape[1], device={src}.device, dtype=torch.long).unsqueeze(0)"
        )
    return lines


__all__ = ["OP_NAME", "interpret", "compile", "uses_node_path"]
