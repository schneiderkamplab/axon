from __future__ import annotations

from typing import Any

import torch

OP_NAME = "causal_mask"


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
    q = model._read_tensor_input(node_spec.get("in"), env)
    key_ref = node_spec.get("key")
    key_tensor = model._read_tensor_input(key_ref, env) if isinstance(key_ref, str) else q
    out_name = model._require_name(node_spec.get("out"), field="causal_mask.out")
    if node_spec.get("window") is None:
        env[out_name] = None
        return
    q_len = q.shape[-2]
    k_len = key_tensor.shape[-2]
    i_idx = torch.arange(q_len, device=q.device).unsqueeze(1)
    j_idx = torch.arange(k_len, device=q.device).unsqueeze(0)
    keep = j_idx <= i_idx
    win = int(model._eval_expr(node_spec.get("window"), env, symbols))
    if win >= k_len and q_len == k_len:
        env[out_name] = None
        return
    keep = keep & (j_idx >= (i_idx - win + 1))
    mask_value = torch.finfo(q.dtype).min
    mask = torch.where(
        keep,
        torch.zeros((), dtype=q.dtype, device=q.device),
        torch.full((), mask_value, dtype=q.dtype, device=q.device),
    )
    env[out_name] = mask.view(1, 1, q_len, k_len)
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

    q = read(str(node_spec.get("in")))
    k_name = node_spec.get("key")
    k = read(str(k_name)) if isinstance(k_name, str) else q
    out_name = str(node_spec.get("out"))
    out_var = assign_out_var(out_name)
    q_len = emitter._fresh("q_len")
    k_len = emitter._fresh("k_len")
    i_idx = emitter._fresh("i_idx")
    j_idx = emitter._fresh("j_idx")
    keep = emitter._fresh("keep")
    mask_val = emitter._fresh("mask_val")
    window_expr = node_spec.get("window")
    if window_expr is None:
        lines.append(f"{indent}{out_var} = None")
        return lines
    lines.append(f"{indent}{q_len} = {q}.shape[-2]")
    lines.append(f"{indent}{k_len} = {k}.shape[-2]")
    lines.append(f"{indent}{i_idx} = torch.arange({q_len}, device={q}.device).unsqueeze(1)")
    lines.append(f"{indent}{j_idx} = torch.arange({k_len}, device={q}.device).unsqueeze(0)")
    lines.append(f"{indent}{keep} = ({j_idx} <= {i_idx})")
    win = emitter._fresh("window")
    window_code = emitter._expr_code(window_expr, env)
    lines.append(f"{indent}{win} = int({window_code})")
    lines.append(f"{indent}if {win} >= {k_len} and {q_len} == {k_len}:")
    lines.append(f"{indent}    {out_var} = None")
    lines.append(f"{indent}else:")
    lines.append(f"{indent}    {keep} = {keep} & ({j_idx} >= ({i_idx} - {win} + 1))")
    lines.append(f"{indent}    {mask_val} = torch.finfo({q}.dtype).min")
    lines.append(
        f"{indent}    {out_var} = torch.where({keep}, torch.zeros((), dtype={q}.dtype, device={q}.device), torch.full((), {mask_val}, dtype={q}.dtype, device={q}.device)).view(1, 1, {q_len}, {k_len})"
    )
    return lines


__all__ = ["OP_NAME", "interpret", "compile", "uses_node_path"]
