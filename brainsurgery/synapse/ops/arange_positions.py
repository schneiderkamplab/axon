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
    mask_ref = node_spec.get("attention_mask")
    mask_tensor = env.get(mask_ref) if isinstance(mask_ref, str) else None
    past = env.get("past_key_values")
    out = model._require_name(node_spec.get("out"), field="arange_positions.out")
    if mask_tensor is not None:
        if not torch.is_tensor(mask_tensor):
            raise ValueError("arange_positions.attention_mask must resolve to tensor or null")
        if mask_tensor.ndim != 2:
            raise ValueError("arange_positions.attention_mask must be rank-2 [batch, seq]")
        if mask_tensor.shape[0] != x.shape[0]:
            raise ValueError("arange_positions.attention_mask batch size must match input")
        full_mask = mask_tensor
        full_pos = full_mask.to(torch.long).cumsum(dim=-1) - 1
        full_pos = full_pos.masked_fill(full_mask == 0, 0)
        if full_mask.shape[1] < seq_len:
            raise ValueError(
                "arange_positions.attention_mask width must be >= input sequence length"
            )
        env[out] = full_pos[:, -seq_len:]
        return

    offset = 0 if past is None else int(past[0][0].shape[-2])
    env[out] = torch.arange(offset, offset + seq_len, device=x.device, dtype=torch.long).unsqueeze(
        0
    )
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
    mask_name = node_spec.get("attention_mask")
    mask = env.get(mask_name) if isinstance(mask_name, str) and mask_name in env else None
    past_var = env.get("past_key_values")
    out_name = str(node_spec.get("out"))
    out_var = assign_out_var(out_name)
    if isinstance(mask, str):
        full_pos = emitter._fresh("full_pos")
        offset = emitter._fresh("pos_offset")
        lines.append(f"{indent}if {mask} is not None:")
        lines.append(f"{indent}    if {mask}.ndim != 2:")
        lines.append(
            f"{indent}        raise ValueError('arange_positions.attention_mask must be rank-2 [batch, seq]')"
        )
        lines.append(f"{indent}    if {mask}.shape[0] != {src}.shape[0]:")
        lines.append(
            f"{indent}        raise ValueError('arange_positions.attention_mask batch size must match input')"
        )
        lines.append(f"{indent}    if {mask}.shape[1] < {src}.shape[1]:")
        lines.append(
            f"{indent}        raise ValueError('arange_positions.attention_mask width must be >= input sequence length')"
        )
        lines.append(f"{indent}    {full_pos} = {mask}.to(torch.long).cumsum(dim=-1) - 1")
        lines.append(f"{indent}    {full_pos} = {full_pos}.masked_fill({mask} == 0, 0)")
        lines.append(f"{indent}    {out_var} = {full_pos}[:, -{src}.shape[1]:]")
        lines.append(f"{indent}else:")
        if isinstance(past_var, str):
            lines.append(
                f"{indent}    {offset} = 0 if {past_var} is None else int({past_var}[0][0].shape[-2])"
            )
            lines.append(
                f"{indent}    {out_var} = torch.arange({offset}, {offset} + {src}.shape[1], device={src}.device, dtype=torch.long).unsqueeze(0)"
            )
        else:
            lines.append(
                f"{indent}    {out_var} = torch.arange({src}.shape[1], device={src}.device, dtype=torch.long).unsqueeze(0)"
            )
        return lines

    if isinstance(past_var, str):
        offset = emitter._fresh("pos_offset")
        lines.append(
            f"{indent}{offset} = 0 if {past_var} is None else int({past_var}[0][0].shape[-2])"
        )
        lines.append(
            f"{indent}{out_var} = torch.arange({offset}, {offset} + {src}.shape[1], device={src}.device, dtype=torch.long).unsqueeze(0)"
        )
        return lines

    lines.append(
        f"{indent}{out_var} = torch.arange({src}.shape[1], device={src}.device, dtype=torch.long).unsqueeze(0)"
    )
    return lines


__all__ = ["OP_NAME", "interpret", "compile", "uses_node_path"]
