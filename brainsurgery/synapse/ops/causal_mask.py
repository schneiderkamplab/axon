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
    padding_ref = node_spec.get("padding_mask")
    padding_mask = env.get(padding_ref) if isinstance(padding_ref, str) else None
    if padding_mask is not None and not torch.is_tensor(padding_mask):
        raise ValueError("causal_mask.padding_mask must resolve to tensor or null")
    out_name = model._require_name(node_spec.get("out"), field="causal_mask.out")
    window_expr = node_spec.get("window")
    if window_expr is None and padding_mask is None:
        env[out_name] = None
        return

    q_len = q.shape[-2]
    k_len = key_tensor.shape[-2]
    i_idx = torch.arange(q_len, device=q.device).unsqueeze(1)
    j_idx = torch.arange(k_len, device=q.device).unsqueeze(0)

    if q_len == 1:
        if window_expr is None:
            keep = torch.ones((q_len, k_len), dtype=torch.bool, device=q.device)
            if padding_mask is None:
                env[out_name] = None
                return
        else:
            win = int(model._eval_expr(window_expr, env, symbols))
            if win >= k_len and padding_mask is None:
                env[out_name] = None
                return
            keep = j_idx >= (k_len - win)
    else:
        if window_expr is None:
            keep = torch.ones((q_len, k_len), dtype=torch.bool, device=q.device)
        else:
            keep = j_idx <= i_idx
            win = int(model._eval_expr(window_expr, env, symbols))
            if win >= k_len and q_len == k_len and padding_mask is None:
                env[out_name] = None
                return
            keep = keep & (j_idx >= (i_idx - win + 1))

    if padding_mask is not None:
        if padding_mask.ndim != 2:
            raise ValueError("causal_mask.padding_mask must be rank-2 [batch, seq]")
        if int(padding_mask.shape[-1]) != k_len:
            raise ValueError("causal_mask.padding_mask width must match key sequence length")
        pad_keep = padding_mask.to(torch.bool).unsqueeze(1).unsqueeze(1)
        keep = keep.unsqueeze(0).unsqueeze(0) & pad_keep
    else:
        keep = keep.view(1, 1, q_len, k_len)

    mask_value = torch.finfo(q.dtype).min
    env[out_name] = torch.where(
        keep,
        torch.zeros((), dtype=q.dtype, device=q.device),
        torch.full((), mask_value, dtype=q.dtype, device=q.device),
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
    padding_name = node_spec.get("padding_mask")
    padding_expr = (
        env.get(padding_name) if isinstance(padding_name, str) and padding_name in env else None
    )
    if window_expr is None and padding_expr is None:
        lines.append(f"{indent}{out_var} = None")
        return lines
    lines.append(f"{indent}{q_len} = {q}.shape[-2]")
    lines.append(f"{indent}{k_len} = {k}.shape[-2]")
    lines.append(f"{indent}{j_idx} = torch.arange({k_len}, device={q}.device).unsqueeze(0)")
    if window_expr is None:
        lines.append(
            f"{indent}{keep} = torch.ones(({q_len}, {k_len}), dtype=torch.bool, device={q}.device)"
        )
    else:
        win = emitter._fresh("window")
        window_code = emitter._expr_code(window_expr, env)
        lines.append(f"{indent}{win} = int({window_code})")
        lines.append(f"{indent}if {q_len} == 1:")
        lines.append(f"{indent}    {keep} = ({j_idx} >= ({k_len} - {win}))")
        lines.append(f"{indent}else:")
        lines.append(f"{indent}    {i_idx} = torch.arange({q_len}, device={q}.device).unsqueeze(1)")
        lines.append(f"{indent}    {keep} = ({j_idx} <= {i_idx})")
        lines.append(f"{indent}    {keep} = {keep} & ({j_idx} >= ({i_idx} - {win} + 1))")

    if padding_expr is not None:
        pad_keep = emitter._fresh("pad_keep")
        lines.append(f"{indent}if {padding_expr} is not None:")
        lines.append(f"{indent}    if {padding_expr}.ndim != 2:")
        lines.append(
            f"{indent}        raise ValueError('causal_mask.padding_mask must be rank-2 [batch, seq]')"
        )
        lines.append(f"{indent}    if int({padding_expr}.shape[-1]) != {k_len}:")
        lines.append(
            f"{indent}        raise ValueError('causal_mask.padding_mask width must match key sequence length')"
        )
        lines.append(
            f"{indent}    {pad_keep} = {padding_expr}.to(torch.bool).unsqueeze(1).unsqueeze(1)"
        )
        lines.append(f"{indent}    {keep} = {keep}.unsqueeze(0).unsqueeze(0) & {pad_keep}")
        lines.append(f"{indent}else:")
        lines.append(f"{indent}    {keep} = {keep}.view(1, 1, {q_len}, {k_len})")
    else:
        lines.append(f"{indent}{keep} = {keep}.view(1, 1, {q_len}, {k_len})")

    lines.append(f"{indent}{mask_val} = torch.finfo({q}.dtype).min")
    lines.append(
        f"{indent}{out_var} = torch.where({keep}, torch.zeros((), dtype={q}.dtype, device={q}.device), torch.full((), {mask_val}, dtype={q}.dtype, device={q}.device))"
    )
    return lines


__all__ = ["OP_NAME", "interpret", "compile", "uses_node_path"]
