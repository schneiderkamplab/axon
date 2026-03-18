from __future__ import annotations

from typing import Any

import torch

OP_NAME = "apply_rope_pair"


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
    ins = node_spec.get("in")
    outs = node_spec.get("out")
    if not isinstance(ins, list) or len(ins) != 2 or not isinstance(outs, list) or len(outs) != 2:
        raise ValueError("apply_rope_pair expects in=[q,k], out=[q_rot,k_rot]")
    q = env[ins[0]]
    k = env[ins[1]]
    theta = float(model._eval_expr(node_spec.get("theta", 10000.0), env, symbols))
    half = q.shape[-1] // 2
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, half, device=q.device, dtype=q.dtype) / float(half))
    )
    offset = int(model._eval_expr(node_spec.get("offset", 0), env, symbols))
    pos = torch.arange(offset, offset + q.shape[-2], device=q.device, dtype=q.dtype)
    ang = torch.einsum("t,d->td", pos, inv_freq)
    cos = torch.cos(ang)[None, None, :, :]
    sin = torch.sin(ang)[None, None, :, :]
    q1, q2 = q[..., :half], q[..., half : 2 * half]
    k1, k2 = k[..., :half], k[..., half : 2 * half]
    env[outs[0]] = torch.cat([q1 * cos - q2 * sin, q1 * sin + q2 * cos], dim=-1)
    env[outs[1]] = torch.cat([k1 * cos - k2 * sin, k1 * sin + k2 * cos], dim=-1)
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

    ins = node_spec.get("in")
    outs = node_spec.get("out")
    if not isinstance(ins, list) or len(ins) != 2 or not isinstance(outs, list) or len(outs) != 2:
        raise ValueError("apply_rope_pair expects in=[q,k], out=[q_rot,k_rot]")
    q = read(str(ins[0]))
    k = read(str(ins[1]))
    q_out = assign_out_var(str(outs[0]))
    k_out = assign_out_var(str(outs[1]))
    theta = emitter._expr_code(node_spec.get("theta", 10000.0), env)
    half = emitter._fresh("half")
    inv_freq = emitter._fresh("inv_freq")
    pos = emitter._fresh("pos")
    ang = emitter._fresh("ang")
    cos = emitter._fresh("cos")
    sin = emitter._fresh("sin")
    q1 = emitter._fresh("q1")
    q2 = emitter._fresh("q2")
    k1 = emitter._fresh("k1")
    k2 = emitter._fresh("k2")
    lines.append(f"{indent}{half} = {q}.shape[-1] // 2")
    lines.append(
        f"{indent}{inv_freq} = 1.0 / (float({theta}) ** (torch.arange(0, {half}, device={q}.device, dtype={q}.dtype) / float({half})))"
    )
    offset = emitter._expr_code(node_spec.get("offset", 0), env)
    lines.append(
        f"{indent}{pos} = torch.arange(int({offset}), int({offset}) + {q}.shape[-2], device={q}.device, dtype={q}.dtype)"
    )
    lines.append(f"{indent}{ang} = torch.einsum('t,d->td', {pos}, {inv_freq})")
    lines.append(f"{indent}{cos} = torch.cos({ang})[None, None, :, :]")
    lines.append(f"{indent}{sin} = torch.sin({ang})[None, None, :, :]")
    lines.append(f"{indent}{q1} = {q}[..., :{half}]")
    lines.append(f"{indent}{q2} = {q}[..., {half}: 2 * {half}]")
    lines.append(f"{indent}{k1} = {k}[..., :{half}]")
    lines.append(f"{indent}{k2} = {k}[..., {half}: 2 * {half}]")
    lines.append(
        f"{indent}{q_out} = torch.cat([{q1} * {cos} - {q2} * {sin}, {q1} * {sin} + {q2} * {cos}], dim=-1)"
    )
    lines.append(
        f"{indent}{k_out} = torch.cat([{k1} * {cos} - {k2} * {sin}, {k1} * {sin} + {k2} * {cos}], dim=-1)"
    )
    return lines


__all__ = ["OP_NAME", "interpret", "compile", "uses_node_path"]
