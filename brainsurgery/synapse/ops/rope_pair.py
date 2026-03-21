from __future__ import annotations

from typing import Any

import torch

OP_NAME = "rope_pair"
LOWERING_ARITY = (2, 2)
LOWERING_ALLOWED_KWARGS: set[str] = {"position_ids", "theta"}
LOWERING_REQUIRED_KWARGS: set[str] = {"position_ids"}
LOWERING_KWARG_KINDS: dict[str, Any] = {"position_ids": "str", "theta": "number"}


def uses_node_path(emitter: Any, node_spec: dict[str, Any]) -> bool:
    del emitter, node_spec
    return False


def lowering_known_output_arity(*, kwargs: dict[str, Any]) -> int:
    del kwargs
    return 2


def interpret(
    model: Any,
    node_spec: dict[str, Any],
    env: dict[str, Any],
    *,
    node_path: str,
    scope: str,
    symbols: dict[str, int],
) -> None:
    ins = node_spec.get("_args")
    outs = node_spec.get("_bind")
    if not isinstance(ins, list) or len(ins) != 2 or not isinstance(outs, list) or len(outs) != 2:
        raise ValueError("rope_pair expects in=[q,k], out=[q_rot,k_rot]")
    q = env[ins[0]]
    k = env[ins[1]]
    if not torch.is_tensor(q) or not torch.is_tensor(k):
        raise ValueError("rope_pair expects tensor inputs for q and k")
    if q.ndim != 4 or k.ndim != 4:
        raise ValueError("rope_pair expects q and k to be rank-4 [batch, heads, seq, head_dim]")
    if int(q.shape[0]) != int(k.shape[0]):
        raise ValueError("rope_pair expects q and k to have matching batch size")
    if int(q.shape[-2]) != int(k.shape[-2]):
        raise ValueError("rope_pair expects q and k to have matching sequence length")
    if int(q.shape[-1]) != int(k.shape[-1]):
        raise ValueError("rope_pair expects q and k to have matching head dimension")
    if int(q.shape[-1]) % 2 != 0:
        raise ValueError("rope_pair expects even head dimension")
    theta = float(model._eval_expr(node_spec.get("theta", 10000.0), env, symbols))
    pos_ref = node_spec.get("position_ids")
    if not isinstance(pos_ref, str):
        raise ValueError("rope_pair.position_ids must be a tensor reference")
    pos_ids = env.get(pos_ref)
    if pos_ids is None:
        raise ValueError("rope_pair.position_ids must not be null")
    if not torch.is_tensor(pos_ids):
        raise ValueError("rope_pair.position_ids must resolve to tensor")
    if pos_ids.ndim != 2:
        raise ValueError("rope_pair.position_ids must be rank-2 [batch, seq]")
    if int(pos_ids.shape[0]) != int(q.shape[0]):
        raise ValueError("rope_pair.position_ids batch size must match q/k batch")
    if int(pos_ids.shape[1]) != int(q.shape[-2]):
        raise ValueError("rope_pair.position_ids width must match q/k sequence length")
    half = q.shape[-1] // 2
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, half, device=q.device, dtype=q.dtype) / float(half))
    )
    pos = pos_ids.to(device=q.device, dtype=q.dtype)
    ang = pos.unsqueeze(-1) * inv_freq.unsqueeze(0).unsqueeze(0)
    cos = torch.cos(ang).unsqueeze(1)
    sin = torch.sin(ang).unsqueeze(1)
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

    def read(name: str) -> str:
        return emitter._read_env_var(env, name)

    ins = node_spec.get("_args")
    outs = node_spec.get("_bind")
    if not isinstance(ins, list) or len(ins) != 2 or not isinstance(outs, list) or len(outs) != 2:
        raise ValueError("rope_pair expects in=[q,k], out=[q_rot,k_rot]")
    q = read(str(ins[0]))
    k = read(str(ins[1]))
    q_out = assign_out_var(str(outs[0]))
    k_out = assign_out_var(str(outs[1]))
    theta = emitter._expr_code(node_spec.get("theta", 10000.0), env)
    pos_name = node_spec.get("position_ids")
    if not isinstance(pos_name, str) or pos_name not in env:
        raise ValueError("rope_pair.position_ids must reference an input tensor name")
    pos_ids = env[pos_name]
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
    lines.append(f"{indent}if {q}.ndim != 4 or {k}.ndim != 4:")
    lines.append(
        f"{indent}    raise ValueError('rope_pair expects q and k to be rank-4 [batch, heads, seq, head_dim]')"
    )
    lines.append(f"{indent}if int({q}.shape[0]) != int({k}.shape[0]):")
    lines.append(
        f"{indent}    raise ValueError('rope_pair expects q and k to have matching batch size')"
    )
    lines.append(f"{indent}if int({q}.shape[-2]) != int({k}.shape[-2]):")
    lines.append(
        f"{indent}    raise ValueError('rope_pair expects q and k to have matching sequence length')"
    )
    lines.append(f"{indent}if int({q}.shape[-1]) != int({k}.shape[-1]):")
    lines.append(
        f"{indent}    raise ValueError('rope_pair expects q and k to have matching head dimension')"
    )
    lines.append(f"{indent}{half} = {q}.shape[-1] // 2")
    lines.append(f"{indent}if int({q}.shape[-1]) % 2 != 0:")
    lines.append(f"{indent}    raise ValueError('rope_pair expects even head dimension')")
    lines.append(
        f"{indent}{inv_freq} = 1.0 / (float({theta}) ** (torch.arange(0, {half}, device={q}.device, dtype={q}.dtype) / float({half})))"
    )
    lines.append(f"{indent}if {pos_ids} is None:")
    lines.append(f"{indent}    raise ValueError('rope_pair.position_ids must not be null')")
    lines.append(f"{indent}if {pos_ids}.ndim != 2:")
    lines.append(
        f"{indent}    raise ValueError('rope_pair.position_ids must be rank-2 [batch, seq]')"
    )
    lines.append(f"{indent}if int({pos_ids}.shape[0]) != int({q}.shape[0]):")
    lines.append(
        f"{indent}    raise ValueError('rope_pair.position_ids batch size must match q/k batch')"
    )
    lines.append(f"{indent}if int({pos_ids}.shape[1]) != int({q}.shape[-2]):")
    lines.append(
        f"{indent}    raise ValueError('rope_pair.position_ids width must match q/k sequence length')"
    )
    lines.append(f"{indent}{pos} = {pos_ids}.to(device={q}.device, dtype={q}.dtype)")
    lines.append(f"{indent}{ang} = {pos}.unsqueeze(-1) * {inv_freq}.unsqueeze(0).unsqueeze(0)")
    lines.append(f"{indent}{cos} = torch.cos({ang}).unsqueeze(1)")
    lines.append(f"{indent}{sin} = torch.sin({ang}).unsqueeze(1)")
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


__all__ = [
    "LOWERING_ARITY",
    "LOWERING_ALLOWED_KWARGS",
    "LOWERING_REQUIRED_KWARGS",
    "LOWERING_KWARG_KINDS",
    "OP_NAME",
    "lowering_known_output_arity",
    "interpret",
    "compile",
    "uses_node_path",
]
