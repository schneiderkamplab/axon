from __future__ import annotations

from typing import Any

import torch

OP_NAME = "moe_scatter_add"
LOWERING_ARITY = (4, 4)
LOWERING_ALLOWED_KWARGS: set[str] = set()
LOWERING_REQUIRED_KWARGS: set[str] = set()
LOWERING_KWARG_KINDS: dict[str, Any] = {}


def uses_node_path(emitter: Any, node_spec: dict[str, Any]) -> bool:
    del emitter, node_spec
    return False


def lowering_validate_signature(
    *, args: list[str], out: str | list[str], kwargs: dict[str, Any], ctx: Any
) -> None:
    del args, kwargs, ctx
    if not isinstance(out, str):
        raise ValueError("moe_scatter_add requires a single scalar output binding")


def _resolve_inputs_and_output(
    node_spec: dict[str, Any], *, strict_out: bool
) -> tuple[list[str], str]:
    ins = node_spec.get("_args")
    if not isinstance(ins, list) or len(ins) != 4 or not all(isinstance(name, str) for name in ins):
        raise ValueError("moe_scatter_add expects in=[accum,token_idx,updates,scores]")
    out_raw = node_spec.get("_bind")
    if not isinstance(out_raw, str):
        if strict_out:
            raise ValueError("moe_scatter_add.out must be a variable name")
        out_raw = str(out_raw)
    return [str(name) for name in ins], out_raw


def _validate_scatter_inputs(
    accum: torch.Tensor,
    token_idx: torch.Tensor,
    updates: torch.Tensor,
    scores: torch.Tensor,
) -> None:
    if accum.ndim < 2:
        raise ValueError("moe_scatter_add accum must be at least rank-2")
    if token_idx.ndim != 1:
        raise ValueError("moe_scatter_add token_idx must be rank-1")
    if updates.ndim != 2:
        raise ValueError("moe_scatter_add updates must be rank-2")
    if scores.ndim != 1:
        raise ValueError("moe_scatter_add scores must be rank-1")
    if token_idx.dtype.is_floating_point or token_idx.dtype.is_complex:
        raise ValueError(
            f"moe_scatter_add token_idx must be an integer tensor, got {token_idx.dtype}"
        )
    n = token_idx.shape[0]
    if updates.shape[0] != n or scores.shape[0] != n:
        raise ValueError("moe_scatter_add token_idx, updates, and scores must align on row count")
    accum_flat = accum.reshape(-1, accum.shape[-1])
    if updates.shape[-1] != accum_flat.shape[-1]:
        raise ValueError("moe_scatter_add updates hidden size must match accum hidden size")
    if token_idx.numel() == 0:
        return
    max_index = int(accum_flat.shape[0]) - 1
    if int(token_idx.min()) < 0 or int(token_idx.max()) > max_index:
        raise ValueError(
            f"moe_scatter_add token_idx contains out-of-range values for accum rows 0..{max_index}"
        )


def interpret(
    model: Any,
    node_spec: dict[str, Any],
    env: dict[str, Any],
    *,
    node_path: str,
    scope: str,
    symbols: dict[str, int],
) -> None:
    ins, out = _resolve_inputs_and_output(node_spec, strict_out=True)
    accum = model._read_tensor_input(ins[0], env)
    token_idx = model._read_tensor_input(ins[1], env)
    updates = model._read_tensor_input(ins[2], env)
    scores = model._read_tensor_input(ins[3], env)
    _validate_scatter_inputs(accum, token_idx, updates, scores)
    if token_idx.numel() == 0:
        env[out] = accum
        return
    accum_flat = accum.reshape(-1, accum.shape[-1])
    weighted = updates * scores.unsqueeze(-1).to(updates.dtype)
    accum_flat.index_add_(0, token_idx, weighted.to(accum_flat.dtype))
    env[out] = accum
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

    ins, out_name = _resolve_inputs_and_output(node_spec, strict_out=False)
    accum = read(ins[0])
    token_idx = read(ins[1])
    updates = read(ins[2])
    scores = read(ins[3])
    out_var = assign_out_var(out_name)
    lines.append(f"{indent}{out_var} = {accum}")
    lines.append(f"{indent}if {token_idx}.numel() != 0:")
    lines.append(f"{indent}    _acc = {out_var}.reshape(-1, {out_var}.shape[-1])")
    lines.append(f"{indent}    _upd = {updates} * {scores}.unsqueeze(-1).to({updates}.dtype)")
    lines.append(f"{indent}    _acc.index_add_(0, {token_idx}, _upd.to(_acc.dtype))")
    return lines


__all__ = [
    "LOWERING_ARITY",
    "LOWERING_ALLOWED_KWARGS",
    "LOWERING_REQUIRED_KWARGS",
    "LOWERING_KWARG_KINDS",
    "OP_NAME",
    "lowering_validate_signature",
    "interpret",
    "compile",
    "uses_node_path",
]
