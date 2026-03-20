from __future__ import annotations

from typing import Any

import torch

OP_NAME = "moe_select_tokens"


def uses_node_path(emitter: Any, node_spec: dict[str, Any]) -> bool:
    del emitter, node_spec
    return False


def _resolve_inputs_and_outputs(node_spec: dict[str, Any]) -> tuple[list[str], list[str]]:
    ins = node_spec.get("in")
    outs = node_spec.get("out")
    if not isinstance(ins, list) or len(ins) != 3 or not all(isinstance(name, str) for name in ins):
        raise ValueError(
            "moe_select_tokens expects in=[hidden,topk_scores,topk_indices], "
            "out=[selected_hidden,token_idx,topk_pos,selected_scores]"
        )
    if (
        not isinstance(outs, list)
        or len(outs) != 4
        or not all(isinstance(name, str) for name in outs)
    ):
        raise ValueError(
            "moe_select_tokens expects in=[hidden,topk_scores,topk_indices], "
            "out=[selected_hidden,token_idx,topk_pos,selected_scores]"
        )
    return [str(name) for name in ins], [str(name) for name in outs]


def _resolve_expert(
    model: Any, node_spec: dict[str, Any], env: dict[str, Any], symbols: dict[str, int]
) -> int:
    expert_raw = model._eval_expr(node_spec.get("expert"), env, symbols)
    if not isinstance(expert_raw, int) or isinstance(expert_raw, bool):
        raise ValueError(
            f"moe_select_tokens expert must evaluate to an integer, got {expert_raw!r}"
        )
    return expert_raw


def _flatten_routing_inputs(
    hidden: torch.Tensor,
    topk_scores: torch.Tensor,
    topk_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if hidden.ndim < 2:
        raise ValueError("moe_select_tokens hidden must be at least rank-2")
    if topk_scores.ndim < 2 or topk_indices.ndim < 2:
        raise ValueError("moe_select_tokens topk_scores/topk_indices must be at least rank-2")
    if topk_scores.shape != topk_indices.shape:
        raise ValueError("moe_select_tokens topk_scores and topk_indices must have the same shape")
    if topk_indices.dtype.is_floating_point or topk_indices.dtype.is_complex:
        raise ValueError(
            f"moe_select_tokens topk_indices must be an integer tensor, got {topk_indices.dtype}"
        )
    hidden_flat = hidden.reshape(-1, hidden.shape[-1])
    topk_scores_flat = topk_scores.reshape(-1, topk_scores.shape[-1])
    topk_indices_flat = topk_indices.reshape(-1, topk_indices.shape[-1])
    if hidden_flat.shape[0] != topk_scores_flat.shape[0]:
        raise ValueError(
            "moe_select_tokens hidden and topk tensors must align on flattened token count"
        )
    return hidden_flat, topk_scores_flat, topk_indices_flat


def interpret(
    model: Any,
    node_spec: dict[str, Any],
    env: dict[str, Any],
    *,
    node_path: str,
    scope: str,
    symbols: dict[str, int],
) -> None:
    ins, outs = _resolve_inputs_and_outputs(node_spec)
    hidden = model._read_tensor_input(ins[0], env)
    topk_scores = model._read_tensor_input(ins[1], env)
    topk_indices = model._read_tensor_input(ins[2], env)
    expert = _resolve_expert(model, node_spec, env, symbols)
    hidden_flat, topk_scores_flat, topk_indices_flat = _flatten_routing_inputs(
        hidden, topk_scores, topk_indices
    )
    expert_pos = (topk_indices_flat == expert).nonzero(as_tuple=False)
    token_idx = expert_pos[:, 0]
    topk_pos = expert_pos[:, 1]
    selected_hidden = hidden_flat[token_idx]
    selected_scores = topk_scores_flat[token_idx, topk_pos].to(selected_hidden.dtype)
    env[outs[0]] = selected_hidden
    env[outs[1]] = token_idx
    env[outs[2]] = topk_pos
    env[outs[3]] = selected_scores
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

    ins, outs = _resolve_inputs_and_outputs(node_spec)
    hidden = read(ins[0])
    topk_scores = read(ins[1])
    topk_indices = read(ins[2])
    selected_hidden = assign_out_var(outs[0])
    token_idx = assign_out_var(outs[1])
    topk_pos = assign_out_var(outs[2])
    selected_scores = assign_out_var(outs[3])
    expert = emitter._expr_code(node_spec.get("expert"), env)
    hidden_flat = emitter._fresh("hidden_flat")
    topk_scores_flat = emitter._fresh("topk_scores_flat")
    topk_indices_flat = emitter._fresh("topk_indices_flat")
    expert_pos = emitter._fresh("expert_pos")
    lines.append(f"{indent}{hidden_flat} = {hidden}.reshape(-1, {hidden}.shape[-1])")
    lines.append(f"{indent}{topk_scores_flat} = {topk_scores}.reshape(-1, {topk_scores}.shape[-1])")
    lines.append(
        f"{indent}{topk_indices_flat} = {topk_indices}.reshape(-1, {topk_indices}.shape[-1])"
    )
    lines.append(
        f"{indent}{expert_pos} = ({topk_indices_flat} == int({expert})).nonzero(as_tuple=False)"
    )
    lines.append(f"{indent}{token_idx} = {expert_pos}[:, 0]")
    lines.append(f"{indent}{topk_pos} = {expert_pos}[:, 1]")
    lines.append(f"{indent}{selected_hidden} = {hidden_flat}[{token_idx}]")
    lines.append(
        f"{indent}{selected_scores} = {topk_scores_flat}[{token_idx}, {topk_pos}].to({selected_hidden}.dtype)"
    )
    return lines


__all__ = ["OP_NAME", "interpret", "compile", "uses_node_path"]
