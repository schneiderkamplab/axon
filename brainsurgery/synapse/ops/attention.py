from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional as F

OP_NAME = "attention"
LOWERING_ARITY = (3, 3)
LOWERING_ALLOWED_KWARGS: set[str] = {"scale", "mask", "causal", "sink", "sink_path"}
LOWERING_REQUIRED_KWARGS: set[str] = set()
LOWERING_KWARG_KINDS: dict[str, Any] = {
    "causal": "bool",
    "mask": "str",
    "scale": "number",
    "sink": "str",
    "sink_path": "str",
}


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
    ins = node_spec.get("_args")
    if not isinstance(ins, list) or len(ins) != 3:
        raise ValueError("attention expects [q, k, v]")
    q = env[ins[0]]
    k_tensor = env[ins[1]]
    v = env[ins[2]]
    mask = None
    mask_name = node_spec.get("mask")
    if isinstance(mask_name, str):
        mask = env.get(mask_name)
    sink_name = node_spec.get("sink")
    sink = None
    if isinstance(sink_name, str):
        sink = env.get(sink_name)
    if sink is None and isinstance(node_spec.get("sink_path"), str):
        sink_scope = model._scope_of(node_path)
        sink_path = model._join(sink_scope, str(node_spec["sink_path"]))
        sink = model._state.get(sink_path)
    scale_expr = node_spec.get("scale")
    scale_value = None if scale_expr is None else float(model._eval_expr(scale_expr, env, symbols))
    if scale_value is None:
        scale_value = float(q.shape[-1]) ** -0.5
    if sink is not None:
        attn_weights = q @ k_tensor.transpose(2, 3)
        attn_weights = attn_weights * scale_value
        if mask is not None:
            attn_weights = attn_weights + mask
        sinks = sink.reshape(1, -1, 1, 1).expand(q.shape[0], -1, q.shape[-2], -1)
        combined_logits = torch.cat([attn_weights, sinks], dim=-1)
        combined_logits = combined_logits - combined_logits.max(dim=-1, keepdim=True).values
        probs = F.softmax(combined_logits, dim=-1, dtype=combined_logits.dtype)
        scores = probs[..., :-1]
        attn_out = scores.to(v.dtype) @ v
        out_name = model._require_name(node_spec.get("_bind"), field="attention._bind")
        env[out_name] = attn_out
        return
    is_causal_flag = bool(node_spec.get("causal", True)) and q.shape[2] > 1 and mask is None
    attn_out = F.scaled_dot_product_attention(
        q,
        k_tensor,
        v,
        attn_mask=mask,
        dropout_p=0.0,
        is_causal=is_causal_flag,
        scale=scale_value,
    )
    out_name = model._require_name(node_spec.get("_bind"), field="attention._bind")
    env[out_name] = attn_out
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

    ins = node_spec.get("_args")
    if not isinstance(ins, list) or len(ins) != 3:
        raise ValueError("attention expects 3 inputs")
    q = read(str(ins[0]))
    k = read(str(ins[1]))
    v = read(str(ins[2]))
    out_name = str(node_spec.get("_bind"))
    out_var = assign_out_var(out_name)
    mask_name = node_spec.get("mask")
    mask_expr = "None"
    if isinstance(mask_name, str) and mask_name in env:
        mask_expr = env[mask_name]
    sink_name = node_spec.get("sink")
    sink_expr = env[sink_name] if isinstance(sink_name, str) and sink_name in env else None
    if sink_expr is None and isinstance(node_spec.get("sink_path"), str):
        sink_scope_expr = f"self._scope_of({node_path_var})"
        sink_path_expr = f"self._join_scope({sink_scope_expr}, {node_spec['sink_path']!r})"
        sink_expr = f"self._state.get({sink_path_expr})"
    if bool(node_spec.get("causal", True)):
        is_causal = f"({q}.shape[-2] > 1 and {mask_expr} is None)"
    else:
        is_causal = "False"
    scale_value = node_spec.get("scale")
    scale_expr = (
        f"({q}.shape[-1] ** -0.5)" if scale_value is None else emitter._expr_code(scale_value, env)
    )
    if sink_expr is None:
        lines.append(
            f"{indent}{out_var} = F.scaled_dot_product_attention({q}, {k}, {v}, attn_mask={mask_expr}, dropout_p=0.0, is_causal={is_causal}, scale={scale_expr})"
        )
        return lines

    sink_var = assign_out_var(f"{out_name}_sink")
    lines.append(f"{indent}{sink_var} = {sink_expr}")
    lines.append(f"{indent}if {sink_var} is None:")
    lines.append(
        f"{indent}    {out_var} = F.scaled_dot_product_attention({q}, {k}, {v}, attn_mask={mask_expr}, dropout_p=0.0, is_causal={is_causal}, scale={scale_expr})"
    )
    lines.append(f"{indent}else:")

    attn_logits_var = assign_out_var(f"{out_name}_logits")
    sinks_var = assign_out_var(f"{out_name}_sinks")
    combined_var = assign_out_var(f"{out_name}_combined")
    probs_var = assign_out_var(f"{out_name}_probs")
    scores_var = assign_out_var(f"{out_name}_scores")
    lines.append(f"{indent}    {attn_logits_var} = ({q} @ {k}.transpose(-2, -1)) * ({scale_expr})")
    lines.append(f"{indent}    if {mask_expr} is not None:")
    lines.append(f"{indent}        {attn_logits_var} = {attn_logits_var} + {mask_expr}")
    lines.append(
        f"{indent}    {sinks_var} = {sink_var}.reshape(1, -1, 1, 1).expand({q}.shape[0], -1, {q}.shape[-2], -1)"
    )
    lines.append(
        f"{indent}    {combined_var} = torch.cat([{attn_logits_var}, {sinks_var}], dim=-1)"
    )
    lines.append(
        f"{indent}    {combined_var} = {combined_var} - {combined_var}.max(dim=-1, keepdim=True).values"
    )
    lines.append(
        f"{indent}    {probs_var} = F.softmax({combined_var}, dim=-1, dtype={combined_var}.dtype)"
    )
    lines.append(f"{indent}    {scores_var} = {probs_var}[..., :-1]")
    lines.append(f"{indent}    {out_var} = {scores_var}.to({v}.dtype) @ {v}")
    return lines


__all__ = [
    "LOWERING_ARITY",
    "LOWERING_ALLOWED_KWARGS",
    "LOWERING_REQUIRED_KWARGS",
    "LOWERING_KWARG_KINDS",
    "OP_NAME",
    "interpret",
    "compile",
    "uses_node_path",
]
