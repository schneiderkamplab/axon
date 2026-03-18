from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional as F

OP_NAME = "moe_expert_ffn"


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
    out = model._require_name(node_spec.get("out"), field="moe_expert_ffn.out")
    expert = int(model._eval_expr(node_spec.get("expert"), env, symbols))
    experts_scope = node_spec.get("experts_scope", "experts")
    if not isinstance(experts_scope, str):
        raise ValueError("moe_expert_ffn experts_scope must be string")
    experts_base = model._join(scope, experts_scope)
    experts_parent = experts_base.rsplit(".", 1)[0] if "." in experts_base else ""
    gate_name = str(node_spec.get("gate_proj_name", "gate_proj.weight"))
    up_name = str(node_spec.get("up_proj_name", "up_proj.weight"))
    down_name = str(node_spec.get("down_proj_name", "down_proj.weight"))
    activation = str(node_spec.get("activation", "silu"))
    packed_gate_up = model._join(experts_base, "gate_up_proj")
    packed_down = model._join(experts_base, "down_proj")
    packed_gate_up_parent = model._join(experts_parent, "gate_up_proj") if experts_parent else ""
    packed_down_parent = model._join(experts_parent, "down_proj") if experts_parent else ""

    def resolve_expert_weight(name: str) -> torch.Tensor:
        candidates = [
            model._join(experts_base, f"{expert}.{name}"),
            model._join(experts_base, name),
        ]
        if experts_parent:
            candidates.append(model._join(experts_parent, f"{expert}.{name}"))
            candidates.append(model._join(experts_parent, name))
        for path in candidates:
            found = model._state.get(path)
            if found is not None:
                return found
        raise KeyError(candidates[0])

    if x.numel() == 0:
        packed_down_key = (
            packed_down
            if packed_down in model._state
            else (packed_down_parent if packed_down_parent in model._state else None)
        )
        if packed_down_key is not None:
            down_w_shape = model._state[packed_down_key].shape
            out_dim = int(down_w_shape[-2])
            env[out] = x.new_zeros((0, out_dim))
        else:
            down_w = resolve_expert_weight(down_name)
            env[out] = x.new_zeros((0, int(down_w.shape[-2])))
        return

    packed_gate_up_key = (
        packed_gate_up
        if packed_gate_up in model._state
        else (packed_gate_up_parent if packed_gate_up_parent in model._state else None)
    )
    packed_down_key = (
        packed_down
        if packed_down in model._state
        else (packed_down_parent if packed_down_parent in model._state else None)
    )
    if packed_gate_up_key is not None and packed_down_key is not None:
        packed_gate_up_w = model._state[packed_gate_up_key]
        if packed_gate_up_w.ndim == 3:
            packed_gate_up_w = packed_gate_up_w[expert]
        gate, up = F.linear(x, packed_gate_up_w, None).chunk(2, dim=-1)
        down_w = model._state[packed_down_key]
        if down_w.ndim == 3:
            down_w = down_w[expert]
    else:
        gate_w = resolve_expert_weight(gate_name)
        up_w = resolve_expert_weight(up_name)
        down_w = resolve_expert_weight(down_name)
        gate = F.linear(x, gate_w, None)
        up = F.linear(x, up_w, None)

    if activation in {"gelu_new", "gelu_pytorch_tanh"}:
        hidden = (
            0.5
            * gate
            * (1.0 + torch.tanh(0.7978845608028654 * (gate + 0.044715 * gate * gate * gate)))
            * up
        )
    elif activation == "gelu":
        hidden = F.gelu(gate) * up
    elif activation == "relu":
        hidden = F.relu(gate) * up
    elif activation == "silu":
        hidden = F.silu(gate) * up
    else:
        raise ValueError(f"Unsupported moe_expert_ffn activation kind: {activation}")

    env[out] = F.linear(hidden, down_w, None)
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
    expert = emitter._expr_code(node_spec.get("expert"), env)
    experts_scope = node_spec.get("experts_scope", "experts")
    if not isinstance(experts_scope, str):
        raise ValueError("moe_expert_ffn experts_scope must be string")
    experts_base = emitter._fresh("experts_base")
    experts_parent = emitter._fresh("experts_parent")
    lines.append(f"{indent}{experts_base} = emitter._join_scope({scope_var}, {experts_scope!r})")
    lines.append(
        f"{indent}{experts_parent} = {experts_base}.rsplit('.', 1)[0] if '.' in {experts_base} else ''"
    )
    gate_name = str(node_spec.get("gate_proj_name", "gate_proj.weight"))
    up_name = str(node_spec.get("up_proj_name", "up_proj.weight"))
    down_name = str(node_spec.get("down_proj_name", "down_proj.weight"))
    activation = str(node_spec.get("activation", "silu"))
    packed_gate_up = emitter._fresh("packed_gate_up")
    packed_down = emitter._fresh("packed_down")
    packed_gate_up_parent = emitter._fresh("packed_gate_up_parent")
    packed_down_parent = emitter._fresh("packed_down_parent")
    packed_gate_up_key = emitter._fresh("packed_gate_up_key")
    packed_down_key = emitter._fresh("packed_down_key")
    packed_gate_up_w = emitter._fresh("packed_gate_up_w")
    down_w = emitter._fresh("down_w")
    gate = emitter._fresh("gate")
    up = emitter._fresh("up")
    hidden_var = emitter._fresh("hidden")
    lines.append(f"{indent}{packed_gate_up} = emitter._join_scope({experts_base}, 'gate_up_proj')")
    lines.append(f"{indent}{packed_down} = self._join_scope({experts_base}, 'down_proj')")
    lines.append(
        f"{indent}{packed_gate_up_parent} = emitter._join_scope({experts_parent}, 'gate_up_proj') if {experts_parent} else ''"
    )
    lines.append(
        f"{indent}{packed_down_parent} = emitter._join_scope({experts_parent}, 'down_proj') if {experts_parent} else ''"
    )
    lines.append(
        f"{indent}{packed_gate_up_key} = {packed_gate_up} if {packed_gate_up} in emitter._state else ({packed_gate_up_parent} if {packed_gate_up_parent} in emitter._state else None)"
    )
    lines.append(
        f"{indent}{packed_down_key} = {packed_down} if {packed_down} in emitter._state else ({packed_down_parent} if {packed_down_parent} in emitter._state else None)"
    )
    lines.append(f"{indent}if {src}.numel() == 0:")
    lines.append(f"{indent}    if {packed_down_key} is not None:")
    lines.append(f"{indent}        _down_shape = self._param({packed_down_key}).shape")
    lines.append(f"{indent}        {out_var} = {src}.new_zeros((0, int(_down_shape[-2])))")
    lines.append(f"{indent}    else:")
    lines.append(
        f"{indent}        _dw_indexed = emitter._join_scope({experts_base}, f'{{int({expert})}}.{down_name}')"
    )
    lines.append(f"{indent}        _dw_scoped = emitter._join_scope({experts_base}, {down_name!r})")
    lines.append(f"{indent}        _dw_candidates = [_dw_indexed, _dw_scoped]")
    lines.append(f"{indent}        if {experts_parent}:")
    lines.append(
        f"{indent}            _dw_candidates.extend([emitter._join_scope({experts_parent}, f'{{int({expert})}}.{down_name}'), emitter._join_scope({experts_parent}, {down_name!r})])"
    )
    lines.append(f"{indent}        _dw = None")
    lines.append(f"{indent}        for _p in _dw_candidates:")
    lines.append(f"{indent}            if _p in self._state:")
    lines.append(f"{indent}                _dw = self._state[_p]")
    lines.append(f"{indent}                break")
    lines.append(f"{indent}        if _dw is None:")
    lines.append(f"{indent}            raise KeyError(_dw_candidates[0])")
    lines.append(f"{indent}        {out_var} = {src}.new_zeros((0, int(_dw.shape[-2])))")
    lines.append(f"{indent}else:")
    lines.append(
        f"{indent}    if {packed_gate_up_key} is not None and {packed_down_key} is not None:"
    )
    lines.append(f"{indent}        {packed_gate_up_w} = self._param({packed_gate_up_key})")
    lines.append(f"{indent}        if {packed_gate_up_w}.ndim == 3:")
    lines.append(f"{indent}            {packed_gate_up_w} = {packed_gate_up_w}[int({expert})]")
    lines.append(
        f"{indent}        {gate}, {up} = F.linear({src}, {packed_gate_up_w}, None).chunk(2, dim=-1)"
    )
    lines.append(f"{indent}        {down_w} = self._param({packed_down_key})")
    lines.append(f"{indent}        if {down_w}.ndim == 3:")
    lines.append(f"{indent}            {down_w} = {down_w}[int({expert})]")
    lines.append(f"{indent}    else:")
    lines.append(
        f"{indent}        _gw_indexed = emitter._join_scope({experts_base}, f'{{int({expert})}}.{gate_name}')"
    )
    lines.append(
        f"{indent}        _uw_indexed = emitter._join_scope({experts_base}, f'{{int({expert})}}.{up_name}')"
    )
    lines.append(
        f"{indent}        _dw_indexed = emitter._join_scope({experts_base}, f'{{int({expert})}}.{down_name}')"
    )
    lines.append(f"{indent}        _gw_scoped = emitter._join_scope({experts_base}, {gate_name!r})")
    lines.append(f"{indent}        _uw_scoped = emitter._join_scope({experts_base}, {up_name!r})")
    lines.append(f"{indent}        _dw_scoped = emitter._join_scope({experts_base}, {down_name!r})")
    lines.append(f"{indent}        _gw_candidates = [_gw_indexed, _gw_scoped]")
    lines.append(f"{indent}        _uw_candidates = [_uw_indexed, _uw_scoped]")
    lines.append(f"{indent}        _dw_candidates = [_dw_indexed, _dw_scoped]")
    lines.append(f"{indent}        if {experts_parent}:")
    lines.append(
        f"{indent}            _gw_candidates.extend([emitter._join_scope({experts_parent}, f'{{int({expert})}}.{gate_name}'), emitter._join_scope({experts_parent}, {gate_name!r})])"
    )
    lines.append(
        f"{indent}            _uw_candidates.extend([emitter._join_scope({experts_parent}, f'{{int({expert})}}.{up_name}'), emitter._join_scope({experts_parent}, {up_name!r})])"
    )
    lines.append(
        f"{indent}            _dw_candidates.extend([emitter._join_scope({experts_parent}, f'{{int({expert})}}.{down_name}'), emitter._join_scope({experts_parent}, {down_name!r})])"
    )
    lines.append(f"{indent}        _gw = None")
    lines.append(f"{indent}        _uw = None")
    lines.append(f"{indent}        {down_w} = None")
    lines.append(f"{indent}        for _p in _gw_candidates:")
    lines.append(f"{indent}            if _p in self._state:")
    lines.append(f"{indent}                _gw = self._state[_p]")
    lines.append(f"{indent}                break")
    lines.append(f"{indent}        for _p in _uw_candidates:")
    lines.append(f"{indent}            if _p in self._state:")
    lines.append(f"{indent}                _uw = self._state[_p]")
    lines.append(f"{indent}                break")
    lines.append(f"{indent}        for _p in _dw_candidates:")
    lines.append(f"{indent}            if _p in self._state:")
    lines.append(f"{indent}                {down_w} = self._state[_p]")
    lines.append(f"{indent}                break")
    lines.append(f"{indent}        if _gw is None:")
    lines.append(f"{indent}            raise KeyError(_gw_candidates[0])")
    lines.append(f"{indent}        if _uw is None:")
    lines.append(f"{indent}            raise KeyError(_uw_candidates[0])")
    lines.append(f"{indent}        if {down_w} is None:")
    lines.append(f"{indent}            raise KeyError(_dw_candidates[0])")
    lines.append(f"{indent}        {gate} = F.linear({src}, _gw, None)")
    lines.append(f"{indent}        {up} = F.linear({src}, _uw, None)")
    if activation in {"gelu_new", "gelu_pytorch_tanh"}:
        lines.append(
            f"{indent}    {hidden_var} = 0.5 * {gate} * (1.0 + torch.tanh(0.7978845608028654 * ({gate} + 0.044715 * {gate} * {gate} * {gate}))) * {up}"
        )
    elif activation == "gelu":
        lines.append(f"{indent}    {hidden_var} = F.gelu({gate}) * {up}")
    elif activation == "relu":
        lines.append(f"{indent}    {hidden_var} = F.relu({gate}) * {up}")
    elif activation == "silu":
        lines.append(f"{indent}    {hidden_var} = F.silu({gate}) * {up}")
    else:
        raise ValueError(f"Unsupported moe_expert_ffn activation kind: {activation}")
    lines.append(f"{indent}    {out_var} = F.linear({hidden_var}, {down_w}, None)")
    return lines


__all__ = ["OP_NAME", "interpret", "compile", "uses_node_path"]
