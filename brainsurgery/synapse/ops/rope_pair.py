from __future__ import annotations

import math
from typing import Any

import torch

OP_NAME = "rope_pair"
LOWERING_ARITY = (2, 2)
LOWERING_ALLOWED_KWARGS: set[str] = {
    "position_ids",
    "theta",
    "scale_factor",
    "low_freq_factor",
    "high_freq_factor",
    "original_context",
    "attention_factor",
    "rope_mode",
    "truncate",
}
LOWERING_REQUIRED_KWARGS: set[str] = {"position_ids"}
LOWERING_KWARG_KINDS: dict[str, Any] = {
    "position_ids": "str",
    "theta": "number",
    "scale_factor": "number",
    "low_freq_factor": "number",
    "high_freq_factor": "number",
    "original_context": "dim",
    "attention_factor": "number",
    "rope_mode": "str",
    "truncate": "bool",
}


def uses_node_path(emitter: Any, node_spec: dict[str, Any]) -> bool:
    del emitter, node_spec
    return False


def lowering_known_output_arity(*, kwargs: dict[str, Any]) -> int:
    del kwargs
    return 2


def _apply_frequency_scaling(
    *,
    inv_freq: torch.Tensor,
    scale_factor: float,
    low_freq_factor: float,
    high_freq_factor: float,
    original_context: int,
) -> torch.Tensor:
    if scale_factor <= 0.0:
        raise ValueError("rope_pair.scale_factor must be > 0")
    if low_freq_factor <= 0.0:
        raise ValueError("rope_pair.low_freq_factor must be > 0")
    if high_freq_factor <= low_freq_factor:
        raise ValueError("rope_pair.high_freq_factor must be > low_freq_factor")
    if original_context <= 0:
        raise ValueError("rope_pair.original_context must be > 0")

    low_freq_wavelen = float(original_context) / low_freq_factor
    high_freq_wavelen = float(original_context) / high_freq_factor
    wavelen = (2.0 * math.pi) / inv_freq

    inv_scaled = torch.where(wavelen > low_freq_wavelen, inv_freq / scale_factor, inv_freq)
    smooth = (float(original_context) / wavelen - low_freq_factor) / (
        high_freq_factor - low_freq_factor
    )
    smoothed = (1.0 - smooth) * (inv_scaled / scale_factor) + smooth * inv_scaled
    is_medium = (~(wavelen < high_freq_wavelen)) & (~(wavelen > low_freq_wavelen))
    return torch.where(is_medium, smoothed, inv_scaled)


def _apply_frequency_scaling_hf_yarn(
    *,
    inv_freq: torch.Tensor,
    scale_factor: float,
    low_freq_factor: float,
    high_freq_factor: float,
    original_context: int,
    truncate: bool,
) -> torch.Tensor:
    if scale_factor <= 0.0:
        raise ValueError("rope_pair.scale_factor must be > 0")
    if low_freq_factor <= 0.0:
        raise ValueError("rope_pair.low_freq_factor must be > 0")
    if high_freq_factor <= low_freq_factor:
        raise ValueError("rope_pair.high_freq_factor must be > low_freq_factor")
    if original_context <= 0:
        raise ValueError("rope_pair.original_context must be > 0")

    # Match HF _compute_yarn_parameters exactly on the inverse-frequency mixing.
    # We reconstruct pos_freqs from inv_freq and then blend interpolation/extrapolation.
    pos_freqs = 1.0 / inv_freq
    dim = int(inv_freq.shape[0] * 2)
    inv_freq_extrapolation = inv_freq
    inv_freq_interpolation = 1.0 / (scale_factor * pos_freqs)

    def _find_correction_dim(
        num_rotations: float, dim_value: int, base: float, max_pos: int
    ) -> float:
        return (dim_value * math.log(max_pos / (num_rotations * 2.0 * math.pi))) / (
            2.0 * math.log(base)
        )

    def _find_correction_range(
        low_rot: float,
        high_rot: float,
        dim_value: int,
        base: float,
        max_pos: int,
        truncate_value: bool,
    ) -> tuple[float, float]:
        low = _find_correction_dim(low_rot, dim_value, base, max_pos)
        high = _find_correction_dim(high_rot, dim_value, base, max_pos)
        if truncate_value:
            low = math.floor(low)
            high = math.ceil(high)
        return max(low, 0.0), min(high, float(dim_value - 1))

    def _linear_ramp_factor(min_value: float, max_value: float, ramp_dim: int) -> torch.Tensor:
        if min_value == max_value:
            max_value += 0.001
        linear = (
            torch.arange(ramp_dim, dtype=torch.float32, device=inv_freq.device) - min_value
        ) / (max_value - min_value)
        return torch.clamp(linear, 0.0, 1.0)

    # Recover theta/base robustly from inv_freq sequence relation:
    # inv_freq[i] = 1 / (theta ** (2i/dim)).
    if inv_freq.shape[0] > 1:
        ratio = float(inv_freq[1] / inv_freq[0])
        if ratio <= 0.0 or ratio == 1.0:
            theta = 10000.0
        else:
            theta = ratio ** (-dim / 2.0)
    else:
        theta = 10000.0
    low, high = _find_correction_range(
        high_freq_factor,
        low_freq_factor,
        dim,
        theta,
        original_context,
        truncate,
    )
    extrapolation_factor = 1.0 - _linear_ramp_factor(low, high, dim // 2).to(
        device=inv_freq.device, dtype=inv_freq.dtype
    )
    return (
        inv_freq_interpolation * (1.0 - extrapolation_factor)
        + inv_freq_extrapolation * extrapolation_factor
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
    attention_factor = float(model._eval_expr(node_spec.get("attention_factor", 1.0), env, symbols))
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
        theta ** (torch.arange(0, half, device=q.device, dtype=torch.float32) / float(half))
    )
    rope_mode = str(node_spec.get("rope_mode", "")).strip().lower()
    if all(
        key in node_spec
        for key in ("scale_factor", "low_freq_factor", "high_freq_factor", "original_context")
    ):
        scale_factor = float(model._eval_expr(node_spec["scale_factor"], env, symbols))
        low_freq_factor = float(model._eval_expr(node_spec["low_freq_factor"], env, symbols))
        high_freq_factor = float(model._eval_expr(node_spec["high_freq_factor"], env, symbols))
        original_context = int(model._eval_expr(node_spec["original_context"], env, symbols))
        if rope_mode == "hf_yarn":
            inv_freq = _apply_frequency_scaling_hf_yarn(
                inv_freq=inv_freq,
                scale_factor=scale_factor,
                low_freq_factor=low_freq_factor,
                high_freq_factor=high_freq_factor,
                original_context=original_context,
                truncate=bool(node_spec.get("truncate", True)),
            )
        else:
            inv_freq = _apply_frequency_scaling(
                inv_freq=inv_freq,
                scale_factor=scale_factor,
                low_freq_factor=low_freq_factor,
                high_freq_factor=high_freq_factor,
                original_context=original_context,
            )
    pos = pos_ids.to(device=q.device, dtype=torch.float32)
    ang = pos.unsqueeze(-1) * inv_freq.unsqueeze(0).unsqueeze(0)
    cos = (torch.cos(ang) * attention_factor).to(dtype=q.dtype).unsqueeze(1)
    sin = (torch.sin(ang) * attention_factor).to(dtype=q.dtype).unsqueeze(1)
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
    attention_factor = emitter._expr_code(node_spec.get("attention_factor", 1.0), env)
    scale_factor = emitter._expr_code(node_spec.get("scale_factor"), env)
    low_freq_factor = emitter._expr_code(node_spec.get("low_freq_factor"), env)
    high_freq_factor = emitter._expr_code(node_spec.get("high_freq_factor"), env)
    original_context = emitter._expr_code(node_spec.get("original_context"), env)
    rope_mode = str(node_spec.get("rope_mode", "")).strip().lower()
    truncate = bool(node_spec.get("truncate", True))
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
        f"{indent}{inv_freq} = 1.0 / (float({theta}) ** (torch.arange(0, {half}, device={q}.device, dtype=torch.float32) / float({half})))"
    )
    if all(
        key in node_spec
        for key in ("scale_factor", "low_freq_factor", "high_freq_factor", "original_context")
    ):
        low_freq_wavelen = emitter._fresh("low_freq_wavelen")
        high_freq_wavelen = emitter._fresh("high_freq_wavelen")
        wavelen = emitter._fresh("wavelen")
        inv_scaled = emitter._fresh("inv_scaled")
        smooth = emitter._fresh("smooth")
        smoothed = emitter._fresh("smoothed")
        is_medium = emitter._fresh("is_medium")
        lines.append(f"{indent}if float({scale_factor}) <= 0.0:")
        lines.append(f"{indent}    raise ValueError('rope_pair.scale_factor must be > 0')")
        lines.append(f"{indent}if float({low_freq_factor}) <= 0.0:")
        lines.append(f"{indent}    raise ValueError('rope_pair.low_freq_factor must be > 0')")
        lines.append(f"{indent}if float({high_freq_factor}) <= float({low_freq_factor}):")
        lines.append(
            f"{indent}    raise ValueError('rope_pair.high_freq_factor must be > low_freq_factor')"
        )
        lines.append(f"{indent}if int({original_context}) <= 0:")
        lines.append(f"{indent}    raise ValueError('rope_pair.original_context must be > 0')")
        lines.append(
            f"{indent}{low_freq_wavelen} = float({original_context}) / float({low_freq_factor})"
        )
        lines.append(
            f"{indent}{high_freq_wavelen} = float({original_context}) / float({high_freq_factor})"
        )
        lines.append(f"{indent}{wavelen} = (2.0 * torch.pi) / {inv_freq}")
        if rope_mode == "hf_yarn":
            theta_var = emitter._fresh("theta")
            low_var = emitter._fresh("low")
            high_var = emitter._fresh("high")
            ramp = emitter._fresh("ramp")
            extrapolation = emitter._fresh("extrapolation")
            inv_interp = emitter._fresh("inv_interp")
            inv_extra = emitter._fresh("inv_extra")
            lines.append(f"{indent}if {inv_freq}.shape[0] > 1:")
            lines.append(f"{indent}    _ratio = float({inv_freq}[1] / {inv_freq}[0])")
            lines.append(f"{indent}    if _ratio <= 0.0 or _ratio == 1.0:")
            lines.append(f"{indent}        {theta_var} = 10000.0")
            lines.append(f"{indent}    else:")
            lines.append(f"{indent}        {theta_var} = _ratio ** (-(2.0 * {half}) / 2.0)")
            lines.append(f"{indent}else:")
            lines.append(f"{indent}    {theta_var} = 10000.0")
            lines.append(
                f"{indent}{low_var} = ((2.0 * {half}) * torch.log(torch.tensor(int({original_context}) / (float({high_freq_factor}) * 2.0 * torch.pi), device={q}.device, dtype=torch.float32))) / (2.0 * torch.log(torch.tensor({theta_var}, device={q}.device, dtype=torch.float32)))"
            )
            lines.append(
                f"{indent}{high_var} = ((2.0 * {half}) * torch.log(torch.tensor(int({original_context}) / (float({low_freq_factor}) * 2.0 * torch.pi), device={q}.device, dtype=torch.float32))) / (2.0 * torch.log(torch.tensor({theta_var}, device={q}.device, dtype=torch.float32)))"
            )
            if truncate:
                lines.append(f"{indent}{low_var} = torch.floor({low_var})")
                lines.append(f"{indent}{high_var} = torch.ceil({high_var})")
            lines.append(
                f"{indent}{low_var} = torch.clamp({low_var}, min=0.0, max=float((2 * {half}) - 1))"
            )
            lines.append(
                f"{indent}{high_var} = torch.clamp({high_var}, min=0.0, max=float((2 * {half}) - 1))"
            )
            lines.append(f"{indent}if float({low_var}) == float({high_var}):")
            lines.append(f"{indent}    {high_var} = {high_var} + 0.001")
            lines.append(
                f"{indent}{ramp} = torch.clamp((torch.arange({half}, dtype=torch.float32, device={q}.device) - {low_var}) / ({high_var} - {low_var}), 0.0, 1.0).to(dtype={inv_freq}.dtype)"
            )
            lines.append(f"{indent}{extrapolation} = 1.0 - {ramp}")
            lines.append(f"{indent}{inv_extra} = {inv_freq}")
            lines.append(f"{indent}{inv_interp} = {inv_freq} / float({scale_factor})")
            lines.append(
                f"{indent}{inv_freq} = {inv_interp} * (1.0 - {extrapolation}) + {inv_extra} * {extrapolation}"
            )
        else:
            lines.append(
                f"{indent}{inv_scaled} = torch.where({wavelen} > {low_freq_wavelen}, {inv_freq} / float({scale_factor}), {inv_freq})"
            )
            lines.append(
                f"{indent}{smooth} = (float({original_context}) / {wavelen} - float({low_freq_factor})) / (float({high_freq_factor}) - float({low_freq_factor}))"
            )
            lines.append(
                f"{indent}{smoothed} = (1.0 - {smooth}) * ({inv_scaled} / float({scale_factor})) + {smooth} * {inv_scaled}"
            )
            lines.append(
                f"{indent}{is_medium} = (~({wavelen} < {high_freq_wavelen})) & (~({wavelen} > {low_freq_wavelen}))"
            )
            lines.append(f"{indent}{inv_freq} = torch.where({is_medium}, {smoothed}, {inv_scaled})")
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
    lines.append(f"{indent}{pos} = {pos_ids}.to(device={q}.device, dtype=torch.float32)")
    lines.append(f"{indent}{ang} = {pos}.unsqueeze(-1) * {inv_freq}.unsqueeze(0).unsqueeze(0)")
    lines.append(
        f"{indent}{cos} = (torch.cos({ang}) * float({attention_factor})).to(dtype={q}.dtype).unsqueeze(1)"
    )
    lines.append(
        f"{indent}{sin} = (torch.sin({ang}) * float({attention_factor})).to(dtype={q}.dtype).unsqueeze(1)"
    )
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
