from __future__ import annotations

import math

import torch

_FP4_VALUES = [
    +0.0,
    +0.5,
    +1.0,
    +1.5,
    +2.0,
    +3.0,
    +4.0,
    +6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
]


def _convert_moe_packed_tensors(
    blocks: torch.Tensor,
    scales: torch.Tensor,
    *,
    dtype: torch.dtype = torch.bfloat16,
    rows_per_chunk: int = 32768 * 1024,
) -> torch.Tensor:
    blocks_u8 = blocks.to(torch.uint8)
    scales_i32 = scales.to(torch.int32) - 127

    if blocks_u8.shape[:-1] != scales_i32.shape:
        raise ValueError(
            f"MXFP4 shape mismatch: blocks.shape[:-1]={blocks_u8.shape[:-1]!r}, "
            f"scales.shape={scales_i32.shape!r}"
        )

    lut = torch.tensor(_FP4_VALUES, dtype=dtype, device=blocks_u8.device)

    *prefix_shape, groups, packed_cols = blocks_u8.shape
    rows_total = math.prod(prefix_shape) * groups

    blocks_2d = blocks_u8.reshape(rows_total, packed_cols)
    scales_2d = scales_i32.reshape(rows_total, 1)
    out = torch.empty(rows_total, packed_cols * 2, dtype=dtype, device=blocks_u8.device)

    for r0 in range(0, rows_total, rows_per_chunk):
        r1 = min(r0 + rows_per_chunk, rows_total)
        blk = blocks_2d[r0:r1]
        exp = scales_2d[r0:r1]
        dst = out[r0:r1]

        idx_lo = (blk & 0x0F).to(torch.int32)
        dst[:, 0::2] = lut[idx_lo]
        idx_hi = (blk >> 4).to(torch.int32)
        dst[:, 1::2] = lut[idx_hi]
        torch.ldexp(dst, exp, out=dst)

    out = out.reshape(*prefix_shape, groups, packed_cols * 2).view(
        *prefix_shape, groups * packed_cols * 2
    )
    return out.transpose(1, 2).contiguous()


def _materialize_mxfp4_aliases(
    state_dict: dict[str, torch.Tensor],
    *,
    dtype: torch.dtype,
) -> None:
    blocks_suffix = "_blocks"

    for key in list(state_dict.keys()):
        if not key.endswith(blocks_suffix):
            continue

        base = key[: -len(blocks_suffix)]
        scales_key = f"{base}_scales"
        if scales_key not in state_dict:
            continue

        weight_key = f"{base}.weight"
        if weight_key not in state_dict:
            dequantized = _convert_moe_packed_tensors(
                state_dict[key],
                state_dict[scales_key],
                dtype=dtype,
            )
            if dequantized.is_floating_point() and dequantized.dtype != dtype:
                dequantized = dequantized.to(dtype=dtype)
            state_dict[weight_key] = dequantized

        bias_src_key = f"{base}_bias"
        bias_dst_key = f"{base}.bias"
        if bias_src_key in state_dict and bias_dst_key not in state_dict:
            bias = state_dict[bias_src_key]
            if bias.is_floating_point() and bias.dtype != dtype:
                bias = bias.to(dtype=dtype)
            state_dict[bias_dst_key] = bias
