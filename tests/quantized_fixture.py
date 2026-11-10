"""Shared numpy fixtures for quantized-checkpoint tests (torch/jax backends).

The canonical packed layout mirrors the one the generated cores hold in memory:

- ``<base>.weight.int8``  uint8 ``(rows, K)`` for 8-bit, ``(rows, K // 2)`` for
  4-bit; for 4-bit, byte ``j`` holds value ``2j`` in its low nibble and value
  ``2j + 1`` in its high nibble.
- ``<base>.weight.scale`` / ``<base>.weight.qbias``: ``(rows, K // group_size)``
  per-group affine scale/bias (dequant: ``q * scale + bias``).

``pack_mlx_layout`` also produces the on-disk ``mlx.nn.quantize`` form (uint32
words, values LSB-first) so tests can exercise the load-time unpack path.
"""

from __future__ import annotations

import numpy as np


def affine_quantize(w, bits, group_size):
    """Per-group affine quantization of a 2-D float array.

    Returns ``(q, scales, biases)`` with ``q`` int64 values in ``[0, 2**bits - 1]``
    shaped ``(rows, K)``, and fp16 ``scales``/``biases`` shaped ``(rows, K // group_size)``.
    """
    w = np.asarray(w, dtype=np.float32)
    rows, K = w.shape
    if K % group_size:
        raise ValueError(f"K={K} is not divisible by group_size={group_size}")
    qmax = (1 << bits) - 1
    groups = w.reshape(rows, K // group_size, group_size)
    amin = groups.min(-1, keepdims=True)
    amax = groups.max(-1, keepdims=True)
    scale = np.maximum((amax - amin) / qmax, 1e-12)
    q = np.clip(np.round((groups - amin) / scale), 0, qmax).astype(np.int64).reshape(rows, K)
    scales = np.broadcast_to(scale.reshape(rows, K // group_size), (rows, K // group_size)).astype(np.float16).copy()
    biases = np.broadcast_to(amin.reshape(rows, K // group_size), (rows, K // group_size)).astype(np.float16).copy()
    return q, scales, biases


def pack_bytes(q, bits):
    """Pack int64 values ``(rows, K)`` into the canonical uint8 layout."""
    rows, K = q.shape
    if bits == 8:
        return q.astype(np.uint8)
    if K % 2:
        raise ValueError("4-bit packing needs an even K")
    return (q[:, 0::2] | (q[:, 1::2] << 4)).astype(np.uint8)


def unpack_bytes(packed, bits):
    """Inverse of ``pack_bytes``: uint8 ``(rows, K // divisor)`` back to values."""
    if bits == 8:
        return packed.astype(np.int64)
    return np.stack([packed & 0xF, packed >> 4], axis=-1).reshape(packed.shape[0], -1)


def dequantize_bytes(packed, scales, biases, bits, group_size):
    """Reference dequantization of the canonical packed layout, in float32."""
    rows = packed.shape[0]
    values = unpack_bytes(packed, bits).shape[1]
    groups = values // group_size
    q = unpack_bytes(packed, bits).reshape(rows, groups, group_size).astype(np.float32)
    s = np.asarray(scales, dtype=np.float32)[:, :, None]
    b = np.asarray(biases, dtype=np.float32)[:, :, None]
    return (q * s + b).reshape(rows, values)


def pack_mlx_layout(q, bits):
    """Pack int64 values ``(rows, K)`` into ``mlx.nn.quantize`` uint32 words (LSB-first)."""
    rows, K = q.shape
    words = (K * bits + 31) // 32
    per_word = 32 // bits
    pad = words * per_word - K
    values = np.concatenate([q, np.zeros((rows, pad), dtype=np.int64)], axis=1)
    groups = values.reshape(rows, words, per_word)
    weights = (1 << (np.arange(per_word, dtype=np.int64) * bits)).reshape(1, 1, per_word)
    return (groups * weights).sum(-1).astype(np.uint32)


def quantized_state_from_float(state, bits, group_size, *, bases=None):
    """Turn a float ``{path.weight: array, ...}`` state dict into the MLX
    ``nn.quantize`` checkpoint layout (uint32-packed ``<base>.weight`` +
    ``<base>.scales`` + ``<base>.biases``), replacing the float weights."""
    out = {}
    for key, value in state.items():
        k = str(key)
        if bases is not None and k not in {f"{base}.weight" for base in bases}:
            out[k] = value
            continue
        array = np.asarray(value)
        if array.ndim != 2 or array.shape[1] % group_size:
            out[k] = value
            continue
        q, scales, biases = affine_quantize(array, bits, group_size)
        base = k[: -len(".weight")]
        out[f"{base}.weight"] = pack_mlx_layout(q, bits)
        out[f"{base}.scales"] = scales
        out[f"{base}.biases"] = biases
    return out


def w8a16_pair(w):
    """W8A16 export layout pair (``<base>.int8`` int8 + ``<base>.scale`` fp32,
    symmetric per-output-channel absmax over dim 1), as in ``synapse/int8.py``."""
    w = np.asarray(w, dtype=np.float32)
    scale = np.abs(w).max(axis=1) / 127.0
    scale = np.maximum(scale, 1e-12).astype(np.float32)
    q = np.clip(np.round(w / scale[:, None]), -127, 127).astype(np.int8)
    return q, scale
