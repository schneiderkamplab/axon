"""Attention masks and attention kernels.

Mask semantics follow ``transformers.masking_utils`` (eager/sdpa paths):

* encoder full layers:    every key, minus padded keys;
* encoder sliding layers: ``|q - k| <= w`` (``sliding_window_bidirectional_overlay``);
* decoder full layers:    ``k <= q``;
* decoder sliding layers: ``q - w < k <= q`` (``sliding_window_overlay`` + causal).

``w`` is ``config.sliding_window`` (= ``local_attention // 2``). A 2D ``attention_mask``
(1 = token, 0 = padding) indexed by absolute position removes padded keys. Rows of
padded *queries* ignore the padding mask instead of being fully masked, so they stay
finite (HF fills them with a uniform softmax over ``finfo.min``); their values are
never read by real tokens.

Every mask is built once per forward call per layer type and closed over by an
``attend(q, k, v, scale) -> (B, Lq, H*D)`` callable that each layer calls.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import mlx.core as mx

Attend = Callable[[mx.array, mx.array, mx.array, float], mx.array]
Mask = mx.array | str | None

# Below this length the dense masked kernel is as fast as the blocked one.
BLOCKED_MIN_LENGTH = 512


def use_blocked(mode: str, length: int, window: int) -> bool:
    if mode == "dense":
        return False
    if mode == "blocked":
        return True
    if mode != "auto":
        raise ValueError(f"local_attention must be 'auto', 'dense' or 'blocked', got {mode!r}")
    return length >= BLOCKED_MIN_LENGTH and length > 4 * window


def build_mask(
    q_start: int,
    q_len: int,
    kv_start: int,
    kv_len: int,
    *,
    causal: bool,
    window: int | None,
    attention_mask: mx.array | None = None,
) -> Mask:
    """Boolean mask (True = attend) broadcastable to ``(B, H, q_len, kv_len)``.

    Queries sit at absolute positions ``[q_start, q_start + q_len)`` and keys at
    ``[kv_start, kv_start + kv_len)``. Returns ``None`` when nothing is masked and
    ``"causal"`` when the plain lower-right causal mask is exact.
    """
    q_last, kv_last = q_start + q_len - 1, kv_start + kv_len - 1
    if causal:
        need_causal = kv_last > q_start
        need_window = window is not None and kv_start <= q_last - window
    else:
        need_causal = False
        need_window = window is not None and max(q_last - kv_start, kv_last - q_start) > window

    if attention_mask is None:
        if not (need_causal or need_window):
            return None
        if need_causal and not need_window and kv_last == q_last:
            return "causal"

    qi = mx.arange(q_start, q_start + q_len)[:, None]
    ki = mx.arange(kv_start, kv_start + kv_len)[None, :]
    mask = None
    if need_causal:
        mask = ki <= qi
    if need_window:
        in_window = (ki > qi - window) if causal else (mx.abs(qi - ki) <= window)
        mask = in_window if mask is None else mask & in_window
    if attention_mask is not None:
        valid = attention_mask.astype(mx.bool_)
        key_ok = valid[:, None, None, kv_start : kv_start + kv_len]
        if mask is None:
            # Unrestricted attention: every query row sees all real keys, so a
            # (B, 1, 1, kv_len) key mask is enough (rows with no real key at all
            # are let through instead of producing NaN).
            return key_ok | ~mx.any(key_ok, axis=-1, keepdims=True)
        query_pad = ~valid[:, None, q_start : q_start + q_len, None]
        mask = mask & (key_ok | query_pad)
    return mask


def dense_attention(mask: Mask) -> Attend:
    def attend(q, k, v, scale):
        B, _, L, _ = q.shape
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
        return out.transpose(0, 2, 1, 3).reshape(B, L, -1)

    return attend


def blocked_local_attention(
    batch: int,
    length: int,
    num_heads: int,
    *,
    window: int,
    causal: bool,
    block: int,
    attention_mask: mx.array | None = None,
) -> Attend:
    """Sliding-window attention over a full sequence in O(L * window).

    Queries are split into blocks of ``block`` rows. Block ``i`` attends to the
    contiguous key span that any of its rows can see: ``[i*block - w, (i+1)*block + w)``
    for the bidirectional window and ``[i*block - w + 1, (i+1)*block)`` for the causal
    one. The spans overlap, so they are exposed as an ``as_strided`` view of the padded
    keys (no copy) and fed to the fused SDPA kernel as a ``(B*H, n_blocks, span, D)``
    batch. The exact window/padding mask is applied inside each span, so the result
    equals dense masked attention up to float summation order.
    """
    n_blocks = math.ceil(length / block)
    padded = n_blocks * block
    left = window if not causal else window - 1
    right = (padded - length) + (0 if causal else window)
    span = block + left + (0 if causal else window)
    total = left + length + right  # padded key length (row stride of the strided view)

    # Absolute key position for (block i, slot j) and query position for (block i, row r).
    k_pos = mx.arange(n_blocks)[:, None] * block - left + mx.arange(span)[None, :]
    q_pos = mx.arange(n_blocks)[:, None] * block + mx.arange(block)[None, :]
    rel = k_pos[:, None, :] - q_pos[:, :, None]  # (n_blocks, block, span)
    mask = (rel <= 0) & (rel > -window) if causal else mx.abs(rel) <= window
    key_ok = (k_pos >= 0) & (k_pos < length)  # (n_blocks, span)
    if attention_mask is None:
        mask = (mask & key_ok[:, None, :])[None]  # (1, n_blocks, block, span)
    else:
        valid = attention_mask.astype(mx.bool_)
        k_valid = mx.take(valid, mx.clip(k_pos, 0, length - 1).reshape(-1), axis=1)
        k_valid = k_valid.reshape(batch, n_blocks, 1, span) & key_ok[None, :, None, :]
        q_valid = mx.pad(valid, [(0, 0), (0, padded - length)]).reshape(batch, n_blocks, block, 1)
        mask = mask[None] & (k_valid | ~q_valid)  # (B, n_blocks, block, span)
        mask = mx.broadcast_to(mask[:, None], (batch, num_heads, n_blocks, block, span))
        mask = mask.reshape(batch * num_heads, n_blocks, block, span)

    def windows(x):
        B, H, _, D = x.shape
        x = mx.pad(x, [(0, 0), (0, 0), (left, right), (0, 0)])
        return mx.as_strided(x, (B * H, n_blocks, span, D), (total * D, block * D, D, 1))

    def attend(q, k, v, scale):
        B, H, L, D = q.shape
        qb = mx.pad(q, [(0, 0), (0, 0), (0, padded - L), (0, 0)]).reshape(B * H, n_blocks, block, D)
        out = mx.fast.scaled_dot_product_attention(qb, windows(k), windows(v), scale=scale, mask=mask)
        out = out.reshape(B, H, padded, D)[:, :, :L]
        return out.transpose(0, 2, 1, 3).reshape(B, L, -1)

    return attend
