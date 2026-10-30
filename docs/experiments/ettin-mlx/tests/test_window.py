"""Test 3 (sliding-window boundary), mask-level: masks equal HF masking_utils, each layer
type sees exactly its window, and the blocked kernel equals dense masked attention."""

import mlx.core as mx
import numpy as np
import pytest
from transformers.masking_utils import (
    bidirectional_mask_function,
    causal_mask_function,
    sdpa_mask,
    sliding_window_bidirectional_mask_function,
    sliding_window_causal_mask_function,
)

from ettin_mlx import EttinConfig, EttinDecoder, EttinEncoder
from ettin_mlx.attention import blocked_local_attention, build_mask, dense_attention
from helpers import model

W = 64  # config.sliding_window for every Ettin checkpoint (local_attention=128)


def materialize(mask, batch, q_start, q_len, kv_start, kv_len) -> np.ndarray:
    if mask is None:
        return np.ones((batch, 1, q_len, kv_len), bool)
    if isinstance(mask, str):
        assert mask == "causal" and kv_start + kv_len == q_start + q_len
        q = np.arange(q_start, q_start + q_len)[:, None]
        k = np.arange(kv_start, kv_start + kv_len)[None]
        return np.broadcast_to(k <= q, (batch, 1, q_len, kv_len))
    return np.broadcast_to(np.array(mask), (batch, 1, q_len, kv_len))


def hf_mask(fn, batch, q_start, q_len, kv_start, kv_len, attention_mask):
    import torch

    am = None if attention_mask is None else torch.from_numpy(attention_mask.astype(bool))
    m = sdpa_mask(
        batch_size=batch, q_length=q_len, kv_length=kv_len, q_offset=q_start, kv_offset=kv_start,
        mask_function=fn, attention_mask=am, allow_is_causal_skip=False, allow_is_bidirectional_skip=False,
    )
    return m.numpy()


# (q_start, q_len, kv_start, kv_len): plain forwards, a prefill, decode steps on a
# sliding cache (kv_start = offset - (W - 1)) and on a full cache.
CAUSAL_SHAPES = [(0, 1, 0, 1), (0, 64, 0, 64), (0, 65, 0, 65), (0, 300, 0, 300), (100, 1, 37, 64),
                 (500, 1, 437, 64), (40, 1, 0, 41), (63, 1, 0, 64), (300, 20, 237, 83), (300, 1, 0, 301)]
BIDIR_SHAPES = [(0, n, 0, n) for n in (1, 7, 64, 65, 66, 129, 130, 300)]


@pytest.mark.parametrize("padded", [False, True])
@pytest.mark.parametrize("kind", ["encoder_sliding", "encoder_full", "decoder_sliding", "decoder_full"])
def test_masks_equal_hf_masking_utils(kind, padded):
    causal = kind.startswith("decoder")
    window = W if kind.endswith("sliding") else None
    fn = {
        "encoder_sliding": sliding_window_bidirectional_mask_function(W),
        "encoder_full": bidirectional_mask_function,
        "decoder_sliding": sliding_window_causal_mask_function(W),
        "decoder_full": causal_mask_function,
    }[kind]
    rng = np.random.default_rng(0)
    for q_start, q_len, kv_start, kv_len in CAUSAL_SHAPES if causal else BIDIR_SHAPES:
        total = max(q_start + q_len, kv_start + kv_len)
        am = None
        if padded:  # right padding for encoders, left padding for decoder decode steps
            am = np.ones((3, total), np.int32)
            for r, n in enumerate(rng.integers(0, total, size=3)):
                am[r, total - n :] = 0 if not causal else 1
                am[r, : n // 2] = 0 if causal else am[r, : n // 2]
        ours = build_mask(q_start, q_len, kv_start, kv_len, causal=causal, window=window,
                          attention_mask=None if am is None else mx.array(am))
        batch = 1 if am is None else am.shape[0]
        ours = materialize(ours, batch, q_start, q_len, kv_start, kv_len)
        theirs = hf_mask(fn, batch, q_start, q_len, kv_start, kv_len, am)
        # Padded query rows are deliberately left unmasked (never read); compare real rows.
        rows = np.ones((batch, q_len), bool) if am is None else am[:, q_start : q_start + q_len].astype(bool)
        assert np.array_equal(ours[:, 0][rows], theirs[:, 0][rows]), (kind, q_start, q_len, kv_start, kv_len)


def tiny_config(model_type: str, window_tokens: int, layer_type: str = "sliding_attention") -> EttinConfig:
    return EttinConfig.from_dict({
        "model_type": model_type, "vocab_size": 101, "hidden_size": 32, "intermediate_size": 48,
        "num_hidden_layers": 1, "num_attention_heads": 2, "layer_types": [layer_type],
        "local_attention": 2 * window_tokens, "global_rope_theta": 160000.0, "local_rope_theta": 10000.0,
    })


@pytest.mark.parametrize("impl", ["dense", "blocked"])
@pytest.mark.parametrize("model_type", ["modernbert", "modernbert-decoder"])
def test_receptive_field_is_exactly_the_window(model_type, impl):
    """One sliding layer: changing token j must change exactly the outputs whose window
    contains j - |q - j| <= w (encoder) or j <= q < j + w (decoder)."""
    mx.random.seed(0)
    cfg = tiny_config(model_type, W)
    m = (EttinDecoder if cfg.is_decoder else EttinEncoder)(cfg, local_attention=impl)
    L = 5 * 2 * W + 17
    ids = mx.random.randint(0, cfg.vocab_size, (1, L))
    run = (lambda x: m.hidden_states(x)) if cfg.is_decoder else m.encode
    base = np.array(run(ids))[0]
    q = np.arange(L)
    for j in [0, 1, W - 1, W, W + 1, 2 * W, 150, L - W - 1, L - 1]:
        changed_ids = np.array(ids)
        changed_ids[0, j] = (changed_ids[0, j] + 1) % cfg.vocab_size
        changed = np.abs(np.array(run(mx.array(changed_ids)))[0] - base).max(-1) > 0
        expected = (q >= j) & (q < j + W) if cfg.is_decoder else np.abs(q - j) <= W
        assert np.array_equal(changed, expected), (j, np.flatnonzero(changed ^ expected))


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("window", [1, 8, 64, 100])
@pytest.mark.parametrize("length", [5, 64, 257, 1000])
@pytest.mark.parametrize("padded", [False, True])
def test_blocked_equals_dense(causal, window, length, padded):
    rng = np.random.default_rng(length + window)
    B, H, D = 3, 2, 16
    q, k, v = (mx.array(rng.standard_normal((B, H, length, D)).astype(np.float32)) for _ in range(3))
    am = None
    if padded:
        lens = [length, max(1, length // 3), max(1, length - 7)]
        am = mx.array(np.array([[1] * n + [0] * (length - n) for n in lens], np.int32))
    dense = dense_attention(build_mask(0, length, 0, length, causal=causal, window=window, attention_mask=am))
    blocked = blocked_local_attention(B, length, H, window=window, causal=causal, block=window, attention_mask=am)
    a, b = np.array(dense(q, k, v, D**-0.5)), np.array(blocked(q, k, v, D**-0.5))
    valid = np.ones((B, length), bool) if am is None else np.array(am).astype(bool)
    assert np.abs(a - b)[valid].max() < 1e-5


@pytest.mark.parametrize("name", ["ettin-encoder-17m", "ettin-decoder-17m"])
@pytest.mark.parametrize("length", [600, 2085])
def test_blocked_equals_dense_on_checkpoint(name, length):
    """End to end the two kernels differ only by float summation order (exactness of the
    window itself is pinned by the kernel-level and receptive-field tests above)."""
    rng = np.random.default_rng(length)
    ids = mx.array(rng.integers(0, 50000, (2, length)))
    am = mx.array(np.array([[1] * length, [1] * (length - 333) + [0] * 333], np.int32))
    outs = []
    for impl in ["dense", "blocked"]:
        m = model(name, local_attention=impl)
        outs.append(np.array(m(ids, am)))
    valid = np.array(am).astype(bool)
    assert np.abs(outs[0] - outs[1])[valid].max() < 1e-3
    assert np.array_equal(outs[0].argmax(-1)[valid], outs[1].argmax(-1)[valid])
