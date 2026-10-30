"""Test 5: KV-cache consistency. Prefill + one-token decode steps give the logits of a single
full forward over the same tokens, at every step, well past the sliding window."""

import mlx.core as mx
import numpy as np
import pytest

from ettin_mlx.cache import KVCache, SlidingKVCache
from helpers import meta, model, models, reference

TOTAL = 400  # > 6 sliding windows, > one KVCache growth step


def tolerance(name: str) -> float:
    """The cached and full paths use different SDPA kernels (single-query vs full), so they
    round differently. Allow up to the model's own fp32 noise on this text (HF fp32 vs HF
    float64 logits on the 2048-token case), floor 1e-4. A window or offset bug shows up as
    O(1) errors past position 64, and argmax must match at every step regardless."""
    return max(1e-4, meta(name)["fp32_vs_fp64"]["long"])


@pytest.mark.parametrize("name", models("decoder"))
@pytest.mark.parametrize("prefill", [1, 63, 64, 65, 150])
@pytest.mark.parametrize("impl", ["auto", "blocked"])
def test_prefill_then_decode_matches_full_forward(name, prefill, impl, report):
    m = model(name, local_attention=impl)
    ids = mx.array(reference(name)["long/input_ids"][:, :TOTAL])
    full = np.array(m(ids))

    cache = m.make_cache()
    steps = [np.array(m(ids[:, :prefill], cache=cache))]
    for t in range(prefill, TOTAL):
        steps.append(np.array(m(ids[:, t : t + 1], cache=cache)))
    cached = np.concatenate(steps, axis=1)

    diff = float(np.abs(cached - full).max())
    report.add(name, f"kv_cache/prefill{prefill}/{impl}", {"max_abs": diff, "steps": TOTAL - prefill})
    assert np.array_equal(cached.argmax(-1), full.argmax(-1))
    assert diff < tolerance(name)
    w = m.config.sliding_window
    for c in cache:
        assert c.offset == TOTAL
        if isinstance(c, SlidingKVCache):
            assert c.keys.shape[2] == w - 1
        else:
            assert isinstance(c, KVCache) and c.keys.shape[2] >= TOTAL


@pytest.mark.parametrize("name", models("decoder"))
def test_chunked_prefill_matches_full_forward(name):
    """Multi-token steps on top of a filled cache (sliding mask with kv offset)."""
    m = model(name)
    ids = mx.array(reference(name)["long/input_ids"][:, :TOTAL])
    full = np.array(m(ids))
    cache = m.make_cache()
    bounds = [0, 70, 75, 200, 201, 333, TOTAL]
    chunked = np.concatenate([np.array(m(ids[:, a:b], cache=cache)) for a, b in zip(bounds, bounds[1:])], axis=1)
    assert np.array_equal(chunked.argmax(-1), full.argmax(-1))
    assert np.abs(chunked - full).max() < tolerance(name)


@pytest.mark.parametrize("name", models("decoder"))
def test_left_padded_batch_decode_matches_single_rows(name):
    """Batched decode with a left-padding mask equals each row decoded alone (RoPE sees
    each row shifted by its pad count, which leaves relative positions unchanged)."""
    m = model(name)
    long_ids = reference(name)["long/input_ids"][0]
    prompts = [long_ids[:30], long_ids[100:180]]
    width, new = 80, 120
    ids = np.stack([np.concatenate([np.full(width - len(p), 50283), p]) for p in prompts]).astype(np.int32)
    mask = (np.arange(width)[None] >= width - np.array([len(p) for p in prompts])[:, None]).astype(np.int32)
    cont = long_ids[200 : 200 + new]

    cache = m.make_cache()
    am = mx.array(mask)
    batched = [np.array(m(mx.array(ids), am, cache=cache))[:, -1]]
    for t in range(new):
        am = mx.concatenate([am, mx.ones((2, 1), am.dtype)], axis=1)
        batched.append(np.array(m(mx.array(np.stack([cont[t : t + 1]] * 2)), am, cache=cache))[:, -1])
    batched = np.stack(batched, axis=1)

    for r, p in enumerate(prompts):
        alone = np.array(m(mx.array(np.concatenate([p, cont])[None])))[0, len(p) - 1 :]
        assert np.abs(batched[r] - alone).max() < 1e-3
        assert np.array_equal(batched[r].argmax(-1), alone.argmax(-1))
