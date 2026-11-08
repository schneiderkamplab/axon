"""Test 2: padding invariance. Each row of a right-padded batch matches the same sequence
run alone, on its non-pad positions."""

import mlx.core as mx
import numpy as np
import pytest

from helpers import meta, model, models, reference

ATOL = 1e-3
SPREAD_FACTOR = 3.0  # same rule as test_parity for ill-conditioned inputs


def forward(m, ids, mask=None):
    return np.array(m(mx.array(ids), None if mask is None else mx.array(mask)))


def batches(name):
    """The reference padded batch, plus a long one that runs the blocked sliding kernel."""
    ref = reference(name)
    yield "padded", ref["padded/input_ids"], ref["padded/attention_mask"]
    long_ids = ref["long/input_ids"][0]
    lens = [2048, 1000, 600]
    ids = np.stack([np.concatenate([long_ids[:n], np.full(2048 - n, 50283)]) for n in lens]).astype(np.int32)
    mask = (np.arange(2048)[None] < np.array(lens)[:, None]).astype(np.int32)
    yield "long_padded", ids, mask


@pytest.mark.parametrize("name", models())
def test_pad_content_cannot_leak(name):
    """Same batch shape, different garbage in the pad slots: real positions must be
    bit-identical (masked keys get exactly zero weight)."""
    m = model(name)
    rng = np.random.default_rng(0)
    for _, ids, mask in batches(name):
        other = np.where(mask == 1, ids, rng.integers(0, 50000, ids.shape)).astype(np.int32)
        a, b = forward(m, ids, mask), forward(m, other, mask)
        valid = mask.astype(bool)
        assert np.array_equal(a[valid], b[valid])


@pytest.mark.parametrize("name", models())
def test_rows_match_unpadded_runs(name, report):
    m = model(name)
    for label, ids, mask in batches(name):
        batched = forward(m, ids, mask)
        worst, top1 = 0.0, 1.0
        for r, n in enumerate(mask.sum(1)):
            alone = forward(m, ids[r : r + 1, :n])[0]
            worst = max(worst, float(np.abs(batched[r, :n] - alone).max()))
            top1 = min(top1, float((batched[r, :n].argmax(-1) == alone.argmax(-1)).mean()))
        report.add(name, f"padding_invariance/{label}", {"max_abs": worst, "top1_agreement": top1})
        assert top1 == 1.0
        if worst >= ATOL:  # see test_parity: only acceptable where fp32 itself is that noisy
            case = "padded" if label == "padded" else "long"
            assert meta(name)["fp32_vs_fp64"][case] >= ATOL / 2, (label, worst)
            assert worst <= SPREAD_FACTOR * meta(name)["fp32_spread"][case], (label, worst)
