"""Tests 1, 3 (HF parity at window edges), 4 and 7: logits vs HF transformers FP32 CPU eager."""

import mlx.core as mx
import numpy as np
import pytest

from helpers import CASES, compare, max_abs, meta, model, models, reference, reference_logits

ATOL = 1e-3
# For ill-conditioned inputs (HF's own fp32 error >= ATOL / 2): allowed distance from the
# float64 truth, as a multiple of the worst error among HF's fp32 variants. Measured
# ratios on the default models are <= 2.1 (see README "Parity").
SPREAD_FACTOR = 3.0


def run(name: str, case: str, dtype: str = "float32") -> tuple[np.ndarray, np.ndarray]:
    ref = reference(name)
    ids, mask = ref[f"{case}/input_ids"], ref[f"{case}/attention_mask"]
    m = model(name, dtype)
    # Pass the mask only for padded batches, as a user would.
    attn = mx.array(mask) if not mask.all() else None
    if m.config.is_decoder:
        hidden = m.hidden_states(mx.array(ids), attn)
    else:
        hidden = m.encode(mx.array(ids), attn)
    logits = m.logits(hidden)
    return np.array(logits.astype(mx.float32)), np.array(hidden.astype(mx.float32))


@pytest.mark.parametrize("name", models())
@pytest.mark.parametrize("case", CASES)
def test_fp32_logits_match_hf(name, case, report):
    """Tests 1 (encoders) and 4 (decoders): 100% top-1 on every real position and
    max abs diff < 1e-3 vs HF fp32.

    Some inputs carry massive activations (|h| ~ 2e3-5e3 after the MLPs of a few
    layers) that make the forward pass ill-conditioned in fp32: HF's own eager logits
    then sit more than 1e-3 from the exact (float64) values and HF's sdpa backend
    differs from HF eager by up to 1e-2. No fp32 implementation can match HF's
    rounding to 1e-3 there, so when HF's error is >= ATOL / 2 the bar becomes: within
    SPREAD_FACTOR x the worst float64 distance of HF's fp32 variants (eager, sdpa,
    ulp-noise runs; ``meta.fp32_spread``).
    """
    ref = reference(name)
    mask = ref[f"{case}/attention_mask"]
    logits, hidden = run(name, case)
    stats = compare(logits, reference_logits(name, case), ref[f"{case}/top1"], mask)

    pos = ref[f"{case}/sample_pos"]
    stats["sampled_hf_logits_max_abs"] = float(np.abs(logits[pos[:, 0], pos[:, 1]] - ref[f"{case}/sample_logits"]).max())
    stats["hidden_max_abs"] = float(np.abs(hidden - ref[f"{case}/hidden"])[mask.astype(bool)].max())
    stats["max_abs_vs_fp64"] = max_abs(logits, reference_logits(name, case, fp64=True), mask)
    stats["hf_fp32_vs_fp64"] = hf_error = meta(name)["fp32_vs_fp64"][case]
    stats["hf_fp32_spread_vs_fp64"] = spread = meta(name)["fp32_spread"][case]
    stats["fp64_error_ratio"] = stats["max_abs_vs_fp64"] / spread
    report.add(name, f"fp32/{case}", stats)

    assert stats["top1_agreement"] == 1.0, stats
    # The rebuilt full logits agree with HF's verbatim logits where both exist.
    assert stats["sampled_hf_logits_max_abs"] <= stats["max_abs"] + 1e-4, stats
    if stats["max_abs"] >= ATOL:
        assert hf_error >= ATOL / 2, f"miss not explained by fp32 conditioning: {stats}"
        assert stats["fp64_error_ratio"] <= SPREAD_FACTOR, stats


@pytest.mark.parametrize("name", models())
def test_window_edges_match_hf(name, report):
    """Test 3: on a 2048-token sequence (> 4 x local_attention), parity specifically at
    positions next to window edges: sequence ends (truncated windows) and every
    multiple of the half-window +- 1 (the blocked kernel's block seams)."""
    ref = reference(name)
    m = model(name)
    w = m.config.sliding_window
    length = ref["long/input_ids"].shape[1]
    assert length > 4 * m.config.local_attention
    edges = {p for k in range(0, length, w) for p in (k - 1, k, k + 1)}
    edges |= set(range(w + 2)) | set(range(length - w - 2, length))
    edges = np.array(sorted(p for p in edges if 0 <= p < length))

    logits, _ = run(name, "long")
    ours = logits[0, edges]
    stats = {
        "max_abs": float(np.abs(ours - reference_logits(name, "long")[0, edges]).max()),
        "max_abs_vs_fp64": float(np.abs(ours - reference_logits(name, "long", fp64=True)[0, edges]).max()),
        "top1_agreement": float((ours.argmax(-1) == ref["long/top1"][0, edges]).mean()),
        "positions": len(edges),
    }
    report.add(name, "fp32/window_edges", stats)
    assert stats["top1_agreement"] == 1.0
    if stats["max_abs"] >= ATOL:
        assert meta(name)["fp32_vs_fp64"]["long"] >= ATOL / 2, stats
        assert stats["max_abs_vs_fp64"] <= SPREAD_FACTOR * meta(name)["fp32_spread"]["long"], stats


@pytest.mark.parametrize("name", models())
@pytest.mark.parametrize("case", CASES)
def test_bf16_vs_fp32_reference(name, case, report):
    """Test 7: BF16 compute against the FP32 HF reference. No threshold; numbers go to the report."""
    ref = reference(name)
    logits, _ = run(name, case, dtype="bfloat16")
    stats = compare(logits, reference_logits(name, case), ref[f"{case}/top1"], ref[f"{case}/attention_mask"])
    report.add(name, f"bf16/{case}", stats)
    assert np.isfinite(logits).all()
