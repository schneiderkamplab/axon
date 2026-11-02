"""Fleet sampler tests: semantics pinned to the Session sampler.

The vendored fleet sampler is the source of truth for fleet decoding. These
tests pin its greedy / top-k / top-p semantics to the Session sampler's numpy
chain (``server.sampling``) on identical logits, so the two engines agree by
construction wherever their surfaces overlap (the OpenAI API is served by
both).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

from server.fleet.layers.sampler import Sampler
from server.fleet.layers.sampler import _apply_top_k as fleet_top_k
from server.fleet.layers.sampler import _apply_top_p as fleet_top_p
from server.sampling import _apply_top_k as session_top_k
from server.sampling import _apply_top_p as session_top_p

V = 64


def _survivor_indices(masked) -> set[int]:
    """The surviving (finite-logit) token ids of one masked logits row."""
    return {int(i) for i, v in enumerate(masked.tolist()) if np.isfinite(v)}


def test_greedy_rows_are_exact_argmax() -> None:
    gen = torch.Generator().manual_seed(7)
    logits = torch.randn(4, V, generator=gen)
    temperatures = torch.tensor([0.0, 0.0, 0.0, 0.0])
    assert Sampler().forward(logits, temperatures).tolist() == logits.argmax(dim=-1).tolist()


def test_mixed_batch_greedy_rows_bypass_the_noise() -> None:
    gen = torch.Generator().manual_seed(11)
    logits = torch.randn(6, V, generator=gen)
    temperatures = torch.tensor([0.0, 0.7, 0.0, 0.7, 0.0, 0.7])
    out = Sampler().forward(logits, temperatures).tolist()
    for row in (0, 2, 4):
        assert out[row] == int(logits[row].argmax()), f"greedy row {row} sampled instead of argmax"


def test_top_k_masking_matches_session_sampler() -> None:
    gen = torch.Generator().manual_seed(21)
    logits = torch.randn(5, V, generator=gen)
    top_ks = torch.tensor([3, 0, V, 7, 1], dtype=torch.int64)
    # Compare in float64: the Session oracle computes in float64, and a
    # float32 cumsum can straddle a threshold differently on boundary tokens.
    fleet = fleet_top_k(logits.double().clone(), top_ks)
    for row, k in enumerate(top_ks.tolist()):
        session = session_top_k(logits[row].double().numpy(), int(k))
        assert _survivor_indices(fleet[row]) == _survivor_indices(torch.from_numpy(session)), (
            f"top_k={k} survivor sets differ"
        )


def test_top_p_masking_matches_session_sampler() -> None:
    gen = torch.Generator().manual_seed(31)
    logits = torch.randn(5, V, generator=gen)
    top_ps = torch.tensor([0.3, 0.9, 1.0, 0.5, 0.99])
    # float64 for the same reason as in the top_k test.
    fleet = fleet_top_p(logits.double().clone(), top_ps)
    for row, p in enumerate(top_ps.tolist()):
        session = session_top_p(logits[row].double().numpy(), float(p))
        assert _survivor_indices(fleet[row]) == _survivor_indices(torch.from_numpy(session)), (
            f"top_p={p} survivor sets differ"
        )


def test_sampled_tokens_stay_within_filtered_support() -> None:
    gen = torch.Generator().manual_seed(41)
    logits = torch.randn(1, V, generator=gen)
    temperatures = torch.tensor([0.8])
    top_ks = torch.tensor([5], dtype=torch.int64)
    top_ps = torch.tensor([0.9])
    support = {
        int(i)
        for i, v in enumerate(
            fleet_top_p(fleet_top_k(logits.clone(), top_ks), top_ps)[0].tolist()
        )
        if np.isfinite(v)
    }
    # The support set must be non-trivial for this assertion to mean anything.
    assert 1 < len(support) <= 5
    sampler = Sampler()
    for _ in range(200):
        assert int(sampler(logits, temperatures, top_ks, top_ps)[0]) in support
