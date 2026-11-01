"""Paged attention runtime tests.

Oracle: plain per-sequence SDPA over exact (contiguous) K/V histories —
the same math the paged path must reproduce when the cache lives in a block
pool addressed through slot mappings and block tables.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
F = torch.nn.functional

from synapse.axon.codegen2_torch.paged_attention import (
    decode_paged_sdpa,
    paged_attention,
    store_kvcache_paged,
    varlen_paged_sdpa,
)
from synapse.axon.codegen2_torch.serving_context import reset_context, set_context

BS = 4  # block size
H, H_KV, D = 4, 2, 8


def _ref_seq(q, k, v, past=0):
    """Causal SDPA for one sequence; ``past`` = cached prefix length."""
    length = q.shape[0]
    qh = q.transpose(0, 1).unsqueeze(0)  # [1, H, L, D]
    kh = k.transpose(0, 1).unsqueeze(0)  # [1, H_kv, L_k, D]
    vh = v.transpose(0, 1).unsqueeze(0)
    q_pos = torch.arange(past, past + length).unsqueeze(1)
    k_pos = torch.arange(k.shape[0]).unsqueeze(0)
    causal = torch.where(q_pos >= k_pos, 0.0, float("-inf"))
    out = F.scaled_dot_product_attention(
        qh, kh, vh, attn_mask=causal.unsqueeze(0).unsqueeze(0),
        enable_gqa=H != H_KV,
    )
    return out.squeeze(0).transpose(0, 1).reshape(length, H * D)


def _ref_decode_rows(q, ks, vs):
    """One fresh query per sequence attending its full history."""
    outs = []
    for i, (k, v) in enumerate(zip(ks, vs)):
        qh = q[i : i + 1].transpose(0, 1).unsqueeze(0)  # [1, H, 1, D]
        kh = k.transpose(0, 1).unsqueeze(0)  # [1, H_kv, L, D]
        vh = v.transpose(0, 1).unsqueeze(0)
        out = F.scaled_dot_product_attention(qh, kh, vh, enable_gqa=H != H_KV)
        outs.append(out.reshape(H * D))
    return torch.stack(outs)


class PagedCase:
    """Builds a pool + block tables for a batch of sequence lengths."""

    def __init__(self, lengths, blocks=32):
        self.lengths = list(lengths)
        self.k_cache = torch.zeros(blocks, BS, H_KV, D)
        self.v_cache = torch.zeros(blocks, BS, H_KV, D)
        next_block = 0
        self.tables = []
        for length in self.lengths:
            n_blocks = (length + BS - 1) // BS
            self.tables.append(list(range(next_block, next_block + n_blocks)))
            next_block += n_blocks

    def slots(self, seq, positions):
        table = self.tables[seq]
        return [table[p // BS] * BS + p % BS for p in positions]

    def pad_tables(self):
        width = max(len(t) for t in self.tables)
        return [t + [-1] * (width - len(t)) for t in self.tables]


def test_store_scatters_rows_into_pool_slots() -> None:
    k_cache = torch.zeros(4, BS, H_KV, D)
    v_cache = torch.zeros(4, BS, H_KV, D)
    key = torch.arange(3 * H_KV * D, dtype=torch.float32).reshape(3, H_KV, D)
    value = torch.arange(3 * H_KV * D, dtype=torch.float32).reshape(3, H_KV, D) + 100
    slots = torch.tensor([0, 5, 9])  # blocks 0, 1, 2 — one row each
    store_kvcache_paged(k_cache, v_cache, key, value, slots)
    assert torch.equal(k_cache.reshape(-1, H_KV, D)[0], key[0])
    assert torch.equal(k_cache.reshape(-1, H_KV, D)[5], key[1])
    assert torch.equal(v_cache.reshape(-1, H_KV, D)[9], value[2])
    assert bool((k_cache.reshape(-1, H_KV, D)[1] == 0).all())  # untouched slot


def test_prefill_varlen_matches_contiguous_reference() -> None:
    lengths = [3, 5, 2]
    case = PagedCase(lengths)
    gen = torch.Generator().manual_seed(7)
    qs, ks, vs = [], [], []
    for length in lengths:
        qs.append(torch.randn(length, H, D, generator=gen))
        ks.append(torch.randn(length, H_KV, D, generator=gen))
        vs.append(torch.randn(length, H_KV, D, generator=gen))
    q = torch.cat(qs)
    k = torch.cat(ks)
    v = torch.cat(vs)

    slot_rows = []
    for seq, length in enumerate(lengths):
        slot_rows.extend(case.slots(seq, range(length)))
    cu = [0]
    for length in lengths:
        cu.append(cu[-1] + length)
    set_context(
        True,
        cu_seqlens_q=torch.tensor(cu),
        cu_seqlens_k=torch.tensor(cu),
        slot_mapping=torch.tensor(slot_rows),
        block_tables=torch.tensor(case.pad_tables()),
    )
    try:
        out = paged_attention(q, k, v, case.k_cache, case.v_cache,
                              num_heads=H, num_kv_heads=H_KV, head_dim=D)
    finally:
        reset_context()
    expected = torch.cat([_ref_seq(qs[i], ks[i], vs[i]) for i in range(len(lengths))])
    assert torch.allclose(out, expected, atol=1e-5), (out - expected).abs().max()


def test_decode_matches_reference_after_prefill() -> None:
    lengths = [3, 5, 2]
    case = PagedCase(lengths)
    gen = torch.Generator().manual_seed(11)
    qs, ks, vs = [], [], []
    for length in lengths:
        qs.append(torch.randn(length, H, D, generator=gen))
        ks.append(torch.randn(length, H_KV, D, generator=gen))
        vs.append(torch.randn(length, H_KV, D, generator=gen))
    slot_rows = []
    for seq, length in enumerate(lengths):
        slot_rows.extend(case.slots(seq, range(length)))
    cu = [0]
    for length in lengths:
        cu.append(cu[-1] + length)
    set_context(
        True,
        cu_seqlens_q=torch.tensor(cu),
        cu_seqlens_k=torch.tensor(cu),
        slot_mapping=torch.tensor(slot_rows),
        block_tables=torch.tensor(case.pad_tables()),
    )
    try:
        paged_attention(
            torch.cat(qs), torch.cat(ks), torch.cat(vs),
            case.k_cache, case.v_cache, num_heads=H, num_kv_heads=H_KV, head_dim=D,
        )
    finally:
        reset_context()

    # Decode step: one fresh query per sequence. context_lens counts the
    # just-stored token (fleet runner semantics).
    q_dec = torch.randn(len(lengths), H, D, generator=gen)
    k_dec = torch.randn(len(lengths), H_KV, D, generator=gen)
    v_dec = torch.randn(len(lengths), H_KV, D, generator=gen)
    set_context(
        False,
        context_lens=torch.tensor([l + 1 for l in lengths]),
        slot_mapping=torch.tensor([case.slots(s, [l])[0] for s, l in enumerate(lengths)]),
        block_tables=torch.tensor(case.pad_tables()),
    )
    try:
        out = paged_attention(q_dec, k_dec, v_dec, case.k_cache, case.v_cache,
                              num_heads=H, num_kv_heads=H_KV, head_dim=D)
    finally:
        reset_context()
    k_full = [torch.cat([ks[i], k_dec[i : i + 1]]) for i in range(len(lengths))]
    v_full = [torch.cat([vs[i], v_dec[i : i + 1]]) for i in range(len(lengths))]
    expected = _ref_decode_rows(q_dec, k_full, v_full)
    assert torch.allclose(out, expected, atol=1e-5), (out - expected).abs().max()


def test_prefill_with_prefix_cache_matches_reference() -> None:
    # Sequence 0 fully prefills (6 tokens). Sequence 1 shares its first block
    # (4 tokens) and appends 2 own tokens: only the tail is scheduled.
    shared = 4
    len0, len1_new = 6, 2
    case = PagedCase([len0, shared + len1_new])
    # Sequence 1 shares sequence 0's first block: its table reuses block 0.
    case.tables[1] = [case.tables[0][0], case.tables[1][1]]
    gen = torch.Generator().manual_seed(23)
    k0 = torch.randn(len0, H_KV, D, generator=gen)
    v0 = torch.randn(len0, H_KV, D, generator=gen)
    q0 = torch.randn(len0, H, D, generator=gen)
    # Sequence 1's prefix equals sequence 0's first block; its tail is new.
    k1_tail = torch.randn(len1_new, H_KV, D, generator=gen)
    v1_tail = torch.randn(len1_new, H_KV, D, generator=gen)
    q1_tail = torch.randn(len1_new, H, D, generator=gen)

    # Prefill sequence 0.
    set_context(
        True,
        cu_seqlens_q=torch.tensor([0, len0]),
        cu_seqlens_k=torch.tensor([0, len0]),
        slot_mapping=torch.tensor(case.slots(0, range(len0))),
        block_tables=torch.tensor(case.pad_tables()),
    )
    try:
        paged_attention(q0, k0, v0, case.k_cache, case.v_cache,
                        num_heads=H, num_kv_heads=H_KV, head_dim=D)
    finally:
        reset_context()

    # Prefill sequence 1: block 0 shared, tail stored at positions 4..5.
    # block_tables has one row per SCHEDULED sequence (fleet semantics).
    set_context(
        True,
        cu_seqlens_q=torch.tensor([0, len1_new]),
        cu_seqlens_k=torch.tensor([0, shared + len1_new]),
        slot_mapping=torch.tensor(case.slots(1, range(shared, shared + len1_new))),
        block_tables=torch.tensor([case.tables[1]]),
    )
    try:
        out = paged_attention(q1_tail, k1_tail, v1_tail, case.k_cache, case.v_cache,
                              num_heads=H, num_kv_heads=H_KV, head_dim=D)
    finally:
        reset_context()
    k1_full = torch.cat([k0[:shared], k1_tail])
    v1_full = torch.cat([v0[:shared], v1_tail])
    expected = _ref_seq(q1_tail, k1_full, v1_full, past=shared)  # queries at abs pos 4..5
    assert torch.allclose(out, expected, atol=1e-5), (out - expected).abs().max()


def test_decode_contiguous_single_stream_matches_reference() -> None:
    # Without block tables the cache is addressed by absolute position —
    # the single-stream contiguous artifact path.
    max_len = 7
    k_cache = torch.zeros(2, BS, H_KV, D)
    v_cache = torch.zeros(2, BS, H_KV, D)
    gen = torch.Generator().manual_seed(31)
    k_hist = torch.randn(max_len, H_KV, D, generator=gen)
    v_hist = torch.randn(max_len, H_KV, D, generator=gen)
    store_kvcache_paged(k_cache, v_cache, k_hist, v_hist,
                        torch.arange(max_len))
    q = torch.randn(1, H, D, generator=gen)
    k_new = torch.randn(1, H_KV, D, generator=gen)
    v_new = torch.randn(1, H_KV, D, generator=gen)
    set_context(
        False,
        context_lens=torch.tensor([max_len + 1]),
        slot_mapping=torch.tensor([max_len]),
        block_tables=None,
    )
    try:
        out = paged_attention(q, k_new, v_new, k_cache, v_cache,
                              num_heads=H, num_kv_heads=H_KV, head_dim=D)
    finally:
        reset_context()
    expected = _ref_decode_rows(
        q, [torch.cat([k_hist, k_new])], [torch.cat([v_hist, v_new])]
    )
    assert torch.allclose(out, expected, atol=1e-5), (out - expected).abs().max()


def test_prefill_helpers_direct_call_matches_paged_attention() -> None:
    # varlen_paged_sdpa/decode_paged_sdpa are callable directly (the emitted
    # module may inline them); the dispatch wrapper must agree.
    lengths = [4, 2]
    case = PagedCase(lengths)
    gen = torch.Generator().manual_seed(41)
    q = torch.randn(sum(lengths), H, D, generator=gen)
    k = torch.randn(sum(lengths), H_KV, D, generator=gen)
    v = torch.randn(sum(lengths), H_KV, D, generator=gen)
    slot_rows = []
    for seq, length in enumerate(lengths):
        slot_rows.extend(case.slots(seq, range(length)))
    cu = [0]
    for length in lengths:
        cu.append(cu[-1] + length)
    set_context(
        True,
        cu_seqlens_q=torch.tensor(cu),
        cu_seqlens_k=torch.tensor(cu),
        slot_mapping=torch.tensor(slot_rows),
        block_tables=torch.tensor(case.pad_tables()),
    )
    try:
        via_wrapper = paged_attention(q, k, v, case.k_cache, case.v_cache,
                                      num_heads=H, num_kv_heads=H_KV, head_dim=D)
        direct = varlen_paged_sdpa(
            q, case.k_cache, case.v_cache,
            num_heads=H, num_kv_heads=H_KV, head_dim=D,
            cu_seqlens_q=torch.tensor(cu), cu_seqlens_k=torch.tensor(cu),
            block_tables=torch.tensor(case.pad_tables()),
        )
    finally:
        reset_context()
    assert torch.allclose(via_wrapper, direct, atol=1e-6)
