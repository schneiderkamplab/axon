"""Metal KV-cache decode kernel tests.

The fused Metal kernel (torch.mps.compile_shader, mps-only) must reproduce
the composed decode path — :func:`decode_paged_sdpa` — at the cache dtype,
and every unsupported configuration must raise loudly (no silent fallbacks).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from synapse.axon.codegen2_torch.paged_attention import (
    PagedAttention,
    decode_paged_sdpa,
    metal_decode_paged_sdpa,
    paged_attention,
)
from synapse.axon.codegen2_torch.serving_context import reset_context, set_context

HQ, HKV, D = 16, 8, 128  # qwen3 head geometry (GQA ratio 2)
BS = 256
SCALE = D**-0.5

_requires_mps = pytest.mark.skipif(
    not getattr(torch.mps, "is_available", lambda: False)(),
    reason="Metal decode kernel requires an mps device",
)


def _case(batch, max_blocks, lens, seed, dtype, device="mps"):
    import numpy as np

    rng = np.random.default_rng(seed)
    to_dev = lambda a: torch.from_numpy(a).to(device, dtype)  # noqa: E731
    k_pool = to_dev(rng.standard_normal((64, BS, HKV, D)).astype(np.float32))
    v_pool = to_dev(rng.standard_normal((64, BS, HKV, D)).astype(np.float32))
    q = to_dev(rng.standard_normal((batch, HQ, D)).astype(np.float32))
    blocks = rng.choice(64, size=(batch, max_blocks), replace=False)
    tables = torch.full((batch, max_blocks), -1, dtype=torch.int64, device=device)
    tables.copy_(torch.from_numpy(blocks.astype(np.int64)).to(device))
    lens_t = torch.tensor(lens, dtype=torch.int64, device=device)
    return k_pool, v_pool, q, tables, lens_t


@_requires_mps
@pytest.mark.parametrize("dtype,tol", [(torch.bfloat16, 2e-2), (torch.float32, 1e-4)])
def test_metal_matches_composed_decode(dtype, tol):
    for batch, max_blocks, lens, seed in (
        (4, 1, [44, 50, 55, 56], 0),
        (8, 1, [56] * 8, 1),
        (8, 2, [300, 400, 500, 512, 256, 350, 450, 510], 2),
    ):
        k_pool, v_pool, q, tables, lens_t = _case(batch, max_blocks, lens, seed, dtype)
        ref = decode_paged_sdpa(
            q, k_pool, v_pool,
            num_heads=HQ, num_kv_heads=HKV, head_dim=D, scale=SCALE,
            context_lens=lens_t, block_tables=tables, context_lens_max=max(lens),
        )
        got = metal_decode_paged_sdpa(
            q, k_pool, v_pool,
            num_heads=HQ, num_kv_heads=HKV, head_dim=D, scale=SCALE,
            context_lens=lens_t, block_tables=tables, context_lens_max=max(lens),
        )
        assert got.shape == (batch, HQ * D)
        assert got.dtype == dtype
        diff = (got.float() - ref.float()).abs().max().item()
        assert diff < tol, f"batch={batch} max_blocks={max_blocks}: diff {diff}"


@_requires_mps
def test_paged_attention_module_routes_to_metal():
    """The engine path: PagedAttention.forward with accelerator=metal must
    match accelerator=torch through the same store + decode step."""
    k_pool, v_pool, q, tables, lens_t = _case(4, 1, [44, 50, 55, 56], 3, torch.bfloat16)
    k4 = torch.randn(4, HKV, 1, D, device="mps").to(torch.bfloat16)
    v4 = torch.randn(4, HKV, 1, D, device="mps").to(torch.bfloat16)
    q4 = q.unsqueeze(2)  # [B, H, 1, D]
    outs = {}
    for accelerator in ("torch", "metal"):
        module = PagedAttention(HQ, HKV, D, accelerator=accelerator)
        module.k_cache = k_pool.unsqueeze(0)  # one-layer stack
        module.v_cache = v_pool.unsqueeze(0)
        set_context(
            False,
            slot_mapping=lens_t,
            context_lens=lens_t,
            block_tables=tables,
            context_lens_max=int(lens_t.max()),
        )
        try:
            outs[accelerator] = module(q4, k4, v4, 0)
        finally:
            reset_context()
    diff = (outs["torch"].float() - outs["metal"].float()).abs().max().item()
    assert diff < 2e-2
    assert outs["metal"].shape == q4.shape


@_requires_mps
def test_paged_attention_accelerator_argument_reaches_decode():
    """accelerator routed through paged_attention() itself (not just the module)."""
    k_pool, v_pool, q, tables, lens_t = _case(2, 1, [40, 48], 4, torch.float32)
    k2 = torch.randn(2, HKV, D, device="mps")   # head-split rows [N, H_kv, D]
    v2 = torch.randn(2, HKV, D, device="mps")
    outs = {}
    for accelerator in ("torch", "metal"):
        set_context(
            False,
            slot_mapping=lens_t,
            context_lens=lens_t,
            block_tables=tables,
            context_lens_max=max(lens_t),
        )
        try:
            outs[accelerator] = paged_attention(
                q, k2, v2, k_pool, v_pool,
                num_heads=HQ, num_kv_heads=HKV, head_dim=D,
                causal=True, accelerator=accelerator,
            )
        finally:
            reset_context()
    assert (outs["torch"].float() - outs["metal"].float()).abs().max().item() < 1e-4


def test_unknown_accelerator_raises():
    with pytest.raises(ValueError, match="unknown PagedAttention accelerator"):
        PagedAttention(HQ, HKV, D, accelerator="cuda")
    with pytest.raises(ValueError, match="unknown paged-attention accelerator"):
        paged_attention(
            torch.zeros(1, HQ, D), torch.zeros(1, HKV, D), torch.zeros(1, HKV, D),
            torch.zeros(2, BS, HKV, D), torch.zeros(2, BS, HKV, D),
            num_heads=HQ, num_kv_heads=HKV, head_dim=D, accelerator="mteal",
        )


@_requires_mps
def test_metal_off_contract_raises():
    """Loud raises for every unsupported configuration (no silent fallbacks)."""
    k_mps, v_mps, q_mps, tables, lens_t = _case(2, 1, [40, 48], 5, torch.bfloat16)
    # non-mps device
    with pytest.raises(ValueError, match="requires mps tensors"):
        metal_decode_paged_sdpa(
            q_mps.cpu(), k_mps.cpu(), v_mps.cpu(),
            num_heads=HQ, num_kv_heads=HKV, head_dim=D, scale=SCALE,
            context_lens=lens_t.cpu(), block_tables=tables.cpu(), context_lens_max=48,
        )
    # unsupported cache dtype (fp16: supported by MPS, not by the kernel)
    with pytest.raises(ValueError, match="bfloat16/float32"):
        metal_decode_paged_sdpa(
            q_mps.half(), k_mps.half(), v_mps.half(),
            num_heads=HQ, num_kv_heads=HKV, head_dim=D, scale=SCALE,
            context_lens=lens_t, block_tables=tables, context_lens_max=48,
        )
    # q/pool dtype mismatch
    with pytest.raises(ValueError, match="pool dtype"):
        metal_decode_paged_sdpa(
            q_mps.float(), k_mps, v_mps,
            num_heads=HQ, num_kv_heads=HKV, head_dim=D, scale=SCALE,
            context_lens=lens_t, block_tables=tables, context_lens_max=48,
        )
    # head_dim not a multiple of 64
    with pytest.raises(ValueError, match="head_dim divisible by 64"):
        metal_decode_paged_sdpa(
            q_mps, k_mps, v_mps,
            num_heads=HQ, num_kv_heads=HKV, head_dim=48, scale=SCALE,
            context_lens=lens_t, block_tables=tables, context_lens_max=48,
        )
    # non-power-of-two block size
    odd_pool = torch.zeros(8, 192, HKV, D, device="mps", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="power-of-two"):
        metal_decode_paged_sdpa(
            q_mps, odd_pool, odd_pool.clone(),
            num_heads=HQ, num_kv_heads=HKV, head_dim=D, scale=SCALE,
            context_lens=lens_t, block_tables=tables, context_lens_max=48,
        )
    # missing block tables
    with pytest.raises(ValueError, match="requires block_tables"):
        metal_decode_paged_sdpa(
            q_mps, k_mps, v_mps,
            num_heads=HQ, num_kv_heads=HKV, head_dim=D, scale=SCALE,
            context_lens=lens_t, block_tables=None, context_lens_max=48,
        )
    # non-int64 metadata
    with pytest.raises(ValueError, match="int64"):
        metal_decode_paged_sdpa(
            q_mps, k_mps, v_mps,
            num_heads=HQ, num_kv_heads=HKV, head_dim=D, scale=SCALE,
            context_lens=lens_t.int(), block_tables=tables.int(), context_lens_max=48,
        )
    # tables too narrow for context_lens_max
    with pytest.raises(ValueError, match="cannot cover context_lens_max"):
        metal_decode_paged_sdpa(
            q_mps, k_mps, v_mps,
            num_heads=HQ, num_kv_heads=HKV, head_dim=D, scale=SCALE,
            context_lens=lens_t, block_tables=tables, context_lens_max=600,
        )


def test_fleet_config_rejects_unknown_kv_kernel():
    from server.fleet.config import Config

    with pytest.raises(AssertionError):
        Config(model_config={}, kv_kernel="tensorrt")
