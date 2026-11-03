"""Paged attention runtime for codegen2-torch serving-profile artifacts.

This is the runtime the opt-in ``__torch_paged_attention`` graph intrinsic
lowers onto: generated serving-profile attention modules call
:func:`paged_attention`, which stores the step's K/V into the engine-owned
pool at ``get_context().slot_mapping`` and reads the cache back through block
tables (or plain positions when the artifact uses the default contiguous
addressing).

Semantics mirror nano-vllm's attention (https://github.com/GeeeekExplorer/
nano-vllm, Copyright (c) 2025 Xingkai Yu — MIT License):

- pool layout ``[num_blocks, block_size, num_kv_heads, head_dim]`` per layer,
  scheduler-owned; a layer's ``k_cache``/``v_cache`` are views into it,
- the store is a per-row scatter at ``slot_mapping`` (upstream uses a Triton
  kernel; the portable store is an indexed copy),
- prefill is varlen over the packed batch (upstream:
  ``flash_attn_varlen_func``; the portable path gathers the cache and builds
  the block-diagonal causal mask explicitly),
- decode gathers K/V via ``block_tables`` and masks by ``context_lens``
  (upstream: ``flash_attn_with_kvcache``),
- GQA is handled by SDPA's ``enable_gqa`` — upstream's
  ``Tensor.repeat`` KV expansion never happens here.

Numerics target: match a contiguous, per-sequence SDPA reference exactly at
the cache dtype (fp32 in tests; parity harness covers real checkpoints).

Scope note: v1 supports full causal masking only. Sliding-window or
block-sparse patterns stay on the Session-style contiguous path until the
mask construction moves into the paged read.
"""

from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional as F

from .serving_context import get_context

__all__ = [
    "PagedAttention",
    "paged_attention",
    "store_kvcache_paged",
    "varlen_paged_sdpa",
    "decode_paged_sdpa",
    "metal_decode_paged_sdpa",
]

_NEG_INF = float("-inf")


def store_kvcache_paged(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    """Scatter this step's K/V rows into the pool at ``slot_mapping``.

    ``k_cache``/``v_cache`` are a layer's pool views with shape
    ``[num_blocks, block_size, num_kv_heads, head_dim]`` (any leading batch
    axes are flattened away); ``key``/``value`` are ``[N, num_kv_heads,
    head_dim]`` rows in the packed batch order; ``slot_mapping`` is ``[N]``
    global slots (``block * block_size + offset``).
    """
    if slot_mapping is None or slot_mapping.numel() == 0:
        return
    flat_k = k_cache.reshape(-1, k_cache.shape[-2], k_cache.shape[-1])
    flat_v = v_cache.reshape(-1, v_cache.shape[-2], v_cache.shape[-1])
    slots = slot_mapping.to(device=flat_k.device, dtype=torch.long)
    flat_k[slots] = key.to(dtype=flat_k.dtype)
    flat_v[slots] = value.to(dtype=flat_v.dtype)


def _gather_kv_via_block_tables(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather K/V for each row of ``block_tables``.

    ``block_tables`` is ``[B, max_blocks]`` padded with ``-1``. Returns K/V as
    ``[B, H_kv, max_blocks * block_size, head_dim]``; padded slots carry
    block-0 garbage and MUST be masked by the caller (context lens).
    """
    # pool: [num_blocks, block_size, H_kv, D] -> key into dim 0 by block id
    blocks = block_tables.clamp_min(0).to(device=k_cache.device, dtype=torch.long)
    k = k_cache[blocks]  # [B, max_blocks, block_size, H_kv, D]
    v = v_cache[blocks]
    max_blocks, block_size = blocks.shape[1], k_cache.shape[1]
    # flatten (block, slot) so position = block * block_size + slot
    k = k.reshape(blocks.shape[0], max_blocks * block_size, k_cache.shape[2], k_cache.shape[3])
    v = v.reshape(blocks.shape[0], max_blocks * block_size, v_cache.shape[2], v_cache.shape[3])
    return k, v


def varlen_paged_sdpa(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    *,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float | None = None,
    cu_seqlens_q: torch.Tensor | None = None,
    cu_seqlens_k: torch.Tensor | None = None,
    block_tables: torch.Tensor | None = None,
    causal: bool = True,
    prefix_lens: torch.Tensor | None = None,
    cu_seqlens_q_host: list[int] | None = None,
    cu_seqlens_k_host: list[int] | None = None,
) -> torch.Tensor:
    """Varlen prefill attention over the pool.

    ``q`` is the packed batch ``[N, num_heads, head_dim]``. Keys/values are
    gathered from the pool per sequence through ``block_tables``: each
    sequence's blocks are gathered and concatenated in position order, then
    trimmed to its ``cu_seqlens_k`` span.

    There is no contiguous fallback: the pool is block-addressed everywhere
    (``store_kvcache_paged`` writes through slot mappings into per-sequence
    blocks), so attention must gather history through the same tables — a
    missing table is a contract violation, not a mode.

    Masking: with ``causal=True`` (the default, and the only mode causal
    models compile to) queries attend to keys at or before their absolute
    position. With ``causal=False`` (PrefixLM artifacts) each sequence
    carries a prefix boundary ``p`` through ``prefix_lens`` and the mask is
    the two-region form ``keep(r, c) = (r < p and c < p) or (r >= p and
    c <= r)``: prompt rows attend bidirectionally within the prompt region,
    response rows causally. ``p`` spanning the whole sequence (fresh prefill)
    degenerates to pure bidirectional attention. ``prefix_lens`` is required
    when ``causal=False`` — there is no implicit boundary.

    Returns the merged output ``[N, num_heads * head_dim]``.
    """
    if block_tables is None:
        raise ValueError(
            "varlen_paged_sdpa requires block_tables: the pool is "
            "block-addressed (store_kvcache_paged writes per-sequence "
            "blocks), so history must be gathered through the same tables"
        )
    if not causal and prefix_lens is None:
        raise ValueError(
            "varlen_paged_sdpa with causal=False (PrefixLM) requires "
            "prefix_lens: the two-region mask has no implicit boundary"
        )
    device = q.device
    dtype = q.dtype
    assert cu_seqlens_q is not None and cu_seqlens_k is not None
    cu_q = cu_seqlens_q.to(device=device, dtype=torch.long)
    cu_k = cu_seqlens_k.to(device=device, dtype=torch.long)
    # Host-side boundaries from the engine's context (see Context):
    # recomputing them here would sync the pipeline once per layer call.
    if cu_seqlens_q_host is not None or cu_seqlens_k_host is not None:
        for name, value in (
            ("cu_seqlens_q_host", cu_seqlens_q_host),
            ("cu_seqlens_k_host", cu_seqlens_k_host),
        ):
            if value is not None and isinstance(value, torch.Tensor):
                raise TypeError(
                    f"{name} must be host-side Python ints, got {type(value).__name__}: "
                    "a device tensor in a *_host field silently degrades the fast "
                    "path into per-layer pipeline syncs (bool()/int() on its "
                    "elements), which is the exact failure this field exists to "
                    "prevent"
                )
    if cu_seqlens_q_host is not None and cu_seqlens_k_host is not None:
        num_seqs = len(cu_seqlens_q_host) - 1
        n = cu_seqlens_q_host[-1]
        total_k = cu_seqlens_k_host[-1]
        k_boundaries = cu_seqlens_k_host
    else:
        num_seqs = cu_q.numel() - 1
        n = int(cu_q[-1].item())
        total_k = int(cu_k[-1].item())
        k_boundaries = [int(cu_k[i].item()) for i in range(num_seqs + 1)]

    # Gather keys/values in position order across the packed batch.
    tables = block_tables.to(device=device, dtype=torch.long)
    pieces_k: list[torch.Tensor] = []
    pieces_v: list[torch.Tensor] = []
    for seq in range(num_seqs):
        start, end = k_boundaries[seq], k_boundaries[seq + 1]
        if end == start:
            continue
        needed = (end - start + k_cache.shape[1] - 1) // k_cache.shape[1]
        table = tables[seq, :needed].clamp_min(0)
        k_seq, v_seq = _gather_kv_via_block_tables(
            k_cache, v_cache, table.unsqueeze(0)
        )
        pieces_k.append(k_seq[0][: end - start])
        pieces_v.append(v_seq[0][: end - start])
    keys = torch.cat(pieces_k, dim=0) if pieces_k else k_cache.new_zeros(0, num_kv_heads, head_dim)
    values = torch.cat(pieces_v, dim=0) if pieces_v else v_cache.new_zeros(0, num_kv_heads, head_dim)

    keys = keys.transpose(0, 1).unsqueeze(0)  # [1, H_kv, total_k, D]
    values = values.transpose(0, 1).unsqueeze(0)
    queries = q.transpose(0, 1).unsqueeze(0)  # [1, H, N, D]

    # Mask over the packed batch. Queries sit at absolute positions
    # `cached + local` within their sequence's key span (cached > 0 with
    # prefix cache or chunked prefill); keys are indexed by their absolute
    # position within the same span.
    q_index = torch.arange(n, device=device)
    k_index = torch.arange(total_k, device=device)
    q_seq = torch.searchsorted(cu_q[1:], q_index, right=True)
    k_seq = torch.searchsorted(cu_k[1:], k_index, right=True)
    q_local = q_index - cu_q[q_seq]
    k_local = k_index - cu_k[k_seq]
    q_span = cu_q[1:] - cu_q[:-1]
    k_span = cu_k[1:] - cu_k[:-1]
    cached = (k_span - q_span).clamp_min(0)[q_seq]
    q_pos = q_local + cached
    same_seq = q_seq.unsqueeze(1) == k_seq.unsqueeze(0)
    if causal:
        keep = q_pos.unsqueeze(1) >= k_local.unsqueeze(0)
    else:
        boundary = prefix_lens.to(device=device, dtype=torch.long)[q_seq]
        q_row = q_pos.unsqueeze(1)
        k_col = k_local.unsqueeze(0)
        prefix_region = (q_row < boundary.unsqueeze(1)) & (k_col < boundary.unsqueeze(1))
        response_region = (q_row >= boundary.unsqueeze(1)) & (k_col <= q_row)
        keep = prefix_region | response_region
    mask = torch.where(
        same_seq & keep,
        torch.zeros((), device=device, dtype=dtype),
        torch.full((), _NEG_INF, device=device, dtype=dtype),
    )
    mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, N, total_k]

    out = F.scaled_dot_product_attention(
        queries,
        keys,
        values,
        attn_mask=mask,
        dropout_p=0.0,
        scale=scale,
        enable_gqa=num_heads != num_kv_heads,
    )
    return out.squeeze(0).transpose(0, 1).reshape(n, num_heads * head_dim)


def decode_paged_sdpa(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    *,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float | None = None,
    context_lens: torch.Tensor | None = None,
    block_tables: torch.Tensor | None = None,
    context_lens_max: int | None = None,
) -> torch.Tensor:
    """Decode attention over the pool.

    ``q`` is ``[B, num_heads, head_dim]`` (one row per sequence). Keys/values
    are gathered through ``block_tables`` ``[B, max_blocks]``; keys at or
    beyond each row's ``context_len`` are masked out. There is no contiguous
    fallback: the pool is block-addressed (store and read must agree on one
    scheme), so a missing table is a contract violation, not a mode.

    Returns the merged output ``[B, num_heads * head_dim]``.
    """
    device = q.device
    dtype = q.dtype
    assert context_lens is not None
    if block_tables is None:
        raise ValueError(
            "decode_paged_sdpa requires block_tables: the pool is "
            "block-addressed (store_kvcache_paged writes per-sequence "
            "blocks), so history must be gathered through the same tables"
        )
    lens = context_lens.to(device=device, dtype=torch.long)
    batch = q.shape[0]
    # context_lens_max arrives from the engine's context (computed host-side
    # once per step); recomputing it here syncs the pipeline once per layer.
    max_len = (
        context_lens_max
        if context_lens_max is not None
        else (int(lens.max().item()) if lens.numel() else 0)
    )

    keys, values = _gather_kv_via_block_tables(
        k_cache, v_cache, block_tables.to(device=device, dtype=torch.long)
    )  # [B, positions, H_kv, D]

    keys = keys[:, :max_len].transpose(1, 2)  # [B, H_kv, max_len, D]
    values = values[:, :max_len].transpose(1, 2)
    k_index = torch.arange(max_len, device=device)
    keep = k_index.unsqueeze(0) < lens.unsqueeze(1)  # [B, max_len]
    mask = torch.where(
        keep.unsqueeze(1).unsqueeze(2),
        torch.zeros((), device=device, dtype=dtype),
        torch.full((), _NEG_INF, device=device, dtype=dtype),
    )  # [B, 1, 1, max_len]

    queries = q.unsqueeze(2)  # [B, H, 1, D]
    out = F.scaled_dot_product_attention(
        queries,
        keys,
        values,
        attn_mask=mask,
        dropout_p=0.0,
        scale=scale,
        enable_gqa=num_heads != num_kv_heads,
    )
    return out.squeeze(2).reshape(batch, num_heads * head_dim)


# --------------------------------------------------------------------------
# Metal decode kernel (opt-in accelerator, mps only).
#
# The composed path above pays, per layer per decode step, a KV gather through
# the block tables, a mask build, dtype alignment, and an SDPA call — separate
# launches over data the GPU already holds. The kernel below does the whole
# decode attention step in ONE launch that reads the pool directly: a
# flash-style online softmax with one simdgroup (32 lanes) per (sequence,
# query head), each lane owning head_dim/32 output dims.
#
# Geometry (head counts, head_dim, block_size) is baked into the source and
# the library is compiled lazily per geometry via torch.mps.compile_shader
# (torch >= 2.11, Apple silicon); kernels dispatch on torch's own MPS stream,
# so ordering against surrounding torch ops is automatic. Dispatch failures
# and unsupported shapes raise — nothing falls back silently.
# --------------------------------------------------------------------------

_METAL_LIBRARIES: dict[tuple[str, int, int, int, int], Any] = {}


def _metal_decode_source(
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
    suffix: str,
) -> str:
    """Generate the (fully unrolled) Metal source for one pool geometry.

    Everything is compile-time: head counts, head_dim, and the block-size
    shift/mask are baked in, because torch.mps.compile_shader does not run
    the optimizer passes that would strength-reduce runtime divisions or
    unroll the per-lane register arrays. ``scale`` stays a runtime scalar.
    """
    dpl = head_dim // 32          # dims per lane
    pairs = dpl // 2              # 4-byte bf16 pairs per lane
    log2 = block_size.bit_length() - 1
    assert head_dim % 64 == 0 and 1 <= pairs <= 4
    bf16 = suffix == "bfloat"
    acc_decls = [f"    float acc{i} = 0.0f;" for i in range(dpl)]

    if bf16:
        # bf16: load/store 4-byte uints, each carrying two bf16 halves that
        # unpack to floats via bit shifts (as_type is a free reinterpret).
        q_loads = [f"    const uint qb{i} = qvec[{i}];" for i in range(pairs)]
        q_unpack = [
            f"    const float qv{i} = "
            + (f"as_type<float>(qb{i // 2} << 16)" if i % 2 == 0
               else f"as_type<float>(qb{i // 2} & 0xffff0000u)") + ";"
            for i in range(dpl)
        ]
        part_terms = [
            f"qv{i} * "
            + (f"as_type<float>(kb{i // 2} << 16)" if i % 2 == 0
               else f"as_type<float>(kb{i // 2} & 0xffff0000u)")
            for i in range(dpl)
        ]
        k_loads = [f"        const uint kb{i} = kvec[{i}];" for i in range(pairs)]
        v_loads = [f"        const uint vb{i} = vvec[{i}];" for i in range(pairs)]
        acc_updates = [
            f"        acc{i} = acc{i} * alpha + w * "
            + (f"as_type<float>(vb{i // 2} << 16)" if i % 2 == 0
               else f"as_type<float>(vb{i // 2} & 0xffff0000u)") + ";"
            for i in range(dpl)
        ]
        store_terms = [
            f"        uptr[{i}] = (as_type<uint>(acc{2 * i} * inv_l) >> 16) |"
            f" (as_type<uint>(acc{2 * i + 1} * inv_l) & 0xffff0000u);"
            for i in range(pairs)
        ]
        qvec_decl = "    const device uint* qvec = (const device uint*)(qrow + d0);"
        kvec_decl = (
            "        const device uint* kvec = (const device uint*)(k_pool + row * D + d0);\n"
            "        const device uint* vvec = (const device uint*)(v_pool + row * D + d0);"
        )
        uptr_decl = "    device uint* uptr = (device uint*)(orow + d0);"
    else:
        # float32: plain element loads and stores (no bit packing).
        q_loads = []
        q_unpack = [f"    const float qv{i} = qrow[d0 + {i}];" for i in range(dpl)]
        part_terms = [f"qv{i} * krow[d0 + {i}]" for i in range(dpl)]
        k_loads = []
        v_loads = []
        acc_updates = [
            f"        acc{i} = acc{i} * alpha + w * vrow[d0 + {i}];"
            for i in range(dpl)
        ]
        store_terms = [
            f"        orow[d0 + {i}] = acc{i} * inv_l;" for i in range(dpl)
        ]
        qvec_decl = ""
        kvec_decl = (
            "        const device float* krow = k_pool + row * D;\n"
            "        const device float* vrow = v_pool + row * D;"
        )
        uptr_decl = ""

    def join(lines: list[str]) -> str:
        return "\n".join(lines) if lines else "        ;"

    return f"""#define HQ {num_heads}u
#define HKV {num_kv_heads}u
#define D {head_dim}u
#define BLOCK_LOG {log2}u
#define BLOCK_MASK {block_size - 1}u
#define GQA_RATIO {num_heads // num_kv_heads}u
#define DPL {dpl}u
#include <metal_stdlib>
using namespace metal;

// Fused paged decode attention over the block-addressed pool.
// q [B, HQ, D]; k/v pools [blocks, block_size, HKV, D];
// block_tables [B, max_blocks] int64 (-1 padding never read: the loop runs
// to each row's context_len); context_lens [B] int64; out [B, HQ, D].
// Grid: (HQ * 32, B, 1) threads, (32, 1, 1) each — one simdgroup per
// (sequence, query head), lane i owns output dims [i*DPL, (i+1)*DPL).
kernel void paged_decode_attention_{suffix}(
    device const {suffix}* q,
    device const {suffix}* k_pool,
    device const {suffix}* v_pool,
    device const int64_t* block_tables,
    device const int64_t* context_lens,
    device {suffix}* out,
    constant uint& max_blocks,
    constant float& scale,
    uint3 tig [[thread_position_in_grid]],
    uint3 tpitg [[thread_position_in_threadgroup]])
{{
    const uint h = tig.x / 32u;
    const uint b = tig.y;
    const uint lane = tpitg.x;
    const uint kvh = h / GQA_RATIO;
    const uint d0 = lane * DPL;
    const int ctx = (int)context_lens[b];
    const device {suffix}* qrow = q + ((uint64_t)b * HQ + h) * D;
    device {suffix}* orow = out + ((uint64_t)b * HQ + h) * D;
{qvec_decl}
{join(q_loads)}
{join(q_unpack)}

    float m = -INFINITY;
    float l = 0.0f;
{join(acc_decls)}

    for (int p = 0; p < ctx; p++) {{
        const uint blk = (uint)block_tables[b * max_blocks + ((uint)p >> BLOCK_LOG)];
        const uint off = (uint)p & BLOCK_MASK;
        const uint64_t row = ((uint64_t)blk * (1u << BLOCK_LOG) + off) * HKV + kvh;
{kvec_decl}
{join(k_loads)}
{join(v_loads)}
        const float part = {' + '.join(part_terms)};
        const float s = simd_sum(part) * scale;
        const float m_new = max(m, s);
        const float alpha = fast::exp(m - m_new);
        const float w = fast::exp(s - m_new);
        l = l * alpha + w;
{join(acc_updates)}
        m = m_new;
    }}

    const float inv_l = 1.0f / l;
{uptr_decl}
{join(store_terms)}
}}
"""


def _metal_decode_library(
    dtype: torch.dtype,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
) -> Any:
    """Compile (once per geometry) and return the Metal kernel library."""
    suffix = "bfloat" if dtype == torch.bfloat16 else "float"
    key = (suffix, num_heads, num_kv_heads, head_dim, block_size)
    lib = _METAL_LIBRARIES.get(key)
    if lib is None:
        compile_shader = getattr(torch.mps, "compile_shader", None)
        if compile_shader is None:
            raise RuntimeError(
                "kv_kernel=metal needs torch.mps.compile_shader (torch >= 2.11 "
                f"on Apple silicon); torch {torch.__version__} does not expose it"
            )
        source = _metal_decode_source(num_heads, num_kv_heads, head_dim, block_size, suffix)
        try:
            lib = compile_shader(source)
        except Exception as exc:
            raise RuntimeError(
                f"compiling the Metal paged-decode kernel failed: {exc}"
            ) from exc
        _METAL_LIBRARIES[key] = lib
    return lib


def metal_decode_paged_sdpa(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    *,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float | None = None,
    context_lens: torch.Tensor | None = None,
    block_tables: torch.Tensor | None = None,
    context_lens_max: int | None = None,
) -> torch.Tensor:
    """Decode attention over the pool as ONE fused Metal kernel launch.

    Same contract as :func:`decode_paged_sdpa` (``q`` ``[B, num_heads,
    head_dim]``, merged ``[B, num_heads * head_dim]`` result, block-addressed
    pool) but executed by the opt-in Metal kernel instead of the composed
    gather+mask+SDPA chain. Loud raises, no fallbacks: non-mps devices,
    unsupported dtypes, non-power-of-two block sizes, and compile failures
    all raise.

    Supported pool dtypes: bfloat16 and float32. ``head_dim`` must be a
    multiple of 64 (the generated kernel unrolls head_dim/32 dims per lane
    in bf16 pairs); ``kvcache_block_size`` must be a power of two (the
    block/offset addressing compiles to shift/mask).
    """
    if q.device.type != "mps":
        raise ValueError(
            "metal_decode_paged_sdpa requires mps tensors; got q on "
            f"{q.device.type} (kv_kernel=metal is mps-only)"
        )
    dtype = k_cache.dtype
    if dtype not in (torch.bfloat16, torch.float32):
        raise ValueError(
            "metal_decode_paged_sdpa supports bfloat16/float32 pools; got "
            f"{dtype} (use kv_kernel=torch for other cache dtypes)"
        )
    if q.dtype != dtype:
        raise ValueError(
            f"metal_decode_paged_sdpa expects q in the pool dtype {dtype}; got {q.dtype}"
        )
    if head_dim % 64 != 0 or head_dim // 32 > 8:
        raise ValueError(
            "metal_decode_paged_sdpa requires head_dim divisible by 64 and "
            f"head_dim/32 <= 8; got {head_dim} (use kv_kernel=torch)"
        )
    block_size = k_cache.shape[1]
    if block_size & (block_size - 1):
        raise ValueError(
            "metal_decode_paged_sdpa requires a power-of-two kvcache_block_size; "
            f"got {block_size} (use kv_kernel=torch)"
        )
    if block_tables is None or context_lens is None:
        raise ValueError(
            "metal_decode_paged_sdpa requires block_tables and context_lens: "
            "the pool is block-addressed, so history must be gathered through "
            "the same tables the store used"
        )
    if block_tables.dtype != torch.int64 or context_lens.dtype != torch.int64:
        raise ValueError(
            "metal_decode_paged_sdpa expects int64 block_tables/context_lens "
            f"(the engine's metadata contract); got {block_tables.dtype}/"
            f"{context_lens.dtype}"
        )
    for name, tensor in (
        ("q", q), ("k_cache", k_cache), ("v_cache", v_cache),
        ("block_tables", block_tables), ("context_lens", context_lens),
    ):
        if not tensor.is_contiguous():
            raise ValueError(f"metal_decode_paged_sdpa requires contiguous {name}")
    batch = q.shape[0]
    max_blocks = block_tables.shape[1]
    if context_lens_max is not None:
        needed = -(-int(context_lens_max) // block_size)
        if max_blocks < needed:
            raise ValueError(
                f"block_tables width {max_blocks} cannot cover context_lens_max "
                f"{int(context_lens_max)} at block_size {block_size}"
            )
    out = torch.empty((batch, num_heads, head_dim), dtype=dtype, device=q.device)
    lib = _metal_decode_library(dtype, num_heads, num_kv_heads, head_dim, block_size)
    getattr(lib, f"paged_decode_attention_{'bfloat' if dtype == torch.bfloat16 else 'float'}")(
        q, k_cache, v_cache, block_tables, context_lens, out,
        int(max_blocks), float(scale if scale is not None else 1.0 / (head_dim**0.5)),
        threads=(num_heads * 32, batch, 1), group_size=(32, 1, 1),
    )
    return out.reshape(batch, num_heads * head_dim)


def paged_attention(
    q: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    *,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float | None = None,
    context: Any | None = None,
    causal: bool = True,
    accelerator: str = "torch",
) -> torch.Tensor:
    """One attention step against the engine-owned pool.

    Stores ``key``/``value`` at ``slot_mapping`` and attends according to the
    ambient context: varlen prefill over the packed batch, or one-row decode
    per sequence. ``q``/``key``/``value`` are head-split tensors (``[N, H,
    head_dim]`` / ``[N, H_kv, head_dim]``); the result is merged
    ``[N, num_heads * head_dim]``. ``causal`` selects the prefill masking
    mode (see :func:`varlen_paged_sdpa`); decode is mask-free either way.

    ``accelerator`` selects the decode kernel: ``"torch"`` (default) runs the
    composed gather+mask+SDPA path; ``"metal"`` runs the fused Metal kernel
    (:func:`metal_decode_paged_sdpa`, mps-only, loud raises when
    unavailable). Prefill always runs the composed varlen path — the Metal
    kernel is decode-only.

    With ``block_tables`` in the context this is paged addressing; the tables
    are required — the pool is block-addressed on both the store and the read
    side, so a missing table is a contract violation, not a mode.
    """
    if context is None:
        context = get_context()
    store_kvcache_paged(k_cache, v_cache, key, value, context.slot_mapping)
    scale = 1.0 / (head_dim**0.5) if scale is None else scale
    if context.is_prefill:
        return varlen_paged_sdpa(
            q,
            k_cache,
            v_cache,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            scale=scale,
            cu_seqlens_q=context.cu_seqlens_q,
            cu_seqlens_k=context.cu_seqlens_k,
            block_tables=context.block_tables,
            causal=causal,
            prefix_lens=context.prefix_lens,
            cu_seqlens_q_host=context.cu_seqlens_q_host,
            cu_seqlens_k_host=context.cu_seqlens_k_host,
        )
    if accelerator == "metal":
        decode = metal_decode_paged_sdpa
    elif accelerator == "torch":
        decode = decode_paged_sdpa
    else:
        raise ValueError(
            f"unknown paged-attention accelerator {accelerator!r} "
            "(expected 'torch' or 'metal')"
        )
    return decode(
        q,
        k_cache,
        v_cache,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        scale=scale,
        context_lens=context.context_lens,
        block_tables=context.block_tables,
        context_lens_max=context.context_lens_max,
    )


class PagedAttention(torch.nn.Module):
    """Generated-artifact wrapper around :func:`paged_attention`.

    The serving-profile emitter lowers the ``__torch_paged_attention``
    intrinsic to one instance per attention site.  The engine binds the
    ``k_cache``/``v_cache`` attributes to this layer stack's pool views
    (``[num_layers, num_blocks, block_size, num_kv_heads, head_dim]``);
    each call slices ``layer_idx`` out of the stack, stores this step's
    K/V rows at ``get_context().slot_mapping`` and attends over the pool.

    Operands are the head-split batched tensors the graph carries:
    ``q`` ``[B, num_heads, S, head_dim]``, ``k``/``v`` ``[B, num_kv_heads,
    S, head_dim]`` (the fresh, un-expanded values).  The result matches the
    plain attention call it replaces: ``[B, num_heads, S, head_dim]``.
    """

    def __init__(
        self,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        scale: float | None = None,
        causal: bool = True,
        accelerator: str = "torch",
    ) -> None:
        super().__init__()
        if num_heads % num_kv_heads != 0:
            raise ValueError(
                f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
            )
        if accelerator not in ("torch", "metal"):
            raise ValueError(
                f"unknown PagedAttention accelerator {accelerator!r} "
                "(expected 'torch' or 'metal')"
            )
        self.num_heads = int(num_heads)
        self.num_kv_heads = int(num_kv_heads)
        self.head_dim = int(head_dim)
        self.scale = scale
        # Prefill masking mode, baked at codegen from the model's masking
        # builtin (causal_mask_for_input vs bidirectional_mask_for_input);
        # never inferred at runtime.
        self.causal = bool(causal)
        # Decode kernel choice: "torch" (composed gather+mask+SDPA) or
        # "metal" (fused Metal kernel, mps-only). The generated artifact
        # leaves this at the default; the engine binds it post-instantiation
        # from the server's kv_kernel config, like the pool views below.
        self.accelerator = accelerator
        # Engine-owned pool views; None until the runner binds them.
        self.k_cache: torch.Tensor | None = None
        self.v_cache: torch.Tensor | None = None

    def _layer_slice(self, cache: torch.Tensor, layer_idx: int) -> torch.Tensor:
        if cache.dim() >= 5:
            return cache[layer_idx]
        return cache

    def extra_repr(self) -> str:
        return (
            f"num_heads={self.num_heads}, num_kv_heads={self.num_kv_heads}, "
            f"head_dim={self.head_dim}, causal={self.causal}, "
            f"accelerator={self.accelerator}"
        )

    def forward(
        self,
        q: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        layer_idx: torch.Tensor | int,
    ) -> torch.Tensor:
        if self.k_cache is None or self.v_cache is None:
            raise RuntimeError(
                "PagedAttention caches are not bound; the engine must set k_cache/v_cache"
            )
        batch, num_heads, seq_len, head_dim = q.shape
        layer = int(layer_idx.item()) if isinstance(layer_idx, torch.Tensor) else int(layer_idx)
        k_cache = self._layer_slice(self.k_cache, layer)
        v_cache = self._layer_slice(self.v_cache, layer)
        # Attention runs in the value dtype, matching the non-paged _sdpa
        # helper's contract. Rope factors are deliberately float32, so q/k
        # arrive promoted to float32 on sub-float32 models; without this
        # alignment the store silently narrows k/v into the pool while q
        # stays wide and SDPA rejects the mix.
        if q.dtype != value.dtype:
            q = q.to(value.dtype)
        if key.dtype != value.dtype:
            key = key.to(value.dtype)
        # [B, H, S, D] -> [B*S, H, D] rows in the packed batch order.
        q_rows = q.transpose(1, 2).reshape(batch * seq_len, num_heads, head_dim)
        k_rows = key.transpose(1, 2).reshape(batch * seq_len, self.num_kv_heads, self.head_dim)
        v_rows = value.transpose(1, 2).reshape(batch * seq_len, self.num_kv_heads, self.head_dim)
        out = paged_attention(
            q_rows,
            k_rows,
            v_rows,
            k_cache,
            v_cache,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            scale=self.scale,
            causal=self.causal,
            accelerator=self.accelerator,
        )
        # [B*S, H*D] merged -> [B, H, S, D] (inverse of merge_heads' permute).
        return out.reshape(batch, seq_len, num_heads, head_dim).transpose(1, 2)
