# Fleet model runner: the seam between the vendored scheduler core and
# Axon-generated context-driven torch models.
#
# Adapted from nano-vllm (https://github.com/GeeeekExplorer/nano-vllm)
# Copyright (c) 2025 Xingkai Yu — MIT License. Rewritten for the Axon serving
# layer:
# - device-parametric (upstream hardcodes CUDA); no nccl / SharedMemory TP
#   machinery in v1 (tensor_parallel_size=1; the upstream structure returns
#   when TP lands, using the vendored linear.py as the reference)
# - the pool is sized from ``config.num_kvcache_blocks``; -1 sizes from the
#   device's recommended memory minus resident weights (the upstream
#   max_model_len x max_num_seqs worst case asks hundreds of GiB on laptops)
# - CUDA-graph capture dropped for now (eager path always; enforce_eager
#   stays a config field for when capture returns)
# - warmup forward dropped (it exists upstream to size CUDA memory)
# - DIVERGENCE FIX: prefill sampling gathers each sequence's LAST scheduled
#   row. Upstream feeds the full [N, V] logits to the sampler with [B]
#   temperatures, which only broadcasts when N == B — multi-sequence prefill
#   batches crash (and mis-sample if they did not).
from __future__ import annotations

import os
from typing import Any

import torch

from ..config import Config
from ..layers.sampler import Sampler
from ..utils.context import get_context, reset_context, set_context
from .sequence import Sequence

__all__ = ["ModelRunner"]


def _device_memory_budget(device: torch.device) -> int:
    """The device's recommended memory ceiling in bytes, for pool sizing.

    MPS reports its recommended max working set (the number macOS actually
    enforces); CUDA reports total device memory; CPU falls back to installed
    RAM. No silent fallbacks: an unknown or unqueryable device raises.
    """
    if device.type == "mps":
        return int(torch.mps.recommended_max_memory())
    if device.type == "cuda":
        return int(torch.cuda.get_device_properties(device).total_memory)
    if device.type == "cpu":
        page_size = os.sysconf("SC_PAGE_SIZE")
        page_count = os.sysconf("SC_PHYS_PAGES")
        if page_size <= 0 or page_count <= 0:
            raise ValueError(
                "cannot determine installed RAM for CPU pool sizing; "
                "pass num_kvcache_blocks explicitly"
            )
        return page_size * page_count
    raise ValueError(
        f"no memory budget known for device {device}; pass num_kvcache_blocks explicitly"
    )


class ModelRunner:

    def __init__(self, config: Config, model: Any) -> None:
        self.config = config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size  # v1: 1
        self.model = model
        self.device = torch.device(config.device)
        if config.kv_kernel == "metal" and self.device.type != "mps":
            raise ValueError(
                f"kv_kernel=metal requires device mps; got {config.device!r} "
                "(the Metal decode kernel is Apple-silicon only)"
            )
        self.sampler = Sampler()
        self.allocate_kv_cache()

    # ── pool ────────────────────────────────────────────────────────────────

    def allocate_kv_cache(self) -> None:
        config = self.config
        hf_config = config.hf_config
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = (
            getattr(hf_config, "head_dim", None)
            or hf_config.hidden_size // hf_config.num_attention_heads
        )
        if config.num_kvcache_blocks < 0:
            config.num_kvcache_blocks = self._autosize_num_blocks(
                hf_config.num_hidden_layers, num_kv_heads, head_dim
            )
        assert config.num_kvcache_blocks > 0
        self.kv_cache = torch.empty(
            2,
            hf_config.num_hidden_layers,
            config.num_kvcache_blocks,
            self.block_size,
            num_kv_heads,
            head_dim,
            dtype=hf_config.dtype,
            device=self.device,
        )
        bound = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                # Cache-carrying modules receive the FULL [L, blocks, bs, H, D]
                # stacks and address their layer through layer_idx at call
                # time: the Axon serving artifact registers one PagedAttention
                # per head geometry (shared across the layer stack), not one
                # module per layer.
                module.k_cache = self.kv_cache[0]
                module.v_cache = self.kv_cache[1]
                # Same post-instantiation binding for the decode-kernel choice:
                # the generated artifact leaves PagedAttention at the "torch"
                # default; the server's kv_kernel config opts into the Metal
                # decode kernel here (the runner already raised if metal was
                # requested off-mps; per-call shape/dtype rules raise loudly
                # inside metal_decode_paged_sdpa).
                module.accelerator = self.config.kv_kernel
                bound += 1
        if bound == 0:
            raise ValueError(
                "the model exposes no KV-cache modules (k_cache/v_cache attributes); "
                "the serving contract requires paged attention modules to bind"
            )

    def _autosize_num_blocks(
        self, num_layers: int, num_kv_heads: int, head_dim: int
    ) -> int:
        """Size the pool from device memory, not from a worst-case shape.

        The upstream formula (every sequence at max_model_len simultaneously)
        asks e.g. 224 GiB on a laptop: a request that either fails loudly or,
        worse, drives the machine into swap. Instead: budget
        ``config.pool_memory_fraction`` of the device's recommended memory
        (the ceiling macOS actually enforces on unified memory) minus the
        resident weights, floor at one full-length sequence, and print the
        derivation. Pass ``num_kvcache_blocks`` explicitly to opt out.
        """
        bytes_per_block = (
            num_layers * 2 * num_kv_heads * head_dim * self.block_size
            * self.config.hf_config.dtype.itemsize
        )
        fraction = self.config.pool_memory_fraction
        budget = _device_memory_budget(self.device)
        weights = sum(p.numel() * p.element_size() for p in self.model.parameters())
        state = getattr(self.model, "state_dict_tensors", None)
        if state:
            weights += sum(
                t.numel() * t.element_size() for t in state.values() if torch.is_tensor(t)
            )
        pool_bytes = int(budget * fraction) - weights
        memory_blocks = pool_bytes // bytes_per_block
        floor_blocks = -(-self.config.max_model_len // self.block_size)
        if memory_blocks < floor_blocks:
            raise ValueError(
                f"device {self.device} has no room for even one max-length sequence: "
                f"pool budget {pool_bytes / 2**30:.2f} GiB ({fraction:.0%} of "
                f"{budget / 2**30:.2f} GiB minus {weights / 2**30:.2f} GiB weights) "
                f"holds {memory_blocks} blocks of {bytes_per_block / 2**20:.1f} MiB, "
                f"but max_model_len {self.config.max_model_len} needs {floor_blocks}; "
                "lower --max-model-len or pass --num-kvcache-blocks on a larger device"
            )
        worst_case = floor_blocks * self.config.max_num_seqs
        blocks = max(floor_blocks, min(worst_case, memory_blocks))
        print(
            f"fleet: KV pool autosized to {blocks} blocks "
            f"({blocks * bytes_per_block / 2**30:.2f} GiB): "
            f"{fraction:.0%} of {budget / 2**30:.1f} GiB device budget "
            f"minus {weights / 2**30:.2f} GiB weights; floor {floor_blocks} blocks "
            f"(one max-length sequence), worst case {worst_case} blocks "
            f"({self.config.max_num_seqs} sequences x {self.config.max_model_len} tokens)"
        )
        return blocks

    # ── batching ────────────────────────────────────────────────────────────

    def _to_device(self, values: list, dtype: torch.dtype) -> torch.Tensor:
        tensor = torch.tensor(values, dtype=dtype, pin_memory=self.device.type == "cuda")
        if self.device.type != "cpu":
            # non_blocking is only sound where the source is pinned (CUDA);
            # CPU->MPS async copies without pinned memory fail with
            # "Placeholder storage has not been allocated on MPS device".
            return tensor.to(self.device, non_blocking=self.device.type == "cuda")
        return tensor

    def prepare_block_tables(self, seqs: list[Sequence]) -> torch.Tensor:
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        # int64: the paged kernels index through these every layer; a narrower
        # dtype would make each layer's .to(torch.long) a real conversion
        # (a tiny kernel launch per layer, 28 per step).
        return self._to_device(block_tables, torch.int64)

    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids: list[int] = []
        positions: list[int] = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping: list[int] = []
        block_tables = None
        sample_rows: list[int] = []  # per-seq last scheduled row (fleet fix)
        for seq in seqs:
            start = seq.num_cached_tokens
            seqlen_q = seq.num_scheduled_tokens
            end = start + seqlen_q
            seqlen_k = end
            input_ids.extend(seq[start:end])
            positions.extend(range(start, end))
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:  # warmup
                continue
            start_block = start // self.block_size
            end_block = (end + self.block_size - 1) // self.block_size
            for i in range(start_block, end_block):
                slot_start = seq.block_table[i] * self.block_size
                if i == start_block:
                    slot_start += start % self.block_size
                if i != end_block - 1:
                    slot_end = seq.block_table[i] * self.block_size + self.block_size
                else:
                    slot_end = seq.block_table[i] * self.block_size + end - i * self.block_size
                slot_mapping.extend(range(slot_start, slot_end))
            sample_rows.append(len(input_ids) - 1)
        # The paged kernel reads history through the pool, so block tables
        # are required on every prefill — not only with a cached prefix
        # (each scheduled sequence's blocks are allocated by the scheduler
        # before this runs).
        block_tables = self.prepare_block_tables(seqs)
        input_ids = self._to_device(input_ids, torch.int64)
        positions = self._to_device(positions, torch.int64)
        # Host lists must be captured BEFORE the names are rebound to device
        # tensors below: the context's *_host fields are the kernel's fast
        # path, and a device tensor smuggled through them would sync the
        # pipeline once per layer (bool()/int() on tensor elements bypass
        # .item()-style instrumentation, so this bug hid from the sync
        # counters until a profiler with aten-level visibility ran).
        cu_seqlens_q_host = cu_seqlens_q
        cu_seqlens_k_host = cu_seqlens_k
        cu_seqlens_q = self._to_device(cu_seqlens_q, torch.int64)
        cu_seqlens_k = self._to_device(cu_seqlens_k, torch.int64)
        slot_mapping = self._to_device(slot_mapping, torch.int64)
        # PrefixLM artifacts carry the per-sequence prompt boundary so the
        # paged kernel builds the two-region mask (prompt rows bidirectional,
        # generated rows causal). Positions are absolute, and a preemption
        # recompute re-prefills prompt+generated with the boundary kept at
        # num_prompt_tokens.
        prefix_lens = None
        if not self.config.prefill_causal:
            prefix_lens = self._to_device([seq.num_prompt_tokens for seq in seqs], torch.int64)
        set_context(
            True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None,
            block_tables, prefix_lens,
            # The cu arrays existed as Python lists a moment ago; carry them so
            # the paged kernel does not re-extract scalars per layer (each
            # extraction is a full MPS/CUDA pipeline sync).
            cu_seqlens_q_host=cu_seqlens_q_host, cu_seqlens_k_host=cu_seqlens_k_host,
        )
        return input_ids, positions, sample_rows

    def prepare_decode(self, seqs: list[Sequence]):
        input_ids: list[int] = []
        positions: list[int] = []
        slot_mapping: list[int] = []
        context_lens: list[int] = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            slot_mapping.append(
                seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1
            )
        max_context_lens = max(context_lens) if context_lens else 0
        input_ids = self._to_device(input_ids, torch.int64)
        positions = self._to_device(positions, torch.int64)
        # int64 metadata: the decode kernel casts context_lens/slot_mapping to
        # long every layer; int64 up front makes those no-ops.
        slot_mapping = self._to_device(slot_mapping, torch.int64)
        context_lens = self._to_device(context_lens, torch.int64)
        block_tables = self.prepare_block_tables(seqs)
        set_context(
            False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables,
            # max(context_lens) is already known host-side; carrying it saves
            # one pipeline sync per layer in the decode kernel.
            context_lens_max=max_context_lens,
        )
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]) -> dict[str, torch.Tensor | bool]:
        # Sentinel convention (sampler-side): top_k <= 0 and top_p outside
        # (0, 1) disable the filter, matching the Session sampler's None.
        # The host-side decisions ride along as bools: the sampler otherwise
        # ends every step with up to three device-side all() checks, each a
        # full pipeline sync ahead of the unavoidable token readback. The
        # top_k test is deliberately conservative (only k <= 0 counts as
        # disabled host-side); a k >= vocab row simply falls back to the
        # device-side check — correct, one sync, and no silently skipped
        # filtering.
        temperatures = [seq.temperature for seq in seqs]
        top_ks = [seq.top_k or 0 for seq in seqs]
        top_ps = [seq.top_p if seq.top_p is not None else 1.0 for seq in seqs]
        return {
            "temperatures": self._to_device(temperatures, torch.float32),
            "top_ks": self._to_device(top_ks, torch.int64),
            "top_ps": self._to_device(top_ps, torch.float32),
            "all_greedy": all(temperature <= 0.0 for temperature in temperatures),
            "any_greedy": any(temperature <= 0.0 for temperature in temperatures),
            "all_topk_disabled": all(top_k <= 0 for top_k in top_ks),
            "all_topp_disabled": all(top_p >= 1.0 for top_p in top_ps),
        }

    # ── forward ─────────────────────────────────────────────────────────────

    @torch.inference_mode()
    def run_model(
        self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool
    ) -> Any:
        # CUDA-graph capture is upstream-only for now; eager always.
        # The Axon serving artifact takes ``positions`` by keyword (its forward
        # forwards extra named inputs), and its graph contract is [B, S]:
        # prefill packs all scheduled rows into one [1, N] view, decode is
        # [num_seqs, 1].
        if is_prefill:
            input_ids = input_ids.view(1, -1)
            positions = positions.view(1, -1)
        else:
            input_ids = input_ids.unsqueeze(-1)
            positions = positions.unsqueeze(-1)
        # Packed logits rows for the flat sample_rows indexing: [B, S, V]
        # from the artifact flattens to [N, V]; already-flat [N, V] mock
        # output is a no-op.
        logits = self.model.compute_logits(self.model(input_ids, positions=positions))
        return logits.reshape(-1, logits.shape[-1])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        if is_prefill:
            input_ids, positions, sample_rows = self.prepare_prefill(seqs)
        else:
            input_ids, positions = self.prepare_decode(seqs)
            sample_rows = None
        temperatures = self.prepare_sample(seqs)
        logits = self.run_model(input_ids, positions, is_prefill)
        if sample_rows is not None:
            logits = logits[sample_rows]  # [B, V]: one row per scheduled seq
        with torch.inference_mode():
            token_ids = self.sampler(logits, **temperatures).tolist()
        reset_context()
        return token_ids
