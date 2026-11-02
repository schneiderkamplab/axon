# Fleet model runner: the seam between the vendored scheduler core and
# Axon-generated context-driven torch models.
#
# Adapted from nano-vllm (https://github.com/GeeeekExplorer/nano-vllm)
# Copyright (c) 2025 Xingkai Yu — MIT License. Rewritten for the Axon serving
# layer:
# - device-parametric (upstream hardcodes CUDA); no nccl / SharedMemory TP
#   machinery in v1 (tensor_parallel_size=1; the upstream structure returns
#   when TP lands, using the vendored linear.py as the reference)
# - the pool is sized from ``config.num_kvcache_blocks`` (no CUDA
#   mem_get_info); -1 falls back to a conservative max_model_len x max_num_seqs
# - CUDA-graph capture dropped for now (eager path always; enforce_eager
#   stays a config field for when capture returns)
# - warmup forward dropped (it exists upstream to size CUDA memory)
# - DIVERGENCE FIX: prefill sampling gathers each sequence's LAST scheduled
#   row. Upstream feeds the full [N, V] logits to the sampler with [B]
#   temperatures, which only broadcasts when N == B — multi-sequence prefill
#   batches crash (and mis-sample if they did not).
from __future__ import annotations

from typing import Any

import torch

from ..config import Config
from ..layers.sampler import Sampler
from ..utils.context import get_context, reset_context, set_context
from .sequence import Sequence

__all__ = ["ModelRunner"]


class ModelRunner:

    def __init__(self, config: Config, model: Any) -> None:
        self.config = config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size  # v1: 1
        self.model = model
        self.device = torch.device(config.device)
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
            blocks_per_seq = -(-config.max_model_len // self.block_size)
            config.num_kvcache_blocks = blocks_per_seq * config.max_num_seqs
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
                bound += 1
        if bound == 0:
            raise ValueError(
                "the model exposes no KV-cache modules (k_cache/v_cache attributes); "
                "the serving contract requires paged attention modules to bind"
            )

    # ── batching ────────────────────────────────────────────────────────────

    def _to_device(self, values: list, dtype: torch.dtype) -> torch.Tensor:
        tensor = torch.tensor(values, dtype=dtype, pin_memory=self.device.type == "cuda")
        if self.device.type != "cpu":
            return tensor.to(self.device, non_blocking=True)
        return tensor

    def prepare_block_tables(self, seqs: list[Sequence]) -> torch.Tensor:
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        return self._to_device(block_tables, torch.int32)

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
        cu_seqlens_q = self._to_device(cu_seqlens_q, torch.int32)
        cu_seqlens_k = self._to_device(cu_seqlens_k, torch.int32)
        slot_mapping = self._to_device(slot_mapping, torch.int32)
        set_context(
            True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables
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
        input_ids = self._to_device(input_ids, torch.int64)
        positions = self._to_device(positions, torch.int64)
        slot_mapping = self._to_device(slot_mapping, torch.int32)
        context_lens = self._to_device(context_lens, torch.int32)
        block_tables = self.prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]) -> torch.Tensor:
        temperatures = [seq.temperature for seq in seqs]
        return self._to_device(temperatures, torch.float32)

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
            token_ids = self.sampler(logits, temperatures).tolist()
        reset_context()
        return token_ids
