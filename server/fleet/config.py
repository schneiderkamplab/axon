# Fleet configuration for the Axon serving layer.
#
# Field names and semantics follow nano-vllm's Config for diffability
# (https://github.com/GeeeekExplorer/nano-vllm, MIT, © 2025 Xingkai Yu), but
# the HF AutoConfig source is replaced by the Axon model_config dict that
# comes from the checkpoint's config.json — the fleet never talks to HF hub
# config loading.
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

__all__ = ["Config", "ModelConfigView"]


class ModelConfigView:
    """Attribute view over the Axon model_config dict (config.json).

    The vendored scheduler/runner code reads ``config.hf_config.<attr>``;
    this view provides those attributes from our dict without HF AutoConfig.
    """

    def __init__(self, model_config: dict[str, Any]) -> None:
        self._config = model_config
        dtype_name = str(model_config.get("torch_dtype", "float16")).removeprefix("torch.")
        self.dtype = getattr(torch, dtype_name)

    def __getattr__(self, name: str) -> Any:
        try:
            return self._config[name]
        except KeyError:
            raise AttributeError(name) from None


@dataclass(slots=True)
class Config:
    model_config: dict[str, Any]
    device: str = "cpu"
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    tensor_parallel_size: int = 1  # reserved; v1 runs TP=1
    enforce_eager: bool = True     # CUDA-graph capture is upstream-only for now
    hf_config: ModelConfigView | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1   # -1: the runner sizes the pool conservatively
    # Fraction of the device's recommended memory budgeted to weights + KV
    # pool when num_kvcache_blocks is -1 (fleet addition, not in upstream
    # nano-vllm). Apple unified memory: the recommended working set is the
    # ceiling macOS actually enforces (~78% of RAM here); 0.6 of it keeps a
    # shared machine comfortable, raise it (e.g. 0.9) on a dedicated box.
    pool_memory_fraction: float = 0.6
    # Prefill masking mode of the served artifact (fleet addition, not in
    # upstream nano-vllm): True for causal decoders, False for PrefixLM
    # artifacts (bidirectional attention over the prompt region). Set by
    # LLMEngine from the artifact's serving_prefill_masking.
    prefill_causal: bool = True

    def __post_init__(self):
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = ModelConfigView(self.model_config)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
