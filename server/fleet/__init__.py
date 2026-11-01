"""Fleet engine: continuous batching over Axon-generated torch models.

Vendored from nano-vllm (https://github.com/GeeeekExplorer/nano-vllm,
MIT License, Copyright (c) 2025 Xingkai Yu) — see file headers for per-file
provenance. Layout mirrors upstream for diffability:

- ``config``            – fleet Config (Axon model_config dict, no AutoConfig)
- ``sampling_params``   – per-request sampling params (vendored)
- ``engine/``           – scheduler, sequence, block_manager (vendored);
                          llm_engine + model_runner rewritten as the seam that
                          drives Axon-generated context-driven models
- ``utils/context``     – the get_context() contract models read (vendored)

Nothing attention-shaped lives here: generated models ship their own
attention and read the paged context through ``get_context()``.
"""

from .config import Config
from .sampling_params import SamplingParams

__all__ = ["Config", "SamplingParams"]
