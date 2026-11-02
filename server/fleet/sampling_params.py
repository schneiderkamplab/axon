# Vendored from nano-vllm (https://github.com/GeeeekExplorer/nano-vllm)
# Copyright (c) 2025 Xingkai Yu — MIT License.
# Modified: upstream forbids greedy (temperature assert) and samples with
# temperature only. The OpenAI surface needs temperature=0 = exact greedy
# (the standard contract) plus top_k/top_p; ``seed`` is deferred until a
# per-request RNG can be threaded through the engine (a fresh generator per
# step would replay the same noise every step).
from dataclasses import dataclass


@dataclass(slots=True)
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int = 64
    top_k: int | None = None
    top_p: float | None = None
    ignore_eos: bool = False

    def __post_init__(self):
        assert self.temperature >= 0.0, "temperature must be >= 0 (0 = greedy)"
