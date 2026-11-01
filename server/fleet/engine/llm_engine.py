# Fleet engine: continuous batching entry point.
#
# Adapted from nano-vllm (https://github.com/GeeeekExplorer/nano-vllm)
# Copyright (c) 2025 Xingkai Yu — MIT License. Rewritten for the Axon serving
# layer:
# - the model is passed in already loaded (Axon-generated torch model); no HF
#   AutoConfig / model-path coupling — config comes from the model_config dict
# - tokenizer injected, not loaded from ``config.model``
# - no multiprocessing spawn for TP ranks in v1 (tensor_parallel_size=1)
# - no tqdm; ``generate`` is silent (the HTTP layer owns progress reporting)
from __future__ import annotations

import atexit
from typing import Any

from ..config import Config
from ..sampling_params import SamplingParams
from .model_runner import ModelRunner
from .scheduler import Scheduler
from .sequence import Sequence

__all__ = ["LLMEngine"]


class LLMEngine:

    def __init__(
        self,
        model: Any,
        *,
        model_config: dict[str, Any],
        tokenizer: Any | None = None,
        **config_kwargs: Any,
    ) -> None:
        config = Config(model_config, **config_kwargs)
        Sequence.block_size = config.kvcache_block_size
        self.config = config
        self.model_runner = ModelRunner(config, model)
        self.tokenizer = tokenizer
        if config.eos == -1 and tokenizer is not None:
            config.eos = getattr(tokenizer, "eos_token_id", -1)
        self.scheduler = Scheduler(config)
        atexit.register(self.exit)

    def exit(self) -> None:
        atexit.unregister(self.exit)
        self.model_runner = None

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams) -> None:
        if isinstance(prompt, str):
            if self.tokenizer is None:
                raise RuntimeError("String prompts require a tokenizer")
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    def step(self) -> tuple[list[tuple[int, list[int]]], int]:
        seqs, is_prefill = self.scheduler.schedule()
        num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
        token_ids = self.model_runner.run(seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids, is_prefill)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        return outputs, num_tokens

    def is_finished(self) -> bool:
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
    ) -> list[dict[str, Any]]:
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs: dict[int, list[int]] = {}
        while not self.is_finished():
            finished, _ = self.step()
            for seq_id, token_ids in finished:
                outputs[seq_id] = token_ids
        return [
            {
                "text": None if self.tokenizer is None else self.tokenizer.decode(token_ids),
                "token_ids": token_ids,
            }
            for token_ids in (outputs[seq_id] for seq_id in sorted(outputs))
        ]
