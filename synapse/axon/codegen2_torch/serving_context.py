"""Ambient batching context for serving-profile artifacts.

This is the runtime contract generated serving-profile (torch) models speak:
attention modules call :func:`get_context` to read the per-step batching
metadata the engine owns (slot mappings, block tables, packed sequence
boundaries). The engine calls :func:`set_context` before each forward and
:func:`reset_context` after.

The module must be a process-wide singleton so the engine and the generated
model observe the same context; import it from exactly this module.

The field set and call pattern mirror nano-vllm's
``nanovllm/utils/context.py`` (https://github.com/GeeeekExplorer/nano-vllm,
Copyright (c) 2025 Xingkai Yu — MIT License), adopted here as the artifact
contract for the serving profile.
"""

from dataclasses import dataclass

import torch


@dataclass(slots=True)
class Context:
    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None
    # Per-sequence PrefixLM boundary (absolute prompt-token count) for
    # non-causal prefill artifacts; the paged kernel builds the two-region
    # mask from it. None for causal models.
    prefix_lens: torch.Tensor | None = None


_CONTEXT = Context()


def get_context() -> Context:
    return _CONTEXT


def set_context(  # noqa: ANN001 - mirroring the upstream call signature
    is_prefill,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    max_seqlen_q=0,
    max_seqlen_k=0,
    slot_mapping=None,
    context_lens=None,
    block_tables=None,
    prefix_lens=None,
):
    global _CONTEXT
    _CONTEXT = Context(
        is_prefill,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        slot_mapping,
        context_lens,
        block_tables,
        prefix_lens,
    )


def reset_context() -> None:
    global _CONTEXT
    _CONTEXT = Context()
