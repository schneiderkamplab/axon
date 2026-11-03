# Vendored from nano-vllm (https://github.com/GeeeekExplorer/nano-vllm)
# Copyright (c) 2025 Xingkai Yu — MIT License.
# Modified: ``@torch.compile`` dropped for v1 (compile cost on CPU tests and
# host-portable paths; re-enable behind a config flag when CUDA-graph capture
# returns). Extended beyond upstream: exact greedy for temperature <= 0 rows
# (argmax, no noise — the OpenAI contract) and per-row top_k / top_p masking
# with the same semantics as the Session sampler (``server/sampling.py``),
# which is the oracle in the tests. This module is the fleet sampler's source
# of truth; ``seed`` stays deferred until a per-request RNG threads through
# the engine.
import torch
from torch import nn


class Sampler(nn.Module):

    def forward(
        self,
        logits: torch.Tensor,
        temperatures: torch.Tensor,
        top_ks: torch.Tensor | None = None,
        top_ps: torch.Tensor | None = None,
        *,
        all_greedy: bool | None = None,
        any_greedy: bool | None = None,
        all_topk_disabled: bool | None = None,
        all_topp_disabled: bool | None = None,
    ) -> torch.Tensor:
        """Sample one token per row.

        The ``all_*``/``any_*`` flags are the host-side decisions the runner
        already knows from the request params. Passing them skips the
        device-side ``bool(tensor.all())`` checks — each of those is a full
        MPS/CUDA pipeline sync, and at step end they would serialize after
        the logits and before the token readback. ``None`` falls back to the
        device-side check for direct callers; results are identical either
        way, only the sync count differs.
        """
        logits = logits.float()
        if all_greedy is None:
            all_greedy = bool((temperatures <= 0.0).all())
        if all_greedy:
            return logits.argmax(dim=-1)
        greedy = temperatures <= 0.0

        # Scale by temperature for the sampled rows; greedy rows are clamped
        # to 1 here and overridden with the exact argmax below (dividing by
        # ~0 would poison the batched softmax).
        safe_t = temperatures.clamp_min(1e-10).unsqueeze(dim=1)
        logits = logits.div_(safe_t)
        if top_ks is not None:
            logits = _apply_top_k(logits, top_ks, all_disabled=all_topk_disabled)
        if top_ps is not None:
            logits = _apply_top_p(logits, top_ps, all_disabled=all_topp_disabled)
        probs = torch.softmax(logits, dim=-1)
        sampled = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        if any_greedy is None:
            any_greedy = bool(greedy.any())
        if any_greedy:
            sampled = torch.where(greedy, logits.argmax(dim=-1), sampled)
        return sampled


def _apply_top_k(
    logits: torch.Tensor, top_ks: torch.Tensor, *, all_disabled: bool | None = None
) -> torch.Tensor:
    """Per-row top-k: keep the k largest logits, mask the rest to -inf.

    Rows with ``top_k <= 0`` (or >= vocab) are untouched, matching the
    Session sampler's ``top_k is None`` / ``top_k <= 0`` semantics.
    ``all_disabled`` is the host-side shortcut (see :meth:`Sampler.forward`).
    """
    vocab = logits.shape[-1]
    if all_disabled is None:
        ks = torch.where(top_ks <= 0, torch.full_like(top_ks, vocab), top_ks).clamp_(max=vocab)
        all_disabled = bool((ks >= vocab).all())
    if all_disabled:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)
    threshold = sorted_logits.gather(1, (ks - 1).unsqueeze(1))  # [B, 1]
    remove_sorted = sorted_logits < threshold  # ties at the threshold survive
    remove = torch.zeros_like(remove_sorted).scatter(1, sorted_indices, remove_sorted)
    return logits.masked_fill(remove, float("-inf"))


def _apply_top_p(
    logits: torch.Tensor, top_ps: torch.Tensor, *, all_disabled: bool | None = None
) -> torch.Tensor:
    """Per-row nucleus: keep the smallest prefix of the softmax holding >= top_p.

    Rows with ``top_p <= 0`` keep only the top token; ``top_p >= 1`` keeps
    everything (same boundaries as the Session sampler). ``all_disabled``
    is the host-side shortcut (see :meth:`Sampler.forward`).
    """
    vocab = logits.shape[-1]
    if all_disabled is None:
        all_disabled = bool((top_ps >= 1.0).all())
    if all_disabled:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)
    cumulative = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
    # Keep every position whose prefix (excluding itself) is still below
    # top_p — i.e. the first token that crosses the threshold is kept:
    # #strictly-below + 1.
    keep_count = ((cumulative < top_ps.unsqueeze(1)).sum(dim=-1, keepdim=True) + 1).clamp_(min=1, max=vocab)
    remove_sorted = torch.arange(vocab, device=logits.device).unsqueeze(0) >= keep_count
    remove = torch.zeros_like(remove_sorted).scatter(1, sorted_indices, remove_sorted)
    return logits.masked_fill(remove, float("-inf"))
