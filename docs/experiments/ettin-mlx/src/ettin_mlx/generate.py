"""Prefill/decode loop for the decoder: greedy, temperature and top-p sampling."""

from __future__ import annotations

from collections.abc import Iterator

import mlx.core as mx

from .decoder import EttinDecoder


def top_p_filter(logits: mx.array, top_p: float) -> mx.array:
    """HF ``TopPLogitsWarper``: drop the least likely tokens whose cumulative probability
    is ``<= 1 - top_p``; the most likely token is always kept."""
    order = mx.argsort(logits, axis=-1)  # ascending
    sorted_logits = mx.take_along_axis(logits, order, axis=-1)
    cumulative = mx.cumsum(mx.softmax(sorted_logits, axis=-1), axis=-1)
    remove_sorted = cumulative <= (1.0 - top_p)
    remove_sorted[..., -1] = False
    remove = mx.put_along_axis(mx.zeros_like(remove_sorted), order, remove_sorted, axis=-1)
    return mx.where(remove, -mx.inf, logits)


def sample(logits: mx.array, temperature: float = 0.0, top_p: float = 1.0, key: mx.array | None = None) -> mx.array:
    """``(B, V)`` logits -> ``(B,)`` token ids. ``temperature == 0`` is greedy."""
    if temperature == 0:
        return mx.argmax(logits, axis=-1)
    logits = logits.astype(mx.float32) / temperature
    if top_p < 1.0:
        logits = top_p_filter(logits, top_p)
    return mx.random.categorical(logits, key=key)


def generate_step(
    model: EttinDecoder,
    input_ids: mx.array,
    max_new_tokens: int,
    *,
    attention_mask: mx.array | None = None,
    temperature: float = 0.0,
    top_p: float = 1.0,
    seed: int | None = None,
) -> Iterator[mx.array]:
    """Yield ``(B,)`` token arrays: one prefill pass over the prompt, then one token per step.

    The next step is scheduled with ``mx.async_eval`` before the current token is
    yielded, so graph construction overlaps with GPU work.
    """
    cache = model.make_cache()
    key = mx.random.key(seed) if seed is not None else None
    batch = input_ids.shape[0]

    def step(ids):
        nonlocal key, attention_mask
        logits = model(ids, attention_mask=attention_mask, cache=cache, logits_to_keep=1)[:, -1]
        sub = None
        if key is not None:
            key, sub = mx.random.split(key)
        return sample(logits, temperature, top_p, sub)

    token = step(input_ids)
    mx.async_eval(token)
    for n in range(max_new_tokens):
        if n + 1 < max_new_tokens:
            if attention_mask is not None:
                attention_mask = mx.concatenate([attention_mask, mx.ones((batch, 1), attention_mask.dtype)], axis=1)
            next_token = step(token[:, None])
            mx.async_eval(next_token)
        yield token
        if n + 1 < max_new_tokens:
            token = next_token


def generate(
    model: EttinDecoder,
    prompts: list[int] | list[list[int]],
    max_new_tokens: int = 64,
    *,
    temperature: float = 0.0,
    top_p: float = 1.0,
    eos_token_id: int | None = None,
    pad_token_id: int = 0,
    seed: int | None = None,
) -> list[list[int]]:
    """Generate continuations for one prompt or a batch of token-id prompts.

    Prompts of different lengths are left-padded and masked. RoPE then sees row ``b``
    shifted by its pad count; attention depends only on relative positions, so this
    matches HF's ``cumsum``-based position ids up to float rounding.
    Returns the new tokens per prompt, cut after the first ``eos_token_id``.
    """
    if prompts and isinstance(prompts[0], int):
        prompts = [prompts]
    if not prompts or any(len(p) == 0 for p in prompts):
        raise ValueError("every prompt needs at least one token (e.g. the BOS token)")
    length = max(len(p) for p in prompts)
    ids = mx.array([[pad_token_id] * (length - len(p)) + list(p) for p in prompts], dtype=mx.int32)
    mask = None
    if any(len(p) != length for p in prompts):
        mask = mx.array([[0] * (length - len(p)) + [1] * len(p) for p in prompts], dtype=mx.int32)

    out: list[list[int]] = [[] for _ in prompts]
    done = [False] * len(prompts)
    for token in generate_step(
        model, ids, max_new_tokens, attention_mask=mask, temperature=temperature, top_p=top_p, seed=seed
    ):
        for row, t in enumerate(token.tolist()):
            if not done[row]:
                out[row].append(t)
                done[row] = t == eos_token_id
        if all(done):
            break
    return out
