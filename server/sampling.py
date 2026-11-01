"""Backend-agnostic sampler chain.

Operates on a 1-D numpy logits vector so the same code serves every backend
(torch, MLX, ...). Logits are tiny (``[vocab]``), so the host-side conversion
cost per step is negligible next to the model forward itself.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["SamplerConfig", "sample_next_token"]


@dataclass(frozen=True)
class SamplerConfig:
    """Sampling parameters for one generation.

    ``temperature <= 0`` selects greedy decoding (argmax), which is the
    behavior parity harnesses rely on.
    """

    temperature: float = 1.0
    top_k: int | None = None
    top_p: float | None = None
    seed: int | None = None

    def rng(self) -> np.random.Generator:
        return np.random.default_rng(self.seed)

    @property
    def greedy(self) -> bool:
        return self.temperature is None or self.temperature <= 0.0


def _apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    if temperature == 1.0:
        return logits
    return logits / max(temperature, 1e-8)


def _apply_top_k(logits: np.ndarray, top_k: int) -> np.ndarray:
    if top_k <= 0 or top_k >= logits.shape[-1]:
        return logits
    kth = np.partition(logits, -top_k)[-top_k]
    return np.where(logits < kth, -np.inf, logits)


def _apply_top_p(logits: np.ndarray, top_p: float) -> np.ndarray:
    if not 0.0 < top_p < 1.0:
        return logits
    sorted_indices = np.argsort(logits)[::-1]
    sorted_logits = logits[sorted_indices]
    probabilities = _softmax(sorted_logits)
    cumulative = np.cumsum(probabilities)
    # Keep the smallest prefix whose cumulative probability >= top_p
    # (always keep at least one token).
    cutoff_index = int(np.searchsorted(cumulative, top_p, side="left")) + 1
    keep = sorted_indices[:cutoff_index]
    filtered = np.full_like(logits, -np.inf)
    filtered[keep] = logits[keep]
    return filtered


def _softmax(logits: np.ndarray) -> np.ndarray:
    max_value = np.max(logits)
    if not np.isfinite(max_value):
        # Degenerate logits (e.g. all -inf): uniform over finite entries.
        finite = np.isfinite(logits)
        if not finite.any():
            return np.full_like(logits, 1.0 / logits.shape[-1])
        uniform = np.zeros_like(logits)
        uniform[finite] = 1.0 / int(finite.sum())
        return uniform
    shifted = logits - max_value
    exponents = np.exp(shifted)
    total = exponents.sum()
    if not np.isfinite(total) or total <= 0.0:
        finite = np.isfinite(logits)
        uniform = np.zeros_like(logits)
        uniform[finite] = 1.0 / int(finite.sum())
        return uniform
    return exponents / total


def sample_next_token(
    logits: np.ndarray,
    config: SamplerConfig,
    rng: np.random.Generator | None = None,
) -> int:
    """Sample one token id from a 1-D logits vector.

    Greedy when ``config.temperature <= 0``; otherwise temperature scaling,
    top-k filtering, nucleus (top-p) filtering, then multinomial sampling.
    """
    values = np.asarray(logits, dtype=np.float64).reshape(-1)
    if config.greedy:
        return int(np.argmax(values))
    generator = rng if rng is not None else config.rng()
    values = _apply_temperature(values, float(config.temperature))
    if config.top_k is not None:
        values = _apply_top_k(values, int(config.top_k))
    if config.top_p is not None:
        values = _apply_top_p(values, float(config.top_p))
    probabilities = _softmax(values)
    return int(generator.choice(values.shape[-1], p=probabilities))
