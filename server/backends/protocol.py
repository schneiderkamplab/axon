"""The ``ModelBackend`` protocol: one generated model instance, one tensor library.

The generated model contract (shared by every backend's codegen):

    forward(input_ids, past_kv=..., use_cache=True) -> logits + updated cache

Backends own everything tensor-shaped: building id tensors, calling forward,
normalizing the result, and moving logits into numpy for the sampler. The
``Session`` engine owns the decode loop and touches no tensor library directly.
"""

from __future__ import annotations

from typing import Any, Protocol

__all__ = ["ModelBackend", "extract_cache", "extract_logits", "last_logits"]


def extract_logits(result: Any) -> Any:
    """Pull the logits tensor out of a forward result (dict / tuple / bare)."""
    if isinstance(result, dict):
        return result.get("logits")
    if isinstance(result, (list, tuple)):
        return result[0]
    return result


def extract_cache(result: Any, previous: Any = None) -> Any:
    """Pull the updated KV cache out of a forward result (dict / tuple / bare)."""
    if isinstance(result, dict):
        for key in ("new_kv", "past_kv", "cache"):
            if key in result and result[key] is not None:
                return result[key]
        return previous
    if isinstance(result, (list, tuple)) and len(result) > 1:
        return result[1]
    return previous


def last_logits(logits: Any) -> Any:
    """Reduce ``[B, S, V]`` logits to the final position ``[B, V]``."""
    if logits is None:
        raise RuntimeError("Model forward returned no logits")
    return logits[:, -1, :]


class ModelBackend(Protocol):
    """Drives one generated model's forward interface for the engine."""

    model: Any

    def ids_2d(self, ids: list[int] | list[list[int]]) -> Any:
        """Build a ``[B, S]`` id tensor from prompt rows."""
        ...

    def step_2d(self, token: int) -> Any:
        """Build the ``[1, 1]`` id tensor for one decode step."""
        ...

    def flat_ids(self, ids: Any) -> list[int]:
        """Read a ``[B, S]`` id tensor back as a flat python list."""
        ...

    def seq_len(self, ids: Any) -> int:
        """Sequence length (axis 1) of a ``[B, S]`` id tensor."""
        ...

    def forward(
        self, input_ids: Any, past_kv: Any = None, use_cache: bool = True
    ) -> Any:
        """One stateful forward; returns logits plus the updated cache."""
        ...

    def extract_logits(self, result: Any) -> Any: ...

    def extract_cache(self, result: Any, previous: Any = None) -> Any: ...

    def last_logits(self, logits: Any) -> Any: ...

    def to_numpy(self, values: Any) -> Any:
        """Move logits somewhere numpy can read (sampler input)."""
        ...
