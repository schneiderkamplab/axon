"""MLX adapter for the ``ModelBackend`` protocol."""

from __future__ import annotations

from typing import Any

from .protocol import extract_cache, extract_logits, last_logits

__all__ = ["MlxBackend"]


class MlxBackend:
    """Drives a generated MLX model: ``mx`` id tensors, dict/tuple results."""

    def __init__(self, model: Any, *, warm_compile: bool = True, max_kv_length: int = 2048) -> None:
        self.model = model
        if warm_compile and hasattr(model, "compile"):
            model.compile(max_kv_length=max_kv_length)

    def _mx(self) -> Any:
        import mlx.core as mx

        return mx

    def ids_2d(self, ids: list[int] | list[list[int]]) -> Any:
        mx = self._mx()
        if ids and isinstance(ids[0], (list, tuple)):
            return mx.array([list(row) for row in ids], dtype=mx.int64)
        return mx.array([list(ids)], dtype=mx.int64)

    def step_2d(self, token: int) -> Any:
        mx = self._mx()
        return mx.array([[int(token)]], dtype=mx.int64)

    def flat_ids(self, ids: Any) -> list[int]:
        return [int(value) for value in list(ids.reshape(-1))]

    def seq_len(self, ids: Any) -> int:
        return int(ids.shape[1])

    def forward(self, input_ids: Any, past_kv: Any = None, use_cache: bool = True) -> Any:
        return self.model.forward(input_ids, past_kv=past_kv, use_cache=use_cache)

    def extract_logits(self, result: Any) -> Any:
        return extract_logits(result)

    def extract_cache(self, result: Any, previous: Any = None) -> Any:
        return extract_cache(result, previous)

    def last_logits(self, logits: Any) -> Any:
        return last_logits(logits)

    def to_numpy(self, values: Any) -> Any:
        import numpy as np

        return np.asarray(values)
