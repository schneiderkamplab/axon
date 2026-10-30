"""KV caches for the decoder.

``KVCache`` (full-attention layers) keeps every key in a buffer that grows in steps,
written in place. ``SlidingKVCache`` mirrors HF's ``DynamicSlidingWindowLayer``: it
stores the last ``window - 1`` keys and returns them concatenated with the new ones,
which is exactly the set a decoder sliding layer (``q - window < k <= q``) can see
for a single new token. Keys are cached after RoPE.
"""

from __future__ import annotations

import mlx.core as mx


class KVCache:
    step = 256

    def __init__(self):
        self.keys: mx.array | None = None
        self.values: mx.array | None = None
        self.offset = 0

    def kv_start(self) -> int:
        """Absolute position of the first key that ``update_and_fetch`` will return."""
        return 0

    def update_and_fetch(self, keys: mx.array, values: mx.array) -> tuple[mx.array, mx.array]:
        prev, n = self.offset, keys.shape[2]
        if self.keys is None or prev + n > self.keys.shape[2]:
            B, H, _, D = keys.shape
            capacity = ((prev + n + self.step - 1) // self.step) * self.step
            new_k = mx.zeros((B, H, capacity, D), keys.dtype)
            new_v = mx.zeros((B, H, capacity, D), values.dtype)
            if self.keys is not None:
                new_k[..., :prev, :] = self.keys[..., :prev, :]
                new_v[..., :prev, :] = self.values[..., :prev, :]
            self.keys, self.values = new_k, new_v
        self.keys[..., prev : prev + n, :] = keys
        self.values[..., prev : prev + n, :] = values
        self.offset += n
        return self.keys[..., : self.offset, :], self.values[..., : self.offset, :]


class SlidingKVCache:
    def __init__(self, window: int):
        self.keep = window - 1
        self.keys: mx.array | None = None
        self.values: mx.array | None = None
        self.offset = 0

    def kv_start(self) -> int:
        return self.offset - (0 if self.keys is None else self.keys.shape[2])

    def update_and_fetch(self, keys: mx.array, values: mx.array) -> tuple[mx.array, mx.array]:
        self.offset += keys.shape[2]
        if self.keys is not None:
            keys = mx.concatenate([self.keys, keys], axis=2)
            values = mx.concatenate([self.values, values], axis=2)
        start = max(keys.shape[2] - self.keep, 0)
        self.keys, self.values = keys[:, :, start:], values[:, :, start:]
        return keys, values
