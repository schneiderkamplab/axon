"""Ettin decoder: HF ``ModernBertDecoderForCausalLM``."""

from __future__ import annotations

import weakref

import mlx.core as mx
import mlx.nn as nn

from .attention import Attend, blocked_local_attention, build_mask, dense_attention, use_blocked
from .cache import KVCache, SlidingKVCache
from .config import FULL, SLIDING, EttinConfig
from .layers import Backbone, LMDecoder, PredictionHead, make_rope

Cache = list[KVCache | SlidingKVCache]


class EttinDecoder(nn.Module):
    """Causal LM with full and sliding-window layers and an optional KV cache.

    ``attention_mask`` follows HF: shape ``(B, past + L)``, 1 = token, 0 = padding.
    Positions are ``arange(L) + past`` (HF's default when no ``position_ids`` are given).
    ``local_attention`` and ``rope`` are as in ``EttinEncoder``.
    """

    def __init__(self, config: EttinConfig, local_attention: str = "auto", rope: str = "precise"):
        super().__init__()
        if not config.is_decoder:
            raise ValueError(f"EttinDecoder needs model_type 'modernbert-decoder', got {config.model_type!r}")
        self.config = config
        self.local_attention = local_attention
        self.rope = rope
        self.model = Backbone(config)
        self.lm_head = PredictionHead(config)
        self.decoder = LMDecoder(config)
        object.__setattr__(self, "_compiled", None)

    def make_cache(self) -> Cache:
        w = self.config.sliding_window
        return [SlidingKVCache(w) if t == SLIDING else KVCache() for t in self.config.layer_types]

    def attention_plan(
        self, batch: int, length: int, attention_mask: mx.array | None, cache: Cache | None
    ) -> dict[str, Attend]:
        cfg = self.config
        types = cfg.layer_types
        offset = cache[0].offset if cache else 0
        if attention_mask is not None and attention_mask.shape[-1] != offset + length:
            raise ValueError(f"attention_mask covers {attention_mask.shape[-1]} positions, expected {offset + length}")
        plan = {}
        if FULL in types:
            mask = build_mask(offset, length, 0, offset + length, causal=True, window=None, attention_mask=attention_mask)
            plan[FULL] = dense_attention(mask)
        if SLIDING in types:
            w = cfg.sliding_window
            kv_start = cache[types.index(SLIDING)].kv_start() if cache else 0
            if offset == 0 and use_blocked(self.local_attention, length, w):
                plan[SLIDING] = blocked_local_attention(
                    batch, length, cfg.num_attention_heads,
                    window=w, causal=True, block=w, attention_mask=attention_mask,
                )
            else:
                mask = build_mask(
                    offset, length, kv_start, offset + length - kv_start,
                    causal=True, window=w, attention_mask=attention_mask,
                )
                plan[SLIDING] = dense_attention(mask)
        return plan

    def hidden_states(
        self, input_ids: mx.array, attention_mask: mx.array | None = None, cache: Cache | None = None
    ) -> mx.array:
        B, L = input_ids.shape
        offset = cache[0].offset if cache else 0
        rope = make_rope(self.rope, self.config, offset, L)
        return self.model(input_ids, self.attention_plan(B, L, attention_mask, cache), rope, cache)

    def logits(self, hidden: mx.array) -> mx.array:
        return self.decoder(self.lm_head(hidden), self.model.embeddings.tok_embeddings.weight)

    def _forward(self, input_ids, attention_mask=None, cache=None, logits_to_keep=0):
        hidden = self.hidden_states(input_ids, attention_mask, cache)
        if logits_to_keep:
            hidden = hidden[:, -logits_to_keep:]
        return self.logits(hidden)

    def compile(self) -> EttinDecoder:
        """Route cache-less forward calls through ``mx.compile``.

        Calls with a cache stay eager: the cache is Python-side state that changes
        shape every step, so a compiled decode step would retrace every token.
        """
        me = weakref.ref(self)  # no self-reference cycle: `del model` frees the weights at once
        compiled = {
            "forward": mx.compile(lambda ids: me()._forward(ids)),
            "forward_mask": mx.compile(lambda ids, m: me()._forward(ids, m)),
        }
        object.__setattr__(self, "_compiled", compiled)
        return self

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: mx.array | None = None,
        cache: Cache | None = None,
        logits_to_keep: int = 0,
    ) -> mx.array:
        if self._compiled is None or cache is not None or logits_to_keep:
            return self._forward(input_ids, attention_mask, cache, logits_to_keep)
        if attention_mask is None:
            return self._compiled["forward"](input_ids)
        return self._compiled["forward_mask"](input_ids, attention_mask)
