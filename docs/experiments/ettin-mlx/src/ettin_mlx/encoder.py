"""Ettin encoder: HF ``ModernBertForMaskedLM``."""

from __future__ import annotations

import weakref

import mlx.core as mx
import mlx.nn as nn

from .attention import Attend, blocked_local_attention, build_mask, dense_attention, use_blocked
from .config import FULL, SLIDING, EttinConfig
from .layers import Backbone, LMDecoder, PredictionHead, make_rope


class EttinEncoder(nn.Module):
    """Bidirectional encoder with an MLM head.

    ``encode`` returns the last hidden state (after ``final_norm``), ``__call__`` the MLM
    logits. Inputs are ``(B, L)`` token ids plus an optional right-padding mask
    (1 = token, 0 = pad). Position ids are ``arange(L)`` regardless of padding, as in HF.

    ``local_attention`` selects the sliding-layer kernel: ``"dense"`` (masked SDPA over
    the whole sequence), ``"blocked"`` (O(L * window)) or ``"auto"``. ``rope`` selects
    ``"precise"`` (HF float32 tables) or ``"fast"`` (``mx.fast.rope``).
    """

    def __init__(self, config: EttinConfig, local_attention: str = "auto", rope: str = "precise"):
        super().__init__()
        if config.is_decoder:
            raise ValueError(f"EttinEncoder needs model_type 'modernbert', got {config.model_type!r}")
        self.config = config
        self.local_attention = local_attention
        self.rope = rope
        self.model = Backbone(config)
        self.head = PredictionHead(config)
        self.decoder = LMDecoder(config)
        # Plain attribute (not a module child): MLX would treat a dict as a subtree.
        object.__setattr__(self, "_compiled", None)

    def attention_plan(self, batch: int, length: int, attention_mask: mx.array | None) -> dict[str, Attend]:
        cfg = self.config
        plan = {}
        if FULL in cfg.layer_types:
            mask = build_mask(0, length, 0, length, causal=False, window=None, attention_mask=attention_mask)
            plan[FULL] = dense_attention(mask)
        if SLIDING in cfg.layer_types:
            w = cfg.sliding_window
            if use_blocked(self.local_attention, length, w):
                plan[SLIDING] = blocked_local_attention(
                    batch, length, cfg.num_attention_heads,
                    window=w, causal=False, block=w, attention_mask=attention_mask,
                )
            else:
                mask = build_mask(0, length, 0, length, causal=False, window=w, attention_mask=attention_mask)
                plan[SLIDING] = dense_attention(mask)
        return plan

    def _encode(self, input_ids: mx.array, attention_mask: mx.array | None = None) -> mx.array:
        B, L = input_ids.shape
        rope = make_rope(self.rope, self.config, 0, L)
        return self.model(input_ids, self.attention_plan(B, L, attention_mask), rope)

    def logits(self, hidden: mx.array) -> mx.array:
        """MLM logits for (a subset of) hidden states, e.g. only the masked positions."""
        return self.decoder(self.head(hidden), self.model.embeddings.tok_embeddings.weight)

    def _forward(self, input_ids: mx.array, attention_mask: mx.array | None = None) -> mx.array:
        return self.logits(self._encode(input_ids, attention_mask))

    def compile(self) -> EttinEncoder:
        """Route ``encode``/``__call__`` through ``mx.compile`` (one trace per input shape)."""
        me = weakref.ref(self)  # no self-reference cycle: `del model` frees the weights at once
        compiled = {
            "encode": mx.compile(lambda ids: me()._encode(ids)),
            "encode_mask": mx.compile(lambda ids, m: me()._encode(ids, m)),
            "forward": mx.compile(lambda ids: me()._forward(ids)),
            "forward_mask": mx.compile(lambda ids, m: me()._forward(ids, m)),
        }
        object.__setattr__(self, "_compiled", compiled)
        return self

    def encode(self, input_ids: mx.array, attention_mask: mx.array | None = None) -> mx.array:
        return self._dispatch("encode", self._encode, input_ids, attention_mask)

    def __call__(self, input_ids: mx.array, attention_mask: mx.array | None = None) -> mx.array:
        return self._dispatch("forward", self._forward, input_ids, attention_mask)

    def _dispatch(self, name, fn, input_ids, attention_mask):
        if self._compiled is None:
            return fn(input_ids, attention_mask)
        if attention_mask is None:
            return self._compiled[name](input_ids)
        return self._compiled[name + "_mask"](input_ids, attention_mask)
