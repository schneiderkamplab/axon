"""Model configuration parsed from a Hugging Face ``config.json``.

Both config generations are accepted:

* old (what the Ettin checkpoints ship): ``global_attn_every_n_layers``,
  ``global_rope_theta``, ``local_rope_theta``;
* new (transformers >= 5): ``layer_types`` and ``rope_parameters`` keyed by layer type.

Defaults mirror ``ModernBertConfig`` / ``ModernBertDecoderConfig`` in transformers 5.x.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ENCODER_MODEL_TYPE = "modernbert"
DECODER_MODEL_TYPE = "modernbert-decoder"
FULL = "full_attention"
SLIDING = "sliding_attention"

# transformers' ModernBert*Config.default_theta
_DEFAULT_THETA = {FULL: 160_000.0, SLIDING: 10_000.0}


@dataclass(frozen=True)
class EttinConfig:
    model_type: str
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    layer_types: tuple[str, ...]
    rope_theta: dict[str, float]
    local_attention: int
    hidden_activation: str = "gelu"
    classifier_activation: str = "gelu"
    norm_eps: float = 1e-5
    norm_bias: bool = False
    attention_bias: bool = False
    mlp_bias: bool = False
    classifier_bias: bool = False
    decoder_bias: bool = True
    tie_word_embeddings: bool = True
    max_position_embeddings: int = 8192
    pad_token_id: int | None = 50283
    bos_token_id: int | None = 50281
    eos_token_id: int | list[int] | None = 50282
    cls_token_id: int | None = 50281
    sep_token_id: int | None = 50282
    raw: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    @property
    def is_decoder(self) -> bool:
        return self.model_type == DECODER_MODEL_TYPE

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def sliding_window(self) -> int:
        """Half of ``local_attention`` (64 for 128), exactly as HF defines it.

        Encoder sliding layers attend to keys with ``|q - k| <= sliding_window``;
        decoder sliding layers to keys with ``q - sliding_window < k <= q``.
        """
        return self.local_attention // 2

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EttinConfig:
        model_type = d.get("model_type")
        if model_type not in (ENCODER_MODEL_TYPE, DECODER_MODEL_TYPE):
            raise ValueError(f"unsupported model_type {model_type!r}")
        if model_type == DECODER_MODEL_TYPE and not d.get("is_causal", True):
            # HF would silently switch the decoder to bidirectional masks; a KV cache
            # makes no sense then, so refuse instead of guessing.
            raise NotImplementedError("modernbert-decoder with is_causal=False is not supported")

        hidden = d.get("hidden_size", 768)
        heads = d.get("num_attention_heads", 12)
        if hidden % heads:
            raise ValueError(f"hidden_size {hidden} is not a multiple of num_attention_heads {heads}")
        n_layers = d.get("num_hidden_layers", 22)

        layer_types = d.get("layer_types")
        if layer_types is None:
            every = d.get("global_attn_every_n_layers", 3)
            layer_types = [SLIDING if i % every else FULL for i in range(n_layers)]
        layer_types = tuple(layer_types)
        if len(layer_types) != n_layers or not set(layer_types) <= {FULL, SLIDING}:
            raise ValueError(f"bad layer_types {layer_types!r} for {n_layers} layers")

        local_attention = d.get("local_attention", 128)
        if SLIDING in layer_types and (local_attention or 0) < 2:
            # sliding_window = local_attention // 2 must be >= 1, or every query is masked out
            raise ValueError(f"sliding_attention layers need local_attention >= 2, got {local_attention!r}")

        return cls(
            model_type=model_type,
            vocab_size=d.get("vocab_size", 50368),
            hidden_size=hidden,
            intermediate_size=int(d.get("intermediate_size", 1152)),
            num_hidden_layers=n_layers,
            num_attention_heads=heads,
            layer_types=layer_types,
            rope_theta=_parse_rope_theta(d),
            local_attention=local_attention or 0,
            hidden_activation=d.get("hidden_activation", "gelu"),
            classifier_activation=d.get("classifier_activation", "gelu"),
            norm_eps=d.get("norm_eps", 1e-5),
            norm_bias=d.get("norm_bias", False),
            attention_bias=d.get("attention_bias", False),
            mlp_bias=d.get("mlp_bias", False),
            classifier_bias=d.get("classifier_bias", False),
            decoder_bias=d.get("decoder_bias", True),
            tie_word_embeddings=d.get("tie_word_embeddings", True),
            max_position_embeddings=d.get("max_position_embeddings", 8192),
            pad_token_id=d.get("pad_token_id", 50283),
            bos_token_id=d.get("bos_token_id", 50281),
            eos_token_id=d.get("eos_token_id", 50282),
            cls_token_id=d.get("cls_token_id", 50281),
            sep_token_id=d.get("sep_token_id", 50282),
            raw=dict(d),
        )

    @classmethod
    def from_file(cls, path: str | Path) -> EttinConfig:
        with open(path) as f:
            return cls.from_dict(json.load(f))


def _parse_rope_theta(d: dict[str, Any]) -> dict[str, float]:
    """Per-layer-type RoPE base, following ``convert_rope_params_to_dict`` in HF."""
    params = {k: dict(v) for k, v in (d.get("rope_parameters") or {}).items() if v is not None}
    scaling = d.get("rope_scaling")
    if scaling:
        for kind in (FULL, SLIDING):
            params.setdefault(kind, {}).update(scaling)
    legacy = {FULL: d.get("global_rope_theta"), SLIDING: d.get("local_rope_theta")}
    theta = {}
    for kind in (FULL, SLIDING):
        p = params.get(kind, {})
        rope_type = p.get("rope_type", p.get("type", "default"))
        if rope_type != "default":
            raise NotImplementedError(f"rope_type {rope_type!r} for {kind} is not supported")
        theta[kind] = float(p.get("rope_theta", legacy[kind] if legacy[kind] is not None else _DEFAULT_THETA[kind]))
    return theta
