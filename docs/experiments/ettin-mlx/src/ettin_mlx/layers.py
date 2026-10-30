"""Building blocks shared by the encoder and the decoder.

Module and parameter names mirror the Hugging Face ModernBERT implementation so that
checkpoint keys map onto them one-to-one (see ``loading.py``).
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

import mlx.core as mx
import mlx.nn as nn

from .config import EttinConfig

# HF ACT2FN name -> MLX function. "gelu" is the exact (erf) GELU in HF.
ACTIVATIONS: dict[str, Callable[[mx.array], mx.array]] = {
    "gelu": nn.gelu,
    "gelu_new": nn.gelu_approx,
    "gelu_pytorch_tanh": nn.gelu_approx,
    "gelu_fast": nn.gelu_approx,
    "quick_gelu": nn.gelu_fast_approx,
    "silu": nn.silu,
    "swish": nn.silu,
    "relu": nn.relu,
}


def get_activation(name: str) -> Callable[[mx.array], mx.array]:
    try:
        return ACTIVATIONS[name]
    except KeyError:
        raise NotImplementedError(f"activation {name!r} is not supported") from None


def layer_norm(config: EttinConfig) -> nn.LayerNorm:
    # nn.LayerNorm calls mx.fast.layer_norm, which accumulates in float32.
    return nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=config.norm_bias)


class Embeddings(nn.Module):
    def __init__(self, config: EttinConfig):
        super().__init__()
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.norm = layer_norm(config)

    def __call__(self, input_ids: mx.array) -> mx.array:
        return self.norm(self.tok_embeddings(input_ids))


class MLP(nn.Module):
    """GeGLU: ``Wo(act(input) * gate)`` where ``input, gate = Wi(x).chunk(2)``."""

    def __init__(self, config: EttinConfig):
        super().__init__()
        self.Wi = nn.Linear(config.hidden_size, 2 * config.intermediate_size, bias=config.mlp_bias)
        self.Wo = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.mlp_bias)
        self.act = get_activation(config.hidden_activation)

    def __call__(self, x: mx.array) -> mx.array:
        input, gate = mx.split(self.Wi(x), 2, axis=-1)
        return self.Wo(self.act(input) * gate)


class PredictionHead(nn.Module):
    """``norm(act(dense(x)))``; HF ``head`` (encoder) / ``lm_head`` (decoder)."""

    def __init__(self, config: EttinConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size, bias=config.classifier_bias)
        self.norm = layer_norm(config)
        self.act = get_activation(config.classifier_activation)

    def __call__(self, x: mx.array) -> mx.array:
        return self.norm(self.act(self.dense(x)))


class LMDecoder(nn.Module):
    """Vocabulary projection. With tied embeddings it owns only the optional bias and
    reads the embedding matrix by reference at call time."""

    def __init__(self, config: EttinConfig):
        super().__init__()
        if not config.tie_word_embeddings:
            self.weight = mx.zeros((config.vocab_size, config.hidden_size))
        if config.decoder_bias:
            self.bias = mx.zeros((config.vocab_size,))

    def __call__(self, x: mx.array, tied_weight: mx.array) -> mx.array:
        weight = self.weight if "weight" in self else tied_weight
        if "bias" in self:
            return mx.addmm(self.bias, x, weight.T)
        return x @ weight.T


class Attention(nn.Module):
    """Multi-head self-attention with a fused ``Wqkv`` projection and rotate-half RoPE.

    The decoder checkpoints store separate ``q_proj``/``k_proj``/``v_proj``; the loader
    concatenates them into ``Wqkv`` so both models run the same code.
    """

    def __init__(self, config: EttinConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim**-0.5
        self.Wqkv = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=config.attention_bias)
        self.Wo = nn.Linear(config.hidden_size, config.hidden_size, bias=config.attention_bias)

    def __call__(self, x: mx.array, attend: Callable, rope: Callable, cache=None) -> mx.array:
        B, L, _ = x.shape
        qkv = self.Wqkv(x).reshape(B, L, 3, self.num_heads, self.head_dim).transpose(2, 0, 3, 1, 4)
        q, k, v = rope(qkv[0]), rope(qkv[1]), qkv[2]
        if cache is not None:
            k, v = cache.update_and_fetch(k, v)
        return self.Wo(attend(q, k, v, self.scale))


class TransformerLayer(nn.Module):
    """Pre-norm block. Layer 0 has no ``attn_norm`` (Identity in HF)."""

    def __init__(self, config: EttinConfig, layer_idx: int):
        super().__init__()
        self.layer_type = config.layer_types[layer_idx]
        if layer_idx != 0:
            self.attn_norm = layer_norm(config)
        self.attn = Attention(config)
        self.mlp_norm = layer_norm(config)
        self.mlp = MLP(config)

    def __call__(self, x: mx.array, attend: Callable, rope: Callable, cache=None) -> mx.array:
        h = self.attn_norm(x) if "attn_norm" in self else x
        x = x + self.attn(h, attend, rope, cache)
        return x + self.mlp(self.mlp_norm(x))


class Backbone(nn.Module):
    """HF ``ModernBertModel`` / ``ModernBertDecoderModel`` minus the mask logic."""

    def __init__(self, config: EttinConfig):
        super().__init__()
        self.embeddings = Embeddings(config)
        self.layers = [TransformerLayer(config, i) for i in range(config.num_hidden_layers)]
        self.final_norm = layer_norm(config)

    def __call__(self, input_ids: mx.array, attend: dict[str, Callable], rope: dict[str, Callable], cache=None):
        x = self.embeddings(input_ids)
        for i, layer in enumerate(self.layers):
            t = layer.layer_type
            x = layer(x, attend[t], rope[t], None if cache is None else cache[i])
        return self.final_norm(x)


# --------------------------------------------------------------------------- RoPE

ROPE_IMPLS = ("precise", "fast")


@mx.compile
def _rotate(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    half = x.shape[-1] // 2
    xf = x.astype(mx.float32)
    x1, x2 = xf[..., :half], xf[..., half:]
    return mx.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1).astype(x.dtype)


def rope_tables(base: float, head_dim: int, offset: int, length: int) -> tuple[mx.array, mx.array]:
    """``cos``/``sin`` of shape ``(length, head_dim // 2)`` computed like HF in float32:
    ``inv_freq = 1 / base ** (arange(0, d, 2) / d)``, ``angle = position * inv_freq``."""
    inv_freq = 1.0 / (base ** (mx.arange(0, head_dim, 2, dtype=mx.float32) / head_dim))
    angle = mx.arange(offset, offset + length, dtype=mx.float32)[:, None] * inv_freq[None, :]
    return mx.cos(angle), mx.sin(angle)


def make_rope(impl: str, config: EttinConfig, offset: int, length: int) -> dict[str, Callable]:
    """One rotate-half RoPE callable per layer type, built once per forward call.

    ``precise`` matches HF's float32 cos/sin tables; ``fast`` uses ``mx.fast.rope``,
    whose fast-math trig drifts by ~1e-3 at positions in the thousands (see README).
    """
    d = config.head_dim
    ropes = {}
    for layer_type in set(config.layer_types):
        base = config.rope_theta[layer_type]
        if impl == "fast":
            ropes[layer_type] = partial(mx.fast.rope, dims=d, traditional=False, base=base, scale=1.0, offset=offset)
        elif impl == "precise":
            cos, sin = rope_tables(base, d, offset, length)
            ropes[layer_type] = lambda x, cos=cos, sin=sin: _rotate(x, cos, sin)
        else:
            raise ValueError(f"rope must be one of {ROPE_IMPLS}, got {impl!r}")
    return ropes
