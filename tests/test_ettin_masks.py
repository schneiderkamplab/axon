"""The Axon mask builtins' ``window=`` boundary equals HF masking_utils for ModernBERT/Ettin.

HF keeps |i - j| <= sliding_window for the bidirectional encoder mask and
0 <= i - j < sliding_window for the causal decoder mask, with sliding_window =
local_attention // 2. The models pass ``window=local_attention / 2`` to the builtins.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from transformers import ModernBertConfig, ModernBertDecoderConfig
from transformers.masking_utils import (
    create_bidirectional_sliding_window_mask,
    create_sliding_window_causal_mask,
    sdpa_mask,
    sliding_window_causal_mask_function,
)

from tests.mlx_core_support import generate_core, load_core

mx = pytest.importorskip("mlx.core")

PROBE = """\
import Masking (bidirectional_mask_for_input, causal_mask_for_input)
import Config
import Tensor

WINDOW = (Config.int @@local_attention default=128) / 2

mask_probe :: TokenIds[B,Q] -> Tensor[B,K] -> Dim -> (Tensor[B,1,Q,Q], Tensor[B,1,Q,K])
mask_probe ids padding key_len = do
  bidirectional <- bidirectional_mask_for_input ids Q padding_mask=padding window=WINDOW
  causal <- causal_mask_for_input ids key_len padding_mask=padding window=WINDOW
  return bidirectional, causal
"""
LENGTH = 300
LOCAL_ATTENTION = 128


@pytest.fixture(scope="module")
def probe(tmp_path_factory):
    directory = tmp_path_factory.mktemp("mask_probe")
    source = directory / "mask_probe.axon"
    source.write_text(PROBE)
    return load_core(generate_core(source, class_name="MaskProbe"), "MaskProbe")({}, config={"local_attention": LOCAL_ATTENTION})


def padding_mask():
    mask = np.ones((2, LENGTH), dtype=np.int64)
    mask[1, 211:] = 0
    return mask


def test_bidirectional_window_matches_hf(probe):
    mask = padding_mask()
    axon, _ = probe._def_mask_probe(mx.zeros(mask.shape, dtype=mx.int64), mx.array(mask), LENGTH)
    config = ModernBertConfig(local_attention=LOCAL_ATTENTION)
    config._attn_implementation = "eager"
    hf = create_bidirectional_sliding_window_mask(config, torch.zeros(2, LENGTH, 8), torch.from_numpy(mask)) == 0
    assert config.sliding_window == 64
    assert np.array_equal(np.array(axon), hf.numpy())


def test_causal_window_matches_hf(probe):
    mask = padding_mask()
    _, axon = probe._def_mask_probe(mx.zeros(mask.shape, dtype=mx.int64), mx.array(mask), LENGTH)
    config = ModernBertDecoderConfig(local_attention=LOCAL_ATTENTION)
    config._attn_implementation = "eager"
    positions = torch.arange(LENGTH)[None].expand(2, -1)
    hf = create_sliding_window_causal_mask(config, torch.zeros(2, LENGTH, 8), torch.from_numpy(mask), None, positions) == 0
    assert config.sliding_window == 64
    assert np.array_equal(np.array(axon), hf.numpy())


@pytest.mark.parametrize("key_len", [64, 65, 128, 129, 300])
def test_causal_window_with_cache_offset_matches_hf(probe, key_len):
    """One query at position key_len - 1 over a cache of key_len keys (a decode step)."""
    _, axon = probe._def_mask_probe(mx.zeros((1, 1), dtype=mx.int64), mx.ones((1, key_len), dtype=mx.int64), key_len)
    hf = sdpa_mask(
        batch_size=1,
        q_length=1,
        kv_length=key_len,
        q_offset=key_len - 1,
        mask_function=sliding_window_causal_mask_function(LOCAL_ATTENTION // 2),
        allow_is_causal_skip=False,
    )
    assert int(np.array(axon).sum()) == min(key_len, LOCAL_ATTENTION // 2)
    assert np.array_equal(np.array(axon), hf.numpy())
