"""Quantized checkpoints (mlx.nn.quantize layout) on generated jax cores.

The generated jax core packs MLX-layout uint32 words into the canonical uint8
layout at load (``quantization={'bits', 'group_size'}``) and lowers quantized
matmuls to a pure-jnp group-partial-sum form (fully jittable, no custom
kernel): ``y = sum_g s_g * sum_{k in g} x_k q_k + b_g * sum_{k in g} x_k``.
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from tests.mlx_core_support import generate_core, load_core
from tests.quantized_fixture import (
    affine_quantize,
    dequantize_bytes,
    pack_bytes,
    quantized_state_from_float,
    w8a16_pair,
)

VOCAB, DIM, LABELS = 31, 128, 5
PROGRAM = f"""import NN
import Tensor

DIM = {DIM}
LABELS = {LABELS}
VOCAB = {VOCAB}

probe :: TokenIds[B,S] -> (Tensor[B,S,LABELS], Tensor[B,S,VOCAB])
probe input_ids = do
  x <- NN.embedding@embed input_ids dim=DIM
  tied <- NN.linear@decoder x dim=VOCAB bias=false weight_path=@@embed.weight
  x <- NN.linear@proj x dim=DIM bias=false
  logits <- NN.linear@classifier x dim=LABELS bias=true
  return logits, tied
"""
IDS = jnp.arange(24).reshape(2, 12) % VOCAB
Q8 = {"bits": 8, "group_size": 64}

_FLOAT_STATE = {
    "embed.weight": np.random.default_rng(0).standard_normal((VOCAB, DIM)) * 0.05,
    "proj.weight": np.random.default_rng(1).standard_normal((DIM, DIM)) * 0.05,
    "classifier.weight": np.random.default_rng(2).standard_normal((LABELS, DIM)) * 0.05,
    "classifier.bias": np.random.default_rng(3).standard_normal(LABELS) * 0.05,
}


def _reference(bits, group_size):
    ref = {}
    for key, value in _FLOAT_STATE.items():
        array = np.asarray(value)
        if array.ndim == 2 and array.shape[1] % group_size == 0:
            q, scales, biases = affine_quantize(array, bits, group_size)
            ref[key] = dequantize_bytes(pack_bytes(q, bits), scales, biases, bits, group_size)
        else:
            ref[key] = array.astype(np.float32)
    return ref


def _expected(ref, ids):
    x = ref["embed.weight"][np.asarray(ids)]
    tied = x @ ref["embed.weight"].T
    hidden = x @ ref["proj.weight"].T
    logits = hidden @ ref["classifier.weight"].T + ref["classifier.bias"]
    return logits.astype(np.float32), tied.astype(np.float32)


def _core(tmp_path, state, *, quantization=Q8):
    source = tmp_path / "probe.axon"
    source.write_text(PROGRAM)
    code = generate_core(source, backend="jax", class_name="Core")
    return load_core(code, "Core").from_state_dict(dict(state), quantization=quantization)


@pytest.mark.parametrize("bits,group_size", [(8, 64), (4, 64), (8, 32)])
def test_logits_match_dequantized_reference(tmp_path, bits, group_size):
    state = quantized_state_from_float(_FLOAT_STATE, bits, group_size)
    core = _core(tmp_path, state, quantization={"bits": bits, "group_size": group_size})
    logits, tied = core._def_probe(IDS)
    logits_ref, tied_ref = _expected(_reference(bits, group_size), IDS)
    assert np.allclose(np.asarray(logits, dtype=np.float32), logits_ref, atol=1e-2)
    assert np.allclose(np.asarray(tied, dtype=np.float32), tied_ref, atol=1e-2)
    # The core holds the canonical packed layout, not the MLX uint32 words.
    assert core._flat_tensors["embed.weight.int8"].dtype == jnp.uint8
    assert core._flat_tensors["embed.weight.scale"].dtype == jnp.float16
    assert core._flat_tensors["proj.weight.qbias"].shape == (DIM, DIM // group_size)


def test_jit_matches_eager(tmp_path):
    state = quantized_state_from_float(_FLOAT_STATE, 8, 64)
    core = _core(tmp_path, state)
    eager = core._def_probe(IDS)
    jitted = jax.jit(core._def_probe)(IDS)
    for got, want in zip(jitted, eager, strict=True):
        # XLA fuses the group-partial form differently inside jit; fp16 outputs
        # differ by at most a couple of ulps, not in structure.
        assert np.allclose(np.asarray(got, dtype=np.float32), np.asarray(want, dtype=np.float32), atol=1e-3)


def test_w8a16_checkpoint_requantizes_to_packed(tmp_path):
    state = {}
    for key, value in _FLOAT_STATE.items():
        array = np.asarray(value)
        if array.ndim == 2 and array.shape[1] % 64 == 0:
            q, scale = w8a16_pair(array)
            state[f"{key}.int8"] = q
            state[f"{key}.scale"] = scale
        else:
            state[key] = value
    core = _core(tmp_path, state)
    assert core._flat_tensors["embed.weight.int8"].dtype == jnp.uint8
    logits, tied = core._def_probe(IDS)
    logits_ref, tied_ref = _expected(_reference(8, 64), IDS)
    assert np.allclose(np.asarray(logits, dtype=np.float32), logits_ref, atol=2e-2)
    assert np.allclose(np.asarray(tied, dtype=np.float32), tied_ref, atol=2e-2)


def test_mlx_layout_without_quantization_kwarg_raises(tmp_path):
    state = quantized_state_from_float(_FLOAT_STATE, 8, 64)
    source = tmp_path / "probe.axon"
    source.write_text(PROGRAM)
    code = generate_core(source, backend="jax", class_name="Core")
    with pytest.raises(ValueError, match="is an mlx.nn.quantize weight: pass quantization="):
        load_core(code, "Core").from_state_dict(dict(state))


def test_quantization_without_quantized_weights_raises(tmp_path):
    source = tmp_path / "probe.axon"
    source.write_text(PROGRAM)
    code = generate_core(source, backend="jax", class_name="Core")
    with pytest.raises(ValueError, match="quantization was given, but the state has no quantized weights"):
        load_core(code, "Core").from_state_dict(dict(_FLOAT_STATE), quantization=dict(Q8))


@pytest.mark.parametrize(
    "quantization,match",
    [
        ({"bits": 5, "group_size": 64}, "unsupported quantization bits 5"),
        ({"bits": 8, "group_size": 16}, "unsupported quantization group_size 16"),
        ({"bits": 8, "group_size": 64, "kernel": "triton"}, "unsupported quantization kernel"),
    ],
)
def test_invalid_quantization_settings_raise(tmp_path, quantization, match):
    state = quantized_state_from_float(_FLOAT_STATE, 8, 64)
    source = tmp_path / "probe.axon"
    source.write_text(PROGRAM)
    code = generate_core(source, backend="jax", class_name="Core")
    with pytest.raises(ValueError, match=match):
        load_core(code, "Core").from_state_dict(dict(state), quantization=quantization)


def test_inconsistent_scales_shape_raises(tmp_path):
    state = quantized_state_from_float(_FLOAT_STATE, 8, 64)
    broken = dict(state)
    broken["proj.scales"] = np.asarray(state["proj.scales"])[:, :1].copy()
    source = tmp_path / "probe.axon"
    source.write_text(PROGRAM)
    code = generate_core(source, backend="jax", class_name="Core")
    with pytest.raises(ValueError, match="scales imply"):
        load_core(code, "Core").from_state_dict(dict(broken), quantization=dict(Q8))


def test_reload_with_float_state_resets_quantized_path(tmp_path):
    state = quantized_state_from_float(_FLOAT_STATE, 8, 64)
    core = _core(tmp_path, state)
    assert core._quantized is True
    core.load_state_dict({key: jnp.asarray(value) for key, value in _FLOAT_STATE.items()})
    assert core._quantized is False
    assert "embed.weight" in core._flat_tensors
    logits, tied = core._def_probe(IDS)
    logits_ref, tied_ref = _expected(_FLOAT_STATE, IDS)
    assert np.allclose(np.asarray(logits, dtype=np.float32), logits_ref, atol=1e-4)
    assert np.allclose(np.asarray(tied, dtype=np.float32), tied_ref, atol=1e-4)
