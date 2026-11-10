"""Quantized checkpoints (mlx.nn.quantize layout) on generated torch cores.

The generated torch core packs MLX-layout uint32 words into the canonical
uint8 layout at load (``quantization={'bits', 'group_size', 'kernel'}``) and
runs every linear/embedding against the packed weights: eager dequant matmul
by default, the fused Triton decode GEMV with ``kernel='triton'``.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

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
IDS = torch.arange(24).reshape(2, 12) % VOCAB
Q8 = {"bits": 8, "group_size": 64}

_FLOAT_STATE = {
    "embed.weight": np.random.default_rng(0).standard_normal((VOCAB, DIM)) * 0.05,
    "proj.weight": np.random.default_rng(1).standard_normal((DIM, DIM)) * 0.05,
    "classifier.weight": np.random.default_rng(2).standard_normal((LABELS, DIM)) * 0.05,
    "classifier.bias": np.random.default_rng(3).standard_normal(LABELS) * 0.05,
}


def _reference(bits, group_size):
    """Dequantized float weights (the quantized state's ground truth), fp32."""
    ref = {}
    for key, value in _FLOAT_STATE.items():
        array = np.asarray(value)
        if array.ndim == 2 and array.shape[1] % group_size == 0:
            q, scales, biases = affine_quantize(array, bits, group_size)
            ref[key] = dequantize_bytes(pack_bytes(q, bits), scales, biases, bits, group_size)
        else:
            ref[key] = array
    return ref


def _torch_state(state):
    return {key: torch.from_numpy(np.ascontiguousarray(value)) for key, value in state.items()}


def _expected(ref, ids):
    x = ref["embed.weight"][ids.numpy()]
    tied = x @ ref["embed.weight"].T
    hidden = x @ ref["proj.weight"].T
    logits = hidden @ ref["classifier.weight"].T + ref["classifier.bias"]
    return logits.astype(np.float32), tied.astype(np.float32)


def _core(tmp_path, state, *, optimized=False, quantization=Q8, param_devices=None):
    source = tmp_path / "probe.axon"
    source.write_text(PROGRAM)
    code = generate_core(source, backend="torch", optimize=optimized, class_name="Core")
    return load_core(code, "Core").from_state_dict(_torch_state(state), quantization=quantization, param_devices=param_devices)


@pytest.mark.parametrize("bits,group_size", [(8, 64), (4, 64), (8, 32)])
@pytest.mark.parametrize("optimized", [False, True], ids=["unfused", "fused"])
def test_logits_match_dequantized_reference(tmp_path, optimized, bits, group_size):
    state = quantized_state_from_float(_FLOAT_STATE, bits, group_size)
    core = _core(tmp_path, state, optimized=optimized, quantization={"bits": bits, "group_size": group_size})
    logits, tied = core._def_probe(IDS)
    logits_ref, tied_ref = _expected(_reference(bits, group_size), IDS)
    assert torch.allclose(logits.float(), torch.from_numpy(logits_ref), atol=1e-2)
    assert torch.allclose(tied.float(), torch.from_numpy(tied_ref), atol=1e-2)
    # The core holds the canonical packed layout, not the MLX uint32 words.
    assert core.state_dict_tensors["embed.weight.int8"].dtype == torch.uint8
    assert core.state_dict_tensors["embed.weight.scale"].dtype == torch.float16
    assert core.state_dict_tensors["proj.weight.qbias"].shape == (DIM, DIM // group_size)


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
    assert core.state_dict_tensors["embed.weight.int8"].dtype == torch.uint8
    assert core.state_dict_tensors["embed.weight.scale"].dtype == torch.float16
    logits, tied = core._def_probe(IDS)
    logits_ref, tied_ref = _expected(_reference(8, 64), IDS)
    # Requantization loses a little more than the original W8A16 grid; fp16
    # group-affine dequant keeps the outputs well inside 2e-2 of the reference.
    assert torch.allclose(logits.float(), torch.from_numpy(logits_ref), atol=2e-2)
    assert torch.allclose(tied.float(), torch.from_numpy(tied_ref), atol=2e-2)


def test_mlx_layout_without_quantization_kwarg_raises(tmp_path):
    state = quantized_state_from_float(_FLOAT_STATE, 8, 64)
    source = tmp_path / "probe.axon"
    source.write_text(PROGRAM)
    code = generate_core(source, backend="torch", class_name="Core")
    core_class = load_core(code, "Core")
    with pytest.raises(ValueError, match="is an mlx.nn.quantize weight: pass quantization="):
        core_class.from_state_dict(_torch_state(state))


def test_quantization_without_quantized_weights_raises(tmp_path):
    source = tmp_path / "probe.axon"
    source.write_text(PROGRAM)
    code = generate_core(source, backend="torch", class_name="Core")
    with pytest.raises(ValueError, match="quantization was given, but the state has no quantized weights"):
        load_core(code, "Core").from_state_dict(_torch_state(dict(_FLOAT_STATE)), quantization=dict(Q8))


@pytest.mark.parametrize(
    "quantization,match",
    [
        ({"bits": 5, "group_size": 64}, "unsupported quantization bits 5"),
        ({"bits": 8, "group_size": 16}, "unsupported quantization group_size 16"),
        ({"bits": 8, "group_size": 64, "kernel": "bogus"}, "unsupported quantization kernel"),
    ],
)
def test_invalid_quantization_settings_raise(tmp_path, quantization, match):
    state = quantized_state_from_float(_FLOAT_STATE, 8, 64)
    source = tmp_path / "probe.axon"
    source.write_text(PROGRAM)
    code = generate_core(source, backend="torch", class_name="Core")
    with pytest.raises(ValueError, match=match):
        load_core(code, "Core").from_state_dict(_torch_state(state), quantization=quantization)


def test_inconsistent_scales_shape_raises(tmp_path):
    state = quantized_state_from_float(_FLOAT_STATE, 8, 64)
    # Break the group count: scales claim one group of 128 per row.
    broken = dict(state)
    broken["proj.scales"] = np.asarray(state["proj.scales"])[:, :1].copy()
    source = tmp_path / "probe.axon"
    source.write_text(PROGRAM)
    code = generate_core(source, backend="torch", class_name="Core")
    with pytest.raises(ValueError, match="scales imply"):
        load_core(code, "Core").from_state_dict(_torch_state(broken), quantization=dict(Q8))


def test_reload_with_float_state_resets_quantized_path(tmp_path):
    state = quantized_state_from_float(_FLOAT_STATE, 8, 64)
    core = _core(tmp_path, state)
    assert core._quantized is True
    core.load_state_dict(_torch_state(dict(_FLOAT_STATE)))
    assert core._quantized is False
    assert "embed.weight" in core.state_dict_tensors
    logits, tied = core._def_probe(IDS)
    logits_ref, tied_ref = _expected({k: np.asarray(v) for k, v in _FLOAT_STATE.items()}, IDS)
    assert torch.allclose(logits.float(), torch.from_numpy(logits_ref), atol=1e-4)
    assert torch.allclose(tied.float(), torch.from_numpy(tied_ref), atol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="triton kernel test needs CUDA")
def test_triton_decode_gemv_matches_eager(tmp_path):
    state = quantized_state_from_float(_FLOAT_STATE, 8, 64)
    core = _core(
        tmp_path,
        state,
        quantization={"bits": 8, "group_size": 64, "kernel": "triton"},
        param_devices=["cuda"],
    )
    ids = IDS.to("cuda")
    eager = _core(tmp_path, state, param_devices=["cuda"])
    logits_triton, tied_triton = core._def_probe(ids)
    logits_eager, tied_eager = eager._def_probe(ids)
    assert torch.allclose(logits_triton, logits_eager, atol=1e-3)
    assert torch.allclose(tied_triton, tied_eager, atol=1e-3)
    # The fused kernel is a 1-D decode GEMV; batched inputs are dispatched to the
    # eager dequant path (documented dispatch, not a silent kernel swap).
    batched = torch.ones(4, DIM, device="cuda")
    assert torch.allclose(
        core._quantized_matmul("proj.weight", batched),
        eager._quantized_matmul("proj.weight", batched),
        atol=1e-3,
    )


def test_triton_kernel_without_cuda_raises(tmp_path):
    state = quantized_state_from_float(_FLOAT_STATE, 8, 64)
    source = tmp_path / "probe.axon"
    source.write_text(PROGRAM)
    code = generate_core(source, backend="torch", class_name="Core")
    core_class = load_core(code, "Core")
    if torch.cuda.is_available():
        pytest.skip("CUDA present: the loud-raise path needs a machine without it")
    with pytest.raises(RuntimeError, match="kernel='triton' requires CUDA"):
        core_class.from_state_dict(_torch_state(state), quantization={"bits": 8, "group_size": 64, "kernel": "triton"})
