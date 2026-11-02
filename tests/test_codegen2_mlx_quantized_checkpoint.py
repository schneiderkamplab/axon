"""``mlx.nn.quantize`` checkpoints on generated MLX cores, against ``mlx.nn``'s own quantized layers."""

from __future__ import annotations

import pytest

from tests.mlx_core_support import generate_core, load_core

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")
tree_flatten = pytest.importorskip("mlx.utils").tree_flatten

VOCAB, DIM, LABELS = 31, 64, 5
PROGRAM = f"""import NN
import Tensor

DIM = {DIM}
LABELS = {LABELS}
VOCAB = {VOCAB}

probe :: TokenIds[B,S] -> (Tensor[B,S,LABELS], Tensor[B,S,VOCAB])
probe input_ids = do
  x <- NN.embedding@embed input_ids dim=DIM
  x <- for@tables i <- [0..2) carry (x) do
    y <- NN.embedding@table input_ids dim=DIM
    x <- x + y
    yield x
  tied <- NN.linear@decoder x dim=VOCAB bias=false weight_path=@@embed.weight
  x <- NN.linear@proj x dim=DIM bias=false
  logits <- NN.linear@classifier x dim=LABELS bias=true
  return logits, tied
"""
IDS = mx.arange(24).reshape(2, 12) % VOCAB
Q8 = {"bits": 8, "group_size": 64}


class Reference(nn.Module):
    """``PROGRAM`` in ``mlx.nn``: a static path, a looped path, a tied decoder and a linear bias."""

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, DIM)
        self.tables = [{"table": nn.Embedding(VOCAB, DIM)} for _ in range(2)]
        self.proj = nn.Linear(DIM, DIM, bias=False)
        self.classifier = nn.Linear(DIM, LABELS)

    def __call__(self, ids):
        x = self.embed(ids)
        for layer in self.tables:
            x = x + layer["table"](ids)
        return self.classifier(self.proj(x)), self.embed.as_linear(x)


def _float():
    mx.random.seed(1729)
    model = Reference()
    return model, dict(tree_flatten(model.parameters()))


def _quantized(bits=8, group_size=64):
    model, _ = _float()
    nn.quantize(model, group_size=group_size, bits=bits, class_predicate=lambda _, m: isinstance(m, (nn.Linear, nn.Embedding)))
    return model, dict(tree_flatten(model.parameters()))


def _core_class(tmp_path, optimized=False):
    source = tmp_path / "probe.axon"
    source.write_text(PROGRAM)
    intrinsics = "codegen2-mlx" if optimized else None
    return load_core(generate_core(source, optimize=optimized, backend_intrinsics=intrinsics, class_name="Core"), "Core")


def _equal(core, model):
    return all(mx.array_equal(a, b) for a, b in zip(core._def_probe(IDS), model(IDS), strict=True))


@pytest.mark.parametrize("bits,group_size", [(8, 64), (4, 64), (8, 32)])
@pytest.mark.parametrize("optimized", [False, True], ids=["unfused", "fused"])
def test_logits_equal_mlx_nn(tmp_path, optimized, bits, group_size):
    model, state = _quantized(bits, group_size)
    core = _core_class(tmp_path, optimized).from_state_dict(
        state, model_config={}, quantization={"bits": bits, "group_size": group_size}
    )
    assert _equal(core, model)


def test_safetensors_load_without_casting_quantized_tensors(tmp_path):
    _, state = _quantized()
    mx.save_safetensors(str(tmp_path / "model.safetensors"), state)
    core = _core_class(tmp_path).from_safetensors(
        [tmp_path / "model.safetensors"], model_config={}, dtype="bfloat16", quantization=Q8
    )
    assert core._flat_tensors["proj.weight.int8"].dtype == mx.uint32
    assert core._flat_tensors["proj.weight.scale"].dtype == mx.float32
    assert core._flat_tensors["proj.weight.qbias"].dtype == mx.float32
    assert core._flat_tensors["classifier.bias"].dtype == mx.bfloat16


def test_quantized_layout_needs_quantization(tmp_path):
    with pytest.raises(ValueError, match="quantization="):
        _core_class(tmp_path).from_state_dict(_quantized()[1], model_config={})


def test_quantization_needs_a_quantized_layout(tmp_path):
    with pytest.raises(ValueError, match="no quantized weights"):
        _core_class(tmp_path).from_state_dict(_float()[1], model_config={}, quantization=Q8)


def test_scales_next_to_a_float_weight_are_an_ordinary_parameter(tmp_path):
    model, state = _float()
    core = _core_class(tmp_path).from_state_dict({**state, "proj.scales": mx.ones((DIM,))}, model_config={})
    assert _equal(core, model)


def test_reloads_start_from_a_clean_state(tmp_path):
    """A forward fills the parameter cache, so a reload must also drop earlier settings."""
    four_bit, four_bit_state = _quantized(4, 64)
    eight_bit, eight_bit_state = _quantized(8, 32)
    plain, plain_state = _float()
    core = _core_class(tmp_path).from_state_dict(four_bit_state, model_config={}, quantization={"bits": 4, "group_size": 64})
    assert _equal(core, four_bit)
    core.load_state_dict(eight_bit_state, quantization={"bits": 8, "group_size": 32})
    assert _equal(core, eight_bit)
    core.load_state_dict(plain_state)
    assert not core._quantized and _equal(core, plain)


def test_int8_layout_keeps_8_bits_group_64(tmp_path):
    _, state = _float()
    core = _core_class(tmp_path).from_state_dict(state, model_config={})
    weight = state["proj.weight"]
    scale = mx.abs(weight).max(axis=1) / 127.0
    pair = {"proj.weight.int8": mx.round(weight / mx.expand_dims(scale, 1)).astype(mx.int8), "proj.weight.scale": scale}
    core.load_state_dict({**{k: v for k, v in state.items() if k != "proj.weight"}, **pair}, quantized=True)
    as_float = pair["proj.weight.int8"].astype(mx.float32) * mx.expand_dims(scale, 1)
    packed, scales, biases = mx.quantize(as_float, group_size=64, bits=8)
    want = mx.quantized_matmul(mx.ones((1, DIM)), packed, scales, biases=biases, transpose=True, group_size=64, bits=8)
    assert mx.array_equal(core._quantized_matmul("proj.weight", mx.ones((1, DIM))), want)
