"""The Ettin models on generated cores, with small synthetic weights and no download.

MLX with and without the optimizer, and Torch, compile the same two ``.axon`` files. They must
agree with each other, and the decoder's cached steps must match its full forward across the
sliding window. ``tests/test_ettin_parity.py`` compares real checkpoints with HF.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from synapse.axon_test import _declared_checkpoints_from_axon, _extract_logits
from tests.mlx_core_support import generate_core, load_core

mx = pytest.importorskip("mlx.core")

ROOT = Path(__file__).parents[1]
ARCHS = {
    "encoder": {"source": ROOT / "synapse/models/bert/generic-modernbert.axon", "entry": "_def_modernbert"},
    "decoder": {"source": ROOT / "synapse/models/ettin/generic-ettin-decoder.axon", "entry": "_def_ettin_decoder"},
}
SIZES = ("17m", "32m", "68m", "150m", "400m", "1b")
CONFIG = {
    "hidden_size": 32,
    "num_attention_heads": 4,
    "num_hidden_layers": 4,
    "intermediate_size": 48,
    "vocab_size": 97,
    "max_position_embeddings": 2048,
    "local_attention": 8,
    "global_attn_every_n_layers": 2,
    "global_rope_theta": 160000.0,
    "local_rope_theta": 10000.0,
    "norm_eps": 1e-5,
    "hidden_activation": "gelu",
    "classifier_activation": "gelu",
}
FUSED = "codegen2-mlx"
ATOL = 2e-3


def _generated(arch, backend, *, optimize, intrinsics=None):
    code = generate_core(
        ARCHS[arch]["source"], backend=backend, backend_intrinsics=intrinsics, optimize=optimize, class_name="Core",
    )
    return load_core(code, "Core"), code


def _shapes(arch):
    """The released parameter layout for CONFIG."""
    d, f, v = CONFIG["hidden_size"], CONFIG["intermediate_size"], CONFIG["vocab_size"]
    shapes = {"model.embeddings.tok_embeddings.weight": (v, d), "model.embeddings.norm.weight": (d,)}
    for i in range(CONFIG["num_hidden_layers"]):
        layer = f"model.layers.{i}"
        if i != 0:
            shapes[f"{layer}.attn_norm.weight"] = (d,)
        if arch == "encoder":
            shapes[f"{layer}.attn.Wqkv.weight"] = (3 * d, d)
        else:
            for projection in ("q_proj", "k_proj", "v_proj"):
                shapes[f"{layer}.attn.{projection}.weight"] = (d, d)
        shapes[f"{layer}.attn.Wo.weight"] = (d, d)
        shapes[f"{layer}.mlp_norm.weight"] = (d,)
        shapes[f"{layer}.mlp.Wi.weight"] = (2 * f, d)
        shapes[f"{layer}.mlp.Wo.weight"] = (d, f)
    head = "head" if arch == "encoder" else "lm_head"
    shapes |= {"model.final_norm.weight": (d,), f"{head}.dense.weight": (d, d), f"{head}.norm.weight": (d,), "decoder.bias": (v,)}
    return shapes


class Encoder:
    def __init__(self, core):
        self.core = core
        self.entry = getattr(core, ARCHS["encoder"]["entry"])

    def __call__(self, ids, mask=None):
        return self.entry(mx.array(ids), None if mask is None else mx.array(mask))


class Decoder:
    """Stateful cache on eager passes. With ``compile``, a pass without a cache runs under
    ``mx.compile``, which needs plain ``(k, v)`` pairs, and the next eager step seeds from them."""

    def __init__(self, core, compile=False):
        self.core = core
        self.entry = getattr(core, ARCHS["decoder"]["entry"])
        self.compile = compile

    def __call__(self, ids, mask=None, cache=None, use_cache=False, logits_to_keep=0):
        ids, mask = mx.array(ids), None if mask is None else mx.array(mask)
        if self.compile and cache is None:
            object.__setattr__(self.core, "_use_kv_cache", False)
            inputs = (ids,) if mask is None else (ids, mask)

            def run(*arrays):
                return self.entry(arrays[0], arrays[1] if mask is not None else None, None, use_cache, logits_to_keep)

            logits, new_cache = mx.compile(run)(*inputs)
            return logits, new_cache if use_cache else None
        object.__setattr__(self.core, "_use_kv_cache", use_cache)
        logits, new_cache = self.entry(ids, mask, cache, use_cache, logits_to_keep)
        return logits, new_cache if use_cache else None


@pytest.fixture(scope="module")
def weights():
    """Deterministic state per arch, in the released parameter layout."""
    rng = np.random.default_rng(1729)
    return {
        arch: {name: rng.normal(0, 0.08, shape).astype(np.float32) for name, shape in sorted(_shapes(arch).items())}
        for arch in ARCHS
    }


@pytest.fixture(scope="module")
def mlx_models(weights):
    models = {}
    for arch, host in (("encoder", Encoder), ("decoder", Decoder)):
        state = {name: mx.array(value) for name, value in weights[arch].items()}
        fused, code = _generated(arch, "mlx", optimize=True, intrinsics=FUSED)
        assert code.count("self._sdpa_band(") == 4
        unfused, _ = _generated(arch, "mlx", optimize=False)
        models[arch, "fused"] = host(fused.from_state_dict(state, model_config=CONFIG))
        models[arch, "unfused"] = host(unfused.from_state_dict(state, model_config=CONFIG))
        compiled = fused.from_state_dict(state, model_config=CONFIG)
        models[arch, "compiled"] = Decoder(compiled, compile=True) if arch == "decoder" else Encoder(compiled)
    return models


@pytest.fixture(scope="module")
def torch_models(weights):
    models = {}
    for arch in ARCHS:
        state = {name: torch.from_numpy(value.copy()) for name, value in weights[arch].items()}
        core, _ = _generated(arch, "torch", optimize=True)
        models[arch] = core.from_state_dict(state, model_config=CONFIG)
    return models


def _ids(length, batch=1):
    return np.random.default_rng(length).integers(0, CONFIG["vocab_size"], (batch, length)).astype(np.int64)


def _padded(length):
    """Two rows, the second with a shorter valid prefix."""
    mask = np.ones((2, length), dtype=np.int64)
    mask[1, length // 2 :] = 0
    return _ids(length, 2), mask


def test_decoder_reads_the_released_parameter_layout(mlx_models, weights):
    """A released checkpoint has separate q, k and v projections, and no packed Wqkv."""
    core_class = type(mlx_models["decoder", "fused"].core)
    state = {name: mx.array(value) for name, value in weights["decoder"].items()}
    assert not any("Wqkv" in name for name in state)
    Decoder(core_class.from_state_dict(state, model_config=CONFIG))(_ids(5))
    del state["model.layers.0.attn.q_proj.weight"]
    with pytest.raises(KeyError, match="q_proj"):
        Decoder(core_class.from_state_dict(state, model_config=CONFIG))(_ids(5))


def _torch_entry(model, arch):
    return getattr(model, ARCHS[arch]["entry"])


def _encoder_logits(model, ids, mask=None):
    with torch.inference_mode():
        return _torch_entry(model, "encoder")(torch.from_numpy(ids), None if mask is None else torch.from_numpy(mask))


def _decoder_logits(model, ids, mask=None):
    with torch.inference_mode():
        return _torch_entry(model, "decoder")(torch.from_numpy(ids), None if mask is None else torch.from_numpy(mask), None, False, 0)


@pytest.mark.parametrize("variant", ["fused", "unfused", "compiled"])
@pytest.mark.parametrize("length", [5, 33])
def test_encoder_backends_agree(mlx_models, torch_models, variant, length):
    ids = _ids(length)
    logits, hidden = mlx_models["encoder", variant](ids)
    want_logits, want_hidden = _encoder_logits(torch_models["encoder"], ids)
    np.testing.assert_allclose(np.array(logits), want_logits.numpy(), atol=ATOL)
    np.testing.assert_allclose(np.array(hidden), want_hidden.numpy(), atol=ATOL)


@pytest.mark.parametrize("variant", ["fused", "unfused", "compiled"])
@pytest.mark.parametrize("length", [5, 33])
def test_decoder_backends_agree(mlx_models, torch_models, variant, length):
    ids = _ids(length)
    logits, _ = mlx_models["decoder", variant](ids)
    want = _decoder_logits(torch_models["decoder"], ids)[0]
    np.testing.assert_allclose(np.array(logits), want.numpy(), atol=ATOL)


@pytest.mark.parametrize("arch", ["encoder", "decoder"])
def test_padded_rows_agree_at_real_tokens(mlx_models, torch_models, arch):
    ids, mask = _padded(21)
    for variant in ("fused", "unfused"):
        logits = mlx_models[arch, variant](ids, mask)[0]
        forward = _encoder_logits if arch == "encoder" else _decoder_logits
        want = forward(torch_models[arch], ids, mask)[0]
        real = mask.astype(bool)
        np.testing.assert_allclose(np.array(logits)[real], want.numpy()[real], atol=ATOL)


def test_band_attention_is_blocked_for_long_inputs(mlx_models):
    """Past 512 tokens the sliding layers take the blocked path, with the same result."""
    ids, mask = _padded(600)
    fused = np.array(mlx_models["encoder", "fused"](ids, mask)[0])
    unfused = np.array(mlx_models["encoder", "unfused"](ids, mask)[0])
    real = mask.astype(bool)
    np.testing.assert_allclose(fused[real], unfused[real], atol=ATOL)


@pytest.mark.parametrize("variant", ["fused", "unfused", "compiled"])
def test_cached_steps_match_the_full_forward_across_the_window(mlx_models, variant):
    """The window is 4 keys: 14 single-token steps cross it, and the rolling cache, many times."""
    model = mlx_models["decoder", variant]
    ids = _ids(20)
    full = np.array(mlx_models["decoder", "unfused"](ids)[0])
    logits, cache = model(ids[:, :6], use_cache=True, logits_to_keep=1)
    np.testing.assert_allclose(np.array(logits)[:, -1], full[:, 5], atol=ATOL)
    for position in range(6, ids.shape[1]):
        logits, cache = model(ids[:, position : position + 1], cache=cache, use_cache=True, logits_to_keep=1)
        np.testing.assert_allclose(np.array(logits)[:, -1], full[:, position], atol=ATOL)


@pytest.mark.parametrize("variant", ["fused", "unfused", "compiled"])
def test_cached_steps_with_a_mask_match_the_full_forward(mlx_models, variant):
    """With a mask, each sliding layer reads the padding of exactly the keys in its rolling cache."""
    model = mlx_models["decoder", variant]
    ids = _ids(20)
    mask = np.ones_like(ids)
    full = np.array(mlx_models["decoder", "unfused"](ids, mask)[0])
    logits, cache = model(ids[:, :6], mask[:, :6], use_cache=True, logits_to_keep=1)
    np.testing.assert_allclose(np.array(logits)[:, -1], full[:, 5], atol=ATOL)
    for position in range(6, ids.shape[1]):
        step = ids[:, position : position + 1]
        logits, cache = model(step, mask[:, : position + 1], cache=cache, use_cache=True, logits_to_keep=1)
        np.testing.assert_allclose(np.array(logits)[:, -1], full[:, position], atol=ATOL)


def test_torch_cached_steps_match_the_full_forward(torch_models):
    entry = _torch_entry(torch_models["decoder"], "decoder")
    ids = _ids(20)
    with torch.inference_mode():
        full = entry(torch.from_numpy(ids), None, None, False, 0)[0].numpy()
        logits, cache = entry(torch.from_numpy(ids[:, :6]), None, None, True, 1)
        np.testing.assert_allclose(logits[:, -1].numpy(), full[:, 5], atol=ATOL)
        for position in range(6, ids.shape[1]):
            logits, cache = entry(torch.from_numpy(ids[:, position : position + 1]), None, cache, True, 1)
            np.testing.assert_allclose(logits[:, -1].numpy(), full[:, position], atol=ATOL)


@pytest.mark.parametrize("variant", ["fused", "unfused"])
def test_generated_forward_runs_the_decoder(mlx_models, variant):
    """The generated class fills unset inputs with null."""
    core = mlx_models["decoder", variant].core
    object.__setattr__(core, "_use_kv_cache", False)
    ids = _ids(12)
    want = np.array(mlx_models["decoder", "unfused"](ids)[0])
    np.testing.assert_allclose(np.array(core.forward(mx.array(ids))["logits"]), want, atol=ATOL)


@pytest.mark.parametrize("variant", ["fused", "unfused"])
@pytest.mark.parametrize("prompt", [3, 12])
def test_generated_generate_matches_full_forwards(mlx_models, variant, prompt):
    """Greedy decoding through the generated class, across the sliding window."""
    ids = _ids(prompt)
    reference = mlx_models["decoder", "unfused"]
    want = ids
    for _ in range(10):
        logits = np.array(reference(want)[0])
        want = np.concatenate([want, logits[:, -1:].argmax(-1).astype(np.int64)], axis=1)
    core = mlx_models["decoder", variant].core
    object.__setattr__(core, "_use_kv_cache", True)
    assert np.array_equal(np.array(core.generate(mx.array(ids), max_new_tokens=10)), want)


@pytest.mark.parametrize("keep,length", [(0, 30), (None, 30), (1, 1), (30, 30), (31, 30), (35, 30)])
def test_logits_to_keep_is_capped_at_the_input(mlx_models, keep, length):
    """As HF's negative slice: more than the input keeps the whole input."""
    logits, _ = mlx_models["decoder", "fused"](_ids(30), logits_to_keep=keep)
    assert logits.shape[1] == length


def test_padding_shorter_than_the_keys_is_refused(mlx_models):
    """The dense mask's reshape refuses it. A negative slice start would use one flag for all keys."""
    model = mlx_models["decoder", "fused"]
    ids = _ids(10, 2)
    _, cache = model(ids, use_cache=True)
    with pytest.raises(ValueError, match="padding"):
        model(ids[:, :1], np.array([[1], [0]]), cache=cache, use_cache=True)


@pytest.mark.parametrize("arch", ["encoder", "decoder"])
def test_every_layer_can_be_full(mlx_models, weights, arch):
    """global_attn_every_n_layers=0 means no sliding layers, without computing i % 0."""
    state = {name: mx.array(value) for name, value in weights[arch].items()}
    core_class = type(mlx_models[arch, "fused"].core)
    host = Encoder if arch == "encoder" else Decoder

    def logits(**changes):
        return np.array(host(core_class.from_state_dict(state, model_config={**CONFIG, **changes}))(_ids(12))[0])

    layer_types = ["full_attention"] * CONFIG["num_hidden_layers"]
    np.testing.assert_allclose(logits(global_attn_every_n_layers=0), logits(layer_types=layer_types), atol=ATOL)


def test_axon_test_reads_the_logits_of_the_tuple_output(torch_models):
    """ModernBERT now returns (logits, hidden); axon-test compares the logits, its first output."""
    ids = _ids(9)
    outputs = _encoder_logits(torch_models["encoder"], ids)
    assert outputs[0].shape == (1, 9, CONFIG["vocab_size"])
    np.testing.assert_array_equal(_extract_logits(tuple(outputs)).numpy(), outputs[0].numpy())


@pytest.mark.parametrize("arch", ["encoder", "decoder"])
def test_released_checkpoints_are_declared(arch):
    declared = {name for name in _declared_checkpoints_from_axon(axon_file=ARCHS[arch]["source"]) if "ettin" in name}
    assert declared == {f"jhu-clsp/ettin-{arch}-{size}" for size in SIZES}
