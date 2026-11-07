"""ModernBERT token classification on generated cores, against small HF models with no download."""

from __future__ import annotations

from functools import cache
from pathlib import Path

import numpy as np
import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import ModernBertConfig, ModernBertForTokenClassification, PreTrainedTokenizerFast

from synapse.axon_test import _should_generate_for_benchmark, _task_pragma_from_axon, run_axon_test
from tests.mlx_core_support import generate_core, load_core

SOURCE = Path(__file__).parents[1] / "synapse/models/bert/generic-modernbert-token-classification.axon"
ENTRY = "_def_modernbert_token_classification"
LABELS = 7
CONFIG = {
    "vocab_size": 97,
    "hidden_size": 32,
    "intermediate_size": 48,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "max_position_embeddings": 1024,
    "local_attention": 8,
    "global_attn_every_n_layers": 2,
    "pad_token_id": 0,
}


@pytest.fixture(scope="module")
def mx():
    return pytest.importorskip("mlx.core")


@cache
def _code(optimized):
    intrinsics = "codegen2-mlx" if optimized else None
    return generate_core(SOURCE, backend_intrinsics=intrinsics, optimize=optimized, class_name="Core")


@cache
def _hf():
    """A small HF model. ``to_dict`` has no ``num_labels``, like a saved config."""
    torch.manual_seed(1729)
    config = ModernBertConfig(**CONFIG, num_labels=LABELS)
    assert "num_labels" not in config.to_dict()
    return ModernBertForTokenClassification(config).eval(), config.to_dict()


def _state(model):
    return {name: value.detach().numpy().copy() for name, value in model.state_dict().items()}


def _mlx_logits(mx, config, state, ids, mask, optimized):
    core = load_core(_code(optimized), "Core").from_state_dict(
        {name: mx.array(value) for name, value in state.items()}, model_config=config, dtype="float32"
    )
    return np.array(getattr(core, ENTRY)(mx.array(ids), mx.array(mask))[0])


def _batch(lengths):
    """Right-padded random ids, as ``(ids, mask)`` with a row per length."""
    ids = np.random.default_rng(sum(lengths)).integers(1, CONFIG["vocab_size"], (len(lengths), max(lengths)))
    mask = (np.arange(max(lengths))[None, :] < np.array(lengths)[:, None]).astype(np.int64)
    return (ids * mask).astype(np.int64), mask


@pytest.mark.parametrize("optimized", [False, True], ids=["unfused", "fused"])
@pytest.mark.parametrize("layer_types", [True, False], ids=["layer_types", "every_n"])
@pytest.mark.parametrize("lengths", [(40,), (40, 23), (600, 517)], ids=["short", "padded", "blocked"])
def test_mlx_logits_match_hf(mx, optimized, layer_types, lengths):
    model, config = _hf()
    config = {key: value for key, value in config.items() if layer_types or key != "layer_types"}
    ids, mask = _batch(lengths)
    got = _mlx_logits(mx, config, _state(model), ids, mask, optimized)
    with torch.inference_mode():
        want = model(input_ids=torch.from_numpy(ids), attention_mask=torch.from_numpy(mask)).logits.numpy()
    valid = mask.astype(bool)  # pad query rows are unspecified
    np.testing.assert_allclose(got[valid], want[valid], atol=2e-4, rtol=0)


def test_classifier_bias_is_read_although_the_config_flag_is_false(mx):
    model, config = _hf()
    assert config["classifier_bias"] is False
    ids, mask = _batch((40,))
    offsets = np.arange(LABELS, dtype=np.float32)
    zero, moved = (
        _mlx_logits(mx, config, {**_state(model), "classifier.bias": bias}, ids, mask, True)
        for bias in (np.zeros_like(offsets), offsets)
    )
    np.testing.assert_allclose(moved - zero, np.broadcast_to(offsets, zero.shape), atol=1e-5, rtol=0)


def test_fused_forward_keeps_the_band_lowering():
    forward = _code(True)[_code(True).index(f"def {ENTRY}(") :]
    assert forward[: forward.index("\n    def ")].count("self._sdpa_band(") == 2  # full and sliding layers


def test_task_is_declared_and_never_generates():
    assert _task_pragma_from_axon(axon_file=SOURCE) == "token_classification"
    assert not _should_generate_for_benchmark(model_task="token_classification", benchmark_mode="auto")


def test_axon_test_runs_the_task_on_a_saved_model(tmp_path):
    model, _ = _hf()
    model.save_pretrained(tmp_path)
    vocabulary = {"<pad>": 0, "<unk>": 1} | {f"t{i}": i for i in range(2, CONFIG["vocab_size"])}
    tokenizer = Tokenizer(WordLevel(vocabulary, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, pad_token="<pad>", unk_token="<unk>").save_pretrained(tmp_path)
    result = run_axon_test(axon_file=SOURCE, weights=tmp_path, text=("t3 t4 t5", "t6 t7"), max_len=8)
    assert bool(result["top1_eq"]) is True
