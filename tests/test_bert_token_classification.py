from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch
from transformers import BertConfig, BertForMaskedLM, BertForTokenClassification

from synapse.axon import (
    GraphOptimizeConfig,
    elaborate_closed_axon_file,
    flatten_closed_axon_file,
    lower_axon_program_to_graph_ir,
    normalize_closed_axon_file,
    optimize_graph_program,
    resolve_axon_program_from_path,
    typecheck2_flat_axon_file,
)
from synapse.axon.codegen2_torch import emit_model_code_from_graph_ir
from synapse.axon_test import (
    _resolve_model_task,
    _should_generate_for_benchmark,
    _task_pragma_from_axon,
)
from synapse.axon_test_matrix import _Pair, _resolve_model_task_for_pair

MODELS = Path(__file__).resolve().parents[1] / "synapse/models/bert"
SOURCE = MODELS / "generic-bert-token-classification.axon"


def graph_for(path, backend=None):
    resolved = resolve_axon_program_from_path(path).ast
    typed = typecheck2_flat_axon_file(
        flatten_closed_axon_file(elaborate_closed_axon_file(normalize_closed_axon_file(resolved)))
    )
    graph = lower_axon_program_to_graph_ir(typed)
    if backend:
        graph = optimize_graph_program(
            graph, config=GraphOptimizeConfig(backend_intrinsics=backend)
        )
    return graph


def generated_model(path, config, state, optimized=False):
    graph = graph_for(path, "codegen2-torch:__torch_sdpa" if optimized else None)
    namespace = {}
    exec(emit_model_code_from_graph_ir(graph, model_config=config.to_dict()), namespace)
    return namespace["GeneratedAxonModel"].from_state_dict(state).eval()


@pytest.mark.parametrize("optimized", [False, True])
@pytest.mark.parametrize("hidden,heads,labels", [(24, 4, 7), (32, 2, 13)])
def test_token_logits_match_hf_for_padding_and_segment_ids(hidden, heads, labels, optimized):
    torch.manual_seed(7)
    config = BertConfig(
        vocab_size=89,
        hidden_size=hidden,
        num_hidden_layers=2,
        num_attention_heads=heads,
        intermediate_size=hidden * 3,
        num_labels=labels,
        layer_norm_eps=1e-5,
    )
    config._attn_implementation = "eager"
    reference = BertForTokenClassification(config).eval()
    model = generated_model(SOURCE, config, reference.state_dict(), optimized)
    ids = torch.randint(1, config.vocab_size, (3, 11))
    mask = torch.ones_like(ids)
    mask[1, 7:] = 0
    mask[2, :4] = 0
    segments = torch.randint(0, 2, ids.shape)
    with torch.inference_mode():
        actual = model(input_ids=ids, attn_mask=mask, token_type_ids=segments)
        expected = reference(input_ids=ids, attention_mask=mask, token_type_ids=segments).logits
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(
            model(input_ids=ids, attn_mask=mask),
            reference(input_ids=ids, attention_mask=mask).logits,
            atol=1e-6,
            rtol=1e-5,
        )
        torch.testing.assert_close(
            model(input_ids=ids), reference(input_ids=ids).logits, atol=1e-6, rtol=1e-5
        )
        changed = ids.clone()
        changed[mask == 0] = 88
        torch.testing.assert_close(
            model(input_ids=changed, attn_mask=mask, token_type_ids=segments)[mask.bool()],
            actual[mask.bool()],
            atol=1e-6,
            rtol=1e-5,
        )
        assert actual.shape == (3, 11, labels)


def test_shared_encoder_preserves_existing_masked_lm_paths_and_logits():
    torch.manual_seed(11)
    config = BertConfig(
        vocab_size=83,
        hidden_size=24,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=48,
    )
    config._attn_implementation = "eager"
    reference = BertForMaskedLM(config).eval()
    # The existing masked-LM definition declares the original gamma/beta paths.
    state = {
        name.replace("LayerNorm.weight", "LayerNorm.gamma").replace(
            "LayerNorm.bias", "LayerNorm.beta"
        ): value
        for name, value in reference.state_dict().items()
    }
    model = generated_model(MODELS / "generic-bert.axon", config, state)
    ids = torch.randint(1, 83, (2, 9))
    mask = torch.ones_like(ids)
    mask[1, 5:] = 0
    with torch.inference_mode():
        torch.testing.assert_close(
            model(input_ids=ids, attn_mask=mask),
            reference(input_ids=ids, attention_mask=mask).logits,
            atol=1e-6,
            rtol=1e-5,
        )


def test_token_classification_task_is_declared_and_never_generates(tmp_path):
    assert _resolve_model_task("token_classification") == "token_classification"
    assert _task_pragma_from_axon(axon_file=SOURCE) == "token_classification"
    renamed = tmp_path / "unrelated-name.axon"
    renamed.write_text(SOURCE.read_text())
    assert _resolve_model_task_for_pair(_Pair(renamed, tmp_path)) == "token_classification"
    assert not _should_generate_for_benchmark(
        model_task="token_classification", benchmark_mode="auto"
    )
    with pytest.raises(ValueError, match="encoder-only"):
        _should_generate_for_benchmark(model_task="token_classification", benchmark_mode="generate")


def test_mlx_source_can_be_generated_without_running_mlx():
    from synapse.axon.codegen2_mlx import emit_model_code_from_graph_ir as emit_mlx

    code = emit_mlx(
        graph_for(SOURCE, "codegen2-mlx:__mlx_sdpa"),
        model_config={
            "hidden_size": 24,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "intermediate_size": 48,
        },
    )
    ast.parse(code)
