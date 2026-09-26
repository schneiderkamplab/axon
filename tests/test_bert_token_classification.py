from __future__ import annotations

import ast
from collections import Counter
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
    assert "_PACKED_PARAMETER_SPECS = ({" in code
    assert "_common_parameter_pack_bindings(tensors, spec)" in code


def test_optimized_encoder_packs_projections_on_load_and_reloads_without_stale_weights():
    from torch.utils._python_dispatch import TorchDispatchMode

    config = BertConfig(
        vocab_size=89,
        hidden_size=24,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=48,
        num_labels=7,
    )
    reference = BertForTokenClassification(config).eval()
    model = generated_model(SOURCE, config, reference.state_dict(), optimized=True)
    assert sum(key.startswith("__packed.") for key in model.state_dict_tensors) == 4
    counts = Counter()

    class CountOps(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            counts[str(func)] += 1
            return func(*args, **(kwargs or {}))

    ids = torch.ones((2, 11), dtype=torch.long)
    with CountOps():
        before = model(input_ids=ids, attn_mask=ids)
    # Two embedding additions and two residuals per encoder layer, each once.
    assert counts["aten.add.Tensor"] == 2 + 2 * config.num_hidden_layers
    assert counts["aten.cat.default"] == 0
    with torch.no_grad():
        reference.bert.encoder.layer[0].attention.self.query.bias.add_(0.5)
    model.load_state_dict(reference.state_dict())
    after = model(input_ids=ids, attn_mask=ids)
    with torch.inference_mode():
        expected = reference(input_ids=ids, attention_mask=ids).logits
    torch.testing.assert_close(after, expected, atol=1e-6, rtol=1e-5)
    assert not torch.equal(before, after)


@pytest.mark.parametrize("inference", [False, True])
def test_generated_eval_forward_supports_fullgraph_aot_with_mask_views(inference):
    config = BertConfig(
        vocab_size=23,
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=16,
        num_labels=3,
    )
    reference = BertForTokenClassification(config).eval()
    model = generated_model(SOURCE, config, reference.state_dict(), optimized=True)
    compiled = torch.compile(model, backend="aot_eager", fullgraph=True)
    with torch.inference_mode() if inference else torch.no_grad():
        ids = torch.ones((2, 7), dtype=torch.long)
        mask = torch.ones_like(ids)
        mask[0, 4:] = 0
        torch.testing.assert_close(
            compiled(input_ids=ids, attn_mask=mask), model(input_ids=ids, attn_mask=mask)
        )


@pytest.mark.parametrize("window", [None, 0, 2])
def test_bidirectional_padding_mask_preserves_shape_window_and_left_crop(tmp_path, window):
    path = tmp_path / "unrelated.axon"
    path.write_text("""import Masking (bidirectional_mask)
main :: Tensor[B,H,Q,D] -> Tensor[B,H,K,D] -> Tensor[B,SM] -> ?Dim -> Tensor[B,1,Q,K]
main q k padding window = bidirectional_mask q k padding_mask=padding window=window
""")
    graph = graph_for(path, "codegen2-torch:__torch_sdpa")
    namespace = {}
    exec(emit_model_code_from_graph_ir(graph), namespace)
    model = namespace["GeneratedAxonModel"].from_state_dict({}).eval()
    padding = torch.tensor([[1, 0, 1, 0, 1, 1], [0, 1, 1, 0, 0, 1]])
    actual = model(
        q=torch.zeros(2, 2, 3, 4), k=torch.zeros(2, 2, 5, 4), padding=padding, window=window
    )
    expected = padding[:, -5:].bool()[:, None, None, :].expand(2, 1, 3, 5)
    if window is not None:
        distance = torch.arange(5)[None, :] - (torch.arange(3)[:, None] + 2)
        expected = expected & (distance.abs() <= window)
    assert actual.shape == (2, 1, 3, 5)
    torch.testing.assert_close(actual, expected)
