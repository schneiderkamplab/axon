from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

import synapse.axon_test as axon_test_module
import tests.conftest as test_fixtures
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
from synapse.axon.codegen2_mlx.core import emit_model_code_from_graph_ir, non_obvious_mlx_ops
from synapse.axon_test_matrix import _Pair, _resolve_model_task_for_pair, run_axon_test_matrix
from tests.model_downloads import MATRIX_AXON_MODEL_DIR_PAIRS, MODEL_SPECS

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = REPO_ROOT / "synapse" / "models" / "pplx-embed"
GENERIC_PII_MASKING = MODEL_DIR / "generic-pplx-pii-masking.axon"
GENERIC_EMBED = MODEL_DIR / "generic-pplx-embed.axon"


def _tiny_pii_masking_payload(num_token_labels: int = 37) -> dict:
    return {
        "model_type": "pii_masking",
        "num_token_labels": num_token_labels,
        "hidden_size": 64,
        "max_seq_len": 4096,
        "backbone": {
            "model_type": "bidirectional_pplx_qwen3",
            "architectures": ["PPLXQwen3Model"],
            "auto_map": {
                "AutoConfig": "configuration.PPLXQwen3Config",
                "AutoModel": "modeling.PPLXQwen3Model",
            },
            "use_bidirectional_attention": True,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "vocab_size": 128,
            "rms_norm_eps": 1e-6,
            "rope_parameters": {"rope_theta": 1000000, "rope_type": "default"},
            "tie_word_embeddings": True,
            "use_cache": False,
        },
    }


def test_pplx_fixtures_and_specs_are_declared() -> None:
    assert hasattr(test_fixtures, "pplx_pii_masking_local_path")
    assert hasattr(test_fixtures, "pplx_embed_local_path")
    assert MODEL_SPECS["pplx_pii_masking"].repo_id == "perplexity-ai/pplx-pii-masking"
    assert MODEL_SPECS["pplx_pii_masking"].local_dir == "pplx_pii_masking"
    assert MODEL_SPECS["pplx_embed"].repo_id == "perplexity-ai/pplx-embed-v1-0.6b"
    assert ("pplx-pii-masking", "pplx_pii_masking") in MATRIX_AXON_MODEL_DIR_PAIRS


def test_matrix_routes_pplx_pii_masking_as_masked_lm(tmp_path: Path) -> None:
    pair = _Pair(
        axon_path=MODEL_DIR / "pplx-pii-masking.axon", model_dir=tmp_path / "checkpoint"
    )
    assert _resolve_model_task_for_pair(pair) == "masked_lm"


def test_matrix_resolves_pplx_pii_masking_pair(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    models_dir = tmp_path / "models"
    (models_dir / "pplx_pii_masking").mkdir(parents=True, exist_ok=True)
    exit_code = run_axon_test_matrix(
        examples_dir=MODEL_DIR,
        models_dir=models_dir,
        dry_run=True,
        include=["pplx_pii_masking"],
    )
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "pplx-pii-masking.axon" in out
    assert "/models/pplx_pii_masking" in out


def test_pii_masking_config_detection() -> None:
    assert axon_test_module._is_pii_masking_config(_tiny_pii_masking_payload())
    assert not axon_test_module._is_pii_masking_config({"model_type": "qwen3", "hidden_size": 64})
    assert not axon_test_module._is_pii_masking_config({"model_type": "pii_masking"})
    assert not axon_test_module._is_pii_masking_config(None)


def test_pii_masking_hf_config_is_a_stock_qwen3_config(tmp_path: Path) -> None:
    payload = _tiny_pii_masking_payload(num_token_labels=237)
    config = axon_test_module._build_pii_masking_hf_config(payload)
    assert config.model_type == "qwen3"
    assert config.hidden_size == 64
    assert config.num_hidden_layers == 2
    assert config.num_key_value_heads == 2
    assert config.head_dim == 16
    assert config.num_token_labels == 237
    assert config.use_bidirectional_attention is True
    assert not getattr(config, "auto_map", None)

    model_dir = tmp_path / "pplx_pii_masking"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps(payload), encoding="utf-8")
    loaded = axon_test_module._load_auto_config_with_compat_fallback(
        model_dir, trust_remote_code=False
    )
    assert loaded.model_type == "qwen3"
    assert loaded.num_token_labels == 237


def test_pii_masking_reference_is_bidirectional_and_padding_invariant() -> None:
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Model

    torch.manual_seed(0)
    config = axon_test_module._build_pii_masking_hf_config(_tiny_pii_masking_payload())
    backbone = Qwen3Model(config).float().eval()
    reference = axon_test_module._PiiMaskingTokenClassificationReference(
        backbone=backbone,
        cls_weight=torch.randn(37, 64),
        cls_bias=torch.randn(37),
    ).eval()

    ids_a = torch.tensor([[5, 6, 7, 8, 9]])
    ids_b = ids_a.clone()
    ids_b[0, -1] = 42
    with torch.no_grad():
        logits_a = reference(input_ids=ids_a, attention_mask=torch.ones_like(ids_a))["logits"]
        logits_b = reference(input_ids=ids_b, attention_mask=torch.ones_like(ids_b))["logits"]
    assert logits_a.shape == (1, 5, 37)
    assert logits_a.dtype == torch.float32
    # Non-causal: changing the last token must change earlier positions.
    assert not torch.allclose(logits_a[0, 0], logits_b[0, 0])

    padded_ids = torch.tensor([[5, 6, 7, 8, 9], [5, 6, 7, 0, 0]])
    padded_mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]])
    with torch.no_grad():
        padded = reference(input_ids=padded_ids, attention_mask=padded_mask)["logits"]
        short = reference(input_ids=padded_ids[1:, :3], attention_mask=padded_mask[1:, :3])[
            "logits"
        ]
    # Padding-only mask: padded keys never influence valid positions.
    assert torch.allclose(padded[1, :3], short[0], atol=1e-5)


def _mlx_graph(axon_path: Path):
    program = resolve_axon_program_from_path(axon_path, strict=False).ast
    program = normalize_closed_axon_file(program)
    program = elaborate_closed_axon_file(program)
    program = flatten_closed_axon_file(program)
    program = typecheck2_flat_axon_file(program)
    graph = lower_axon_program_to_graph_ir(program)
    return optimize_graph_program(
        graph, config=GraphOptimizeConfig(backend_intrinsics="codegen2-mlx")
    )


@pytest.mark.parametrize(
    "axon_path",
    [GENERIC_PII_MASKING, GENERIC_EMBED, MODEL_DIR / "pplx-pii-masking.axon",
     MODEL_DIR / "eraser.axon", MODEL_DIR / "pplx-embed-v1-0.6b.axon"],
    ids=["pii_masking", "embed", "pii_materialized", "eraser", "embed_materialized"],
)
def test_pplx_files_lower_to_native_mlx_gqa(axon_path: Path) -> None:
    graph = _mlx_graph(axon_path)
    assert non_obvious_mlx_ops(graph) == ()
    nodes = [node for module in graph.modules for node in module.nodes]
    attention = [node for node in nodes if node.op.name == "__mlx_sdpa"]
    assert attention
    assert all(node.inputs[0].type_expr.dims[1] != node.inputs[1].type_expr.dims[1]
               for node in attention)
    assert not any(node.op.name == "_repeat" for node in nodes)
    code = emit_model_code_from_graph_ir(graph)
    assert "class AxonMlxModel" in code
    assert "import mlx.core as mx" in code


@pytest.mark.parametrize("encoder_only", [False, True], ids=["classifier", "embedding"])
def test_pplx_mlx_gqa_matches_qwen3_reference(encoder_only: bool) -> None:
    mx = pytest.importorskip("mlx.core", exc_type=ImportError)
    import numpy as np
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Model

    from synapse.axon.codegen2_mlx import torch_state_dict_to_mlx

    torch.manual_seed(17)
    payload = _tiny_pii_masking_payload()
    # As in the real checkpoint, the query projection is wider than hidden_size.
    payload["backbone"]["head_dim"] = 32
    config = axon_test_module._build_pii_masking_hf_config(payload)
    reference = axon_test_module._PiiMaskingTokenClassificationReference(
        backbone=Qwen3Model(config).float().eval(),
        cls_weight=torch.randn(37, 64),
        cls_bias=torch.randn(37),
    ).eval()
    sensitivity_weight = torch.randn(1, 64) * 0.1
    sensitivity_bias = torch.randn(1) * 0.1
    state = dict(reference.backbone.state_dict())
    if not encoder_only:
        state = {f"backbone.{key}": value for key, value in state.items()}
        state.update({
            "token_cls_head.weight": reference.cls_weight.detach(),
            "token_cls_head.bias": reference.cls_bias.detach(),
            "sensitivity_head.weight": sensitivity_weight,
            "sensitivity_head.bias": sensitivity_bias,
        })
    graph = _mlx_graph(GENERIC_EMBED if encoder_only else GENERIC_PII_MASKING)
    namespace = {}
    exec(emit_model_code_from_graph_ir(graph), namespace)
    model = namespace["AxonMlxModel"].from_state_dict(
        torch_state_dict_to_mlx(state),
        model_config=payload["backbone"] if encoder_only else payload,
    )
    hidden = []
    hook = reference.backbone.register_forward_hook(
        lambda _module, _args, output: hidden.append(output.last_hidden_state)
    )
    try:
        cases = [
            ([[5]], None),
            ([[5, 6, 7, 8, 9]], None),
            ([[5, 6, 7, 8, 9], [5, 6, 7, 0, 0]], [[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]]),
            ([[0, 0, 5, 6, 7]], [[0, 0, 1, 1, 1]]),
        ]
        for ids_list, mask_list in cases:
            ids = torch.tensor(ids_list)
            mask = torch.tensor(mask_list) if mask_list is not None else None
            hidden.clear()
            with torch.no_grad():
                expected = reference(input_ids=ids, attention_mask=mask)["logits"]
                h = hidden[0]
                sensitivity = torch.sigmoid(torch.nn.functional.linear(
                    h.mean(dim=1), sensitivity_weight, sensitivity_bias
                ))
                if mask is None:
                    embedding = h.mean(dim=1)
                else:
                    mf = mask.float().unsqueeze(-1)
                    embedding = (h * mf).sum(dim=1) / mf.sum(dim=1)
            actual = model.forward(
                input_ids=mx.array(ids.numpy()),
                attn_mask=mx.array(mask.numpy()) if mask is not None else None,
            )
            mx.eval(actual)
            if encoder_only:
                np.testing.assert_allclose(np.array(actual), embedding.numpy(), atol=1e-5, rtol=1e-5)
            else:
                assert set(actual) == {"logits", "sensitivity"}
                np.testing.assert_allclose(np.array(actual["logits"]), expected.numpy(), atol=2e-5, rtol=1e-5)
                np.testing.assert_allclose(np.array(actual["sensitivity"]), sensitivity.numpy(), atol=1e-6, rtol=1e-5)
    finally:
        hook.remove()
