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
    assert ("pplx_pii_masking", "pplx_pii_masking") in MATRIX_AXON_MODEL_DIR_PAIRS


def test_matrix_routes_pplx_pii_masking_as_masked_lm(tmp_path: Path) -> None:
    pair = _Pair(
        axon_path=tmp_path / "pplx_pii_masking.axon", model_dir=tmp_path / "pplx_pii_masking"
    )
    assert _resolve_model_task_for_pair(pair) == "masked_lm"


def test_matrix_resolves_pplx_pii_masking_pair(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    examples_dir = tmp_path / "examples"
    models_dir = tmp_path / "models"
    examples_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    (examples_dir / "pplx_pii_masking.axon").write_text(
        "pplx_pii_masking :: Tensor -> Tensor\npplx_pii_masking x = x\n",
        encoding="utf-8",
    )
    (models_dir / "pplx_pii_masking").mkdir(parents=True, exist_ok=True)
    exit_code = run_axon_test_matrix(
        examples_dir=examples_dir,
        models_dir=models_dir,
        dry_run=True,
        include=["pplx_pii_masking"],
    )
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "pplx_pii_masking.axon" in out
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
    "axon_path", [GENERIC_PII_MASKING, GENERIC_EMBED], ids=["pii_masking", "embed"]
)
def test_pplx_generic_files_lower_to_mlx(axon_path: Path) -> None:
    graph = _mlx_graph(axon_path)
    assert non_obvious_mlx_ops(graph) == ()
    code = emit_model_code_from_graph_ir(graph)
    assert "class AxonMlxModel" in code
    assert "import mlx.core as mx" in code
