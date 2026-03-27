from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from brainsurgery.synapse import lower_axon_program_to_synapse_spec, parse_axon_program_from_path
from tests.test_flags import LONG_TEST_ENV, run_long_tests_enabled

_RUN_LONG = run_long_tests_enabled()


def _load_axon_spec(path: Path) -> dict[str, Any]:
    modules = parse_axon_program_from_path(path)
    return lower_axon_program_to_synapse_spec(modules)


@pytest.mark.skipif(not _RUN_LONG, reason=f"set {LONG_TEST_ENV}=1 to enable long tests")
def test_deepseek_v2_lite_axon_key_alignment(
    repo_root: Path, deepseek_v2_lite_local_path: Path
) -> None:
    index_path = deepseek_v2_lite_local_path / "model.safetensors.index.json"
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = payload.get("weight_map")
    assert isinstance(weight_map, dict)
    keys = set(weight_map.keys())

    # Core top-level paths.
    assert "model.embed_tokens.weight" in keys
    assert "model.norm.weight" in keys
    assert "lm_head.weight" in keys

    # Layer 0 is dense-only.
    assert "model.layers.0.mlp.gate_proj.weight" in keys
    assert "model.layers.0.mlp.up_proj.weight" in keys
    assert "model.layers.0.mlp.down_proj.weight" in keys
    assert "model.layers.0.mlp.gate.weight" not in keys
    assert "model.layers.0.mlp.shared_experts.gate_proj.weight" not in keys
    assert "model.layers.0.self_attn.q_proj.weight" in keys
    assert "model.layers.0.self_attn.kv_a_proj_with_mqa.weight" in keys
    assert "model.layers.0.self_attn.kv_a_layernorm.weight" in keys
    assert "model.layers.0.self_attn.kv_b_proj.weight" in keys
    assert "model.layers.0.self_attn.o_proj.weight" in keys

    # Later layers are MoE with routed + shared experts.
    assert "model.layers.1.mlp.gate.weight" in keys
    assert "model.layers.1.mlp.shared_experts.gate_proj.weight" in keys
    assert "model.layers.1.mlp.experts.0.gate_proj.weight" in keys
    assert "model.layers.1.mlp.experts.63.down_proj.weight" in keys

    spec = _load_axon_spec(repo_root / "examples" / "deepseek_v2_lite.axon")
    model = spec.get("model", {})
    blocks = model.get("blocks", {})
    assert "deepseek_v2_lite_dense_block" in blocks
    assert "deepseek_v2_lite_moe_block" in blocks
