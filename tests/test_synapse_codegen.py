from __future__ import annotations

from pathlib import Path

import pytest
import torch
import typer
from omegaconf import OmegaConf

from brainsurgery.cli.synapse import axon_to_synapse, emit_generic, synapse_to_axon
from brainsurgery.synapse import emit_model_code_from_synapse_spec


def _spec_dict() -> dict[str, object]:
    return {
        "synapse": 1,
        "model": {
            "symbols": {"D": 16, "V": 32, "C": 12, "L": 2, "H": 4, "M": 64},
            "params": {
                "activation": "gelu_new",
                "layer_norm_epsilon": 1e-5,
                "attn_backend": "sdpa",
            },
            "graph": [],
        },
    }


def _spec_yaml() -> str:
    return """synapse: 1
model:
  symbols:
    D: 16
    V: 32
    C: 12
    L: 2
    H: 4
    M: 64
  params:
    activation: gelu_new
    layer_norm_epsilon: 1.0e-5
    attn_backend: sdpa
  graph: []
"""


def test_cli_emit_synapse_writes_python_file(tmp_path: Path) -> None:
    spec_path = tmp_path / "spec.yaml"
    out_path = tmp_path / "generated_model.py"
    spec_path.write_text(_spec_yaml(), encoding="utf-8")

    emit_generic(spec_path=spec_path, output_path=out_path, class_name="FromCli", force=False)

    assert out_path.exists()
    contents = out_path.read_text(encoding="utf-8")
    assert "class FromCli(nn.Module):" in contents
    assert "def from_state_dict(cls, state_dict" in contents


def test_cli_emit_requires_force_for_existing_output(tmp_path: Path) -> None:
    spec_path = tmp_path / "spec.yaml"
    out_path = tmp_path / "generated_model.py"
    spec_path.write_text(_spec_yaml(), encoding="utf-8")
    out_path.write_text("# existing\n", encoding="utf-8")

    with pytest.raises(typer.BadParameter) as exc_info:
        emit_generic(spec_path=spec_path, output_path=out_path, class_name="FromCli", force=False)

    assert "overwrite" in str(exc_info.value)


def test_emit_accepts_minimal_op_map() -> None:
    bad_op_map = {
        "ops": {
            "embedding": {"target": "torch.nn.Embedding"},
            "linear": {"target": "torch.nn.Linear"},
            "layernorm": {"target": "torch.nn.LayerNorm"},
            "attention": {"target": "torch.nn.MultiheadAttention"},
        }
    }
    source = emit_model_code_from_synapse_spec(_spec_dict(), class_name="BadMap", op_map=bad_op_map)
    assert "class BadMap(nn.Module):" in source


def test_emit_generic_from_gemma3_spec(tmp_path: Path) -> None:
    spec_path = Path(__file__).resolve().parents[1] / "examples" / "gemma3_270m_synapse.yaml"
    out_path = tmp_path / "gemma_model.py"
    emit_generic(spec_path=spec_path, output_path=out_path, class_name="Gemma3Synapse", force=False)
    contents = out_path.read_text(encoding="utf-8")
    assert "class Gemma3Synapse(nn.Module):" in contents
    assert "'D': 640" in contents


def test_emit_model_code_from_synapse_spec_generic() -> None:
    source = emit_model_code_from_synapse_spec(_spec_dict(), class_name="GenericSynapse")
    assert "class GenericSynapse(nn.Module):" in source
    assert "def generate(self, input_ids: torch.Tensor" in source


def test_cli_synapse_to_axon_and_back_roundtrip(tmp_path: Path) -> None:
    spec_path = tmp_path / "spec.yaml"
    axon_path = tmp_path / "spec.axon"
    lowered_path = tmp_path / "lowered.yaml"
    spec_path.write_text(_spec_yaml(), encoding="utf-8")

    synapse_to_axon(
        spec_path=spec_path,
        output_path=axon_path,
        module_name="tiny",
        force=False,
    )
    assert axon_path.exists()
    axon_text = axon_path.read_text(encoding="utf-8")
    assert axon_text.startswith("tiny :: ")
    assert "meta __inputs" not in axon_text
    assert "meta __outputs" not in axon_text

    axon_to_synapse(axon_path=axon_path, output_path=lowered_path, force=False)
    assert lowered_path.exists()
    lowered = OmegaConf.to_container(OmegaConf.load(lowered_path), resolve=True)
    assert isinstance(lowered, dict)
    assert lowered.get("synapse") == 1
    assert lowered.get("model", {}).get("symbols") is None


def test_cli_synapse_to_axon_requires_force_for_existing_output(tmp_path: Path) -> None:
    spec_path = tmp_path / "spec.yaml"
    axon_path = tmp_path / "spec.axon"
    spec_path.write_text(_spec_yaml(), encoding="utf-8")
    axon_path.write_text("# existing\n", encoding="utf-8")

    with pytest.raises(typer.BadParameter) as exc_info:
        synapse_to_axon(
            spec_path=spec_path,
            output_path=axon_path,
            module_name="tiny",
            force=False,
        )
    assert "overwrite" in str(exc_info.value)


def test_cli_axon_to_synapse_requires_yaml_output(tmp_path: Path) -> None:
    axon_path = tmp_path / "spec.axon"
    bad_output = tmp_path / "lowered.txt"
    axon_path.write_text(
        "tiny :: Tensor -> Tensor\ntiny x = do\n  y <- x\n  return y\n", encoding="utf-8"
    )

    with pytest.raises(typer.BadParameter) as exc_info:
        axon_to_synapse(axon_path=axon_path, output_path=bad_output, force=False)
    assert ".yaml" in str(exc_info.value)


def test_optional_input_defaults_to_none_in_emitted_code() -> None:
    spec = {
        "synapse": 1,
        "model": {
            "symbols": {},
            "inputs": {
                "x": {"shape": [], "optional": True},
            },
            "graph": [],
            "outputs": {"x_out": "x"},
        },
    }
    source = emit_model_code_from_synapse_spec(spec, class_name="OptionalModel")
    assert "x = env.get('x')" in source


def test_index_on_none_collection_is_none_safe() -> None:
    spec = {
        "synapse": 1,
        "model": {
            "symbols": {},
            "inputs": {"collection": {"shape": [], "optional": True}},
            "graph": [{"at0": {"op": "index", "in": ["collection", 0], "out": "x0"}}],
            "outputs": {"x0": "x0"},
        },
    }
    source = emit_model_code_from_synapse_spec(spec, class_name="IndexSafeModel")
    namespace: dict[str, object] = {}
    exec(source, namespace)  # noqa: S102 - generated test code
    model = namespace["IndexSafeModel"]()
    out = model()
    assert out["x0"] is None


def test_emit_repeat_block_single_output_loop_carry() -> None:
    spec = {
        "synapse": 1,
        "model": {
            "symbols": {"L": 3},
            "inputs": {"zero": {"shape": []}, "one_seed": {"shape": []}},
            "blocks": {
                "step": {
                    "inputs": {"x": {"shape": []}, "one": {"shape": []}},
                    "graph": [{"inc": {"op": "add", "in": ["x", "one"], "out": "y"}}],
                    "outputs": {"y": "y"},
                }
            },
            "graph": [
                {"init": {"op": "add", "in": ["zero", "zero"], "out": "x"}},
                {"one_make": {"op": "add", "in": ["zero", "one_seed"], "out": "one"}},
                {
                    "loop": {
                        "op": "repeat",
                        "var": "i",
                        "range": "L",
                        "body": [
                            {
                                "blk": {
                                    "use": "step",
                                    "in": {"x": "x", "one": "one"},
                                    "out": {"y": "x"},
                                }
                            }
                        ],
                    }
                },
            ],
            "outputs": {"result": "x"},
        },
    }
    source = emit_model_code_from_synapse_spec(spec, class_name="LoopModel")
    namespace: dict[str, object] = {}
    exec(source, namespace)  # noqa: S102 - generated test code
    model = namespace["LoopModel"]()
    out = model(zero=torch.tensor(0.0), one_seed=torch.tensor(1.0))
    assert torch.is_tensor(out["result"])
    assert float(out["result"]) == 3.0
