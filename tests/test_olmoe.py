from __future__ import annotations

from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from brainsurgery.synapse import SynapseProgramModel
from tests.synapse_test_utils import (
    assert_logits_close,
    build_codegen_model,
    extract_logits,
    load_yaml_mapping,
)
from tests.test_flags import LONG_TEST_ENV, run_long_tests_enabled

_RUN_LONG = run_long_tests_enabled()


def _build_runtime_model_from_spec(
    spec_path: Path, state_dict: dict[str, torch.Tensor]
) -> SynapseProgramModel:
    return SynapseProgramModel.from_spec(load_yaml_mapping(spec_path), state_dict=state_dict)


def _load_olmoe_state_dict_from_safetensors(
    olmoe_local_path: Path, device: torch.device
) -> dict[str, torch.Tensor]:
    payload = OmegaConf.to_container(
        OmegaConf.load(olmoe_local_path / "model.safetensors.index.json"), resolve=True
    )
    assert isinstance(payload, dict)
    weight_map = payload.get("weight_map")
    assert isinstance(weight_map, dict)
    shard_names = sorted({str(name) for name in weight_map.values()})
    safetensors = pytest.importorskip("safetensors")
    state_dict: dict[str, torch.Tensor] = {}
    for shard_name in shard_names:
        st = safetensors.safe_open(str(olmoe_local_path / shard_name), framework="pt")
        for key in st.keys():
            state_dict[key] = st.get_tensor(key).to(device)
    return state_dict


@pytest.mark.skipif(not _RUN_LONG, reason=f"set {LONG_TEST_ENV}=1 to enable long tests")
def test_generated_olmoe_matches_hf(repo_root: Path, olmoe_local_path: Path) -> None:
    transformers = pytest.importorskip("transformers")
    device = torch.device("cpu")

    spec_path = repo_root / "examples" / "olmoe_1b_7b_0924_synapse.yaml"
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        str(olmoe_local_path), local_files_only=True
    )
    hf_model = (
        transformers.AutoModelForCausalLM.from_pretrained(
            str(olmoe_local_path), local_files_only=True
        )
        .to(device)
        .eval()
    )

    state_dict = _load_olmoe_state_dict_from_safetensors(olmoe_local_path, device)
    synapse_model = (
        build_codegen_model(load_yaml_mapping(spec_path), "OlmoeGenerated", state_dict)
        .to(device)
        .eval()
    )
    runtime_model = _build_runtime_model_from_spec(spec_path, state_dict).to(device).eval()

    inputs = tokenizer("I eat my own", return_tensors="pt")
    syn_inputs = {k: v.to(device) for k, v in inputs.items()}
    hf_inputs = {k: v.to(device) for k, v in inputs.items()}
    input_ids = syn_inputs["input_ids"]
    max_len = input_ids.shape[1] + 8

    with torch.no_grad():
        synapse_logits = extract_logits(synapse_model(syn_inputs["input_ids"]))
        runtime_logits = extract_logits(runtime_model(syn_inputs["input_ids"]))
        hf_logits = hf_model(input_ids=hf_inputs["input_ids"]).logits

    synapse_logits = synapse_logits.to(hf_logits.device)
    runtime_logits = runtime_logits.to(hf_logits.device)
    assert_logits_close(synapse_logits, hf_logits, mean_tol=0.1, max_tol=1.0)
    assert_logits_close(runtime_logits, hf_logits, mean_tol=0.1, max_tol=1.0)
    assert_logits_close(synapse_logits, runtime_logits, mean_tol=1.0e-4, max_tol=2.0e-3)

    synapse_generated = synapse_model.generate(
        syn_inputs["input_ids"], eos_token_id=tokenizer.eos_token_id, max_len=max_len
    )
    runtime_generated = runtime_model.generate(
        syn_inputs["input_ids"], eos_token_id=tokenizer.eos_token_id, max_len=max_len
    )
    hf_generated = hf_model.generate(
        **hf_inputs,
        max_new_tokens=max(1, max_len - int(hf_inputs["input_ids"].shape[1])),
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )
    synapse_generated = synapse_generated.to(hf_generated.device)
    runtime_generated = runtime_generated.to(hf_generated.device)
    assert torch.equal(synapse_generated, hf_generated)
    assert torch.equal(runtime_generated, hf_generated)
    assert torch.equal(runtime_generated, synapse_generated)
