from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from omegaconf import OmegaConf

from brainsurgery.synapse import SynapseProgramModel, emit_model_code_from_synapse_spec


def _extract_logits(output: Any) -> torch.Tensor:
    if isinstance(output, dict):
        logits = output.get("logits")
        assert isinstance(logits, torch.Tensor)
        return logits
    assert isinstance(output, torch.Tensor)
    return output


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    loaded = OmegaConf.load(path)
    data = OmegaConf.to_container(loaded, resolve=True)
    assert isinstance(data, dict)
    return {str(key): value for key, value in data.items()}


def _build_model_from_spec(
    spec_path: Path, class_name: str, state_dict: dict[str, torch.Tensor]
) -> Any:
    source = emit_model_code_from_synapse_spec(_load_yaml_mapping(spec_path), class_name=class_name)
    namespace: dict[str, Any] = {}
    exec(source, namespace)  # noqa: S102 - test-controlled generated source
    model_cls = namespace[class_name]
    return model_cls.from_state_dict(state_dict)


def _build_runtime_model_from_spec(
    spec_path: Path, state_dict: dict[str, torch.Tensor]
) -> SynapseProgramModel:
    return SynapseProgramModel.from_spec(_load_yaml_mapping(spec_path), state_dict=state_dict)


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

    state_dict = {
        (key[len("model.") :] if key.startswith("model.") else key): value.to(device)
        for key, value in hf_model.state_dict().items()
    }
    synapse_model = (
        _build_model_from_spec(spec_path, "OlmoeGenerated", state_dict).to(device).eval()
    )
    runtime_model = _build_runtime_model_from_spec(spec_path, state_dict).to(device).eval()

    inputs = tokenizer("I eat my own", return_tensors="pt")
    syn_inputs = {k: v.to(device) for k, v in inputs.items()}
    hf_inputs = {k: v.to(device) for k, v in inputs.items()}
    input_ids = syn_inputs["input_ids"]
    max_len = input_ids.shape[1] + 8

    with torch.no_grad():
        synapse_logits = _extract_logits(synapse_model(syn_inputs["input_ids"]))
        runtime_logits = _extract_logits(runtime_model(syn_inputs["input_ids"]))
        hf_logits = hf_model(input_ids=hf_inputs["input_ids"]).logits

    synapse_logits = synapse_logits.to(hf_logits.device)
    runtime_logits = runtime_logits.to(hf_logits.device)
    diff = (synapse_logits - hf_logits).abs()
    assert float(diff.mean()) < 0.1
    assert float(diff.max()) < 1.0
    assert torch.equal(synapse_logits[:, -1, :].argmax(-1), hf_logits[:, -1, :].argmax(-1))
    runtime_diff = (runtime_logits - hf_logits).abs()
    assert float(runtime_diff.mean()) < 0.1
    assert float(runtime_diff.max()) < 1.0
    assert torch.equal(runtime_logits[:, -1, :].argmax(-1), hf_logits[:, -1, :].argmax(-1))
    syn_rt_diff = (synapse_logits - runtime_logits).abs()
    assert float(syn_rt_diff.mean()) < 1.0e-4
    assert float(syn_rt_diff.max()) < 2.0e-3

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
