from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from omegaconf import OmegaConf

from brainsurgery.synapse import emit_model_code_from_synapse_spec


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


def test_generated_olmoe_matches_hf(repo_root: Path, olmoe_local_path: Path) -> None:
    transformers = pytest.importorskip("transformers")

    spec_path = repo_root / "examples" / "olmoe_1b_7b_0924_synapse.yaml"
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        str(olmoe_local_path), local_files_only=True
    )
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        str(olmoe_local_path), local_files_only=True
    )
    hf_model.eval()

    state_dict = {
        (key[len("model.") :] if key.startswith("model.") else key): value
        for key, value in hf_model.state_dict().items()
    }
    synapse_model = _build_model_from_spec(spec_path, "OlmoeGenerated", state_dict)
    synapse_model.eval()

    inputs = tokenizer("I eat my own", return_tensors="pt")
    input_ids = inputs["input_ids"]
    max_len = input_ids.shape[1] + 8

    with torch.no_grad():
        synapse_logits = synapse_model(input_ids)
        hf_logits = hf_model(input_ids=input_ids).logits

    diff = (synapse_logits - hf_logits).abs()
    assert float(diff.mean()) < 0.1
    assert float(diff.max()) < 1.0
    assert torch.equal(synapse_logits[:, -1, :].argmax(-1), hf_logits[:, -1, :].argmax(-1))

    synapse_generated = synapse_model.generate(
        input_ids, eos_token_id=tokenizer.eos_token_id, max_len=max_len
    )
    hf_generated = hf_model.generate(
        **inputs,
        max_new_tokens=max(1, max_len - int(input_ids.shape[1])),
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )
    assert torch.equal(synapse_generated, hf_generated)
