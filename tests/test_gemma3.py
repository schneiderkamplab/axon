from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from omegaconf import OmegaConf

from brainsurgery.synapse import SynapseProgramModel, emit_model_code_from_synapse_spec


def _auto_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


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


def test_generated_gemma3_matches_hf(repo_root: Path, gemma3_local_path: Path) -> None:
    transformers = pytest.importorskip("transformers")
    safetensors = pytest.importorskip("safetensors")
    device = _auto_device()

    spec_path = repo_root / "examples" / "gemma3_270m_synapse.yaml"
    weights_path = gemma3_local_path / "model.safetensors"
    if not weights_path.exists():
        pytest.skip(f"missing Gemma3 checkpoint: {weights_path}")

    st = safetensors.safe_open(str(weights_path), framework="pt")
    state_dict = {
        (key[len("model.") :] if key.startswith("model.") else key): st.get_tensor(key).to(
            device=device, dtype=torch.float32
        )
        for key in st.keys()
    }

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        str(gemma3_local_path), local_files_only=True
    )
    hf_model = (
        transformers.AutoModelForCausalLM.from_pretrained(
            str(gemma3_local_path), local_files_only=True, dtype=torch.float32
        )
        .to(device)
        .eval()
    )

    # Match deterministic greedy decoding behavior used by the synapse generated model.
    hf_model.generation_config.do_sample = False
    hf_model.generation_config.top_p = None
    hf_model.generation_config.top_k = None

    synapse_model = (
        _build_model_from_spec(spec_path, "Gemma3Generated", state_dict).to(device).eval()
    )
    runtime_model = _build_runtime_model_from_spec(spec_path, state_dict).to(device).eval()

    inputs = tokenizer("I eat my own", return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    max_len = input_ids.shape[1] + 12

    with torch.no_grad():
        synapse_logits = synapse_model(input_ids)
        runtime_logits = runtime_model(input_ids)
        hf_logits = hf_model(input_ids=input_ids, use_cache=False).logits

    diff = (synapse_logits - hf_logits).abs()
    assert float(diff.mean()) < 1.0e-4
    assert float(diff.max()) < 5.0e-4
    assert torch.equal(synapse_logits[:, -1, :].argmax(-1), hf_logits[:, -1, :].argmax(-1))
    runtime_diff = (runtime_logits - hf_logits).abs()
    assert float(runtime_diff.mean()) < 1.0e-4
    assert float(runtime_diff.max()) < 5.0e-4
    assert torch.equal(runtime_logits[:, -1, :].argmax(-1), hf_logits[:, -1, :].argmax(-1))
    syn_rt_diff = (synapse_logits - runtime_logits).abs()
    assert float(syn_rt_diff.mean()) < 1.0e-6
    assert float(syn_rt_diff.max()) < 1.0e-5

    synapse_generated = synapse_model.generate(
        input_ids, eos_token_id=tokenizer.eos_token_id, max_len=max_len
    )
    runtime_generated = runtime_model.generate(
        input_ids, eos_token_id=tokenizer.eos_token_id, max_len=max_len
    )
    hf_generated = hf_model.generate(
        **inputs,
        max_new_tokens=max(1, max_len - int(input_ids.shape[1])),
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )
    assert torch.equal(synapse_generated, hf_generated)
    assert torch.equal(runtime_generated, hf_generated)
    assert torch.equal(runtime_generated, synapse_generated)
