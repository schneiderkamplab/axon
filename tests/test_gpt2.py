from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from omegaconf import OmegaConf

from brainsurgery.synapse import emit_model_code_from_synapse_spec


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


def _extract_logits(output: Any) -> torch.Tensor:
    if isinstance(output, dict):
        logits = output.get("logits")
        assert isinstance(logits, torch.Tensor)
        return logits
    assert isinstance(output, torch.Tensor)
    return output


@pytest.mark.parametrize(
    ("spec_name", "class_name"),
    [
        ("gpt2_synapse.yaml", "GPT2SynapseGenerated"),
        ("gpt2_kv_synapse.yaml", "GPT2KVSynapseGenerated"),
    ],
)
def test_generated_gpt2_variants_match_hf(
    repo_root: Path,
    gpt2_local_paths: tuple[Path, Path],
    spec_name: str,
    class_name: str,
) -> None:
    transformers = pytest.importorskip("transformers")
    safetensors = pytest.importorskip("safetensors")

    device = _auto_device()

    synapse_weights, hf_model_dir = gpt2_local_paths
    spec_path = repo_root / "examples" / spec_name

    st = safetensors.safe_open(str(synapse_weights), framework="pt")
    state_dict = {
        key: st.get_tensor(key).to(device=device, dtype=torch.float32) for key in st.keys()
    }

    tokenizer = transformers.AutoTokenizer.from_pretrained(str(hf_model_dir), local_files_only=True)
    hf_model = (
        transformers.AutoModelForCausalLM.from_pretrained(
            str(hf_model_dir), local_files_only=True, dtype=torch.float32
        )
        .to(device)
        .eval()
    )

    synapse_model = _build_model_from_spec(spec_path, class_name, state_dict).to(device).eval()

    inputs = tokenizer("I eat my own", return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    max_len = input_ids.shape[1] + 12

    with torch.no_grad():
        synapse_logits = _extract_logits(synapse_model(input_ids))
        hf_logits = hf_model(input_ids=input_ids).logits

    diff = (synapse_logits - hf_logits).abs()
    assert float(diff.mean()) < 1.0e-4
    assert float(diff.max()) < 5.0e-4
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
