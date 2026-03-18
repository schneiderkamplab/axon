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


def _spec_uses_model_prefix_for_embed(spec: dict[str, Any]) -> bool:
    graph = spec.get("model", {}).get("graph", [])
    if not isinstance(graph, list):
        return False
    for node in graph:
        if not isinstance(node, dict):
            continue
        node_spec = next(iter(node.values()), None)
        if not isinstance(node_spec, dict):
            continue
        tie_weight = node_spec.get("tie_weight")
        if isinstance(tie_weight, str) and tie_weight == "model.embed_tokens.weight":
            return True
        if node_spec.get("op") != "embedding":
            continue
        return node_spec.get("weight") == "model.embed_tokens.weight"
    return False


def _masked_logits_diff(
    lhs: torch.Tensor, rhs: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    token_mask = attention_mask.to(torch.bool).unsqueeze(-1).expand_as(lhs)
    return (lhs - rhs).abs()[token_mask]


def _extract_logits(output: Any) -> torch.Tensor:
    if isinstance(output, dict):
        logits = output.get("logits")
        assert isinstance(logits, torch.Tensor)
        return logits
    assert isinstance(output, torch.Tensor)
    return output


@pytest.mark.parametrize(
    ("texts",),
    [
        (["I eat my own"],),
        (["The future of AI is", "Hello world"],),
    ],
)
def test_generated_gemma3_matches_hf(
    repo_root: Path, gemma3_local_path: Path, texts: list[str]
) -> None:
    transformers = pytest.importorskip("transformers")
    safetensors = pytest.importorskip("safetensors")
    device = _auto_device()

    spec_path = repo_root / "examples" / "gemma3_270m_synapse.yaml"
    spec = _load_yaml_mapping(spec_path)
    weights_path = gemma3_local_path / "model.safetensors"
    if not weights_path.exists():
        pytest.skip(f"missing Gemma3 checkpoint: {weights_path}")

    st = safetensors.safe_open(str(weights_path), framework="pt")
    if _spec_uses_model_prefix_for_embed(spec):
        state_dict = {
            key: st.get_tensor(key).to(device=device, dtype=torch.float32) for key in st.keys()
        }
    else:
        state_dict = {
            (key[len("model.") :] if key.startswith("model.") else key): st.get_tensor(key).to(
                device=device, dtype=torch.float32
            )
            for key in st.keys()
        }

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        str(gemma3_local_path), local_files_only=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

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

    inputs = tokenizer(texts, return_tensors="pt", padding=True).to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")
    assert attention_mask is not None
    max_len = input_ids.shape[1] + 12

    with torch.no_grad():
        synapse_logits = _extract_logits(synapse_model(input_ids, attn_mask=attention_mask))
        runtime_logits = _extract_logits(runtime_model(input_ids, attn_mask=attention_mask))
        hf_logits = hf_model(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=False
        ).logits

    syn_hf_diff = _masked_logits_diff(synapse_logits, hf_logits, attention_mask)
    assert float(syn_hf_diff.mean()) < 1.0e-4
    assert float(syn_hf_diff.max()) < 5.0e-4
    rt_hf_diff = _masked_logits_diff(runtime_logits, hf_logits, attention_mask)
    assert float(rt_hf_diff.mean()) < 1.0e-4
    assert float(rt_hf_diff.max()) < 5.0e-4
    syn_rt_diff = _masked_logits_diff(synapse_logits, runtime_logits, attention_mask)
    assert float(syn_rt_diff.mean()) < 1.0e-6
    assert float(syn_rt_diff.max()) < 1.0e-5

    b_idx = torch.arange(attention_mask.shape[0], device=attention_mask.device)
    seq_positions = torch.arange(attention_mask.shape[1], device=attention_mask.device).unsqueeze(0)
    last_idx = torch.where(attention_mask.to(torch.bool), seq_positions, -1).max(dim=-1).values
    syn_last_top1 = synapse_logits[b_idx, last_idx, :].argmax(-1)
    rt_last_top1 = runtime_logits[b_idx, last_idx, :].argmax(-1)
    hf_last_top1 = hf_logits[b_idx, last_idx, :].argmax(-1)
    assert torch.equal(syn_last_top1, hf_last_top1)
    assert torch.equal(rt_last_top1, hf_last_top1)
    assert torch.equal(rt_last_top1, syn_last_top1)

    synapse_generated = synapse_model.generate(
        input_ids,
        attn_mask=attention_mask,
        eos_token_id=tokenizer.eos_token_id,
        max_len=max_len,
    )
    runtime_generated = runtime_model.generate(
        input_ids,
        attn_mask=attention_mask,
        eos_token_id=tokenizer.eos_token_id,
        max_len=max_len,
    )
    hf_generated = hf_model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max(1, max_len - int(input_ids.shape[1])),
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )
    assert torch.equal(synapse_generated, hf_generated)
    assert torch.equal(runtime_generated, hf_generated)
    assert torch.equal(runtime_generated, synapse_generated)
