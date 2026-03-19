from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from omegaconf import OmegaConf

from brainsurgery.synapse import (
    SynapseProgramModel,
    emit_model_code_from_synapse_spec,
    lower_axon_program_to_synapse_spec,
    parse_axon_program,
)


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


def _load_axon_spec(path: Path) -> dict[str, Any]:
    modules = parse_axon_program(path.read_text(encoding="utf-8"))
    return lower_axon_program_to_synapse_spec(modules)


def _axon_uses_model_prefix_for_embed(repo_root: Path, axon_name: str) -> bool:
    spec = _load_axon_spec(repo_root / "examples" / axon_name)
    graph = spec.get("model", {}).get("graph", [])

    def _walk(items: list[Any], prefix: str) -> bool:
        for item in items:
            if not isinstance(item, dict) or len(item) != 1:
                continue
            node_name, node_spec = next(iter(item.items()))
            if not isinstance(node_spec, dict):
                continue
            node_path = f"{prefix}.{node_name}" if prefix else str(node_name)
            if node_spec.get("op") == "embedding" and node_path.startswith("model."):
                return True
            tie_weight = node_spec.get("tie_weight")
            if isinstance(tie_weight, str) and tie_weight.startswith("model."):
                return True
            nested = node_spec.get("graph")
            if isinstance(nested, list) and _walk(nested, node_path):
                return True
            body = node_spec.get("body")
            if isinstance(body, list) and _walk(body, node_path):
                return True
        return False

    return _walk(graph if isinstance(graph, list) else [], "")


def _build_codegen_model(
    spec: dict[str, Any], class_name: str, state_dict: dict[str, torch.Tensor]
) -> Any:
    source = emit_model_code_from_synapse_spec(spec, class_name=class_name)
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
    ("yaml_name", "axon_name", "expected_symbols"),
    [
        (
            "gpt2_synapse.yaml",
            "gpt2.axon",
            {
                "T": 1024,
                "D": 768,
                "L": 12,
                "H": 12,
                "V": None,
                "B": None,
                "S": None,
            },
        ),
        (
            "gemma3_270m_synapse.yaml",
            "gemma3_270m.axon",
            {
                "D": 640,
                "V": 262208,
                "L": 18,
                "H": 4,
                "KVH": 1,
                "QD": 1024,
                "KVD": 256,
                "FFN": 2048,
                "ROPE_PERIOD": 6,
                "THETA_BASE": 10000.0,
                "THETA_LONG": 1000000.0,
                "WIN_LOCAL": 512,
                "WIN_LONG": 32768,
                "ATTN_SCALE": 0.0625,
                "EMB_SCALE": 25.298221281347036,
                "B": None,
                "S": None,
            },
        ),
        (
            "olmoe_1b_7b_0924_synapse.yaml",
            "olmoe_1b_7b_0924.axon",
            {
                "D": 2048,
                "V": 50304,
                "L": 16,
                "H": 16,
                "HD": 128,
                "E": 64,
                "EPT": 8,
                "C": 4096,
                "EPS": 1.0e-05,
                "THETA": 10000.0,
                "ATTN_SCALE": 0.08838834764831845,
                "B": None,
                "S": None,
            },
        ),
    ],
)
def test_axon_files_roundtrip_to_original_synapse_spec(
    repo_root: Path, yaml_name: str, axon_name: str, expected_symbols: dict[str, object] | None
) -> None:
    expected = _load_yaml_mapping(repo_root / "examples" / yaml_name)
    lowered = _load_axon_spec(repo_root / "examples" / axon_name)
    assert lowered.get("synapse") == expected.get("synapse") == 1
    assert set(lowered.get("model", {}).get("inputs", {}).keys()) == set(
        expected.get("model", {}).get("inputs", {}).keys()
    )
    assert lowered.get("model", {}).get("outputs") == expected.get("model", {}).get("outputs")
    assert lowered.get("model", {}).get("symbols") == expected_symbols
    assert set(lowered.get("model", {}).get("blocks", {}).keys()) == set(
        expected.get("model", {}).get("blocks", {}).keys()
    )


@pytest.mark.parametrize(
    ("axon_name", "prompt"),
    [
        ("gpt2.axon", "I eat my own"),
        ("gemma3_270m.axon", "I eat my own"),
    ],
)
def test_codegen_and_runtime_from_axon_match_hf_decoder_models(
    repo_root: Path,
    gpt2_local_paths: tuple[Path, Path],
    gemma3_local_path: Path,
    axon_name: str,
    prompt: str,
) -> None:
    transformers = pytest.importorskip("transformers")
    safetensors = pytest.importorskip("safetensors")
    device = _auto_device()

    if axon_name == "gpt2.axon":
        weights_path, hf_dir = gpt2_local_paths
        tokenizer = transformers.AutoTokenizer.from_pretrained(str(hf_dir), local_files_only=True)
        st = safetensors.safe_open(str(weights_path), framework="pt")
        state_dict = {
            key: st.get_tensor(key).to(device=device, dtype=torch.float32) for key in st.keys()
        }
    else:
        hf_dir = gemma3_local_path
        tokenizer = transformers.AutoTokenizer.from_pretrained(str(hf_dir), local_files_only=True)
        st = safetensors.safe_open(str(hf_dir / "model.safetensors"), framework="pt")
        if _axon_uses_model_prefix_for_embed(repo_root, axon_name):
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

    hf_model = (
        transformers.AutoModelForCausalLM.from_pretrained(
            str(hf_dir), local_files_only=True, dtype=torch.float32
        )
        .to(device)
        .eval()
    )
    hf_model.generation_config.do_sample = False
    hf_model.generation_config.top_p = None
    hf_model.generation_config.top_k = None

    spec = _load_axon_spec(repo_root / "examples" / axon_name)
    cg_model = (
        _build_codegen_model(spec, f"Axon{axon_name.replace('.', '_')}", state_dict)
        .to(device)
        .eval()
    )
    rt_model = SynapseProgramModel.from_spec(spec, state_dict=state_dict).to(device).eval()

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    max_len = input_ids.shape[1] + 8

    with torch.no_grad():
        cg_logits = _extract_logits(cg_model(input_ids))
        rt_logits = _extract_logits(rt_model(input_ids))
        hf_logits = hf_model(input_ids=input_ids, use_cache=False).logits

    diff_cg = (cg_logits - hf_logits).abs()
    diff_rt = (rt_logits - hf_logits).abs()
    assert float(diff_cg.mean()) < 1.0e-4
    assert float(diff_cg.max()) < 5.0e-4
    assert float(diff_rt.mean()) < 1.0e-4
    assert float(diff_rt.max()) < 5.0e-4
    assert torch.equal(cg_logits[:, -1, :].argmax(-1), hf_logits[:, -1, :].argmax(-1))
    assert torch.equal(rt_logits[:, -1, :].argmax(-1), hf_logits[:, -1, :].argmax(-1))

    cg_gen = cg_model.generate(input_ids, eos_token_id=tokenizer.eos_token_id, max_len=max_len)
    rt_gen = rt_model.generate(input_ids, eos_token_id=tokenizer.eos_token_id, max_len=max_len)
    hf_gen = hf_model.generate(
        **inputs,
        max_new_tokens=max(1, max_len - int(input_ids.shape[1])),
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )
    assert torch.equal(cg_gen, hf_gen)
    assert torch.equal(rt_gen, hf_gen)


def test_codegen_and_runtime_from_axon_match_hf_olmoe_cpu(
    repo_root: Path, olmoe_local_path: Path
) -> None:
    transformers = pytest.importorskip("transformers")
    safetensors = pytest.importorskip("safetensors")
    device = torch.device("cpu")

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

    index_path = olmoe_local_path / "model.safetensors.index.json"
    payload = OmegaConf.to_container(OmegaConf.load(index_path), resolve=True)
    assert isinstance(payload, dict)
    weight_map = payload.get("weight_map")
    assert isinstance(weight_map, dict)
    shard_names = sorted({str(name) for name in weight_map.values()})
    state_dict: dict[str, torch.Tensor] = {}
    for shard_name in shard_names:
        st = safetensors.safe_open(str(olmoe_local_path / shard_name), framework="pt")
        for key in st.keys():
            mapped = (
                key
                if _axon_uses_model_prefix_for_embed(repo_root, "olmoe_1b_7b_0924.axon")
                else (key[len("model.") :] if key.startswith("model.") else key)
            )
            state_dict[mapped] = st.get_tensor(key).to(device)
    spec = _load_axon_spec(repo_root / "examples" / "olmoe_1b_7b_0924.axon")
    cg_model = _build_codegen_model(spec, "AxonOlmoeGenerated", state_dict).to(device).eval()
    rt_model = SynapseProgramModel.from_spec(spec, state_dict=state_dict).to(device).eval()

    inputs = tokenizer("The future of AI is", return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    max_len = input_ids.shape[1] + 8

    with torch.no_grad():
        cg_logits = _extract_logits(cg_model(input_ids))
        rt_logits = _extract_logits(rt_model(input_ids))
        hf_logits = hf_model(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=False
        ).logits

    diff_cg = (cg_logits - hf_logits).abs()
    diff_rt = (rt_logits - hf_logits).abs()
    assert float(diff_cg.mean()) < 0.1
    assert float(diff_cg.max()) < 1.0
    assert float(diff_rt.mean()) < 0.1
    assert float(diff_rt.max()) < 1.0
    assert torch.equal(cg_logits[:, -1, :].argmax(-1), hf_logits[:, -1, :].argmax(-1))
    assert torch.equal(rt_logits[:, -1, :].argmax(-1), hf_logits[:, -1, :].argmax(-1))

    cg_gen = cg_model.generate(input_ids, eos_token_id=tokenizer.eos_token_id, max_len=max_len)
    rt_gen = rt_model.generate(input_ids, eos_token_id=tokenizer.eos_token_id, max_len=max_len)
    hf_gen = hf_model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max(1, max_len - int(input_ids.shape[1])),
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )
    assert torch.equal(cg_gen, hf_gen)
    assert torch.equal(rt_gen, hf_gen)
