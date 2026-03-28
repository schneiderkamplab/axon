from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any

import pytest
import torch

from brainsurgery.synapse import (
    SynapseProgramModel,
    lower_axon_program_to_synapse_spec,
    parse_axon_program_from_path,
)
from tests.synapse_test_utils import extract_logits
from tests.test_flags import LONG_TEST_ENV, run_long_tests_enabled

_RUN_LONG = run_long_tests_enabled()


def _normalize_rope_numeric_fields(config: Any) -> Any:
    def _normalize_dict(mapping: Any) -> None:
        if not isinstance(mapping, dict):
            return
        for key in ("factor", "beta_fast", "beta_slow", "mscale", "mscale_all_dim"):
            value = mapping.get(key)
            if isinstance(value, int) and not isinstance(value, bool):
                mapping[key] = float(value)

    _normalize_dict(getattr(config, "rope_scaling", None))
    _normalize_dict(getattr(config, "rope_parameters", None))
    return config


def _load_axon_spec(path: Path) -> dict[str, Any]:
    modules = parse_axon_program_from_path(path)
    return lower_axon_program_to_synapse_spec(modules)


def _load_state_dict_from_index(
    model_dir: Path, *, device: torch.device, dtype: torch.dtype
) -> dict[str, torch.Tensor]:
    safetensors = pytest.importorskip("safetensors")
    payload = json.loads((model_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))
    weight_map = payload.get("weight_map")
    assert isinstance(weight_map, dict)
    shard_names = sorted({str(name) for name in weight_map.values()})
    state_dict: dict[str, torch.Tensor] = {}
    for shard_name in shard_names:
        st = safetensors.safe_open(str(model_dir / shard_name), framework="pt")
        for key in st.keys():
            tensor = st.get_tensor(key)
            if tensor.is_floating_point():
                tensor = tensor.to(device=device, dtype=dtype)
            else:
                tensor = tensor.to(device=device)
            state_dict[key] = tensor
    return state_dict


@pytest.mark.skipif(not _RUN_LONG, reason=f"set {LONG_TEST_ENV}=1 to enable long tests")
def test_deepseek_v2_lite_runtime_logits_align_hf(
    repo_root: Path, deepseek_v2_lite_local_path: Path
) -> None:
    transformers = pytest.importorskip("transformers")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        str(deepseek_v2_lite_local_path), local_files_only=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    hf_config = transformers.AutoConfig.from_pretrained(
        str(deepseek_v2_lite_local_path), local_files_only=True
    )
    hf_config = _normalize_rope_numeric_fields(hf_config)
    hf_model = (
        transformers.AutoModelForCausalLM.from_pretrained(
            str(deepseek_v2_lite_local_path),
            local_files_only=True,
            torch_dtype=dtype,
            config=hf_config,
        )
        .to(device)
        .eval()
    )
    inputs = tokenizer(
        ["The future of AI is", "DeepSeek V2 Lite uses"], return_tensors="pt", padding=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    attention_mask = inputs.get("attention_mask")
    assert attention_mask is not None
    with torch.no_grad():
        hf_logits = hf_model(
            input_ids=inputs["input_ids"], attention_mask=attention_mask, use_cache=False
        ).logits
    del hf_model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    spec = _load_axon_spec(repo_root / "examples" / "deepseek_v2_lite.axon")
    state_dict = _load_state_dict_from_index(
        deepseek_v2_lite_local_path, device=device, dtype=dtype
    )
    runtime_model = SynapseProgramModel.from_spec(spec, state_dict=state_dict).to(device).eval()

    with torch.no_grad():
        runtime_logits = extract_logits(
            runtime_model(inputs["input_ids"], attn_mask=attention_mask)
        )

    runtime_logits = runtime_logits.to(hf_logits.device, dtype=hf_logits.dtype)
    diff = (runtime_logits - hf_logits).abs()
    mean_diff = float(diff.mean())
    max_diff = float(diff.max())
    top1_eq = bool((runtime_logits[:, -1, :].argmax(-1) == hf_logits[:, -1, :].argmax(-1)).all())
    assert mean_diff < 0.2 and max_diff < 2.0 and top1_eq, (
        "DeepSeek-V2-Lite logits are not aligned yet: "
        f"mean={mean_diff:.6f} max={max_diff:.6f} top1_eq={top1_eq}"
    )
