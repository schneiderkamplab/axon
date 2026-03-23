from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest

_HF_API = "https://huggingface.co/api/models"
_HF_RESOLVE = "https://huggingface.co/{repo_id}/resolve/main/{filename}"
_ESSENTIAL_TEXT_FILES = {
    "config.json",
    "generation_config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
}


@dataclass(frozen=True)
class ModelDownloadSpec:
    local_dir: str
    repo_id: str
    require_tokenizer: bool = True


MODEL_SPECS: dict[str, ModelDownloadSpec] = {
    "gpt2.old": ModelDownloadSpec(local_dir="gpt2.old", repo_id="openai-community/gpt2"),
    "gemma3": ModelDownloadSpec(local_dir="gemma3", repo_id="google/gemma-3-270m"),
    "olmoe_1b_7b_0924": ModelDownloadSpec(
        local_dir="olmoe_1b_7b_0924", repo_id="allenai/OLMoE-1B-7B-0924"
    ),
    "falcon_rw_1b": ModelDownloadSpec(local_dir="falcon_rw_1b", repo_id="tiiuae/falcon-rw-1b"),
    "llama3_2_1b": ModelDownloadSpec(local_dir="llama3_2_1b", repo_id="meta-llama/Llama-3.2-1B"),
    "mistral_7b_v0_1": ModelDownloadSpec(
        local_dir="mistral_7b_v0_1", repo_id="mistralai/Mistral-7B-v0.1"
    ),
    "qwen2_5_0_5b": ModelDownloadSpec(local_dir="qwen2_5_0_5b", repo_id="Qwen/Qwen2.5-0.5B"),
    "flexmath": ModelDownloadSpec(local_dir="flexmath", repo_id="allenai/Flex-math-2x7B-1T"),
}

MATRIX_AXON_TO_MODEL_DIR: dict[str, str] = {
    "falcon_rw_1b": "falcon_rw_1b",
    "flexolmo": "flexmath",
    "gemma3_270m": "gemma3",
    "gpt2": "gpt2.old",
    "gpt2_kv": "gpt2.old",
    "llama3_2_1b": "llama3_2_1b",
    "mistral_7b_v0_1": "mistral_7b_v0_1",
    "olmoe_1b_7b_0924": "olmoe_1b_7b_0924",
    "qwen2_5_0_5b": "qwen2_5_0_5b",
}


def _status(config: pytest.Config, message: str) -> None:
    reporter = config.pluginmanager.get_plugin("terminalreporter")
    if reporter is not None:
        reporter.write_line(f"[model-download] {message}")
    else:
        print(f"[model-download] {message}", flush=True)


def _run_curl(
    *,
    url: str,
    out_path: Path,
    headers: list[str] | None = None,
    resume: bool,
    cwd: Path,
) -> None:
    cmd = ["curl", "-fL"]
    if resume:
        cmd.extend(["-C", "-"])
    if headers:
        for h in headers:
            cmd.extend(["-H", h])
    cmd.extend(["-o", str(out_path), url])
    run = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if run.returncode != 0:
        raise RuntimeError(f"curl failed for {url}\nstdout:\n{run.stdout}\nstderr:\n{run.stderr}")


def _auth_headers() -> list[str]:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        return []
    return [f"Authorization: Bearer {token}"]


def _load_hf_siblings(*, repo_id: str, cwd: Path) -> list[str]:
    headers = _auth_headers()
    api_url = f"{_HF_API}/{repo_id}"
    tmp = cwd / ".tmp_hf_model_api.json"
    _run_curl(url=api_url, out_path=tmp, headers=headers, resume=False, cwd=cwd)
    try:
        payload = json.loads(tmp.read_text(encoding="utf-8"))
    finally:
        tmp.unlink(missing_ok=True)
    siblings = payload.get("siblings")
    if not isinstance(siblings, list):
        raise RuntimeError(f"Unexpected HF API payload for {repo_id}: missing siblings")
    out: list[str] = []
    for item in siblings:
        if isinstance(item, dict) and isinstance(item.get("rfilename"), str):
            out.append(item["rfilename"])
    return out


def _is_complete_model_dir(model_dir: Path, *, require_tokenizer: bool) -> bool:
    index_path = model_dir / "model.safetensors.index.json"
    single_path = model_dir / "model.safetensors"

    has_weights = False
    if index_path.exists():
        try:
            payload = json.loads(index_path.read_text(encoding="utf-8"))
            weight_map = payload.get("weight_map")
            if not isinstance(weight_map, dict):
                return False
            shard_names = {str(v) for v in weight_map.values()}
            if not shard_names:
                return False
            has_weights = all((model_dir / shard).exists() for shard in shard_names)
        except json.JSONDecodeError:
            return False
    elif single_path.exists():
        has_weights = True

    if not has_weights:
        return False

    if not (model_dir / "config.json").exists():
        return False
    if require_tokenizer and not (model_dir / "tokenizer_config.json").exists():
        return False
    if require_tokenizer:
        has_tokenizer = (model_dir / "tokenizer.json").exists() or (
            (model_dir / "vocab.json").exists() and (model_dir / "merges.txt").exists()
        )
        if not has_tokenizer:
            return False

    return True


def ensure_model_downloaded(
    *,
    repo_root: Path,
    config: pytest.Config,
    spec: ModelDownloadSpec,
) -> Path:
    if shutil.which("curl") is None:
        raise RuntimeError("curl is required to download test models")

    model_dir = repo_root / "models" / spec.local_dir
    model_dir.mkdir(parents=True, exist_ok=True)

    if _is_complete_model_dir(model_dir, require_tokenizer=spec.require_tokenizer):
        _status(config, f"{spec.local_dir}: already complete, skipping download")
        return model_dir

    _status(config, f"{spec.local_dir}: fetching sibling manifest from {spec.repo_id}")
    siblings = _load_hf_siblings(repo_id=spec.repo_id, cwd=repo_root)

    shard_files = sorted(name for name in siblings if name.endswith(".safetensors"))
    index_files = [name for name in siblings if name == "model.safetensors.index.json"]
    selected_files = set(index_files + shard_files)
    selected_files.update(name for name in siblings if name in _ESSENTIAL_TEXT_FILES)

    headers = _auth_headers()
    for name in sorted(selected_files):
        dst = model_dir / name
        if dst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        _status(config, f"{spec.local_dir}: downloading {name}")
        _run_curl(
            url=_HF_RESOLVE.format(repo_id=spec.repo_id, filename=name),
            out_path=dst,
            headers=headers,
            resume=True,
            cwd=repo_root,
        )

    if not _is_complete_model_dir(model_dir, require_tokenizer=spec.require_tokenizer):
        raise RuntimeError(
            f"Model download incomplete for {spec.local_dir} ({spec.repo_id}) at {model_dir}"
        )

    _status(config, f"{spec.local_dir}: download complete")
    return model_dir


def ensure_gpt2_weights_alias(repo_root: Path, config: pytest.Config) -> Path:
    src = repo_root / "models" / "gpt2.old" / "model.safetensors"
    if not src.exists():
        raise RuntimeError(f"Missing source GPT-2 model.safetensors at {src}")

    dst = repo_root / "models" / "gpt2" / "model.safetensors"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        _status(config, "gpt2/model.safetensors already present, skipping")
        return dst

    _status(config, "creating gpt2/model.safetensors alias from gpt2.old/model.safetensors")
    shutil.copy2(src, dst)
    return dst


def ensure_matrix_models(repo_root: Path, config: pytest.Config) -> None:
    required_dirs = sorted(set(MATRIX_AXON_TO_MODEL_DIR.values()))
    for model_dir in required_dirs:
        spec = MODEL_SPECS.get(model_dir)
        if spec is None:
            raise RuntimeError(f"No download spec registered for matrix model dir: {model_dir}")
        ensure_model_downloaded(repo_root=repo_root, config=config, spec=spec)

    # Keep legacy GPT-2 single-file path working for tests that use models/gpt2/model.safetensors.
    ensure_gpt2_weights_alias(repo_root, config)
