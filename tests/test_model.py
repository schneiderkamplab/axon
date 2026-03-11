from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from brainsurgery.engine.checkpoint_io import (
    load_state_dict_from_directory,
    load_state_dict_from_file,
    resolve_safetensor_shards_from_index,
    save_sharded_safetensors,
)


def test_resolve_safetensor_shards_from_index_rejects_unsafe_path(tmp_path: Path) -> None:
    index_path = tmp_path / "model.safetensors.index.json"
    index_path.write_text(
        json.dumps({"weight_map": {"x": "../outside.safetensors"}}),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="unsafe shard path"):
        resolve_safetensor_shards_from_index(index_path, tmp_path)


def test_save_sharded_safetensors_writes_index(tmp_path: Path) -> None:
    state_dict = {
        "a.weight": torch.ones(8, dtype=torch.float32),
        "b.weight": torch.zeros(8, dtype=torch.float32),
    }
    index_path = save_sharded_safetensors(
        state_dict=state_dict,
        output_dir=tmp_path,
        max_shard_size=32,
        max_io_workers=1,
    )

    assert index_path.exists()
    data = json.loads(index_path.read_text(encoding="utf-8"))
    assert "weight_map" in data
    assert set(data["weight_map"]) == set(state_dict)


def test_resolve_safetensor_shards_from_index_requires_weight_map(tmp_path: Path) -> None:
    index_path = tmp_path / "model.safetensors.index.json"
    index_path.write_text(json.dumps({"meta": {}}), encoding="utf-8")

    with pytest.raises(RuntimeError, match="missing weight_map"):
        resolve_safetensor_shards_from_index(index_path, tmp_path)


def test_load_state_dict_from_directory_rejects_mixed_file_types(tmp_path: Path) -> None:
    torch.save({"a": torch.ones(1)}, tmp_path / "model.pt")
    save_sharded_safetensors(
        state_dict={"b": torch.ones(1)},
        output_dir=tmp_path,
        max_shard_size=1024,
        max_io_workers=1,
    )

    with pytest.raises(RuntimeError, match="contains both torch and safetensors files"):
        load_state_dict_from_directory(tmp_path, {}, max_io_workers=1)


def test_load_state_dict_from_file_rejects_duplicate_tensor_key(tmp_path: Path) -> None:
    file_path = tmp_path / "one.pt"
    torch.save({"dup": torch.ones(1)}, file_path)
    with pytest.raises(RuntimeError, match="duplicate tensor key"):
        load_state_dict_from_file(file_path, {"dup": torch.zeros(1)})
