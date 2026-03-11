from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from brainsurgery.engine.checkpoint_io import (
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
