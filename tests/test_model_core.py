from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import save_file as save_safetensors_file

from brainsurgery.engine.tensor_files import infer_tensor_file_format, load_tensor_from_path
from brainsurgery.engine.output_paths import parse_shard_size, resolve_output_destination
from brainsurgery.engine.checkpoint_io import shard_state_dict, validate_state_dict_mapping
from brainsurgery.engine.plan import OutputSpec


def test_resolve_output_destination_infers_directory_and_explicit_torch(tmp_path: Path) -> None:
    directory = tmp_path / "outdir"
    directory.mkdir()

    resolved = resolve_output_destination(OutputSpec(path=directory), default_shard_size="none")
    assert resolved == (directory / "model.safetensors", "safetensors", None)

    explicit = resolve_output_destination(
        OutputSpec(path=tmp_path / "model.pt", format="torch"),
        default_shard_size="none",
    )
    assert explicit == (tmp_path / "model.pt", "torch", None)


def test_parse_shard_size_and_shard_state_dict_cover_boundaries() -> None:
    assert parse_shard_size("2KB") == 2048
    assert parse_shard_size("none") is None

    state_dict = {
        "a": torch.ones(4, dtype=torch.float32),
        "b": torch.ones(4, dtype=torch.float32),
        "c": torch.ones(4, dtype=torch.float32),
    }
    shards = shard_state_dict(state_dict, max_shard_size=16)
    assert [sorted(shard) for shard in shards] == [["a"], ["b"], ["c"]]

    with pytest.raises(RuntimeError, match="expected values like"):
        parse_shard_size("5XB")


def test_load_tensor_from_path_supports_numpy_safetensors_and_torch(tmp_path: Path) -> None:
    numpy_path = tmp_path / "tensor.npy"
    np.save(numpy_path, np.array([[1, 2], [3, 4]], dtype=np.float32))
    assert torch.equal(load_tensor_from_path(numpy_path), torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    assert infer_tensor_file_format(numpy_path) == "numpy"

    safe_path = tmp_path / "tensor.safetensors"
    save_safetensors_file({"weight": torch.ones(2, 2)}, str(safe_path))
    assert torch.equal(load_tensor_from_path(safe_path), torch.ones(2, 2))
    assert infer_tensor_file_format(safe_path) == "safetensors"

    torch_path = tmp_path / "tensor.pt"
    torch.save({"state_dict": {"weight": torch.arange(3)}}, torch_path)
    assert torch.equal(load_tensor_from_path(torch_path), torch.arange(3))
    assert infer_tensor_file_format(torch_path) == "torch"


def test_validate_state_dict_mapping_rejects_non_tensor_values(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="plain tensor state_dict"):
        validate_state_dict_mapping({"bad": 1}, tmp_path / "bad.pt")
