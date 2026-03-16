from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import save_file as save_safetensors_file

from brainsurgery.engine.checkpoint_io import shard_state_dict
from brainsurgery.engine.output_paths import (
    _resolve_output_destination,
    _resolve_sharded_output_directory,
    parse_shard_size,
)
from brainsurgery.engine.plan import _OutputSpec
from brainsurgery.engine.tensor_files import load_tensor_from_path, save_tensor_to_path
from brainsurgery.io import infer_tensor_file_format
from brainsurgery.io import torch as torch_io


def test_resolve_output_destination_infers_directory_and_explicit_torch(tmp_path: Path) -> None:
    directory = tmp_path / "outdir"
    directory.mkdir()

    resolved = _resolve_output_destination(_OutputSpec(path=directory), default_shard_size="none")
    assert resolved == (directory / "model.safetensors", "safetensors", None)

    explicit = _resolve_output_destination(
        _OutputSpec(path=tmp_path / "model.pt", format="torch"),
        default_shard_size="none",
    )
    assert explicit == (tmp_path / "model.pt", "torch", None)

    dcp_explicit = _resolve_output_destination(
        _OutputSpec(path=tmp_path / "dcp_out", format="dcp"),
        default_shard_size="none",
    )
    assert dcp_explicit == (tmp_path / "dcp_out", "dcp", None)


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
        torch_io._validate_state_dict_mapping({"bad": 1}, tmp_path / "bad.pt")


def test_resolve_output_destination_rejects_incompatible_explicit_safetensors(
    tmp_path: Path,
) -> None:
    with pytest.raises(RuntimeError, match="incompatible"):
        _resolve_output_destination(
            _OutputSpec(path=tmp_path / "model.pt", format="safetensors"),
            default_shard_size="none",
        )


def test_resolve_output_destination_rejects_torch_directory_path(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    with pytest.raises(RuntimeError, match="requires a file path"):
        _resolve_output_destination(
            _OutputSpec(path=output_dir, format="torch"),
            default_shard_size="none",
        )


def test_resolve_output_destination_rejects_dcp_file_style_path(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="directory-style path"):
        _resolve_output_destination(
            _OutputSpec(path=tmp_path / "out.pt", format="dcp"),
            default_shard_size="none",
        )
    existing_file = tmp_path / "out_dcp"
    existing_file.write_text("x", encoding="utf-8")
    with pytest.raises(RuntimeError, match="directory path, not a file"):
        _resolve_output_destination(
            _OutputSpec(path=existing_file, format="dcp"),
            default_shard_size="none",
        )


def test_resolve_sharded_output_directory_rejects_file_style_path(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="directory-style output path"):
        _resolve_sharded_output_directory(
            original_path=tmp_path / "model.safetensors",
            resolved_path=tmp_path / "model.safetensors",
        )


def test_load_tensor_from_path_covers_npz_and_error_branches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    npz_path = tmp_path / "one.npz"
    np.savez(npz_path, only=np.array([1, 2], dtype=np.float32))
    assert torch.equal(load_tensor_from_path(npz_path), torch.tensor([1.0, 2.0]))

    multi_npz = tmp_path / "many.npz"
    np.savez(multi_npz, a=np.array([1]), b=np.array([2]))
    with pytest.raises(RuntimeError, match="exactly one array"):
        load_tensor_from_path(multi_npz)

    monkeypatch.setattr("brainsurgery.io.npy.np.load", lambda *args, **kwargs: object())
    with pytest.raises(RuntimeError, match="unsupported numpy payload"):
        load_tensor_from_path(tmp_path / "x.npy")


def test_load_tensor_from_path_covers_safetensors_and_torch_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    safe_path = tmp_path / "many.safetensors"
    save_safetensors_file({"a": torch.ones(1), "b": torch.zeros(1)}, str(safe_path))
    with pytest.raises(RuntimeError, match="exactly one tensor"):
        load_tensor_from_path(safe_path)

    torch_single = tmp_path / "single.pt"
    torch.save(torch.tensor([1, 2]), torch_single)
    assert torch.equal(load_tensor_from_path(torch_single), torch.tensor([1, 2]))

    torch_numpy = tmp_path / "array.pt"
    torch_numpy.write_bytes(b"placeholder")
    monkeypatch.setattr(
        "brainsurgery.io.torch.torch.load",
        lambda *args, **kwargs: np.array([1, 2], dtype=np.float32),
    )
    assert torch.equal(load_tensor_from_path(torch_numpy), torch.tensor([1.0, 2.0]))

    torch_many = tmp_path / "many.pt"
    torch_many.write_bytes(b"placeholder")
    monkeypatch.setattr(
        "brainsurgery.io.torch.torch.load",
        lambda *args, **kwargs: {"state_dict": {"a": torch.ones(1), "b": torch.zeros(1)}},
    )
    with pytest.raises(RuntimeError, match="exactly one tensor"):
        load_tensor_from_path(torch_many)

    bad = tmp_path / "bad.pt"
    bad.write_bytes(b"placeholder")
    monkeypatch.setattr("brainsurgery.io.torch.torch.load", lambda *args, **kwargs: "bad")
    with pytest.raises(RuntimeError, match="unsupported tensor payload"):
        load_tensor_from_path(bad)


def test_save_tensor_to_path_covers_all_formats(tmp_path: Path) -> None:
    tensor = torch.arange(4, dtype=torch.float32).reshape(2, 2)
    torch_out = tmp_path / "x.pt"
    np_out = tmp_path / "x.npy"
    safe_out = tmp_path / "x.safetensors"

    save_tensor_to_path("x", tensor, torch_out, format="torch")
    assert torch.equal(torch.load(torch_out), tensor)

    save_tensor_to_path("x", tensor, np_out, format="numpy")
    assert torch.equal(torch.from_numpy(np.load(np_out)), tensor)

    save_tensor_to_path("x", tensor, safe_out, format="safetensors")
    assert torch.equal(load_tensor_from_path(safe_out, format="safetensors"), tensor)


def test_output_paths_additional_branches(tmp_path: Path) -> None:
    out_file = tmp_path / "model.bin"
    resolved = _resolve_output_destination(_OutputSpec(path=out_file), default_shard_size="none")
    assert resolved == (out_file, "torch", None)

    with pytest.raises(RuntimeError, match="unsupported output format"):
        _resolve_output_destination(
            _OutputSpec(path=tmp_path / "out.abc"), default_shard_size="none"
        )

    with pytest.raises(RuntimeError, match="only supported for safetensors"):
        _resolve_output_destination(
            _OutputSpec(path=tmp_path / "out.pt", format="torch", shard="1KB"),
            default_shard_size="none",
        )
    with pytest.raises(RuntimeError, match="only supported for safetensors"):
        _resolve_output_destination(
            _OutputSpec(path=tmp_path / "dcp_out", format="dcp", shard="1KB"),
            default_shard_size="none",
        )

    dir_style = tmp_path / "dirstyle"
    assert _resolve_output_destination(
        _OutputSpec(path=dir_style, format="safetensors"),
        default_shard_size="none",
    ) == (dir_style / "model.safetensors", "safetensors", None)

    with pytest.raises(RuntimeError, match="requires a .pt, .pth, or .bin"):
        _resolve_output_destination(
            _OutputSpec(path=tmp_path / "out.safetensors", format="torch"),
            default_shard_size="none",
        )

    with pytest.raises(RuntimeError, match="non-empty string"):
        parse_shard_size("")

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    assert _resolve_sharded_output_directory(out_dir, out_dir / "model.safetensors") == out_dir
