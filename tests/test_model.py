from __future__ import annotations

import builtins
import json
import warnings
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file as save_safetensors_file

import brainsurgery.engine.checkpoint_io as checkpoint_io
from brainsurgery.engine.checkpoint_io import (
    detect_torch_distributed_checkpoint_layout,
    load_state_dict_from_directory,
    load_state_dict_from_file,
    load_state_dict_from_torch_distributed_checkpoint,
    resolve_dcp_output_directory,
    resolve_safetensor_shards_from_index,
    save_sharded_safetensors,
    save_state_dict_to_path,
)
from brainsurgery.io import is_full_torch_distributed_tensor_storage_metadata


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


def test_load_state_dict_from_directory_loads_sharded_safetensors_with_index(
    tmp_path: Path,
) -> None:
    state_dict = {
        "a.weight": torch.ones(2, dtype=torch.float32),
        "b.weight": torch.zeros(2, dtype=torch.float32),
    }
    save_sharded_safetensors(
        state_dict=state_dict,
        output_dir=tmp_path,
        max_shard_size=8,
        max_io_workers=2,
    )

    loaded: dict[str, torch.Tensor] = {}
    load_state_dict_from_directory(tmp_path, loaded, max_io_workers=2)
    assert set(loaded) == set(state_dict)


def test_load_state_dict_from_directory_loads_safetensors_without_index(tmp_path: Path) -> None:
    save_safetensors_file({"x": torch.arange(3)}, str(tmp_path / "model.safetensors"))
    loaded: dict[str, torch.Tensor] = {}
    load_state_dict_from_directory(tmp_path, loaded, max_io_workers=1)
    assert torch.equal(loaded["x"], torch.arange(3))


def test_load_state_dict_from_directory_rejects_empty_checkpoint_file(tmp_path: Path) -> None:
    torch.save({}, tmp_path / "empty.pt")
    with pytest.raises(RuntimeError, match="contained zero tensors"):
        load_state_dict_from_directory(tmp_path, {}, max_io_workers=1)


def test_resolve_safetensor_shards_from_index_rejects_invalid_json(tmp_path: Path) -> None:
    index_path = tmp_path / "model.safetensors.index.json"
    index_path.write_text("{not json", encoding="utf-8")
    with pytest.raises(RuntimeError, match="failed to parse safetensors index"):
        resolve_safetensor_shards_from_index(index_path, tmp_path)


def test_resolve_safetensor_shards_from_index_rejects_missing_shard(tmp_path: Path) -> None:
    index_path = tmp_path / "model.safetensors.index.json"
    index_path.write_text(
        json.dumps({"weight_map": {"w": "missing.safetensors"}}),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="references missing shard"):
        resolve_safetensor_shards_from_index(index_path, tmp_path)


def test_load_state_dict_from_file_rejects_non_mapping_payload(tmp_path: Path) -> None:
    file_path = tmp_path / "bad.pt"
    torch.save(123, file_path)
    with pytest.raises(RuntimeError, match="is not a state_dict mapping"):
        load_state_dict_from_file(file_path, {})


def test_load_state_dict_from_directory_loads_torch_distributed_checkpoint(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "dcp"
    state_dict = {
        "a.weight": torch.arange(4, dtype=torch.float32),
        "b.bias": torch.ones(2, dtype=torch.float32),
    }
    _save_realistic_dcp_checkpoint(checkpoint_dir, state_dict)

    loaded: dict[str, torch.Tensor] = {}
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        load_state_dict_from_directory(checkpoint_dir, loaded, max_io_workers=1)

    messages = [str(w.message) for w in captured]
    assert not any("assuming the intent is to load in a single process" in msg for msg in messages)
    assert not any("TypedStorage is deprecated" in msg for msg in messages)

    assert set(loaded) == set(state_dict)
    for key, tensor in state_dict.items():
        assert torch.equal(loaded[key], tensor)


def test_detect_torch_distributed_checkpoint_layout_reports_full(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "dcp_layout"
    _save_realistic_dcp_checkpoint(checkpoint_dir, {"x": torch.arange(4, dtype=torch.float32)})

    layout = detect_torch_distributed_checkpoint_layout(checkpoint_dir)
    assert layout == "full"


def test_load_torch_distributed_checkpoint_falls_back_to_conversion(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    checkpoint_dir = tmp_path / "dcp_fallback"
    expected = {
        "x": torch.arange(3, dtype=torch.float32),
        "y": torch.ones(2, dtype=torch.float32),
    }
    _save_realistic_dcp_checkpoint(checkpoint_dir, expected)

    monkeypatch.setattr(
        "brainsurgery.engine.checkpoint_io._load_state_dict_from_torch_distributed_checkpoint_direct",
        lambda path: (_ for _ in ()).throw(RuntimeError("direct not supported")),
    )

    loaded: dict[str, torch.Tensor] = {}
    load_state_dict_from_torch_distributed_checkpoint(checkpoint_dir, loaded)

    assert set(loaded) == set(expected)
    for key, tensor in expected.items():
        assert torch.equal(loaded[key], tensor)


def test_load_torch_distributed_checkpoint_raises_when_both_paths_fail(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    checkpoint_dir = tmp_path / "dcp_both_fail"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(
        "brainsurgery.engine.checkpoint_io._load_state_dict_from_torch_distributed_checkpoint_direct",
        lambda path: (_ for _ in ()).throw(RuntimeError("direct fail")),
    )
    monkeypatch.setattr(
        "brainsurgery.engine.checkpoint_io._load_state_dict_from_torch_distributed_checkpoint_via_conversion",
        lambda path: (_ for _ in ()).throw(RuntimeError("conversion fail")),
    )
    with pytest.raises(RuntimeError, match="failed to load torch distributed checkpoint"):
        load_state_dict_from_torch_distributed_checkpoint(checkpoint_dir, {})


def test_load_torch_distributed_checkpoint_rejects_zero_tensors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    checkpoint_dir = tmp_path / "dcp_empty"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        "brainsurgery.engine.checkpoint_io._load_state_dict_from_torch_distributed_checkpoint_direct",
        lambda path: {},
    )
    with pytest.raises(RuntimeError, match="contained zero tensors"):
        load_state_dict_from_torch_distributed_checkpoint(checkpoint_dir, {})


def test_detect_torch_distributed_checkpoint_layout_returns_unknown_on_import_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    original_import = builtins.__import__

    def _raising_import(name, globals=None, locals=None, fromlist=(), level=0):  # type: ignore[no-untyped-def]
        if name == "torch.distributed.checkpoint":
            raise ImportError("no dcp")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _raising_import)
    assert detect_torch_distributed_checkpoint_layout(tmp_path / "missing") == "unknown"


def test_detect_torch_distributed_checkpoint_layout_returns_unknown_on_reader_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class _BrokenReader:
        def __init__(self, _path: str):
            pass

        def read_metadata(self) -> object:
            raise RuntimeError("boom")

    monkeypatch.setattr("torch.distributed.checkpoint.FileSystemReader", _BrokenReader)
    assert detect_torch_distributed_checkpoint_layout(tmp_path / "missing") == "unknown"


def test_detect_torch_distributed_checkpoint_layout_handles_non_dict_and_empty_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class _ReaderNonDict:
        def __init__(self, _path: str):
            pass

        def read_metadata(self) -> object:
            return SimpleNamespace(state_dict_metadata=[])

    class _ReaderEmpty:
        def __init__(self, _path: str):
            pass

        def read_metadata(self) -> object:
            return SimpleNamespace(state_dict_metadata={})

    monkeypatch.setattr("torch.distributed.checkpoint.FileSystemReader", _ReaderNonDict)
    assert detect_torch_distributed_checkpoint_layout(tmp_path / "x") == "unknown"

    monkeypatch.setattr("torch.distributed.checkpoint.FileSystemReader", _ReaderEmpty)
    assert detect_torch_distributed_checkpoint_layout(tmp_path / "x") == "unknown"


def test_detect_torch_distributed_checkpoint_layout_reports_sharded_and_mixed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from torch.distributed.checkpoint.metadata import (
        BytesStorageMetadata,
        ChunkStorageMetadata,
        TensorProperties,
        TensorStorageMetadata,
    )

    full = TensorStorageMetadata(
        properties=TensorProperties(dtype=torch.float32),
        size=torch.Size([2, 2]),
        chunks=[ChunkStorageMetadata(offsets=torch.Size([0, 0]), sizes=torch.Size([2, 2]))],
    )
    sharded = TensorStorageMetadata(
        properties=TensorProperties(dtype=torch.float32),
        size=torch.Size([2, 2]),
        chunks=[ChunkStorageMetadata(offsets=torch.Size([1, 0]), sizes=torch.Size([1, 2]))],
    )

    class _ReaderSharded:
        def __init__(self, _path: str):
            pass

        def read_metadata(self) -> object:
            return SimpleNamespace(state_dict_metadata={"x": sharded})

    class _ReaderMixed:
        def __init__(self, _path: str):
            pass

        def read_metadata(self) -> object:
            return SimpleNamespace(state_dict_metadata={"a": full, "meta": BytesStorageMetadata()})

    class _ReaderOther:
        def __init__(self, _path: str):
            pass

        def read_metadata(self) -> object:
            return SimpleNamespace(state_dict_metadata={"unknown": object()})

    monkeypatch.setattr("torch.distributed.checkpoint.FileSystemReader", _ReaderSharded)
    assert detect_torch_distributed_checkpoint_layout(tmp_path / "x") == "sharded"

    monkeypatch.setattr("torch.distributed.checkpoint.FileSystemReader", _ReaderMixed)
    assert detect_torch_distributed_checkpoint_layout(tmp_path / "x") == "mixed"

    monkeypatch.setattr("torch.distributed.checkpoint.FileSystemReader", _ReaderOther)
    assert detect_torch_distributed_checkpoint_layout(tmp_path / "x") == "sharded"


def test_is_full_tensor_storage_metadata_handles_invalid_chunks() -> None:
    entry = SimpleNamespace(chunks="not-a-list", size=torch.Size([1]))
    assert is_full_torch_distributed_tensor_storage_metadata(entry) is False


def test_load_dcp_direct_rejects_invalid_metadata_non_dict(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class _Reader:
        def __init__(self, _path: str):
            pass

        def read_metadata(self) -> object:
            return SimpleNamespace(state_dict_metadata=[])

    monkeypatch.setattr("torch.distributed.checkpoint.FileSystemReader", _Reader)
    with pytest.raises(RuntimeError, match="invalid torch distributed checkpoint metadata"):
        checkpoint_io._load_state_dict_from_torch_distributed_checkpoint_direct(tmp_path / "dcp")


def test_load_dcp_direct_rejects_invalid_key_and_non_tensor_entry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from torch.distributed.checkpoint.metadata import (
        ChunkStorageMetadata,
        TensorProperties,
        TensorStorageMetadata,
    )

    tensor_meta = TensorStorageMetadata(
        properties=TensorProperties(dtype=torch.float32),
        size=torch.Size([1]),
        chunks=[ChunkStorageMetadata(offsets=torch.Size([0]), sizes=torch.Size([1]))],
    )

    class _ReaderBadKey:
        def __init__(self, _path: str):
            pass

        def read_metadata(self) -> object:
            return SimpleNamespace(state_dict_metadata={1: tensor_meta})

    class _ReaderNonTensor:
        def __init__(self, _path: str):
            pass

        def read_metadata(self) -> object:
            return SimpleNamespace(state_dict_metadata={"x": object()})

    monkeypatch.setattr("torch.distributed.checkpoint.FileSystemReader", _ReaderBadKey)
    with pytest.raises(RuntimeError, match="invalid torch distributed checkpoint tensor key"):
        checkpoint_io._load_state_dict_from_torch_distributed_checkpoint_direct(tmp_path / "dcp")

    monkeypatch.setattr("torch.distributed.checkpoint.FileSystemReader", _ReaderNonTensor)
    with pytest.raises(RuntimeError, match="contains non-tensor entry"):
        checkpoint_io._load_state_dict_from_torch_distributed_checkpoint_direct(tmp_path / "dcp")


def test_load_dcp_direct_rejects_missing_dtype(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from torch.distributed.checkpoint.metadata import (
        ChunkStorageMetadata,
        TensorProperties,
        TensorStorageMetadata,
    )

    tensor_meta = TensorStorageMetadata(
        properties=TensorProperties(dtype=torch.float32),
        size=torch.Size([1]),
        chunks=[ChunkStorageMetadata(offsets=torch.Size([0]), sizes=torch.Size([1]))],
    )

    class _Reader:
        def __init__(self, _path: str):
            pass

        def read_metadata(self) -> object:
            return SimpleNamespace(state_dict_metadata={"x": tensor_meta})

    monkeypatch.setattr("torch.distributed.checkpoint.FileSystemReader", _Reader)
    monkeypatch.setattr(
        checkpoint_io.torch, "dtype", tuple
    )  # force isinstance(dtype, torch.dtype) to fail
    with pytest.raises(RuntimeError, match="missing dtype"):
        checkpoint_io._load_state_dict_from_torch_distributed_checkpoint_direct(tmp_path / "dcp")


def test_load_dcp_via_conversion_import_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    original_import = builtins.__import__

    def _raising_import(name, globals=None, locals=None, fromlist=(), level=0):  # type: ignore[no-untyped-def]
        if name == "torch.distributed.checkpoint.format_utils":
            raise ImportError("missing format utils")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _raising_import)
    with pytest.raises(RuntimeError, match="requires torch.distributed.checkpoint.format_utils"):
        checkpoint_io._load_state_dict_from_torch_distributed_checkpoint_via_conversion(
            tmp_path / "dcp"
        )


def test_load_dcp_via_conversion_unwraps_wrapped_state_dict(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    checkpoint_dir = tmp_path / "dcp_convert"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(
        "torch.distributed.checkpoint.format_utils.dcp_to_torch_save",
        lambda src, dst: None,
    )
    monkeypatch.setattr(
        checkpoint_io.torch,
        "load",
        lambda _path, map_location="cpu": {"state_dict": {"x": torch.ones(2)}},
    )

    loaded = checkpoint_io._load_state_dict_from_torch_distributed_checkpoint_via_conversion(
        checkpoint_dir
    )
    assert set(loaded) == {"x"}


def test_resolve_dcp_output_directory_rejects_file_and_extension(tmp_path: Path) -> None:
    file_path = tmp_path / "out.pt"
    file_path.write_text("x", encoding="utf-8")
    with pytest.raises(RuntimeError, match="directory path"):
        resolve_dcp_output_directory(file_path)
    with pytest.raises(RuntimeError, match="directory-style path"):
        resolve_dcp_output_directory(tmp_path / "out.safetensors")


def test_save_state_dict_to_path_dcp_rejects_file_path(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="directory-style path"):
        save_state_dict_to_path(
            {"x": torch.ones(1)},
            tmp_path / "dcp.pt",
            format="dcp",
        )


def test_save_state_dict_to_torch_distributed_checkpoint_import_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    original_import = builtins.__import__

    def _raising_import(name, globals=None, locals=None, fromlist=(), level=0):  # type: ignore[no-untyped-def]
        if name == "torch.distributed.checkpoint":
            raise ImportError("missing dcp")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _raising_import)
    with pytest.raises(RuntimeError, match="requires torch.distributed.checkpoint"):
        checkpoint_io.save_state_dict_to_torch_distributed_checkpoint(
            {"x": torch.ones(1)}, tmp_path / "dcp_out"
        )


def _save_realistic_dcp_checkpoint(path: Path, state_dict: dict[str, torch.Tensor]) -> None:
    dcp = pytest.importorskip("torch.distributed.checkpoint")
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=(
                r"torch\.distributed is disabled, unavailable or uninitialized, "
                r"assuming the intent is to save in a single process\."
            ),
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=r"TypedStorage is deprecated\..*",
            category=UserWarning,
        )
        dcp.save(state_dict, checkpoint_id=path, no_dist=True)
