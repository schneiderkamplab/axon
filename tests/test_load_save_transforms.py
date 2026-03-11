from __future__ import annotations

from pathlib import Path

import json
import pytest
import torch
from safetensors.torch import load_file as load_safetensors_file
from safetensors.torch import save_file as save_safetensors_file

from brainsurgery.engine.execution import execute_transform_pairs
from brainsurgery.engine.plan import compile_plan
from brainsurgery.engine import create_state_dict_provider
import brainsurgery.transforms.save as save_module

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore[assignment]


def _write_checkpoint(path: Path, values: dict[str, torch.Tensor]) -> None:
    save_safetensors_file(values, str(path))


@pytest.mark.parametrize("provider_name", ["inmemory", "arena"])
def test_load_state_dict_then_save_state_dict(provider_name: str, tmp_path: Path) -> None:
    base_path = tmp_path / "base.safetensors"
    in_path = tmp_path / "additional.safetensors"
    out_path = tmp_path / "saved.pt"
    _write_checkpoint(base_path, {"base": torch.tensor([0.0], dtype=torch.float32)})
    _write_checkpoint(in_path, {"x": torch.tensor([1.0, 2.0], dtype=torch.float32)})

    raw = {
        "inputs": [f"base::{base_path}"],
        "transforms": [
            {"load": {"path": str(in_path), "alias": "extra"}},
            {"save": {"path": str(out_path), "alias": "extra", "format": "torch"}},
        ],
    }
    plan = compile_plan(raw)
    provider = create_state_dict_provider(
        provider=provider_name,
        model_paths=plan.inputs,
        max_io_workers=2,
        arena_root=tmp_path / "arena",
        arena_segment_size="1MB",
    )
    try:
        should_continue, executed = execute_transform_pairs(
            zip(raw["transforms"], plan.transforms, strict=False),
            provider,
            interactive=False,
        )
    finally:
        provider.close()

    assert should_continue is True
    assert len(executed) == 2
    assert out_path.exists()
    saved = torch.load(out_path, map_location="cpu")
    assert isinstance(saved, dict)
    assert "x" in saved
    assert torch.equal(saved["x"], torch.tensor([1.0, 2.0], dtype=torch.float32))


@pytest.mark.skipif(np is None, reason="numpy not available")
@pytest.mark.parametrize("provider_name", ["inmemory", "arena"])
def test_load_tensor_npy_then_save_tensor_npy(provider_name: str, tmp_path: Path) -> None:
    assert np is not None
    base_path = tmp_path / "base.safetensors"
    in_path = tmp_path / "tensor.npy"
    out_path = tmp_path / "tensor_out.npy"

    _write_checkpoint(base_path, {"base": torch.tensor([0.0], dtype=torch.float32)})
    np.save(in_path, np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))

    raw = {
        "inputs": [f"base::{base_path}"],
        "transforms": [
            {"load": {"path": str(in_path), "to": "base::loaded.tensor", "format": "numpy"}},
            {"save": {"path": str(out_path), "target": "base::loaded.tensor", "format": "numpy"}},
        ],
    }
    plan = compile_plan(raw)
    provider = create_state_dict_provider(
        provider=provider_name,
        model_paths=plan.inputs,
        max_io_workers=2,
        arena_root=tmp_path / "arena",
        arena_segment_size="1MB",
    )
    try:
        should_continue, executed = execute_transform_pairs(
            zip(raw["transforms"], plan.transforms, strict=False),
            provider,
            interactive=False,
        )
    finally:
        provider.close()

    assert should_continue is True
    assert len(executed) == 2
    assert out_path.exists()
    loaded = np.load(out_path, allow_pickle=False)
    assert loaded.shape == (2, 2)
    assert loaded.dtype == np.float32
    assert float(loaded[1, 1]) == 4.0


def test_save_requires_alias_when_multiple_models(tmp_path: Path) -> None:
    left = tmp_path / "left.safetensors"
    right = tmp_path / "right.safetensors"
    out = tmp_path / "out.safetensors"
    _write_checkpoint(left, {"x": torch.tensor([1.0], dtype=torch.float32)})
    _write_checkpoint(right, {"y": torch.tensor([2.0], dtype=torch.float32)})

    raw = {
        "inputs": [f"left::{left}", f"right::{right}"],
        "transforms": [
            {"save": {"path": str(out)}},
        ],
    }
    plan = compile_plan(raw)
    provider = create_state_dict_provider(
        provider="inmemory",
        model_paths=plan.inputs,
        max_io_workers=2,
        arena_root=tmp_path / "arena",
        arena_segment_size="1MB",
    )
    try:
        with pytest.raises(RuntimeError, match="save.alias is required"):
            execute_transform_pairs(
                zip(raw["transforms"], plan.transforms, strict=False),
                provider,
                interactive=False,
            )
    finally:
        provider.close()


def test_save_default_format_is_safetensors(tmp_path: Path) -> None:
    in_path = tmp_path / "in.safetensors"
    out_path = tmp_path / "out.safetensors"
    _write_checkpoint(in_path, {"x": torch.tensor([3.0], dtype=torch.float32)})

    raw = {
        "inputs": [str(in_path)],
        "transforms": [
            {"save": {"path": str(out_path)}},
        ],
    }
    plan = compile_plan(raw)
    provider = create_state_dict_provider(
        provider="inmemory",
        model_paths=plan.inputs,
        max_io_workers=2,
        arena_root=tmp_path / "arena",
        arena_segment_size="1MB",
    )
    try:
        should_continue, executed = execute_transform_pairs(
            zip(raw["transforms"], plan.transforms, strict=False),
            provider,
            interactive=False,
        )
    finally:
        provider.close()

    assert should_continue is True
    assert len(executed) == 1
    saved = load_safetensors_file(str(out_path))
    assert torch.equal(saved["x"], torch.tensor([3.0], dtype=torch.float32))


@pytest.mark.parametrize("provider_name", ["inmemory", "arena"])
def test_save_state_dict_sharded_writes_index(provider_name: str, tmp_path: Path) -> None:
    in_path = tmp_path / "in.safetensors"
    out_dir = tmp_path / "out_sharded"
    _write_checkpoint(
        in_path,
        {
            "a": torch.arange(600, dtype=torch.float32),
            "b": torch.arange(700, dtype=torch.float32),
            "c": torch.arange(800, dtype=torch.float32),
        },
    )

    raw = {
        "inputs": [str(in_path)],
        "transforms": [
            {"save": {"path": str(out_dir), "shard": "2KB"}},
        ],
    }
    plan = compile_plan(raw)
    provider = create_state_dict_provider(
        provider=provider_name,
        model_paths=plan.inputs,
        max_io_workers=3,
        arena_root=tmp_path / "arena",
        arena_segment_size="1MB",
    )
    try:
        should_continue, executed = execute_transform_pairs(
            zip(raw["transforms"], plan.transforms, strict=False),
            provider,
            interactive=False,
        )
    finally:
        provider.close()

    assert should_continue is True
    assert len(executed) == 1
    index_path = out_dir / "model.safetensors.index.json"
    assert index_path.exists()
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = payload["weight_map"]
    assert set(weight_map) == {"a", "b", "c"}
    assert len(set(weight_map.values())) >= 2


def test_save_shard_rejects_tensor_target(tmp_path: Path) -> None:
    in_path = tmp_path / "in.safetensors"
    _write_checkpoint(in_path, {"x": torch.tensor([3.0], dtype=torch.float32)})

    with pytest.raises(RuntimeError, match="save.shard is only supported for state_dict save"):
        compile_plan(
            {
                "inputs": [str(in_path)],
                "transforms": [{"save": {"path": str(tmp_path / "x"), "target": "model::x", "shard": "1MB"}}],
            }
        )


def test_save_shard_rejects_torch_format(tmp_path: Path) -> None:
    in_path = tmp_path / "in.safetensors"
    _write_checkpoint(in_path, {"x": torch.tensor([3.0], dtype=torch.float32)})

    with pytest.raises(
        RuntimeError,
        match="save.shard is only supported for safetensors state_dict save",
    ):
        compile_plan(
            {
                "inputs": [str(in_path)],
                "transforms": [{"save": {"path": str(tmp_path / "x.pt"), "format": "torch", "shard": "1MB"}}],
            }
        )


def test_save_shard_uses_provider_max_io_workers(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    in_path = tmp_path / "in.safetensors"
    out_dir = tmp_path / "out_sharded"
    _write_checkpoint(
        in_path,
        {"x": torch.arange(256, dtype=torch.float32)},
    )

    captured: dict[str, int] = {}

    def _fake_persist_state_dict(
        state_dict: dict[str, torch.Tensor],
        *,
        output_path: Path,
        output_format: str,
        shard_size: int | None,
        sharded_output_root: Path,
        max_io_workers: int,
    ) -> Path:
        del state_dict, output_path, output_format, shard_size, sharded_output_root
        captured["max_io_workers"] = max_io_workers
        return out_dir / "model.safetensors.index.json"

    monkeypatch.setattr(save_module, "persist_state_dict", _fake_persist_state_dict)

    raw = {
        "inputs": [str(in_path)],
        "transforms": [
            {"save": {"path": str(out_dir), "shard": "1KB"}},
        ],
    }
    plan = compile_plan(raw)
    provider = create_state_dict_provider(
        provider="inmemory",
        model_paths=plan.inputs,
        max_io_workers=7,
        arena_root=tmp_path / "arena",
        arena_segment_size="1MB",
    )
    try:
        should_continue, executed = execute_transform_pairs(
            zip(raw["transforms"], plan.transforms, strict=False),
            provider,
            interactive=False,
        )
    finally:
        provider.close()

    assert should_continue is True
    assert len(executed) == 1
    assert captured["max_io_workers"] == 7
