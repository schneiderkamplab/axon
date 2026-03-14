from __future__ import annotations

import gc
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file as save_safetensors_file

from brainsurgery.engine import create_state_dict_provider
from brainsurgery.engine.execution import execute_transform_pairs
from brainsurgery.engine.plan import compile_plan

def _toy_tensors() -> dict[str, torch.Tensor]:
    return {
        "wte.weight": torch.arange(24, dtype=torch.float32).reshape(6, 4),
        "ln_f.weight": torch.arange(4, dtype=torch.float32),
        "h.0.attn.c_attn.weight": torch.arange(48, dtype=torch.float32).reshape(4, 12),
        "h.0.attn.c_proj.weight": torch.arange(16, dtype=torch.float32).reshape(4, 4),
        "h.0.mlp.c_fc.bias": torch.arange(8, dtype=torch.float32),
        "h.0.mlp.c_proj.bias": torch.arange(4, dtype=torch.float32),
        "mat.left": torch.arange(6, dtype=torch.float32).reshape(2, 3),
        "mat.right": torch.arange(6, dtype=torch.float32).reshape(3, 2),
    }

def _write_checkpoint(path: Path, tensors: dict[str, torch.Tensor] | None = None) -> None:
    save_safetensors_file(_toy_tensors() if tensors is None else tensors, str(path))

def _execute_plan(
    provider_name: str,
    *,
    raw: dict[str, object],
    tmp_path: Path,
    arena_label: str,
    arena_segment_size: str = "1KB",
):
    provider = create_state_dict_provider(
        provider=provider_name,
        model_paths=compile_plan(raw).inputs,
        max_io_workers=2,
        arena_root=tmp_path / f"arena_{arena_label}",
        arena_segment_size=arena_segment_size,
    )
    try:
        should_continue, executed = execute_transform_pairs(
            zip(raw["transforms"], compile_plan(raw).transforms, strict=False),  # type: ignore[index]
            provider,
            interactive=False,
        )
        assert should_continue is True
        assert len(executed) == len(raw["transforms"])  # type: ignore[index]

        snapshots: dict[str, dict[str, torch.Tensor]] = {}
        counts: dict[str, dict[str, dict[str, int]]] = {}
        for alias in ("base", "work"):
            if alias in compile_plan(raw).inputs or alias == "work":
                try:
                    state_dict = provider.get_state_dict(alias)
                except Exception:
                    continue
                snapshots[alias] = {name: state_dict[name].clone() for name in sorted(state_dict.keys())}
                counts[alias] = {name: state_dict.access_counts(name) for name in sorted(state_dict.keys())}

        arena_root = None
        if provider_name == "arena":
            arena_root = provider.arena.root
        return snapshots, counts, arena_root, provider
    except Exception:
        provider.close()
        raise

def test_arena_temp_files_are_deleted_after_provider_lifecycle(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.safetensors"
    _write_checkpoint(checkpoint)
    raw = {
        "inputs": [f"base::{checkpoint}"],
        "transforms": [
            {"prefixes": {"mode": "add", "alias": "work"}},
            {"copy": {"from": r"base::(.+)", "to": r"work::\1"}},
            {"scale_": {"target": "work::wte.weight", "by": 1.0}},
        ],
    }

    _, _, arena_root, provider = _execute_plan("arena", raw=raw, tmp_path=tmp_path, arena_label="cleanup")
    assert arena_root is not None and arena_root.exists()
    provider.close()
    del provider
    gc.collect()
    assert not arena_root.exists()

def test_arena_and_inmemory_match_after_global_read_write_sequence(tmp_path: Path) -> None:
    checkpoint = tmp_path / "global_ops.safetensors"
    _write_checkpoint(checkpoint)
    raw = {
        "inputs": [f"base::{checkpoint}"],
        "transforms": [
            {"prefixes": {"mode": "add", "alias": "work"}},
            {"copy": {"from": r"base::(.+)", "to": r"work::\1"}},
            {"fill": {"from": "work::ln_f.weight", "to": "work::delta", "mode": "constant", "value": 2.0}},
            {"add_": {"from": "work::delta", "to": "work::ln_f.weight"}},
            {"scale_": {"target": "work::wte.weight", "by": 0.5}},
            {"reshape": {"from": "work::mat.left", "to": "work::mat.left.flat", "shape": [6]}},
            {"reshape_": {"target": "work::mat.left.flat", "shape": [2, 3]}},
            {"delete": {"target": "work::delta"}},
        ],
    }

    inmemory_snap, _, _, p_inmemory = _execute_plan("inmemory", raw=raw, tmp_path=tmp_path, arena_label="eq_inmemory")
    p_inmemory.close()
    arena_snap, _, _, p_arena = _execute_plan("arena", raw=raw, tmp_path=tmp_path, arena_label="eq_arena")
    p_arena.close()

    assert set(inmemory_snap["work"]) == set(arena_snap["work"])
    for name in sorted(inmemory_snap["work"]):
        assert torch.equal(inmemory_snap["work"][name], arena_snap["work"][name]), name

def test_arena_and_inmemory_have_same_access_counts_after_sequence(tmp_path: Path) -> None:
    checkpoint = tmp_path / "counts.safetensors"
    _write_checkpoint(checkpoint)
    raw = {
        "inputs": [f"base::{checkpoint}"],
        "transforms": [
            {"prefixes": {"mode": "add", "alias": "work"}},
            {"copy": {"from": r"base::(.+)", "to": r"work::\1"}},
            {"dump": {"target": "work::wte.weight", "format": "compact", "verbosity": "shape"}},
            {"assert": {"reads": {"of": "work::wte.weight", "ge": 1}}},
            {"assert": {"writes": {"of": "work::wte.weight", "ge": 1}}},
        ],
    }

    _, inmemory_counts, _, p_inmemory = _execute_plan("inmemory", raw=raw, tmp_path=tmp_path, arena_label="counts_inmemory")
    p_inmemory.close()
    _, arena_counts, _, p_arena = _execute_plan("arena", raw=raw, tmp_path=tmp_path, arena_label="counts_arena")
    p_arena.close()

    assert inmemory_counts["work"]["wte.weight"] == arena_counts["work"]["wte.weight"]

def test_arena_rolls_over_to_multiple_segments_for_large_writes(tmp_path: Path) -> None:
    checkpoint = tmp_path / "large.safetensors"
    tensors = _toy_tensors()
    tensors["big.weight"] = torch.arange(4096, dtype=torch.float32).reshape(1024, 4)
    tensors["big2.weight"] = torch.arange(4096, dtype=torch.float32).reshape(1024, 4)
    _write_checkpoint(checkpoint, tensors=tensors)

    raw = {
        "inputs": [f"base::{checkpoint}"],
        "transforms": [
            {"prefixes": {"mode": "add", "alias": "work"}},
            {"copy": {"from": r"base::(.+)", "to": r"work::\1"}},
        ],
    }

    snapshots, _, arena_root, provider = _execute_plan(
        "arena",
        raw=raw,
        tmp_path=tmp_path,
        arena_label="segments",
        arena_segment_size="20KB",
    )
    segment_files = sorted(arena_root.glob("segment-*.bin")) if arena_root is not None else []
    provider.close()
    assert len(segment_files) >= 2
    assert torch.equal(snapshots["work"]["big.weight"], tensors["big.weight"])

def test_arena_alias_isolation_matches_inmemory_for_move_delete(tmp_path: Path) -> None:
    checkpoint = tmp_path / "isolation.safetensors"
    original = _toy_tensors()
    _write_checkpoint(checkpoint, tensors=original)
    raw = {
        "inputs": [f"base::{checkpoint}"],
        "transforms": [
            {"prefixes": {"mode": "add", "alias": "work"}},
            {"copy": {"from": r"base::(.+)", "to": r"work::\1"}},
            {"move": {"from": "work::h.0.mlp.c_fc.bias", "to": "work::moved.bias"}},
            {"delete": {"target": "work::h.0.mlp.c_proj.bias"}},
        ],
    }

    inmemory_snap, _, _, p_inmemory = _execute_plan("inmemory", raw=raw, tmp_path=tmp_path, arena_label="iso_inmemory")
    p_inmemory.close()
    arena_snap, _, _, p_arena = _execute_plan("arena", raw=raw, tmp_path=tmp_path, arena_label="iso_arena")
    p_arena.close()

    assert set(inmemory_snap["work"]) == set(arena_snap["work"])
    assert "h.0.mlp.c_fc.bias" not in arena_snap["work"]
    assert "h.0.mlp.c_proj.bias" not in arena_snap["work"]
    assert torch.equal(arena_snap["work"]["moved.bias"], original["h.0.mlp.c_fc.bias"])

    # Base alias must remain untouched for both providers.
    for name, tensor in original.items():
        assert torch.equal(inmemory_snap["base"][name], tensor), name
        assert torch.equal(arena_snap["base"][name], tensor), name
