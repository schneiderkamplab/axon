from __future__ import annotations

import pytest
import torch

from brainsurgery.engine import (
    RuntimeFlagLifecycleScope,
    reset_runtime_flags_for_scope,
    set_runtime_flag,
)
from brainsurgery.engine.arena import _SegmentedFileBackedArena
from brainsurgery.engine.state_dicts import _ArenaStateDict, _InMemoryStateDict


def test_inmemory_dry_run_overlay_paths() -> None:
    reset_runtime_flags_for_scope(RuntimeFlagLifecycleScope.CLI_RUN)
    sd = _InMemoryStateDict()
    sd["x"] = torch.tensor([1.0])

    set_runtime_flag("dry_run", True)

    # __setitem__ dry-run path
    sd["y"] = torch.tensor([2.0])
    assert "y" in set(sd.keys())

    # slot() dry-run path delegates to __getitem__
    assert torch.equal(sd.slot("x"), torch.tensor([1.0]))

    # bind_slot() dry-run path
    sd.bind_slot("z", torch.tensor([3.0]))
    assert torch.equal(sd["z"], torch.tensor([3.0]))

    # __delitem__ dry-run and deleted-key read path
    del sd["z"]
    with pytest.raises(KeyError):
        _ = sd["z"]

    # mark read/write should not update counters in dry-run
    before = sd.access_counts("x")
    _ = sd["x"]
    sd.mark_write("x")
    sd._mark_read("x")  # exercise explicit dry-run read-counter no-op branch
    assert sd.access_counts("x") == before

    # leaving dry-run clears overlay
    reset_runtime_flags_for_scope(RuntimeFlagLifecycleScope.CLI_RUN)
    assert "y" not in sd
    assert "z" not in sd


def test_arena_dry_run_overlay_paths(tmp_path) -> None:
    reset_runtime_flags_for_scope(RuntimeFlagLifecycleScope.CLI_RUN)
    arena = _SegmentedFileBackedArena(tmp_path, segment_size_bytes=1024, alignment=16)
    sd = _ArenaStateDict(arena)
    sd["x"] = torch.tensor([1.0])

    set_runtime_flag("dry_run", True)

    # __getitem__ dry-run clones from slot once
    assert torch.equal(sd["x"], torch.tensor([1.0]))

    # __setitem__ dry-run branch
    sd["y"] = torch.tensor([2.0])
    assert torch.equal(sd["y"], torch.tensor([2.0]))

    # bind_slot dry-run branch
    sd.bind_slot("z", sd.slot("x"))
    assert torch.equal(sd["z"], torch.tensor([1.0]))

    # deleted-key read branch in dry-run
    del sd["z"]
    with pytest.raises(KeyError):
        _ = sd["z"]

    reset_runtime_flags_for_scope(RuntimeFlagLifecycleScope.CLI_RUN)
    assert "y" not in sd
    assert "z" not in sd
    arena.close()
