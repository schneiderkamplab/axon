from __future__ import annotations

import pytest
import torch

from brainsurgery.engine.arena import ProviderError
from brainsurgery.engine.arena import _SegmentedFileBackedArena
from brainsurgery.engine.state_dicts import _ArenaStateDict
from brainsurgery.engine.state_dicts import _InMemoryStateDict
from brainsurgery.engine.providers import InMemoryStateDictProvider
from brainsurgery.engine import (
    create_state_dict_provider,
)

def test_inmemory_state_dict_enforces_tensor_values() -> None:
    state_dict = _InMemoryStateDict()
    state_dict["weight"] = torch.ones(2)
    assert torch.equal(state_dict["weight"], torch.ones(2))

    with pytest.raises(ProviderError, match="not a tensor"):
        state_dict["bad"] = object()  # type: ignore[assignment]

def test_inmemory_state_dict_tracks_reads_and_writes() -> None:
    state_dict = _InMemoryStateDict()

    state_dict["weight"] = torch.ones(2)
    _ = state_dict["weight"]
    state_dict.mark_write("weight")

    assert state_dict.access_counts("weight") == {"reads": 1, "writes": 2}

def test_arena_state_dict_tracks_reads_and_writes(tmp_path) -> None:
    arena = _SegmentedFileBackedArena(tmp_path, segment_size_bytes=1024, alignment=16)
    state_dict = _ArenaStateDict(arena)

    state_dict["weight"] = torch.ones(2)
    _ = state_dict["weight"]
    state_dict.mark_write("weight")

    assert state_dict.access_counts("weight") == {"reads": 1, "writes": 2}
    arena.close()

def test_provider_get_or_create_alias_state_dict_creates_new_alias() -> None:
    provider = InMemoryStateDictProvider({}, max_io_workers=1)
    created = provider.get_or_create_alias_state_dict("scratch")
    created["weight"] = torch.ones(1)

    assert provider.has_model_alias("scratch") is True
    assert torch.equal(provider.get_state_dict("scratch")["weight"], torch.ones(1))

def test_create_state_dict_provider_supports_inmemory_and_arena(tmp_path) -> None:
    inmemory = create_state_dict_provider(
        provider="inmemory",
        model_paths={},
        max_io_workers=1,
        arena_root=tmp_path,
        arena_segment_size="1KB",
    )
    assert isinstance(inmemory, InMemoryStateDictProvider)

    arena_provider = create_state_dict_provider(
        provider="arena",
        model_paths={},
        max_io_workers=1,
        arena_root=tmp_path,
        arena_segment_size="1KB",
    )
    arena_provider.close()

    with pytest.raises(ProviderError, match="either 'inmemory' or 'arena'"):
        create_state_dict_provider(
            provider="unknown",
            model_paths={},
            max_io_workers=1,
            arena_root=tmp_path,
            arena_segment_size="1KB",
        )
