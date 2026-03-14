from __future__ import annotations

import pytest
import torch

from brainsurgery.engine.arena import ProviderError, _SegmentedFileBackedArena, ensure_supported_dtype, prod

def test_segmented_file_backed_arena_stores_and_reads_tensor_roundtrip(tmp_path) -> None:
    arena = _SegmentedFileBackedArena(tmp_path, segment_size_bytes=1024, alignment=16)

    source = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    slot = arena.store_tensor(source)
    restored = arena.tensor_from_slot(slot)

    assert torch.equal(restored, source)
    assert slot.offset % 16 == 0
    arena.close()

def test_arena_allocate_rejects_invalid_sizes(tmp_path) -> None:
    arena = _SegmentedFileBackedArena(tmp_path, segment_size_bytes=32, alignment=8)

    with pytest.raises(ProviderError, match="zero-byte tensors"):
        arena.allocate(0)

    with pytest.raises(ProviderError, match="exceeds segment size"):
        arena.allocate(64)

    arena.close()

def test_arena_dtype_helpers_cover_supported_and_rejected_values() -> None:
    ensure_supported_dtype(torch.float32)
    assert prod((2, 3, 4)) == 24

    with pytest.raises(ProviderError, match="unsupported dtype"):
        ensure_supported_dtype(torch.complex64)
