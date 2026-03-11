from __future__ import annotations

from pathlib import Path

import pytest

from brainsurgery.providers.arena import ProviderError
from brainsurgery.providers import InMemoryStateDictProvider


def test_get_state_dict_rejects_unknown_model_alias(tmp_path: Path) -> None:
    provider = InMemoryStateDictProvider({"known": tmp_path / "model.safetensors"}, max_io_workers=1)
    with pytest.raises(ProviderError, match="unknown model alias"):
        provider.get_state_dict("unknown")
