from __future__ import annotations

from pathlib import Path

import pytest
import torch

from brainsurgery.io import dcp as dcp_io
from brainsurgery.io import safetensors as safetensors_io
from brainsurgery.io import torch as torch_io

@pytest.mark.parametrize(
    "validator",
    [
        dcp_io._validate_state_dict_mapping,
        safetensors_io._validate_state_dict_mapping,
        torch_io._validate_state_dict_mapping,
    ],
)
def test_state_dict_validators_cover_non_mapping_non_tensor_and_success(validator) -> None:
    path = Path("/tmp/x")

    with pytest.raises(RuntimeError, match="not a state_dict mapping"):
        validator(123, path)

    with pytest.raises(RuntimeError, match="plain tensor state_dict"):
        validator({"bad": 1}, path)

    good = {"x": torch.ones(1)}
    loaded = validator(good, path)
    assert isinstance(loaded, dict)
    assert set(loaded) == {"x"}
    assert torch.equal(loaded["x"], torch.ones(1))
