from __future__ import annotations

import pytest
import torch

from brainsurgery.dtypes import parse_torch_dtype
from brainsurgery.transform_types import TransformError


class _Error(TransformError):
    pass


def test_parse_torch_dtype_supports_aliases_and_rejects_unknown() -> None:
    assert parse_torch_dtype("fp16", error_type=_Error, op_name="cast", field_name="dtype") is torch.float16
    assert parse_torch_dtype("long", error_type=_Error, op_name="cast", field_name="dtype") is torch.int64

    with pytest.raises(_Error, match="unsupported dtype"):
        parse_torch_dtype("complex64", error_type=_Error, op_name="cast", field_name="dtype")
