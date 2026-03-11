from __future__ import annotations

import pytest
import torch

from brainsurgery.core import (
    PhloraSvdCache,
    compute_phlora_factors,
    reconstruct_phlora_rank,
    require_positive_rank,
)
from brainsurgery.core import TransformError
from brainsurgery.core.phlora import _require_matrix, _resolve_effective_rank


def test_require_positive_rank_rejects_non_integral_value() -> None:
    with pytest.raises(TransformError, match="positive integer"):
        require_positive_rank(1.5, error_type=TransformError, op_name="phlora", key="rank")


def test_require_matrix_rejects_non_2d_tensor() -> None:
    with pytest.raises(TransformError, match="must be 2D"):
        _require_matrix(
            torch.ones(3),
            error_type=TransformError,
            op_name="phlora",
            tensor_name="model::x",
        )


def test_compute_phlora_factors_shapes_match_requested_rank() -> None:
    source = torch.tensor([[3.0, 0.0], [0.0, 2.0]], dtype=torch.float32)
    factor_a, factor_b = compute_phlora_factors(
        source,
        1,
        cache=PhloraSvdCache(),
        cache_key="model::w",
        error_type=TransformError,
        op_name="phlora",
        tensor_name="model::w",
    )

    assert factor_a.shape == (1, 2)
    assert factor_b.shape == (2, 1)


def test_reconstruct_phlora_rank_truncates_to_effective_rank() -> None:
    source = torch.tensor([[3.0, 0.0], [0.0, 2.0]], dtype=torch.float32)
    reconstructed = reconstruct_phlora_rank(
        source,
        1,
        cache=PhloraSvdCache(),
        cache_key="model::w",
        error_type=TransformError,
        op_name="phlora_",
        tensor_name="model::w",
    )

    assert torch.allclose(reconstructed, torch.tensor([[3.0, 0.0], [0.0, 0.0]]))


def test_resolve_effective_rank_caps_to_tensor_shape() -> None:
    source = torch.ones((2, 3), dtype=torch.float32)
    assert _resolve_effective_rank(
        source,
        5,
        error_type=TransformError,
        op_name="phlora",
        tensor_name="model::w",
    ) == 2
