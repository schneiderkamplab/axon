import torch

from brainsurgery.tensor_checks import require_same_shape_dtype_device, require_same_shape_dtype_device3
from brainsurgery.transform_types import TransformError


def test_require_same_shape_dtype_device_accepts_matching_tensors() -> None:
    left = torch.ones((2,), dtype=torch.float32)
    right = torch.ones((2,), dtype=torch.float32)
    require_same_shape_dtype_device(
        left,
        right,
        error_type=TransformError,
        op_name="assigning",
        left_name="a",
        right_name="b",
    )


def test_require_same_shape_dtype_device3_rejects_shape_mismatch() -> None:
    try:
        require_same_shape_dtype_device3(
            torch.ones((2,)),
            torch.ones((1,)),
            torch.ones((2,)),
            error_type=TransformError,
            op_name="adding",
            first_name="a",
            second_name="b",
            dest_name="dst",
            symbol="+",
        )
    except TransformError as exc:
        assert "shape mismatch" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected shape mismatch")
