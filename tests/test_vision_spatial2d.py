"""The channels-last 2D primitives: output-size typing and the Torch and MLX runtimes."""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from synapse.axon.ast import DimExprBinary
from synapse.axon.codegen2_torch import spatial2d as torch_runtime
from synapse.ops import OP_MODULES
from synapse.ops._spatial2d_common import transposed_out_dim, window_out_dim

# CUDA builds of MLX round convolutions to TF32 unless this is set before MLX starts.
os.environ.setdefault("MLX_ENABLE_TF32", "0")

WINDOWS = [(3, 2, 1, 1, 1), (2, 1, 0, 1, 1), (3, 3, 0, 0, 1), (7, 1, 3, 3, 1), (3, 2, 0, 0, 2), (2, 2, 0, 0, 1)]
CONVS = [
    # out, groups, kh, kw, stride_h, stride_w, top, bottom, left, right, dilation, bias
    (16, 1, 3, 3, 2, 2, 1, 1, 1, 1, 1, True),
    (8, 8, 3, 3, 1, 1, 1, 1, 1, 1, 1, True),
    (12, 1, 2, 2, 1, 1, 0, 1, 0, 1, 1, True),
    (8, 8, 3, 3, 2, 1, 1, 1, 1, 1, 1, False),
    (24, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, True),
    (24, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, False),
    (8, 8, 1, 7, 1, 1, 0, 0, 3, 3, 1, True),
    (16, 2, 3, 3, 1, 1, 2, 2, 2, 2, 2, True),
    (16, 1, 1, 1, 2, 2, 0, 0, 0, 0, 1, True),
    (8, 8, 1, 1, 1, 1, 0, 0, 0, 0, 1, True),
    (8, 8, 7, 7, 1, 1, 3, 3, 3, 3, 1, True),
    (8, 8, 3, 3, 2, 2, 1, 1, 1, 1, 1, True),
    (8, 8, 2, 2, 1, 1, 0, 1, 0, 1, 1, False),
    (8, 8, 3, 3, 1, 2, 2, 2, 2, 2, 2, True),
]
TRANSPOSED = [(6, 2, 2, 0, True), (1, 2, 2, 0, False), (6, 3, 2, 1, True), (6, 1, 1, 0, True), (5, 3, 3, 0, True)]
MAX_POOLS = [(2, 2, 1, 1, 0, 1, 0, 1), (3, 3, 2, 2, 1, 1, 1, 1), (2, 2, 2, 2, 0, 0, 0, 0)]
AVG_POOLS = [(3, 2, 3, 2), (2, 2, 2, 2), (3, 3, 1, 1)]


def _eval_dim(dim, size: int) -> int:
    if isinstance(dim, DimExprBinary):
        left, right = _eval_dim(dim.left, size), _eval_dim(dim.right, size)
        return {"+": left + right, "-": left - right, "*": left * right, "/": left // right}[dim.op]
    return size if dim == "H" else dim


@pytest.mark.parametrize("kernel,stride,before,after,dilation", WINDOWS)
def test_window_extent_matches_torch(kernel: int, stride: int, before: int, after: int, dilation: int) -> None:
    geometry = dict(kernel=kernel, stride=stride, pad_before=before, pad_after=after, dilation=dilation, axis="h")
    symbolic = window_out_dim("H", **geometry)
    for size in range(dilation * (kernel - 1) + 1, 40):
        padded = torch.zeros(1, 1, size + before + after)
        want = F.conv1d(padded, torch.zeros(1, 1, kernel), stride=stride, dilation=dilation).shape[-1]
        assert window_out_dim(size, **geometry) == want
        assert _eval_dim(symbolic, size) == want


def test_window_extent_keeps_same_padding_and_names_unknown_geometry() -> None:
    same = dict(kernel=3, stride=1, pad_before=1, pad_after=1, dilation=1, axis="h")
    assert window_out_dim("H", **same) == "H"
    unknown = dict(kernel="K", stride=1, pad_before=1, pad_after=1, dilation=1, axis="h")
    opaque = window_out_dim("H", **unknown)
    assert isinstance(opaque, str) and opaque != "H"
    assert window_out_dim("H", **unknown) == opaque
    assert window_out_dim("W", **unknown) != opaque


@pytest.mark.parametrize("kernel,stride,pad", [(2, 2, 0), (3, 2, 1), (4, 2, 1), (1, 1, 0)])
def test_transposed_extent_matches_torch(kernel: int, stride: int, pad: int) -> None:
    symbolic = transposed_out_dim("H", kernel=kernel, stride=stride, pad=pad, axis="h")
    for size in range(1, 12):
        want = F.conv_transpose1d(torch.zeros(1, 1, size), torch.zeros(1, 1, kernel), stride=stride, padding=pad).shape[-1]
        assert transposed_out_dim(size, kernel=kernel, stride=stride, pad=pad, axis="h") == want
        assert _eval_dim(symbolic, size) == want


def test_spatial_primitives_are_registered_and_emitted() -> None:
    assert torch_runtime.SPATIAL2D_PRIMITIVES <= set(OP_MODULES)
    assert set(torch_runtime.SPATIAL2D_RUNTIME) == torch_runtime.SPATIAL2D_PRIMITIVES
    from synapse.axon.codegen2_mlx import SUPPORTED_MLX_PRIMITIVES

    assert torch_runtime.SPATIAL2D_PRIMITIVES <= SUPPORTED_MLX_PRIMITIVES


def _inputs() -> tuple[np.random.Generator, np.ndarray]:
    rng = np.random.default_rng(0)
    return rng, rng.standard_normal((2, 13, 17, 8)).astype(np.float32)


def _first(x: torch.Tensor) -> torch.Tensor:
    return x.permute(0, 3, 1, 2)


def _last(x: torch.Tensor) -> np.ndarray:
    return x.permute(0, 2, 3, 1).numpy()


def test_torch_runtime_matches_channels_first_torch() -> None:
    rng, x = _inputs()
    xt = torch.from_numpy(x)
    for out, groups, kh, kw, sh, sw, top, bottom, left, right, dilation, bias in CONVS:
        w = torch.from_numpy(rng.standard_normal((out, 8 // groups, kh, kw)).astype(np.float32))
        b = torch.from_numpy(rng.standard_normal(out).astype(np.float32)) if bias else None
        want = F.conv2d(F.pad(_first(xt), (left, right, top, bottom)), w, b, stride=(sh, sw), dilation=dilation, groups=groups)
        got = torch_runtime._spatial_conv2d(xt, w, b, sh, sw, top, bottom, left, right, dilation, groups)
        np.testing.assert_allclose(got.numpy(), _last(want), atol=1e-5)
    for kh, kw, sh, sw, top, bottom, left, right in MAX_POOLS:
        want = F.max_pool2d(F.pad(_first(xt), (left, right, top, bottom), value=float("-inf")), (kh, kw), (sh, sw))
        got = torch_runtime._spatial_max_pool2d(xt, kh, kw, sh, sw, top, bottom, left, right)
        np.testing.assert_array_equal(got.numpy(), _last(want))
    np.testing.assert_allclose(
        torch_runtime._spatial_global_avg_pool2d(xt).numpy(), x.mean(axis=(1, 2), keepdims=True), atol=1e-6
    )
    ref = torch.zeros(2, 26, 51, 3)
    want = F.interpolate(_first(xt), size=(26, 51), mode="nearest")
    np.testing.assert_array_equal(torch_runtime._spatial_upsample_nearest2d_like(xt, ref).numpy(), _last(want))
    with pytest.raises(ValueError, match="integer scale"):
        torch_runtime._spatial_upsample_nearest2d_like(xt, torch.zeros(2, 20, 51, 3))


@pytest.mark.parametrize("shift_taps", [0, 49], ids=["grouped-conv", "shifted-sum"])
def test_mlx_runtime_matches_torch_runtime(shift_taps: int, monkeypatch: pytest.MonkeyPatch) -> None:
    mx = pytest.importorskip("mlx.core", exc_type=ImportError)
    from synapse.axon.codegen2_mlx import spatial2d as mlx_runtime

    monkeypatch.setattr(mlx_runtime, "_SPATIAL_DEPTHWISE_SHIFT_TAPS", shift_taps)

    rng, x = _inputs()
    xt, xm = torch.from_numpy(x), mx.array(x)

    def both(name: str, torch_args: tuple, mlx_args: tuple) -> None:
        want = getattr(torch_runtime, name)(xt, *torch_args)
        got = getattr(mlx_runtime, name)(xm, *mlx_args)
        assert tuple(got.shape) == tuple(want.shape), name
        np.testing.assert_allclose(np.array(got), want.numpy(), atol=2e-5, rtol=1e-5, err_msg=name)

    for out, groups, kh, kw, *geometry, bias in CONVS:
        w = rng.standard_normal((out, 8 // groups, kh, kw)).astype(np.float32)
        b = rng.standard_normal(out).astype(np.float32) if bias else None
        both(
            "_spatial_conv2d",
            (torch.from_numpy(w), None if b is None else torch.from_numpy(b), *geometry, groups),
            (mx.array(w), None if b is None else mx.array(b), *geometry, groups),
        )
    for out, kernel, stride, pad, bias in TRANSPOSED:
        w = rng.standard_normal((8, out, kernel, kernel)).astype(np.float32)
        b = rng.standard_normal(out).astype(np.float32) if bias else None
        both(
            "_spatial_conv_transpose2d",
            (torch.from_numpy(w), None if b is None else torch.from_numpy(b), stride, stride, pad, pad),
            (mx.array(w), None if b is None else mx.array(b), stride, stride, pad, pad),
        )
    for window in MAX_POOLS:
        both("_spatial_max_pool2d", window, window)
    for window in AVG_POOLS:
        both("_spatial_avg_pool2d", window, window)
    both("_spatial_global_avg_pool2d", (), ())
    both("_spatial_upsample_nearest2d_like", (torch.zeros(2, 26, 51, 3),), (mx.zeros((2, 26, 51, 3)),))


def test_global_avg_pool_does_not_overflow_in_half_precision() -> None:
    mx = pytest.importorskip("mlx.core", exc_type=ImportError)
    from synapse.axon.codegen2_mlx import spatial2d as mlx_runtime

    # 160 * 240 values of 4.0 sum to 153600, above the float16 maximum of 65504.
    x = np.full((1, 160, 240, 2), 4.0, dtype=np.float16)
    got = mlx_runtime._spatial_global_avg_pool2d(mx.array(x))
    assert got.dtype == mx.float16
    np.testing.assert_array_equal(np.array(got), np.full((1, 1, 1, 2), 4.0, dtype=np.float16))
    half = torch_runtime._spatial_global_avg_pool2d(torch.from_numpy(x))
    assert half.dtype == torch.float16
    np.testing.assert_array_equal(half.numpy(), np.full((1, 1, 1, 2), 4.0, dtype=np.float16))
