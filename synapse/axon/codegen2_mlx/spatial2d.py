"""MLX runtime for the channels-last 2D primitives.

The functions are embedded by source into emitted modules, so they depend only on the
``mx`` and ``nn`` names of the emitted module. MLX convolutions are channels-last, which
is also the Axon layout, so activations are never transposed; only the weights are moved
from the checkpoint layout, once per weight.
"""

from __future__ import annotations

try:
    import mlx.core as mx
    import mlx.nn as nn
except ImportError:  # the emitter itself runs without MLX installed
    mx = None
    nn = None

# (id(weight), form, dtype) -> (weight, derived weight). Holding the source weight keeps its
# id from being reused.
_SPATIAL_WEIGHTS: dict = {}

# Depthwise convolutions with at most this many kernel taps are computed as a sum of
# shifted, per-channel scaled copies of the input instead of a grouped convolution. The
# two forms are equal up to rounding; which is faster depends on the MLX backend, so the
# default (0) keeps the convolution and a host opts in by setting this module global.
_SPATIAL_DEPTHWISE_SHIFT_TAPS = 0


def _spatial_weight(weight, form, dtype, build):
    key = (id(weight), form, dtype)
    cached = _SPATIAL_WEIGHTS.get(key)
    if cached is None or cached[0] is not weight:
        derived = build(weight.astype(dtype))
        mx.eval(derived)
        cached = (weight, derived)
        _SPATIAL_WEIGHTS[key] = cached
    return cached[1]


def _spatial_bias(y, bias):
    return y if bias is None else y + bias.astype(y.dtype)


def _spatial_conv2d(
    x, weight, bias, stride_h, stride_w, pad_top, pad_bottom, pad_left, pad_right, dilation, groups
):
    """x: [B,H,W,C]; weight: checkpoint layout [O,C/groups,KH,KW]."""
    out, per_group, kernel_h, kernel_w = weight.shape
    stride_h, stride_w, dilation, groups = int(stride_h), int(stride_w), int(dilation), int(groups)
    top, bottom, left, right = int(pad_top), int(pad_bottom), int(pad_left), int(pad_right)
    if kernel_h == 1 and kernel_w == 1 and groups == 1 and not (top or bottom or left or right):
        # A pointwise convolution is a matmul over the channel axis.
        w = _spatial_weight(weight, "pointwise", x.dtype, lambda w: w.reshape(out, per_group).T)
        if stride_h != 1 or stride_w != 1:
            x = x[:, ::stride_h, ::stride_w, :]
        return x @ w if bias is None else mx.addmm(bias.astype(x.dtype), x, w)
    padded = top != bottom or left != right
    depthwise = per_group == 1 and groups == out and out == x.shape[-1]
    if depthwise and kernel_h * kernel_w <= _SPATIAL_DEPTHWISE_SHIFT_TAPS:
        # y[h, w, c] = sum over taps (i, j) of x[h * stride + i * dilation, ..., c] * k[i, j, c].
        taps = _spatial_weight(weight, "taps", x.dtype, lambda w: w.reshape(out, kernel_h, kernel_w))
        if top or bottom or left or right:
            x = mx.pad(x, ((0, 0), (top, bottom), (left, right), (0, 0)))
        span_h = x.shape[1] - dilation * (kernel_h - 1)
        span_w = x.shape[2] - dilation * (kernel_w - 1)
        y = None
        for i in range(kernel_h):
            for j in range(kernel_w):
                rows = slice(i * dilation, i * dilation + span_h, stride_h)
                cols = slice(j * dilation, j * dilation + span_w, stride_w)
                term = x[:, rows, cols, :] * taps[:, i, j]
                y = term if y is None else y + term
        return _spatial_bias(y, bias)
    w = _spatial_weight(weight, "conv", x.dtype, lambda w: w.transpose(0, 2, 3, 1))
    if padded:
        x = mx.pad(x, ((0, 0), (top, bottom), (left, right), (0, 0)))
    y = mx.conv2d(
        x,
        w,
        stride=(stride_h, stride_w),
        padding=(0, 0) if padded else (top, left),
        dilation=dilation,
        groups=groups,
    )
    return _spatial_bias(y, bias)


def _spatial_conv_transpose2d(x, weight, bias, stride_h, stride_w, pad_h, pad_w):
    """x: [B,H,W,C]; weight: checkpoint layout [C,O,KH,KW]."""
    channels, out, kernel_h, kernel_w = weight.shape
    stride_h, stride_w, pad_h, pad_w = int(stride_h), int(stride_w), int(pad_h), int(pad_w)
    if (kernel_h, kernel_w) == (stride_h, stride_w) and pad_h == 0 and pad_w == 0:
        # Windows do not overlap: each input pixel writes its own kernel-sized output
        # block, so this is a matmul to kernel_h * kernel_w * out channels and a
        # depth-to-space rearrangement.
        w = _spatial_weight(
            weight,
            "blocks",
            x.dtype,
            lambda w: w.transpose(0, 2, 3, 1).reshape(channels, kernel_h * kernel_w * out),
        )
        batch, height, width, _ = x.shape
        y = (x @ w).reshape(batch, height, width, kernel_h, kernel_w, out)
        y = y.transpose(0, 1, 3, 2, 4, 5).reshape(batch, height * kernel_h, width * kernel_w, out)
        return _spatial_bias(y, bias)
    w = _spatial_weight(weight, "transposed", x.dtype, lambda w: w.transpose(1, 2, 3, 0))
    y = mx.conv_transpose2d(x, w, stride=(stride_h, stride_w), padding=(pad_h, pad_w))
    return _spatial_bias(y, bias)


def _spatial_max_pool2d(
    x, kernel_h, kernel_w, stride_h, stride_w, pad_top, pad_bottom, pad_left, pad_right
):
    """x: [B,H,W,C]; padding never wins the maximum."""
    pads = ((0, 0), (int(pad_top), int(pad_bottom)), (int(pad_left), int(pad_right)), (0, 0))
    if any(any(pair) for pair in pads):
        x = mx.pad(x, pads, constant_values=float("-inf"))
    pool = nn.MaxPool2d(
        kernel_size=(int(kernel_h), int(kernel_w)), stride=(int(stride_h), int(stride_w))
    )
    return pool(x)


def _spatial_avg_pool2d(x, kernel_h, kernel_w, stride_h, stride_w):
    """x: [B,H,W,C]; unpadded windows."""
    pool = nn.AvgPool2d(
        kernel_size=(int(kernel_h), int(kernel_w)), stride=(int(stride_h), int(stride_w))
    )
    return pool(x)


def _spatial_global_avg_pool2d(x):
    """x: [B,H,W,C] -> [B,1,1,C]. Accumulates in float32: a half-precision sum over a
    large map overflows."""
    if x.dtype == mx.float32:
        return x.mean(axis=(1, 2), keepdims=True)
    return x.astype(mx.float32).mean(axis=(1, 2), keepdims=True).astype(x.dtype)


def _spatial_upsample_nearest2d_like(x, ref):
    """x: [B,H,W,C] -> [B,Ho,Wo,C] for ref [B,Ho,Wo,D]; Ho and Wo are multiples of H and W."""
    scale_h, rem_h = divmod(int(ref.shape[1]), int(x.shape[1]))
    scale_w, rem_w = divmod(int(ref.shape[2]), int(x.shape[2]))
    if rem_h or rem_w:
        raise ValueError(
            f"upsample_nearest2d_like needs an integer scale: {tuple(x.shape)} -> {tuple(ref.shape)}"
        )
    if scale_h == 1 and scale_w == 1:
        return x
    # One broadcast and one copy instead of a repeat per axis.
    batch, height, width, channels = x.shape
    y = mx.broadcast_to(
        x[:, :, None, :, None, :], (batch, height, scale_h, width, scale_w, channels)
    )
    return y.reshape(batch, height * scale_h, width * scale_w, channels)


SPATIAL2D_SOURCE_FUNCTIONS = (
    _spatial_weight,
    _spatial_bias,
    _spatial_conv2d,
    _spatial_conv_transpose2d,
    _spatial_max_pool2d,
    _spatial_avg_pool2d,
    _spatial_global_avg_pool2d,
    _spatial_upsample_nearest2d_like,
)

__all__ = ["SPATIAL2D_SOURCE_FUNCTIONS"]
