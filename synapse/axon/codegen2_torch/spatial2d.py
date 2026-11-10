"""Torch runtime for the channels-last 2D primitives.

The functions are used directly by the graph interpreter and are embedded by source into
emitted modules, so they depend only on ``torch`` and ``F``.
"""

from __future__ import annotations

from torch.nn import functional as F

SPATIAL2D_PRIMITIVES: frozenset[str] = frozenset(
    {
        "conv2d",
        "conv_transpose2d",
        "max_pool2d",
        "avg_pool2d",
        "global_avg_pool2d",
        "upsample_nearest2d_like",
    }
)


def _spatial_like(value, x):
    if value is None:
        return None
    return value.to(device=x.device, dtype=x.dtype)


def _spatial_conv2d(
    x, weight, bias, stride_h, stride_w, pad_top, pad_bottom, pad_left, pad_right, dilation, groups
):
    """x: [B,H,W,C]; weight: [O,C/groups,KH,KW]."""
    x = x.permute(0, 3, 1, 2)
    top, bottom, left, right = int(pad_top), int(pad_bottom), int(pad_left), int(pad_right)
    padding = (top, left)
    if top != bottom or left != right:
        x = F.pad(x, (left, right, top, bottom))
        padding = (0, 0)
    y = F.conv2d(
        x,
        _spatial_like(weight, x),
        _spatial_like(bias, x),
        stride=(int(stride_h), int(stride_w)),
        padding=padding,
        dilation=int(dilation),
        groups=int(groups),
    )
    return y.permute(0, 2, 3, 1)


def _spatial_conv_transpose2d(x, weight, bias, stride_h, stride_w, pad_h, pad_w):
    """x: [B,H,W,C]; weight: [C,O,KH,KW]."""
    y = F.conv_transpose2d(
        x.permute(0, 3, 1, 2),
        _spatial_like(weight, x),
        _spatial_like(bias, x),
        stride=(int(stride_h), int(stride_w)),
        padding=(int(pad_h), int(pad_w)),
    )
    return y.permute(0, 2, 3, 1)


def _spatial_max_pool2d(
    x, kernel_h, kernel_w, stride_h, stride_w, pad_top, pad_bottom, pad_left, pad_right
):
    """x: [B,H,W,C]; padding never wins the maximum."""
    x = x.permute(0, 3, 1, 2)
    pads = (int(pad_left), int(pad_right), int(pad_top), int(pad_bottom))
    if any(pads):
        x = F.pad(x, pads, value=float("-inf"))
    y = F.max_pool2d(
        x, kernel_size=(int(kernel_h), int(kernel_w)), stride=(int(stride_h), int(stride_w))
    )
    return y.permute(0, 2, 3, 1)


def _spatial_avg_pool2d(x, kernel_h, kernel_w, stride_h, stride_w):
    """x: [B,H,W,C]; unpadded windows."""
    y = F.avg_pool2d(
        x.permute(0, 3, 1, 2),
        kernel_size=(int(kernel_h), int(kernel_w)),
        stride=(int(stride_h), int(stride_w)),
    )
    return y.permute(0, 2, 3, 1)


def _spatial_global_avg_pool2d(x):
    """x: [B,H,W,C] -> [B,1,1,C]. Accumulates in float32: a half-precision sum over a
    large map overflows."""
    if x.dtype.itemsize >= 4:
        return x.mean(dim=(1, 2), keepdim=True)
    return x.float().mean(dim=(1, 2), keepdim=True).to(dtype=x.dtype)


def _spatial_upsample_nearest2d_like(x, ref):
    """x: [B,H,W,C] -> [B,Ho,Wo,C] for ref [B,Ho,Wo,D]; Ho and Wo are multiples of H and W."""
    scale_h, rem_h = divmod(int(ref.shape[1]), int(x.shape[1]))
    scale_w, rem_w = divmod(int(ref.shape[2]), int(x.shape[2]))
    if rem_h or rem_w:
        raise ValueError(
            f"upsample_nearest2d_like needs an integer scale: {tuple(x.shape)} -> {tuple(ref.shape)}"
        )
    if scale_h != 1:
        x = x.repeat_interleave(scale_h, dim=1)
    if scale_w != 1:
        x = x.repeat_interleave(scale_w, dim=2)
    return x


SPATIAL2D_RUNTIME = {
    "conv2d": _spatial_conv2d,
    "conv_transpose2d": _spatial_conv_transpose2d,
    "max_pool2d": _spatial_max_pool2d,
    "avg_pool2d": _spatial_avg_pool2d,
    "global_avg_pool2d": _spatial_global_avg_pool2d,
    "upsample_nearest2d_like": _spatial_upsample_nearest2d_like,
}
SPATIAL2D_SOURCE_FUNCTIONS = (
    _spatial_like,
    _spatial_conv2d,
    _spatial_conv_transpose2d,
    _spatial_max_pool2d,
    _spatial_avg_pool2d,
    _spatial_global_avg_pool2d,
    _spatial_upsample_nearest2d_like,
)

__all__ = ["SPATIAL2D_PRIMITIVES", "SPATIAL2D_RUNTIME", "SPATIAL2D_SOURCE_FUNCTIONS"]
