"""NumPy/float64 normalization, ported from Mistral Timestral (Apache-2.0).

See NOTICE and LICENSE. Keep statistics on the host: Metal has no float64 tensors.
"""

import numpy as np


def causal_patch_loc_scale(data, observed, patch_size, minimum_scale):
    x = data.astype(np.float64)
    w = observed.astype(np.float64)
    count = np.maximum(np.cumsum(w, axis=-1), 1.0)
    means = np.cumsum(w * x, axis=-1) / count
    previous = np.concatenate([np.zeros_like(means[..., :1]), means[..., :-1]], axis=-1)
    m2 = np.cumsum((x - previous) * (x - means) * w, axis=-1)
    scales = np.maximum(np.sqrt(m2 / np.maximum(count - 1.0, 1.0)), minimum_scale)
    loc = np.repeat(means[..., patch_size - 1 :: patch_size], patch_size, axis=-1)
    scale = np.repeat(scales[..., patch_size - 1 :: patch_size], patch_size, axis=-1)
    return loc.astype(data.dtype), scale.astype(data.dtype)


def apply_affine_asinh_masked(data, loc, scale, observed, use_asinh):
    scaled = (data - loc) / np.maximum(scale, np.finfo(np.float32).eps)
    scaled = np.where(observed, scaled, 0.0)
    return np.arcsinh(scaled) if use_asinh else scaled


def clamp_nonfinite(values):
    finite = np.isfinite(values)
    hi = np.where(finite, values, -np.inf).max(axis=-1, keepdims=True)
    lo = np.where(finite, values, np.inf).min(axis=-1, keepdims=True)
    result = np.where(np.isposinf(values), hi, np.where(np.isneginf(values), lo, values))
    return np.where(np.isinf(result), 0.0, result)


def denormalize_quantiles(preds, loc, scale, use_asinh):
    with np.errstate(over="ignore", invalid="ignore"):
        values = np.sinh(preds) if use_asinh else preds
        return clamp_nonfinite(values * scale + loc)


def welford_combine(w_a, mean_a, m2_a, w_b, mean_b, m2_b):
    weight = w_a + w_b
    safe = np.maximum(weight, np.finfo(weight.dtype).tiny)
    delta = mean_b - mean_a
    mean = mean_a + delta * w_b / safe
    m2 = m2_a + m2_b + delta**2 * w_a * w_b / safe
    return weight, np.where(weight == 0, 0.0, mean), np.where(weight == 0, 0.0, m2)


def cumulative_welford(weight, mean, m2):
    distance = 1
    while distance < weight.shape[1]:
        left = [np.pad(s[:, :-distance], ((0, 0), (distance, 0))) for s in (weight, mean, m2)]
        weight, mean, m2 = welford_combine(*left, weight, mean, m2)
        distance *= 2
    return weight, mean, m2


def advancing_block_frames(context, observed, sinh_median, minimum_scale):
    rows, horizon, patch_size = sinh_median.shape
    blank = np.zeros((rows, horizon, patch_size), dtype=np.float64)
    values = np.concatenate(
        [context.reshape(rows, -1, patch_size).astype(np.float64), blank], axis=1
    )
    weights = np.concatenate(
        [observed.reshape(rows, -1, patch_size).astype(np.float64), blank], axis=1
    )
    count = weights.sum(-1)
    mean = (values * weights).sum(-1) / np.maximum(count, 1.0)
    m2 = (weights * (values - mean[..., None]) ** 2).sum(-1)
    mean = np.where(count == 0, 0.0, mean)
    m2 = np.where(count == 0, 0.0, m2)
    count, mean, m2 = [s[:, -horizon - 1] for s in cumulative_welford(count, mean, m2)]
    offset = mean
    clip = float(np.finfo(np.float32).max) ** 0.5 / 1e3
    locs, scales = [], []
    for index in range(horizon):
        var = np.maximum(m2 / np.maximum(count - 1.0, 1.0), 0.0)
        scale = np.maximum(np.sqrt(var), minimum_scale)
        locs.append(mean)
        scales.append(scale)
        folded = sinh_median[:, index] * scale[:, None] + mean[:, None] - offset[:, None]
        folded = np.clip(folded, -clip, clip) + offset[:, None]
        block_mean = folded.mean(-1)
        block_m2 = ((folded - block_mean[:, None]) ** 2).sum(-1)
        count, mean, m2 = welford_combine(
            count, mean, m2, np.full_like(count, patch_size), block_mean, block_m2
        )
    return np.stack(locs, axis=1), np.stack(scales, axis=1)


def instance_norm_stats(values, observed, minimum_scale):
    weights = observed.astype(values.dtype)
    count = np.maximum(weights.sum(-1, keepdims=True), 1.0)
    loc = (values * weights).sum(-1, keepdims=True) / count
    var = (((values - loc) * weights) ** 2).sum(-1, keepdims=True) / count
    scale = np.maximum(np.sqrt(var), minimum_scale)
    binary = ((values == 0.0) | (values == 1.0) | ~observed).all(-1, keepdims=True)
    binary &= observed.any(-1, keepdims=True)
    return np.where(binary, 0.0, loc), np.where(binary, 1.0, scale), binary
