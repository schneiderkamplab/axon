"""Host preparation around an Axon-generated Timestral neural core.

Derived from mistralai/timestral; see NOTICE and LICENSE.
"""

import numpy as np

from .masks import (
    build_forecast_time_mask,
    build_two_stream_time_mask,
    build_two_stream_variate_mask,
    compute_enc_key_validity,
    first_observed_positions,
)
from .scaler import (
    apply_affine_asinh_masked,
    causal_patch_loc_scale,
    denormalize_quantiles,
    instance_norm_stats,
)

ROLE_TARGET, ROLE_PAST_COV, ROLE_KNOWN_COV = 0, 1, 2


def monotone_quantiles(raw, levels, factor):
    median_index = levels.index(0.5)
    widths = np.zeros(len(levels), dtype=np.float32)
    widths[:median_index] = np.diff(levels[: median_index + 1])
    widths[median_index + 1 :] = np.diff(levels[median_index:])
    # Stable softplus, including the same linear branch as torch's beta=1 default.
    softplus = np.log1p(np.exp(-np.abs(raw))) + np.maximum(raw, 0)
    softplus = np.where(raw > 20.0, raw, softplus)
    increments = factor * widths * softplus
    median = raw[..., median_index : median_index + 1]
    down = np.cumsum(increments[..., :median_index][..., ::-1], axis=-1)[..., ::-1]
    up = np.cumsum(increments[..., median_index + 1 :], axis=-1)
    result = np.concatenate([median - down, median, median + up], axis=-1)
    big = np.finfo(np.float32).max
    return np.clip(result, -big, big)


class TimestralModel:
    def __init__(self, config, core):
        self.config = config
        self.core = core
        inverse = np.zeros(config.head_dim // 2, dtype=np.float32)
        indices = np.arange(0, config.rope_dim, 2, dtype=np.float32)
        inverse[: config.rope_dim // 2] = 1.0 / (config.rope_theta ** (indices / config.rope_dim))
        self.inv_freq = inverse

    def scale(self, inputs, observed):
        c = self.config
        loc, scale = causal_patch_loc_scale(inputs, observed, c.patch_size, c.minimum_scale)
        scaled = apply_affine_asinh_masked(inputs, loc, scale, observed, c.use_asinh)
        return scaled, loc, scale

    def predict_block(
        self,
        inputs,
        observed,
        group_ids,
        horizon_patches,
        role_ids=None,
        known_future=None,
        known_future_mask=None,
    ):
        c = self.config
        rows, length = inputs.shape
        ps = c.patch_size
        if length % ps or horizon_patches < 1:
            raise ValueError("predict_block requires patch-aligned context and a positive horizon")
        context_patches = length // ps
        window = role_ids is not None and c.covariate_enabled and np.any(role_ids != ROLE_TARGET)
        target_rows = known_rows = None
        if window:
            target_rows = role_ids == ROLE_TARGET
            known_rows = role_ids == ROLE_KNOWN_COV
            horizon_length = horizon_patches * ps
            if known_future is None:
                known_future = np.zeros((rows, horizon_length), dtype=np.float32)
                known_future_mask = np.zeros_like(known_future, dtype=bool)
            if known_future_mask is None:
                raise ValueError("known_future_mask is required with known_future")
            inputs = np.concatenate([inputs, known_future], axis=-1)
            observed = np.concatenate([observed, known_future_mask], axis=-1)
        scaled, loc, scale = self.scale(inputs, observed)
        if window and known_rows.any():
            loc_c, scale_c, binary = instance_norm_stats(inputs, observed, c.minimum_scale)
            cov_scaled = apply_affine_asinh_masked(inputs, loc_c, scale_c, observed, False)
            if c.use_asinh:
                cov_scaled = np.where(observed & ~binary, np.arcsinh(cov_scaled), cov_scaled)
            scaled = np.where(known_rows[:, None], cov_scaled, scaled)
            loc = np.where(known_rows[:, None], loc_c, loc)
            scale = np.where(known_rows[:, None], scale_c, scale)
        n = inputs.shape[1] // ps
        n_dec = n if window else horizon_patches
        patches = np.concatenate(
            [
                scaled.reshape(rows, n, ps),
                (~observed).reshape(rows, n, ps).astype(np.float32),
            ],
            axis=-1,
        )
        obs_per_patch = observed.reshape(rows, n, ps).any(-1)
        key_valid = compute_enc_key_validity(obs_per_patch, c.attend_missing_patches)
        if window:
            time_keep = build_two_stream_time_mask(
                np.full(rows, context_patches), n, c.block_size, key_valid, known_rows
            )
        else:
            time_keep = build_forecast_time_mask(n, n_dec, key_valid)
        variate_keep = build_two_stream_variate_mask(
            group_ids, key_valid, n_dec, target_rows, known_rows
        )
        positions = first_observed_positions(obs_per_patch, group_ids, n_dec, window)
        angles = positions.astype(np.float32)[..., None] * self.inv_freq
        angles = np.concatenate([angles, angles], axis=-1)[:, None]
        raw = self.core(
            patches=patches,
            forecast_shape=np.zeros((rows, n_dec, 1), dtype=np.float32),
            time_keep=time_keep,
            variate_keep=variate_keep,
            cos=np.cos(angles),
            sin=np.sin(angles),
        )
        preds = monotone_quantiles(
            np.asarray(raw, dtype=np.float32), c.quantile_knots, c.quantile_width_factor
        )
        if window:
            preds = preds[:, context_patches * ps :]
        anchor_loc = loc[:, context_patches * ps - 1, None, None]
        anchor_scale = scale[:, context_patches * ps - 1, None, None]
        return denormalize_quantiles(preds, anchor_loc, anchor_scale, c.use_asinh)
