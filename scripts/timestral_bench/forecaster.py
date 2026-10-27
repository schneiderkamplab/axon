"""Full NumPy forecasting API around the generated MLX network.

Ported from Mistral Timestral; see NOTICE and LICENSE.
"""

import math
from dataclasses import dataclass

import numpy as np

from .covariate_encoding import target_encode
from .model import ROLE_KNOWN_COV, ROLE_PAST_COV, ROLE_TARGET
from .scaler import advancing_block_frames, clamp_nonfinite


@dataclass(frozen=True)
class Forecast:
    quantiles: np.ndarray
    quantile_levels: tuple[float, ...]

    @property
    def median(self):
        return self.quantiles[..., self.quantile_levels.index(0.5)]

    def quantile(self, level):
        if level not in self.quantile_levels:
            raise ValueError(f"level {level} is not one of {self.quantile_levels}")
        return self.quantiles[..., self.quantile_levels.index(level)]


def ffill_with_mask(inputs, mask):
    index = np.where(mask, np.arange(inputs.shape[-1]), 0)
    index = np.maximum.accumulate(index, axis=-1)
    return np.nan_to_num(np.take_along_axis(inputs, index, axis=-1), nan=0.0)


def pad_to_patch(inputs, mask, patch_size):
    pad = (-inputs.shape[-1]) % patch_size
    padding = [(0, 0)] * (inputs.ndim - 1) + [(pad, 0)]
    return np.pad(inputs, padding), np.pad(mask, padding)


def replace_extreme_values(values, threshold):
    return np.where(np.isfinite(values) & (np.abs(values) < threshold), values, 0.0)


def cat_rows(a, b):
    if a is None or b is None:
        return a if b is None else b
    return np.concatenate([a, b], axis=1)


class TimestralForecaster:
    def __init__(
        self,
        model,
        max_context=None,
        per_shot_horizon_patches=32,
        quantile_real_cap_k=1e4,
        extreme_value_threshold=1e30,
        max_batch_rows=128,
        max_batch_tokens=4096,
    ):
        self.model = model
        self.config = model.config
        self.max_context = max_context if max_context is not None else self.config.context_length
        if per_shot_horizon_patches < 1 or self.max_context < 1:
            raise ValueError("context and per-shot horizon must be positive")
        self.per_shot_horizon_patches = min(per_shot_horizon_patches, self.config.block_size)
        self.quantile_real_cap_k = quantile_real_cap_k
        self.extreme_value_threshold = extreme_value_threshold
        for name, limit in (("max_batch_rows", max_batch_rows), ("max_batch_tokens", max_batch_tokens)):
            if limit is not None and (not isinstance(limit, (int, np.integer)) or limit < 1):
                raise ValueError(f"{name} must be a positive integer or None")
        self.max_batch_rows = max_batch_rows
        self.max_batch_tokens = max_batch_tokens
        self.quantile_levels = tuple(self.config.quantile_knots)
        self.median_index = self.config.median_index

    @classmethod
    def from_pretrained(
        cls,
        path_or_repo_id,
        *,
        revision=None,
        cache_dir=None,
        token=None,
        local_files_only=False,
        core_module=None,
        prune_final_block=False,
        **forecast_options,
    ):
        from .loading import load_model

        model = load_model(
            path_or_repo_id,
            revision=revision,
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_files_only,
            core_module=core_module,
            prune_final_block=prune_final_block,
        )
        return cls(model, **forecast_options)

    @property
    def patch_size(self):
        return self.config.patch_size

    @property
    def supports_covariates(self):
        return self.config.covariate_enabled

    @property
    def supports_known_covariates(self):
        return self.config.supports_known_covariates

    def forecast(
        self,
        context,
        prediction_length,
        observation_mask=None,
        past_covariates=None,
        known_covariates=None,
        categorical_past_covariates=None,
        categorical_known_covariates=None,
        symmetric_averaging=True,
        clip_nonnegative=True,
    ):
        if not isinstance(prediction_length, (int, np.integer)) or prediction_length < 1:
            raise ValueError("prediction_length must be a positive integer")
        inputs, mask = self._as_context(context, observation_mask)
        batch, _, length = inputs.shape
        past = self._as_covariates(past_covariates, batch, length, "past_covariates")
        known = self._as_covariates(
            known_covariates, batch, length + prediction_length, "known_covariates"
        )
        cat_past = self._as_categorical(
            categorical_past_covariates, batch, length, "categorical_past_covariates"
        )
        cat_known = self._as_categorical(
            categorical_known_covariates,
            batch,
            length + prediction_length,
            "categorical_known_covariates",
        )
        if cat_past is not None or cat_known is not None:
            target = self._encoding_target(inputs, mask)
            past = cat_rows(past, self._encode_categorical(cat_past, target))
            known = cat_rows(known, self._encode_categorical(cat_known, target))
        if known is not None and not self.supports_known_covariates:
            raise ValueError("this checkpoint cannot condition on known-future covariates")
        if symmetric_averaging:
            levels = self.quantile_levels
            if not all(abs(lo + hi - 1) < 1e-9 for lo, hi in zip(levels, reversed(levels))):
                raise ValueError("symmetric averaging requires quantile levels symmetric about 0.5")
        variables = inputs.shape[1] + (0 if past is None else past.shape[1]) + (0 if known is None else known.shape[1])
        groups = self._series_per_call(
            batch * (2 if symmetric_averaging else 1), variables, length,
            prediction_length, past is not None or known is not None,
        )
        paired = symmetric_averaging and groups >= 2
        size = max(1, groups // (2 if paired else 1))
        quantiles = np.empty(
            (*inputs.shape[:2], prediction_length, self.config.num_quantiles), dtype=np.float32
        )
        for start in range(0, batch, size):
            stop = min(start + size, batch)
            quantiles[start:stop] = self._forecast_chunk(
                inputs[start:stop], mask[start:stop], prediction_length,
                None if past is None else past[start:stop],
                None if known is None else known[start:stop],
                symmetric_averaging, paired,
            )
        if clip_nonnegative:
            eligible = mask.any(-1) & ((inputs >= 0) | ~mask).all(-1)
            np.maximum(quantiles, 0, out=quantiles, where=eligible[..., None, None])
        return Forecast(quantiles, self.quantile_levels)

    def _series_per_call(self, groups, variables, length, prediction_length, windowed):
        """Bound rows and transformer tokens while keeping each series intact."""
        ps = self.patch_size
        context = math.ceil(min(length, self.max_context) / ps)
        remaining = math.ceil(prediction_length / ps)
        peak_tokens = 0
        while remaining:
            block = min(remaining, self.per_shot_horizon_patches)
            peak_tokens = max(peak_tokens, context + block)
            remaining -= block
            context += block
            if context * ps > self.max_context:
                context = max(1, self.max_context // ps)
        if windowed:
            peak_tokens *= 2  # Encoder and decoder each contain the full window.
        if self.max_batch_rows is not None:
            groups = min(groups, self.max_batch_rows // variables)
        if self.max_batch_tokens is not None:
            groups = min(groups, self.max_batch_tokens // (variables * peak_tokens))
        # One multivariate series is indivisible; these are scheduling limits,
        # not a hard allocator cap.
        return max(1, groups)

    def _forecast_chunk(self, inputs, mask, prediction_length, past, known, symmetric, paired):
        if paired:
            both = self._forecast_raw(
                np.concatenate([inputs, -inputs], axis=0),
                np.concatenate([mask, mask], axis=0),
                prediction_length,
                None if past is None else np.concatenate([past, -past], axis=0),
                None if known is None else np.concatenate([known, -known], axis=0),
            )
            batch = len(inputs)
            return 0.5 * (both[:batch] - both[batch:, ..., ::-1])
        quantiles = self._forecast_raw(inputs, mask, prediction_length, past, known)
        if symmetric:
            mirrored = self._forecast_raw(
                -inputs, mask, prediction_length,
                None if past is None else -past, None if known is None else -known,
            )
            quantiles = 0.5 * (quantiles - mirrored[..., ::-1])
        return quantiles

    def _as_context(self, context, observation_mask):
        values = np.asarray(context, dtype=np.float32)
        if values.ndim == 1:
            values = values[None, None]
        elif values.ndim == 2:
            values = values[:, None]
        if values.ndim != 3 or any(size == 0 for size in values.shape):
            raise ValueError(
                f"context must be nonempty (T,), (B,T), or (B,V,T), got {values.shape}"
            )
        mask = np.isfinite(values)
        if observation_mask is not None:
            mask &= np.asarray(observation_mask, dtype=bool).reshape(values.shape)
        return np.where(mask, values, 0.0), mask

    def _as_covariates(self, values, batch, length, name, dtype=np.float32):
        if values is None:
            return None
        if not self.supports_covariates:
            raise ValueError(f"checkpoint was not trained with covariates; got {name}")
        values = np.asarray(values, dtype=dtype)
        if values.ndim == 2:
            values = values[:, None]
        if values.ndim != 3 or values.shape[0] != batch or values.shape[-1] != length:
            raise ValueError(f"{name} must be (B={batch},C,{length}), got {values.shape}")
        return values if values.shape[1] else None

    def _as_categorical(self, values, batch, length, name):
        values = self._as_covariates(values, batch, length, name, dtype=np.float64)
        if values is not None and np.isinf(values).any():
            raise ValueError(f"{name} must hold finite codes or NaN")
        return values

    def _encoding_target(self, inputs, mask):
        if inputs.shape[1] != 1:
            return None
        started = np.maximum.accumulate(mask[:, 0], axis=-1)
        length = inputs.shape[-1]
        in_window = np.arange(length) >= length - self.max_context
        return np.where(started & in_window, ffill_with_mask(inputs, mask)[:, 0], np.nan)

    def _encode_categorical(self, codes, target):
        if codes is None or target is None:
            return None if codes is None else codes.astype(np.float32)
        length = target.shape[-1]
        out = np.empty(codes.shape, dtype=np.float32)
        for row in range(codes.shape[0]):
            for channel in range(codes.shape[1]):
                future = codes[row, channel, length:] if codes.shape[-1] > length else None
                past_enc, future_enc = target_encode(
                    target[row], codes[row, channel, :length], future
                )
                padding = np.isnan(target[row]) & np.isnan(codes[row, channel, :length])
                out[row, channel, :length] = np.where(padding, np.nan, past_enc)
                if future_enc is not None:
                    out[row, channel, length:] = future_enc
        return out

    def _forecast_raw(self, inputs, mask, prediction_length, past=None, known=None):
        rows, mask, roles, future, future_mask, targets = self._assemble_rows(
            inputs, mask, prediction_length, past, known
        )
        quantiles, context_scale = self._block_ar_decode(
            rows, mask, prediction_length, roles, future, future_mask
        )
        quantiles = quantiles[:, :targets]
        if self.quantile_real_cap_k > 0:
            quantiles = self._cap_quantiles(
                quantiles, rows[:, :targets], mask[:, :targets], context_scale[:, :targets]
            )
        if np.isnan(quantiles).any():
            raise ValueError(
                f"forecast produced {np.isnan(quantiles).sum()} NaNs; input is degenerate"
            )
        return quantiles

    def _cap_quantiles(self, quantiles, inputs, mask, context_scale):
        hi = np.max(np.where(mask, inputs, -np.inf), axis=-1, keepdims=True)
        lo = np.min(np.where(mask, inputs, np.inf), axis=-1, keepdims=True)
        hi = np.nan_to_num(hi, neginf=0.0)
        lo = np.nan_to_num(lo, posinf=0.0)
        radius = context_scale[..., None] * self.quantile_real_cap_k
        return np.clip(quantiles, (lo - radius)[..., None], (hi + radius)[..., None])

    def _assemble_rows(self, inputs, mask, prediction_length, past, known):
        batch, targets, _ = inputs.shape
        horizon = math.ceil(prediction_length / self.patch_size) * self.patch_size
        rows, masks = [ffill_with_mask(inputs, mask)], [mask]
        roles = [ROLE_TARGET] * targets
        n_past = 0
        if past is not None:
            n_past = past.shape[1]
            rows.append(np.nan_to_num(past, nan=0.0))
            masks.append(~np.isnan(past))
            roles += [ROLE_PAST_COV] * n_past
        if known is not None:
            context = known[..., :-prediction_length]
            rows.append(np.nan_to_num(context, nan=0.0))
            masks.append(~np.isnan(context))
            roles += [ROLE_KNOWN_COV] * known.shape[1]
        rows, masks = np.concatenate(rows, axis=1), np.concatenate(masks, axis=1)
        rows, masks = rows[..., -self.max_context :], masks[..., -self.max_context :]
        rows, masks = pad_to_patch(rows, masks, self.patch_size)
        if past is None and known is None:
            return rows, masks, None, None, None, targets
        role_ids = np.tile(np.asarray(roles), batch)
        future = future_mask = None
        if known is not None:
            variates = rows.shape[1]
            future = np.zeros((batch, variates, horizon), dtype=np.float32)
            future_mask = np.zeros_like(future, dtype=bool)
            start = targets + n_past
            values = known[..., -prediction_length:]
            future[:, start:, :prediction_length] = np.nan_to_num(values, nan=0.0)
            future_mask[:, start:, :prediction_length] = ~np.isnan(values)
            future = future.reshape(batch * variates, horizon)
            future_mask = future_mask.reshape(batch * variates, horizon)
        return rows, masks, role_ids, future, future_mask, targets

    def _block_ar_decode(self, inputs, observed, prediction_length, roles, future, future_mask):
        batch, variates, length = inputs.shape
        context = inputs.reshape(batch * variates, length)
        mask = observed.reshape(context.shape)
        groups = np.repeat(np.arange(batch), variates)
        targets = None if roles is None else roles == ROLE_TARGET
        remaining = math.ceil(prediction_length / self.patch_size)
        offset, blocks = 0, []
        while remaining:
            block = min(remaining, self.per_shot_horizon_patches)
            block_length = block * self.patch_size
            kf = None if future is None else future[:, offset : offset + block_length]
            kfm = None if future_mask is None else future_mask[:, offset : offset + block_length]
            quantiles, loc0, scale0 = self.model.predict_block(
                context, mask, groups, block, roles, kf, kfm
            )
            if not blocks:
                context_scale = scale0.reshape(batch, variates)
            quantiles = self._refine_block(quantiles, context, mask, targets, loc0, scale0)
            blocks.append(quantiles)
            remaining -= block
            offset += block_length
            if not remaining:
                break
            feedback = replace_extreme_values(
                quantiles[..., self.median_index], self.extreme_value_threshold
            )
            feedback_mask = np.ones_like(feedback, dtype=bool)
            if roles is not None:
                known = roles == ROLE_KNOWN_COV
                past = (roles != ROLE_TARGET) & ~known
                if kf is not None:
                    feedback = np.where(known[:, None], kf, feedback)
                    feedback_mask = np.where(known[:, None], kfm, feedback_mask)
                feedback = np.where(past[:, None], 0.0, feedback)
                feedback_mask &= ~past[:, None]
            context = np.concatenate([context, feedback], axis=-1)
            mask = np.concatenate([mask, feedback_mask], axis=-1)
            if context.shape[-1] > self.max_context:
                keep = max(self.patch_size, (self.max_context // self.patch_size) * self.patch_size)
                context, mask = context[..., -keep:], mask[..., -keep:]
        packed = np.concatenate(blocks, axis=1)[:, :prediction_length]
        return packed.reshape(batch, variates, prediction_length, -1), context_scale

    def _refine_block(self, quantiles, context, mask, targets, loc0, scale0):
        ps = self.patch_size
        n_pred = quantiles.shape[1] // ps
        if n_pred <= 1:
            return quantiles
        median = (
            quantiles[..., self.median_index]
            .reshape(context.shape[0], n_pred, ps)
            .astype(np.float64)
        )
        normalized = (median - loc0[:, None, None]) / scale0[:, None, None]
        loc_j, scale_j = advancing_block_frames(
            context, mask, normalized, self.config.minimum_scale
        )
        loc_j, scale_j = loc_j.astype(quantiles.dtype), scale_j.astype(quantiles.dtype)
        loc_j[:, 0], scale_j[:, 0] = loc0, scale0
        loc_j, scale_j = np.repeat(loc_j, ps, axis=-1), np.repeat(scale_j, ps, axis=-1)
        normalized = (quantiles - loc0[:, None, None]) / scale0[:, None, None]
        refined = clamp_nonfinite(normalized * scale_j[..., None] + loc_j[..., None])
        return refined if targets is None else np.where(targets[:, None, None], refined, quantiles)
