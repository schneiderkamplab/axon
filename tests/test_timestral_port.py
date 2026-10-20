from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from scripts.timestral_axon import TimestralConfig
from scripts.timestral_axon.build import generate_core
from scripts.timestral_axon.covariate_encoding import target_encode
from scripts.timestral_axon.forecaster import Forecast, TimestralForecaster
from scripts.timestral_axon.loading import expected_shapes, validate_weights
from scripts.timestral_axon.masks import (
    build_forecast_time_mask,
    build_two_stream_time_mask,
    build_two_stream_variate_mask,
)
from scripts.timestral_axon.model import TimestralModel, monotone_quantiles
from scripts.timestral_axon.scaler import (
    causal_patch_loc_scale,
    clamp_nonfinite,
    instance_norm_stats,
)


@pytest.fixture
def config():
    return TimestralConfig(
        patch_size=8,
        d_model=32,
        num_heads=4,
        mlp_hidden_dim=64,
        block_schedule=("time", "time_variate", "variate_time", "variate"),
        quantile_knots=(0.01, 0.1, 0.5, 0.9, 0.99),
        block_size=4,
        context_length=32,
    )


def test_causal_statistics_ignore_missing_and_use_sample_variance():
    values = np.array([[1.0, 3.0, 5.0, 0.0]], dtype=np.float32)
    observed = np.array([[True, True, True, False]])
    loc, scale = causal_patch_loc_scale(values, observed, 2, 1e-6)
    np.testing.assert_allclose(loc, [[2, 2, 3, 3]])
    np.testing.assert_allclose(scale, [[2**0.5, 2**0.5, 2, 2]])
    assert loc.dtype == scale.dtype == np.float32


def test_known_indicator_rows_keep_identity_frame():
    values = np.array([[0, 1, 0, 1], [2, 4, 6, 8]], dtype=np.float32)
    loc, scale, binary = instance_norm_stats(values, np.ones_like(values, dtype=bool), 1e-6)
    np.testing.assert_array_equal(binary[:, 0], [True, False])
    np.testing.assert_allclose(loc[:, 0], [0, 5])
    np.testing.assert_allclose(scale[:, 0], [1, 5**0.5])


def test_quantile_head_is_stable_and_respects_uneven_knots():
    raw = np.array([[1000, 0, 2, 0, -1000]], dtype=np.float32)
    levels = (0.01, 0.1, 0.5, 0.9, 0.99)
    result = monotone_quantiles(raw, levels, 3.0)
    median = result[0, 2]
    assert median == 2
    assert np.isfinite(result).all()
    assert (np.diff(result) >= 0).all()
    np.testing.assert_allclose(result[0, 3] - median, 3 * 0.4 * np.log(2), rtol=1e-6)
    np.testing.assert_allclose(result[0, 1] - result[0, 0], 3 * 0.09 * 1000, rtol=1e-6)


def test_nonfinite_clamp_preserves_nan_and_zeros_fully_infinite_rows():
    result = clamp_nonfinite(np.array([[-np.inf, 1, 3, np.inf], [np.inf] * 4, [np.nan, 1, 2, 3]]))
    np.testing.assert_array_equal(result[:2], [[1, 1, 3, 3], [0, 0, 0, 0]])
    assert np.isnan(result[2, 0])


def test_categorical_encoding_retains_large_codes_and_unseen_mean():
    base = 2**25
    past, future = target_encode(
        np.array([1.0, 3.0, 7.0, 9.0]),
        np.array([base, base, base + 1, base + 1], dtype=np.float64),
        np.array([base + 2, base, base + 1], dtype=np.float64),
    )
    np.testing.assert_allclose(past, [3, 3, 7, 7])
    np.testing.assert_allclose(future, [5, 3, 7])


def test_context_cannot_read_horizon_or_leading_padding():
    valid = np.array([[False, True, True]])
    mask = build_forecast_time_mask(3, 2, valid)[0, 0]
    assert not mask[:3, 3:].any()
    assert not mask[1:, 0].any()
    assert mask.diagonal().all()
    assert not mask[3, 4]
    assert mask[4, 3]


def test_known_rows_read_future_but_causal_rows_cannot():
    valid = np.ones((3, 5), dtype=bool)
    mask = build_two_stream_time_mask(np.full(3, 3), 5, 2, valid, np.array([False, False, True]))
    assert not mask[0, 0, 0, 4]
    assert mask[2, 0, 0, 4]
    assert not mask[:, :, :5, 5:].any()
    assert mask[:, 0, 8, 2].all()
    assert not mask[:, 0, 8, 3].any()


def test_variate_firewall_isolates_groups_and_targets():
    groups = np.array([0, 0, 0, 1])
    targets = np.array([True, False, False, True])
    known = np.array([False, False, True, False])
    mask = build_two_stream_variate_mask(groups, np.ones((4, 3), bool), 2, targets, known)
    assert not mask[:, 0, :3, 3].any()
    assert not mask[:, 0, 3, :3].any()
    assert mask[:, 0, 0, 1:3].all()
    assert not mask[:, 0, 1:3, 0].any()
    assert not mask[:, 0, 2, 1].any()
    assert np.diagonal(mask, axis1=-2, axis2=-1).all()


@pytest.mark.parametrize("field", ["missing", "extra", "shape"])
def test_loading_rejects_checkpoint_mismatches(config, field):
    state = {key: np.zeros(shape, np.float32) for key, shape in expected_shapes(config).items()}
    validate_weights(state, config)
    if field == "missing":
        state.pop("representation.forecast_query")
    elif field == "extra":
        state["unrecognized.weight"] = np.zeros(1)
    else:
        state["representation.forecast_query"] = np.zeros(3)
    with pytest.raises(ValueError, match="checkpoint mismatch"):
        validate_weights(state, config)


@pytest.mark.parametrize("backend", ["torch", "mlx"])
def test_neural_core_compiles_without_runtime_imports(backend):
    tree = ast.parse(generate_core(backend))
    assert any(
        isinstance(node, ast.ClassDef) and node.name == "TimestralCore" for node in tree.body
    )


def test_forecaster_validates_before_running_network(config):
    def unexpected(**kwargs):
        raise AssertionError("invalid inputs reached network")

    forecaster = TimestralForecaster(TimestralModel(config, unexpected))
    with pytest.raises(ValueError, match="positive integer"):
        forecaster.forecast(np.ones(3), 0)
    with pytest.raises(ValueError, match="nonempty"):
        forecaster.forecast(np.empty(0), 3)
    with pytest.raises(ValueError, match="known_covariates"):
        forecaster.forecast(np.ones(3), 3, known_covariates=np.ones((1, 1, 5)))
    with pytest.raises(ValueError, match="finite codes"):
        forecaster.forecast(np.ones(3), 3, categorical_past_covariates=np.full((1, 1, 3), np.inf))
    disabled = TimestralForecaster(
        TimestralModel(replace(config, covariate_enabled=False), unexpected)
    )
    with pytest.raises(ValueError, match="not trained"):
        disabled.forecast(np.ones(3), 3, past_covariates=np.ones((1, 1, 3)))


def test_forecast_quantile_selection():
    values = np.arange(15).reshape(1, 1, 5, 3)
    result = Forecast(values, (0.1, 0.5, 0.9))
    np.testing.assert_array_equal(result.median, values[..., 1])
    np.testing.assert_array_equal(result.quantile(0.9), values[..., 2])
    with pytest.raises(ValueError, match="not one of"):
        result.quantile(0.8)


def test_model_uses_builtins_without_primitives():
    path = Path(__file__).parents[1] / "synapse/models/timestral/generic-timestral.axon"
    source = path.read_text()
    assert "import Attention" in source
    import re

    assert not re.search(r"(?<![\w.@])_[a-z]\w*\s", source)
