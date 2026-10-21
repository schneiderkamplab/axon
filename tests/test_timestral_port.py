from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from scripts.timestral_axon import TimestralConfig
from scripts.timestral_axon.build import SOURCE, generate_core
from scripts.timestral_axon.covariate_encoding import target_encode
from scripts.timestral_axon.forecaster import Forecast, TimestralForecaster
from scripts.timestral_axon.loading import core_config, expected_shapes, validate_weights
from scripts.timestral_axon.masks import (
    build_forecast_time_mask,
    build_two_stream_time_mask,
    build_two_stream_variate_mask,
)
from scripts.timestral_axon.model import TimestralModel
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


@pytest.fixture(scope="module", params=["torch", "mlx"])
def quantile_core(request, tmp_path_factory):
    runtime = pytest.importorskip("torch" if request.param == "torch" else "mlx.core")
    path = tmp_path_factory.mktemp("timestral_quantiles") / "head.axon"
    path.write_text(SOURCE.read_text().replace('MAIN "timestral_core"', 'MAIN "monotone_quantiles"'))
    namespace = {}
    exec(compile(generate_core(request.param, path), str(path), "exec"), namespace)
    return request.param, runtime, namespace["TimestralCore"]


@pytest.mark.parametrize(
    "levels",
    [(0.01, 0.1, 0.5, 0.9, 0.99), (0.1, 0.5, 0.7, 0.8), (0.5, 0.9), (0.1, 0.5), (0.5,)],
)
def test_generated_quantile_head_is_stable_and_respects_uneven_knots(quantile_core, config, levels):
    backend, runtime, core_class = quantile_core
    config = replace(config, quantile_knots=levels, quantile_width_factor=3.0)
    raw = np.random.default_rng(120).normal(size=(2, 4, len(levels))).astype(np.float32)
    raw[0, 0] = 1000
    raw[0, 1] = -1000
    model = core_class.from_state_dict({}, model_config=core_config(config))
    if backend == "torch":
        result = model.forward(raw=runtime.from_numpy(raw)).numpy()
    else:
        result = runtime.compile(model.forward)(raw=runtime.array(raw))
        runtime.eval(result)
        result = np.asarray(result)
    mid = levels.index(0.5)
    np.testing.assert_array_equal(result[..., mid], raw[..., mid])
    assert np.isfinite(result).all()
    assert (np.diff(result) >= 0).all()
    # Independent float64 reference verifies widths, direction, and both tails.
    expected = np.broadcast_to(raw[..., mid, None], raw.shape).astype(np.float64).copy()
    smooth = np.logaddexp(0.0, raw.astype(np.float64))
    for i in range(mid - 1, -1, -1):
        expected[..., i] = expected[..., i + 1] - 3 * (levels[i + 1] - levels[i]) * smooth[..., i]
    for i in range(mid + 1, len(levels)):
        expected[..., i] = expected[..., i - 1] + 3 * (levels[i] - levels[i - 1]) * smooth[..., i]
    np.testing.assert_allclose(result, expected, rtol=2e-6, atol=1e-5)


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
    groups = np.repeat(np.arange(2), 3)
    targets = np.tile([True, False, False], 2)
    known = np.tile([False, False, True], 2)
    mask = build_two_stream_variate_mask(groups, np.ones((6, 3), bool), 2, targets, known)
    assert mask.shape == (2, 5, 1, 3, 3)
    assert mask[:, :, 0, 0, 1:3].all()
    assert not mask[:, :, 0, 1:3, 0].any()
    assert not mask[:, :, 0, 2, 1].any()
    assert np.diagonal(mask, axis1=-2, axis2=-1).all()


@pytest.mark.parametrize("batch,variables", [(1, 1), (3, 2), (8, 5)])
def test_grouped_masks_match_dense_attention_blocks(batch, variables):
    rng = np.random.default_rng(191)
    rows = batch * variables
    groups = np.repeat(np.arange(batch), variables)
    valid = rng.random((rows, 7)) > 0.3
    roles = rng.integers(0, 3, rows)
    targets, known = roles == 0, roles == 2
    compact = build_two_stream_variate_mask(groups, valid, 3, targets, known)
    key_valid = np.concatenate([valid, np.ones((rows, 3), bool)], axis=1).T
    dense = (groups[:, None] == groups[None])[None, None] & key_valid[:, None, None]
    dense &= ~(~targets[:, None] & targets[None])[None, None]
    dense &= ~(known[:, None] & ~known[None])[None, None]
    dense |= np.eye(rows, dtype=bool)[None, None]
    for group in range(batch):
        indices = slice(group * variables, (group + 1) * variables)
        np.testing.assert_array_equal(compact[group], dense[:, :, indices, indices])


@pytest.mark.parametrize("groups", [[0, 0, 1], [0, 1, 0, 1], [1, 1, 0, 0]])
def test_grouped_masks_reject_invalid_layouts(groups):
    with pytest.raises(ValueError, match="variate groups"):
        build_two_stream_variate_mask(np.array(groups), np.ones((len(groups), 3), bool), 2)


def test_context_statistics_precede_known_covariate_normalization(config):
    captured = {}

    def core(**inputs):
        captured.update(inputs)
        rows, patches, _ = inputs["output_shape"].shape
        return np.zeros((rows, patches * config.patch_size, config.num_quantiles), np.float32)

    model = TimestralModel(config, core)
    inputs = np.arange(32, dtype=np.float32).reshape(2, 16)
    observed = np.ones_like(inputs, bool)
    future = np.full((2, 8), 1000, np.float32)
    future_mask = np.zeros_like(future, bool)
    future_mask[1] = True
    result, loc, scale = model.predict_block(
        inputs, observed, np.zeros(2, int), 1, np.array([0, 2]), future, future_mask
    )
    expected_loc, expected_scale = causal_patch_loc_scale(inputs, observed, 8, config.minimum_scale)
    np.testing.assert_array_equal(loc, expected_loc[:, -1])
    np.testing.assert_array_equal(scale, expected_scale[:, -1])
    assert result.shape == (2, 8, config.num_quantiles)
    assert captured["forecast_shape"].shape == (2, 3, 1)
    assert captured["output_shape"].shape == (2, 1, 1)


@pytest.mark.parametrize("symmetric", [True, False])
@pytest.mark.parametrize("rows,tokens", [(13, None), (25, None), (None, 84), (7, 1)])
def test_bounded_forecasts_preserve_order_covariates_and_symmetry(
    config, monkeypatch, symmetric, rows, tokens
):
    context = np.arange(190, dtype=np.float32).reshape(5, 2, 19) / 10 - 8
    covariates = {
        "past_covariates": np.arange(95, dtype=np.float32).reshape(5, 1, 19),
        "known_covariates": np.arange(320, dtype=np.float32).reshape(5, 1, 64),
        "categorical_past_covariates": np.ones((5, 1, 19)),
        "categorical_known_covariates": np.full((5, 1, 64), 3),
    }
    calls = []

    def raw(inputs, mask, prediction_length, past, known):
        calls.append((inputs.copy(), past.copy(), known.copy()))
        base = inputs[..., -1] + past.mean((1, 2))[:, None] + known.mean((1, 2))[:, None]
        offsets = np.arange(config.num_quantiles, dtype=np.float32) / 10
        return np.broadcast_to(
            base[..., None, None] + offsets, (*base.shape, prediction_length, len(offsets))
        ).copy()

    def forecast(max_rows, max_tokens):
        forecaster = TimestralForecaster(
            TimestralModel(config, None), max_batch_rows=max_rows, max_batch_tokens=max_tokens
        )
        monkeypatch.setattr(forecaster, "_forecast_raw", raw)
        return forecaster.forecast(
            context, 45, **covariates, symmetric_averaging=symmetric, clip_nonnegative=False
        ).quantiles

    expected = forecast(None, None)
    calls.clear()
    actual = forecast(rows, tokens)
    np.testing.assert_array_equal(actual, expected)
    assert len(calls) > 1
    # Each original series retains all two targets and four covariates.
    assert all(x.shape[1] == 2 and p.shape[1] == k.shape[1] == 2 for x, p, k in calls)
    if rows is not None:
        assert all(len(x) * 6 <= rows for x, _, _ in calls)


@pytest.mark.parametrize("max_context", [7, 17, 32])
def test_memory_limits_cover_growing_and_truncated_decode_windows(config, max_context):
    shapes = []

    def core(**inputs):
        rows, patches, _ = inputs["output_shape"].shape
        tokens = inputs["patches"].shape[1] + inputs["forecast_shape"].shape[1]
        groups = inputs["variate_keep"].shape[0]
        shapes.append((rows, tokens, groups))
        return np.zeros((rows, patches * config.patch_size, config.num_quantiles), np.float32)

    forecaster = TimestralForecaster(
        TimestralModel(config, core), max_context=max_context,
        per_shot_horizon_patches=2, max_batch_rows=17, max_batch_tokens=100,
    )
    result = forecaster.forecast(
        np.arange(190, dtype=np.float32).reshape(5, 2, 19), 81,
        known_covariates=np.ones((5, 1, 100), np.float32),
    )
    assert result.quantiles.shape == (5, 2, 81, config.num_quantiles)
    assert np.isfinite(result.quantiles).all()
    assert all(rows <= 17 and rows * tokens <= 100 or groups == 1 for rows, tokens, groups in shapes)


@pytest.mark.parametrize("name", ["max_batch_rows", "max_batch_tokens"])
@pytest.mark.parametrize("value", [0, -1, 1.5])
def test_memory_limits_must_be_positive_integers(config, name, value):
    with pytest.raises(ValueError, match=name):
        TimestralForecaster(TimestralModel(config, None), **{name: value})


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


@pytest.mark.parametrize("prune", [False, True])
def test_pruning_is_an_explicit_load_option(config, monkeypatch, prune):
    from scripts.timestral_axon import loading

    captured = {}

    def load(path, **options):
        captured.update(options)
        return TimestralModel(config, None)

    monkeypatch.setattr(loading, "load_model", load)
    options = {"prune_final_block": True} if prune else {}
    forecaster = TimestralForecaster.from_pretrained(
        "local-checkpoint", local_files_only=True, max_batch_rows=16, **options
    )
    assert captured["prune_final_block"] is prune
    assert captured["local_files_only"] is True
    assert "max_batch_rows" not in captured
    assert forecaster.max_batch_rows == 16
    assert core_config(config)["prune_final_block"] is False
    assert core_config(config, prune_final_block=prune)["prune_final_block"] is prune
    assert "prune_final_block" not in config.to_dict()


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
