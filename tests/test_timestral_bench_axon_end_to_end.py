"""Build a fused MLX core, load a tiny synthetic checkpoint, forecast, and replay the recorded core."""

from __future__ import annotations

import json

import numpy as np
import pytest

from scripts.timestral_bench import TimestralConfig, TimestralForecaster
from scripts.timestral_bench.build import write_cores
from scripts.timestral_bench.loading import expected_shapes, load_model

CONFIG = TimestralConfig(
    patch_size=8,
    d_model=32,
    num_heads=4,
    mlp_hidden_dim=64,
    block_schedule=("time", "time_variate", "variate", "variate_time"),
    quantile_knots=(0.01, 0.1, 0.5, 0.9, 0.99),
    block_size=4,
    context_length=64,
)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    pytest.importorskip("mlx.core")
    from safetensors.numpy import save_file

    root = tmp_path_factory.mktemp("timestral_bench_e2e")
    cores = write_cores(root / "run")
    info = json.loads((cores / "build-info.json").read_text())
    assert set(info["variants"]) == {"unfused", "fused"}
    assert (cores / "_axon_common.py").exists()
    checkpoint = root / "checkpoint"
    checkpoint.mkdir()
    CONFIG.save(checkpoint / "config.json")
    rng = np.random.default_rng(5)
    state = {}
    for name, shape in expected_shapes(CONFIG).items():
        value = rng.normal(0, 0.06, shape).astype(np.float32)
        if name.endswith(".scale"):
            value += 1
        state[name] = value
    save_file(state, str(checkpoint / "model.safetensors"))
    return cores, checkpoint


@pytest.mark.parametrize("variant", ["fused", "unfused"])
def test_generated_core_forecasts_and_replays_without_the_compiler(built, variant):
    cores, checkpoint = built
    model = load_model(checkpoint, core_module=cores / f"axon_core_{variant}.py")
    forecaster = TimestralForecaster(model)
    context = np.sin(np.arange(64, dtype=np.float32) / 5)
    result = forecaster.forecast(context, 24)
    assert result.quantiles.shape == (1, 1, 24, 5)
    assert np.isfinite(result.quantiles).all()
    assert (np.diff(result.quantiles, axis=-1) >= 0).all()
    again = forecaster.forecast(context, 24).quantiles
    np.testing.assert_array_equal(again, result.quantiles)

    core = model.core
    core.calls.clear()
    core.recording = True
    try:
        forecaster.forecast(context, 24)
    finally:
        core.recording = False
    # One block with paired symmetric averaging is a single network call.
    assert len(core.calls) == 1
    outputs = core.replay()
    assert len(outputs) == len(core.calls)
    assert all(np.isfinite(np.array(output)).all() for output in outputs)
    core.calls.clear()
    core.recording = True
    try:
        longer = forecaster.forecast(context, 40).quantiles  # two decode blocks
    finally:
        core.recording = False
    assert longer.shape == (1, 1, 40, 5)
    assert len(core.calls) == 2
    assert len(core.replay()) == 2


def test_pruned_core_matches_default_within_tolerance(built):
    cores, checkpoint = built
    default = TimestralForecaster(load_model(checkpoint, core_module=cores / "axon_core_fused.py"))
    pruned = TimestralForecaster(
        load_model(checkpoint, core_module=cores / "axon_core_fused.py", prune_final_block=True)
    )
    context = np.cumsum(np.random.default_rng(9).normal(size=(2, 2, 48)).astype(np.float32), axis=-1)
    a = default.forecast(context, 20).quantiles
    b = pruned.forecast(context, 20).quantiles
    assert a.shape == b.shape == (2, 2, 20, 5)
    np.testing.assert_allclose(b, a, rtol=2e-4, atol=2e-4)
