"""Pruning must preserve the full network's forecast for every layer schedule."""

from dataclasses import replace

import numpy as np
import pytest

from scripts.timestral_axon import TimestralConfig
from scripts.timestral_axon.build import SOURCE, generate_core
from scripts.timestral_axon.loading import core_config, expected_shapes
from scripts.timestral_axon.model import TimestralModel


@pytest.fixture(scope="module", params=["torch", "mlx"])
def pruning_cores(request):
    backend = request.param
    runtime = pytest.importorskip("torch" if backend == "torch" else "mlx.core")
    namespace = {}
    exec(compile(generate_core(backend, optimize=True), str(SOURCE), "exec"), namespace)
    return backend, runtime, namespace["TimestralCore"]


@pytest.mark.parametrize("last_kind,bias_and_no_gates,variables", [
    (kind, bias, variables)
    for kind in ("time", "variate", "time_variate", "variate_time")
    for bias in (False, True)
    for variables in (1, 3)
] + [(None, False, 1)])
@pytest.mark.parametrize("prune_final_block", [False, True])
def test_pruned_outputs_match_full_network(
    pruning_cores, last_kind, bias_and_no_gates, variables, prune_final_block
):
    backend, runtime, cls = pruning_cores
    config = TimestralConfig(
        patch_size=8,
        d_model=32,
        num_heads=4,
        mlp_hidden_dim=64,
        block_schedule=() if last_kind is None else ("time", "time_variate", last_kind),
        quantile_knots=(0.01, 0.1, 0.5, 0.9, 0.99),
        block_size=4,
        context_length=64,
    )
    if bias_and_no_gates:
        config = replace(config, attn_bias=True, mlp_bias=True,
                         time_attn_gate="none", variate_attn_gate="none")
    rng = np.random.default_rng(819)
    state = {name: rng.normal(0, 0.06, shape).astype(np.float32)
             for name, shape in expected_shapes(config).items()}
    for name, values in state.items():
        if name.endswith(".scale"):
            values += 1
    if backend == "torch":
        state = {name: runtime.from_numpy(value) for name, value in state.items()}
    else:
        state = {name: runtime.array(value) for name, value in state.items()}
        runtime.eval(state)
    rows = 2 * variables
    values = rng.normal(size=(rows, 40)).astype(np.float32)
    observed = rng.random(values.shape) > 0.15
    groups = np.repeat(np.arange(2), variables)
    roles = np.tile([0, 1, 2], 2) if variables == 3 else None
    future = rng.normal(size=(rows, 16)).astype(np.float32) if roles is not None else None
    future_mask = np.broadcast_to((roles == 2)[:, None], future.shape) if roles is not None else None
    results = []
    for full_output in (False, True):
        generated = cls.from_state_dict(
            state, model_config=core_config(config, prune_final_block=prune_final_block)
        )
        forward = runtime.compile(generated.forward) if backend == "mlx" else generated.forward

        def core(**inputs):
            forecast_steps = inputs["output_shape"].shape[1] * config.patch_size
            if full_output:
                # Request every position, then select forecast outputs outside
                # the compiled network. No block may prune in this reference.
                sequence = inputs["patches"].shape[1] + inputs["forecast_shape"].shape[1]
                inputs["output_shape"] = np.zeros((rows, sequence, 1), np.float32)
            if backend == "torch":
                with runtime.inference_mode():
                    result = forward(**{k: runtime.from_numpy(v.copy()) for k, v in inputs.items()}).numpy()
                    return result[:, -forecast_steps:]
            result = forward(**{k: runtime.array(v) for k, v in inputs.items()})
            runtime.eval(result)
            return np.asarray(result)[:, -forecast_steps:]

        prediction, _, _ = TimestralModel(config, core).predict_block(
            values, observed, groups, 2, roles, future, future_mask
        )
        results.append(prediction)
    assert results[0].shape == (rows, 16, config.num_quantiles)
    assert np.isfinite(results[0]).all()
    assert (np.diff(results[0], axis=-1) >= 0).all()
    np.testing.assert_allclose(results[0], results[1], rtol=2e-4, atol=2e-4)
