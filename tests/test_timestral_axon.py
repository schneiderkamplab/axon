"""Test the Timestral Axon core directly, without a forecasting package."""

import re
from pathlib import Path

import numpy as np
import pytest

from synapse.axon import (
    GraphOptimizeConfig,
    elaborate_closed_axon_file,
    flatten_closed_axon_file,
    lower_axon_program_to_graph_ir,
    normalize_closed_axon_file,
    optimize_graph_program,
    resolve_axon_program_from_path,
    typecheck2_flat_axon_file,
)

SOURCE = Path(__file__).parents[1] / "synapse/models/timestral/generic-timestral.axon"
AXES = {
    "time": ("time",),
    "variate": ("variate",),
    "time_variate": ("time", "variate"),
    "variate_time": ("variate", "time"),
}


@pytest.fixture(scope="module", params=["torch", "mlx"])
def cores(request, tmp_path_factory):
    backend = request.param
    runtime = pytest.importorskip("torch" if backend == "torch" else "mlx.core")
    if backend == "torch":
        from synapse.axon.codegen2_torch import emit_model_code_from_graph_ir
    else:
        from synapse.axon.codegen2_mlx import emit_model_code_from_graph_ir

    classes = {}
    directory = tmp_path_factory.mktemp("timestral_cores")
    for main in ("timestral_core", "monotone_quantiles"):
        source = directory / f"{main}.axon"
        source.write_text(SOURCE.read_text().replace('MAIN "timestral_core"', f'MAIN "{main}"'))
        program = resolve_axon_program_from_path(source).ast
        for transform in (
            normalize_closed_axon_file, elaborate_closed_axon_file,
            flatten_closed_axon_file, typecheck2_flat_axon_file,
        ):
            program = transform(program, main_module=main)
        graph = lower_axon_program_to_graph_ir(program)
        intrinsics = f"codegen2-{backend}:__{backend}_sdpa"
        if backend == "mlx":
            intrinsics += ",__mlx_rmsnorm_scaled"
        graph = optimize_graph_program(
            graph, config=GraphOptimizeConfig(backend_intrinsics=intrinsics)
        )
        namespace = {}
        exec(emit_model_code_from_graph_ir(graph, class_name="TimestralCore"), namespace)
        classes[main] = namespace["TimestralCore"]
    return backend, runtime, classes


def _config(schedule=(), bias=False):
    return dict(
        patch_size=8, d_model=32, num_heads=4, mlp_hidden_dim=64,
        block_schedule=list(schedule), quantile_knots=[0.01, 0.1, 0.5, 0.9, 0.99],
        median_index=2, attn_bias=bias, mlp_bias=bias,
        time_attn_gate="none" if bias else "sdpa_sigmoid",
        variate_attn_gate="none" if bias else "sdpa_sigmoid",
    )


def _weights(config, rng):
    """Tiny deterministic checkpoint in the released parameter layout."""
    d, ps, ff = config["d_model"], config["patch_size"], config["mlp_hidden_dim"]
    nq = len(config["quantile_knots"])
    shapes = {"representation.forecast_query": (d,), "transformer.out_norm.scale": (d,)}

    def linear(path, inputs, outputs, bias):
        shapes[path + ".weight"] = (outputs, inputs)
        if bias:
            shapes[path + ".bias"] = (outputs,)

    for path, inputs, outputs, bias in (
        ("representation.patch_proj", 2 * ps, d, True),
        ("head.projection", d, ps * nq, False),
    ):
        linear(path + ".hidden_layer", inputs, 4 * d, bias)
        linear(path + ".output_layer", 4 * d, outputs, bias)
        linear(path + ".residual_layer", inputs, outputs, bias)
    for index, kind in enumerate(config["block_schedule"]):
        path = f"transformer.layers.{index}"
        for axis, name in enumerate(AXES[kind]):
            shapes[f"{path}.attn_norms.{axis}.scale"] = (d,)
            attention = f"{path}.attns.{axis}"
            linear(attention + ".wQKV", d, 3 * d, config["attn_bias"])
            linear(attention + ".wO", d, d, config["attn_bias"])
            if config[name + "_attn_gate"] == "sdpa_sigmoid":
                linear(attention + "._gate", d, d, config["attn_bias"])
        shapes[path + ".mlp_norm.scale"] = (d,)
        linear(path + ".mlp.w12", d, 2 * ff, config["mlp_bias"])
        linear(path + ".mlp.w3", ff, d, config["mlp_bias"])
    state = {name: rng.normal(0, 0.06, shape).astype(np.float32)
             for name, shape in shapes.items()}
    for name, value in state.items():
        if name.endswith(".scale"):
            value += 1
    return state


def _forward(cores, main, config, state, inputs):
    backend, runtime, classes = cores
    array = runtime.from_numpy if backend == "torch" else runtime.array
    state = {name: array(value) for name, value in state.items()}
    if backend == "mlx":
        runtime.eval(state)
    model = classes[main].from_state_dict(state, model_config=config)
    inputs = {name: array(value) for name, value in inputs.items()}
    if backend == "torch":
        with runtime.inference_mode():
            return model.forward(**inputs).numpy()
    result = runtime.compile(model.forward)(**inputs)
    runtime.eval(result)
    return np.asarray(result)


@pytest.mark.parametrize("last_kind,bias,variables", [
    (kind, bias, variables)
    for kind in AXES for bias in (False, True) for variables in (1, 3)
] + [(None, False, 1)])
@pytest.mark.parametrize("prune", [False, True])
def test_requested_outputs_match_full_sequence(cores, last_kind, bias, variables, prune):
    schedule = () if last_kind is None else ("time", "time_variate", last_kind)
    config = _config(schedule, bias)
    if prune:
        config["prune_final_block"] = True
    rng = np.random.default_rng(819)
    state = _weights(config, rng)
    rows, encoder, decoder, requested = 2 * variables, 7, 7, 2
    length = encoder + decoder
    patches = rng.normal(size=(rows, encoder, 16)).astype(np.float32)
    patches[..., 8:] = rng.random((rows, encoder, 8)) < 0.15
    time_keep = np.broadcast_to(np.tri(length, dtype=bool), (rows, 1, length, length)).copy()
    variate_keep = np.ones((2, length, 1, variables, variables), dtype=bool)
    if variables == 3:
        # Known covariates can read the encoder window; role firewalls apply
        # independently inside each group of target/past/known variates.
        time_keep[2::3, 0, :encoder, :encoder] = True
        variate_keep[:, :, 0, 1:, 0] = False
        variate_keep[:, :, 0, 2, :2] = False
    angles = rng.normal(size=(rows, 1, length, 4)).astype(np.float32)
    angles = np.concatenate([angles, angles], axis=-1)
    inputs = dict(
        patches=patches, forecast_shape=np.zeros((rows, decoder, 1), np.float32),
        time_keep=time_keep, variate_keep=variate_keep,
        cos=np.cos(angles), sin=np.sin(angles),
        output_shape=np.zeros((rows, requested, 1), np.float32),
    )
    actual = _forward(cores, "timestral_core", config, state, inputs)
    # Request every position, then select the forecast outside the network.
    inputs["output_shape"] = np.zeros((rows, length, 1), np.float32)
    full = _forward(cores, "timestral_core", config, state, inputs)
    expected = full[:, -requested * config["patch_size"]:]
    assert actual.shape == (rows, 16, 5)
    assert np.isfinite(actual).all()
    assert (np.diff(actual, axis=-1) >= 0).all()
    np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-4)


@pytest.mark.parametrize("levels", [
    (0.01, 0.1, 0.5, 0.9, 0.99), (0.1, 0.5, 0.7, 0.8), (0.5, 0.9), (0.1, 0.5), (0.5,),
])
def test_quantile_head_matches_float64_reference(cores, levels):
    config = _config()
    mid = levels.index(0.5)
    config.update(quantile_knots=list(levels), median_index=mid, quantile_width_factor=3.0)
    raw = np.random.default_rng(120).normal(size=(2, 4, len(levels))).astype(np.float32)
    raw[0, 0] = 1000
    raw[0, 1] = -1000
    actual = _forward(cores, "monotone_quantiles", config, {}, {"raw": raw})
    expected = np.broadcast_to(raw[..., mid, None], raw.shape).astype(np.float64).copy()
    smooth = np.logaddexp(0.0, raw.astype(np.float64))
    for i in range(mid - 1, -1, -1):
        expected[..., i] = expected[..., i + 1] - 3 * (levels[i + 1] - levels[i]) * smooth[..., i]
    for i in range(mid + 1, len(levels)):
        expected[..., i] = expected[..., i - 1] + 3 * (levels[i] - levels[i - 1]) * smooth[..., i]
    np.testing.assert_array_equal(actual[..., mid], raw[..., mid])
    assert np.isfinite(actual).all()
    assert (np.diff(actual, axis=-1) >= 0).all()
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=1e-5)


def test_model_uses_builtins_without_direct_primitives():
    source = SOURCE.read_text()
    assert "import Attention" in source
    assert not re.search(r"(?<![\w.@])_[a-z]\w*\s", source)
