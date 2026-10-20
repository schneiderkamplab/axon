"""Checkpoint and MLX loading. The exported package does not import PyTorch."""

from functools import lru_cache
from pathlib import Path

import numpy as np

from .config import LAYER_KIND_AXES, TimestralConfig
from .model import TimestralModel


def expected_shapes(config):
    c = config
    d = c.d_model
    shapes = {"representation.forecast_query": (d,), "transformer.out_norm.scale": (d,)}

    def linear(path, inputs, outputs, bias):
        shapes[f"{path}.weight"] = (outputs, inputs)
        if bias:
            shapes[f"{path}.bias"] = (outputs,)

    def residual(path, inputs, outputs, bias):
        linear(f"{path}.hidden_layer", inputs, 4 * d, bias)
        linear(f"{path}.output_layer", 4 * d, outputs, bias)
        linear(f"{path}.residual_layer", inputs, outputs, bias)

    residual("representation.patch_proj", 2 * c.patch_size, d, True)
    residual("head.projection", d, c.patch_size * c.num_quantiles, False)
    for index, kind in enumerate(c.block_schedule):
        prefix = f"transformer.layers.{index}"
        for axis, is_time in enumerate(LAYER_KIND_AXES[kind]):
            shapes[f"{prefix}.attn_norms.{axis}.scale"] = (d,)
            path = f"{prefix}.attns.{axis}"
            linear(f"{path}.wQKV", d, 3 * d, c.attn_bias)
            linear(f"{path}.wO", d, d, c.attn_bias)
            gate = c.time_attn_gate if is_time else c.variate_attn_gate
            if gate == "sdpa_sigmoid":
                linear(f"{path}._gate", d, d, c.attn_bias)
        shapes[f"{prefix}.mlp_norm.scale"] = (d,)
        linear(f"{prefix}.mlp.w12", d, 2 * c.mlp_hidden_dim, c.mlp_bias)
        linear(f"{prefix}.mlp.w3", c.mlp_hidden_dim, d, c.mlp_bias)
    return shapes


def validate_weights(state, config):
    expected = expected_shapes(config)
    missing, extra = set(expected) - set(state), set(state) - set(expected)
    wrong = [
        key for key in expected.keys() & state.keys() if tuple(state[key].shape) != expected[key]
    ]
    if missing or extra or wrong:
        raise ValueError(
            f"checkpoint mismatch: missing={sorted(missing)}, unexpected={sorted(extra)}, wrong_shapes={sorted(wrong)}"
        )


def resolve_checkpoint(path_or_repo_id, **hub_options):
    path = Path(path_or_repo_id).expanduser()
    if path.is_dir():
        return path
    if path.exists():
        raise NotADirectoryError(str(path))
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            str(path_or_repo_id), allow_patterns=["config.json", "model.safetensors"], **hub_options
        )
    )


@lru_cache(maxsize=1)
def _model_class():
    # The exported module is the normal deployment path. In a source checkout,
    # build lazily so the same forecast API is usable without an export step.
    import importlib.util

    name = f"{__package__}._generated"
    if importlib.util.find_spec(name) is not None:
        from ._generated import TimestralCore

        return TimestralCore
    from .build import generate_core

    namespace = {}
    exec(compile(generate_core(), "<axon-timestral-mlx>", "exec"), namespace)
    return namespace["TimestralCore"]


def load_model(path_or_repo_id, **hub_options):
    import mlx.core as mx

    path = resolve_checkpoint(path_or_repo_id, **hub_options)
    config = TimestralConfig.load(path / "config.json")
    state = mx.load(str(path / "model.safetensors"))
    validate_weights(state, config)
    # Upstream validates fp32 inference. Keep this port in fp32 until reduced
    # precision has its own parity evidence.
    state = {key: value.astype(mx.float32) for key, value in state.items()}
    generated = _model_class().from_state_dict(state, model_config=config.to_dict())
    mx.eval(generated.parameters())

    def core(**inputs):
        arrays = {key: mx.array(value) for key, value in inputs.items()}
        # Named tensor arguments preserve floating patch values; the generic
        # codegen's positional input_ids convention is integer-valued.
        result = generated.forward(**arrays)
        mx.eval(result)
        return np.array(result)

    return TimestralModel(config, core)
