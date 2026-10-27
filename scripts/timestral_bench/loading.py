"""Checkpoint and MLX loading for the benchmark host. Worker processes import no PyTorch."""

import hashlib
import importlib.util
import sys
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


def core_config(config, *, prune_final_block=False):
    """Adapt checkpoint configuration to the dimensions used by the Axon graph."""
    return config.to_dict() | {
        "median_index": config.median_index,
        "prune_final_block": prune_final_block,
    }


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


def load_core_class(core_module=None):
    """Return the generated ``TimestralCore`` class.

    ``core_module`` is a Python file written by ``build.write_cores``; its directory is
    put on ``sys.path`` so the generated ``from _axon_common import ...`` resolves. When
    it is ``None`` the fused core is generated in-process, which imports the compiler.
    """
    if core_module is None:
        from .build import generate_core

        namespace = {}
        exec(compile(generate_core("fused"), "<axon-timestral-mlx>", "exec"), namespace)
        return namespace["TimestralCore"]
    core_module = Path(core_module).resolve()
    if not core_module.is_file():
        raise FileNotFoundError(f"generated core not found: {core_module}; run the build step")
    helpers = core_module.parent / "_axon_common.py"
    if not helpers.is_file():
        raise FileNotFoundError(f"helper module missing beside the generated core: {helpers}")
    # Bind the generated module's ``from _axon_common import ...`` to the sibling copy,
    # whichever ``_axon_common`` another directory may have registered earlier.
    sys.modules["_axon_common"] = _import_file("_axon_common", helpers)
    directory = str(core_module.parent)
    if directory in sys.path:
        sys.path.remove(directory)
    sys.path.insert(0, directory)
    tag = hashlib.sha256(str(core_module).encode()).hexdigest()[:12]
    return _import_file(f"_timestral_core_{core_module.stem}_{tag}", core_module).TimestralCore


def _import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _synchronize(mx):
    synchronize = getattr(mx, "synchronize", None)
    if synchronize is not None:
        synchronize()


class MlxCore:
    """Compiled generated forward with NumPy in/out and optional call recording.

    Recording keeps the converted MLX inputs of every call so ``replay`` can time the
    generated network alone, without host preparation, transfers, or output conversion.
    """

    def __init__(self, generated):
        import mlx.core as mx

        self.mx = mx
        self.generated = generated
        self.forward = mx.compile(generated.forward)
        self.recording = False
        self.calls = []

    def __call__(self, **inputs):
        mx = self.mx
        arrays = {key: mx.array(value) for key, value in inputs.items()}
        if self.recording:
            mx.eval(*arrays.values())
            self.calls.append(arrays)
        result = self.forward(**arrays)
        mx.eval(result)
        return np.array(result)

    def replay(self):
        """Run every recorded call on its preloaded arrays, evaluating each result."""
        mx = self.mx
        outputs = []
        for arrays in self.calls:
            result = self.forward(**arrays)
            mx.eval(result)
            outputs.append(result)
        _synchronize(mx)
        return outputs


def load_model(path_or_repo_id, *, core_module=None, prune_final_block=False, **hub_options):
    import mlx.core as mx

    path = resolve_checkpoint(path_or_repo_id, **hub_options)
    config = TimestralConfig.load(path / "config.json")
    state = mx.load(str(path / "model.safetensors"))
    validate_weights(state, config)
    # Upstream validates fp32 inference. Keep this port in fp32 until reduced
    # precision has its own parity evidence.
    state = {key: value.astype(mx.float32) for key, value in state.items()}
    # Generated weights live in a flat dictionary, outside nn.Module's parameter
    # traversal. Materialize them before tracing, so compiled shapes share the
    # same evaluated buffers instead of repeating lazy checkpoint loads.
    mx.eval(state)
    generated = load_core_class(core_module).from_state_dict(
        state, model_config=core_config(config, prune_final_block=prune_final_block)
    )
    return TimestralModel(config, MlxCore(generated))
