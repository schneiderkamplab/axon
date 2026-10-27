"""Uniform access to the three implementations. Heavy imports happen inside ``load``.

Every adapter exposes ``load(checkpoint_dir)``, ``forecast(inputs, constructor=None)``
returning float32 NumPy quantiles, ``assert_isolation()``, ``memory_reset()`` and
``memory_info()``. ``inputs`` is ``{"arrays": {...}, "kwargs": {...}}`` and is passed
as ``forecast(**arrays, **kwargs)``; ``constructor`` holds per-case forecaster options.
"""

import sys

import numpy as np

TORCH_MODULES = ("torch",)
MLX_MODULES = ("mlx", "mlx.core")


def _loaded(names):
    return [name for name in names if name in sys.modules]


def _synchronize(mx):
    synchronize = getattr(mx, "synchronize", None)
    if synchronize is not None:
        synchronize()


def _mlx_peak_memory(mx):
    getter = getattr(mx, "get_peak_memory", None) or getattr(getattr(mx, "metal", None), "get_peak_memory", None)
    return int(getter()) if getter is not None else None


def _mlx_reset_peak_memory(mx):
    reset = getattr(mx, "reset_peak_memory", None) or getattr(getattr(mx, "metal", None), "reset_peak_memory", None)
    if reset is not None:
        reset()


class TorchAdapter:
    """Original Mistral implementation on CPU with a fixed intra-op thread count."""

    runtime = "torch"
    supports_core_replay = False

    def __init__(self, threads):
        self.threads = int(threads)
        self.name = f"torch-{self.threads}"

    def load(self, checkpoint_dir):
        import torch

        torch.set_num_threads(self.threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass  # already initialized by an earlier parallel call
        from timestral import TimestralForecaster
        from timestral.loading import load_model

        self.torch = torch
        self.forecaster_class = TimestralForecaster
        self.model = load_model(str(checkpoint_dir), device="cpu")
        self.default = TimestralForecaster(self.model)
        return {
            "torch_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
            "device": "cpu",
        }

    def forecast(self, inputs, constructor=None):
        forecaster = self.default if not constructor else self.forecaster_class(self.model, **constructor)
        result = forecaster.forecast(**inputs["arrays"], **inputs["kwargs"])
        return np.asarray(result.quantiles.detach().cpu().numpy(), dtype=np.float32)

    def assert_isolation(self):
        loaded = _loaded(MLX_MODULES)
        if loaded:
            raise RuntimeError(f"torch worker imported MLX modules: {loaded}")

    def memory_reset(self):
        return None

    def memory_info(self):
        return {}


class VikiivalAdapter:
    """vikiival/timestral-mlx with its shipped defaults."""

    runtime = "mlx"
    supports_core_replay = False
    name = "vikiival"

    def load(self, checkpoint_dir):
        import mlx.core as mx
        from timestral_mlx import TimestralForecaster
        from timestral_mlx.loading import load_model

        self.mx = mx
        self.forecaster_class = TimestralForecaster
        self.model = load_model(str(checkpoint_dir))
        self.default = TimestralForecaster(self.model)
        return {
            "max_batch_rows": self.default.max_batch_rows,
            "max_batch_tokens": self.default.max_batch_tokens,
        }

    def forecast(self, inputs, constructor=None):
        forecaster = self.default if not constructor else self.forecaster_class(self.model, **constructor)
        result = forecaster.forecast(**inputs["arrays"], **inputs["kwargs"])
        output = np.asarray(result.quantiles, dtype=np.float32)
        _synchronize(self.mx)
        return output

    def assert_isolation(self):
        loaded = _loaded(TORCH_MODULES)
        if loaded:
            raise RuntimeError(f"MLX worker imported PyTorch modules: {loaded}")

    def memory_reset(self):
        _mlx_reset_peak_memory(self.mx)

    def memory_info(self):
        return {"mlx_peak_memory_bytes": _mlx_peak_memory(self.mx)}


class AxonAdapter:
    """Axon-generated MLX core with the NumPy host from this package."""

    runtime = "mlx"
    supports_core_replay = True

    def __init__(self, variant, *, core_module, prune_final_block=False):
        self.variant = variant
        self.core_module = core_module
        self.prune_final_block = bool(prune_final_block)
        self.name = f"axon-{variant}" + ("-pruned" if self.prune_final_block else "")

    def load(self, checkpoint_dir):
        import mlx.core as mx

        from .forecaster import TimestralForecaster
        from .loading import load_model

        self.mx = mx
        self.forecaster_class = TimestralForecaster
        self.model = load_model(
            checkpoint_dir, core_module=self.core_module, prune_final_block=self.prune_final_block
        )
        self.default = TimestralForecaster(self.model)
        return {
            "core_module": str(self.core_module),
            "prune_final_block": self.prune_final_block,
            "max_batch_rows": self.default.max_batch_rows,
            "max_batch_tokens": self.default.max_batch_tokens,
        }

    def forecast(self, inputs, constructor=None):
        forecaster = self.default if not constructor else self.forecaster_class(self.model, **constructor)
        result = forecaster.forecast(**inputs["arrays"], **inputs["kwargs"])
        output = np.asarray(result.quantiles, dtype=np.float32)
        _synchronize(self.mx)
        return output

    def record(self, inputs, constructor=None):
        """Run one forecast while recording every generated-core call; return the call count."""
        core = self.model.core
        core.calls.clear()
        core.recording = True
        try:
            self.forecast(inputs, constructor)
        finally:
            core.recording = False
        return len(core.calls)

    def replay(self):
        return self.model.core.replay()

    def clear_recording(self):
        self.model.core.calls.clear()

    def assert_isolation(self):
        loaded = _loaded(TORCH_MODULES) + (["synapse"] if "synapse" in sys.modules else [])
        if loaded:
            raise RuntimeError(f"MLX worker imported forbidden modules: {loaded}")

    def memory_reset(self):
        _mlx_reset_peak_memory(self.mx)

    def memory_info(self):
        return {"mlx_peak_memory_bytes": _mlx_peak_memory(self.mx)}
