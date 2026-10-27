"""Timestral forecasting host around an Axon-generated MLX core, kept for benchmarking.

Benchmark-only code. It compares Axon-generated MLX against the original PyTorch
implementation and the vikiival MLX port; see ``wiki/scripts.md``. The learned
network lives in ``synapse/models/timestral/generic-timestral.axon``.
"""

from .config import TimestralConfig
from .forecaster import Forecast, TimestralForecaster

__all__ = ["Forecast", "TimestralConfig", "TimestralForecaster"]
