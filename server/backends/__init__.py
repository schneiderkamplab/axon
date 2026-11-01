"""Backend adapters: drive a generated model's stateful forward contract.

``Session`` (the decode loop) is backend-agnostic and speaks to a
``ModelBackend`` protocol only. Each adapter owns tensor construction, result
normalization, and warm-up for its tensor library, so adding a backend never
touches the engine.
"""

from .mlx import MlxBackend
from .protocol import ModelBackend
from .torch import TorchBackend

__all__ = ["MlxBackend", "ModelBackend", "TorchBackend"]
