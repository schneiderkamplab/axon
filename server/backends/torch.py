"""Torch adapter for the ``ModelBackend`` protocol (CPU-first, device-parametric).

Generated torch models share the MLX contract exactly:
``forward(input_ids, past_kv=..., use_cache=...)`` -> ``{'logits', 'new_kv'}``,
with the attention mask defaulting to causal when omitted. This adapter is the
agnosticsm proof: the same ``Session`` runs over it unmodified.
"""

from __future__ import annotations

from typing import Any

from .protocol import extract_cache, extract_logits, last_logits

__all__ = ["TorchBackend"]


class TorchBackend:
    """Drives a generated torch model: ``torch`` id tensors, ``.eval()`` inference."""

    def __init__(self, model: Any, *, device: str = "cpu") -> None:
        import torch

        self.model = model.to(device).eval()
        self.device = torch.device(device)

    def ids_2d(self, ids: list[int] | list[list[int]]) -> Any:
        import torch

        if ids and isinstance(ids[0], (list, tuple)):
            return torch.tensor([list(row) for row in ids], dtype=torch.long, device=self.device)
        return torch.tensor([list(ids)], dtype=torch.long, device=self.device)

    def step_2d(self, token: int) -> Any:
        import torch

        return torch.tensor([[int(token)]], dtype=torch.long, device=self.device)

    def flat_ids(self, ids: Any) -> list[int]:
        return [int(value) for value in ids.reshape(-1).tolist()]

    def seq_len(self, ids: Any) -> int:
        return int(ids.shape[1])

    def forward(self, input_ids: Any, past_kv: Any = None, use_cache: bool = True) -> Any:
        return self.model.forward(input_ids, past_kv=past_kv, use_cache=use_cache)

    def extract_logits(self, result: Any) -> Any:
        return extract_logits(result)

    def extract_cache(self, result: Any, previous: Any = None) -> Any:
        return extract_cache(result, previous)

    def last_logits(self, logits: Any) -> Any:
        return last_logits(logits)

    def to_numpy(self, values: Any) -> Any:
        import numpy as np
        import torch

        if isinstance(values, torch.Tensor):
            values = values.detach().to("cpu")
        return np.asarray(values)
