from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf

from brainsurgery.synapse import emit_model_code_from_synapse_spec


def auto_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_yaml_mapping(path: Path) -> dict[str, Any]:
    loaded = OmegaConf.load(path)
    data = OmegaConf.to_container(loaded, resolve=True)
    assert isinstance(data, dict)
    return {str(key): value for key, value in data.items()}


def build_codegen_model(
    spec: dict[str, Any], class_name: str, state_dict: dict[str, torch.Tensor]
) -> Any:
    source = emit_model_code_from_synapse_spec(spec, class_name=class_name)
    namespace: dict[str, Any] = {}
    exec(source, namespace)  # noqa: S102 - test-controlled generated source
    model_cls = namespace[class_name]
    return model_cls.from_state_dict(state_dict)


def extract_logits(output: Any) -> torch.Tensor:
    if isinstance(output, dict):
        logits = output.get("logits")
        assert isinstance(logits, torch.Tensor)
        return logits
    assert isinstance(output, torch.Tensor)
    return output


def masked_logits_diff(
    lhs: torch.Tensor, rhs: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    token_mask = attention_mask.to(torch.bool).unsqueeze(-1).expand_as(lhs)
    return (lhs - rhs).abs()[token_mask]


def assert_logits_close(
    actual: torch.Tensor,
    reference: torch.Tensor,
    *,
    mean_tol: float,
    max_tol: float,
) -> None:
    diff = (actual - reference).abs()
    assert float(diff.mean()) < mean_tol
    assert float(diff.max()) < max_tol
    assert torch.equal(actual[:, -1, :].argmax(-1), reference[:, -1, :].argmax(-1))


def assert_masked_logits_close(
    actual: torch.Tensor,
    reference: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    mean_tol: float,
    max_tol: float,
) -> None:
    diff = masked_logits_diff(actual, reference, attention_mask)
    assert float(diff.mean()) < mean_tol
    assert float(diff.max()) < max_tol
    b_idx = torch.arange(attention_mask.shape[0], device=attention_mask.device)
    seq_positions = torch.arange(attention_mask.shape[1], device=attention_mask.device).unsqueeze(0)
    last_idx = torch.where(attention_mask.to(torch.bool), seq_positions, -1).max(dim=-1).values
    assert torch.equal(
        actual[b_idx, last_idx, :].argmax(-1),
        reference[b_idx, last_idx, :].argmax(-1),
    )


__all__ = [
    "auto_device",
    "load_yaml_mapping",
    "build_codegen_model",
    "extract_logits",
    "masked_logits_diff",
    "assert_logits_close",
    "assert_masked_logits_close",
]
