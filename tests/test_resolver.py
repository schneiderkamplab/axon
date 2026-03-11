from __future__ import annotations

import pytest
import torch

from brainsurgery.providers import InMemoryStateDict
from brainsurgery.core import TensorRef
from brainsurgery.core.resolver import (
    _resolve_tensor_mappings as resolve_tensor_mappings,
    _resolve_tensors as resolve_tensors,
    resolve_single_tensor,
    resolve_target_names,
)
from brainsurgery.core import TransformError


class _ResolverError(TransformError):
    pass


class _Provider:
    def __init__(self) -> None:
        self.state_dicts = {
            "src": InMemoryStateDict(),
            "dst": InMemoryStateDict(),
        }
        self.state_dicts["src"]["weight.0"] = torch.arange(6).reshape(2, 3)
        self.state_dicts["src"]["weight.1"] = torch.ones(2, 3)
        self.state_dicts["dst"]["copy.0"] = torch.zeros(2, 3)

    def get_state_dict(self, model: str) -> InMemoryStateDict:
        return self.state_dicts[model]


def _resolve_names(ref: TensorRef, provider: _Provider, *, op_name: str) -> list[str]:
    del op_name
    return resolve_target_names(
        target_ref=ref,
        provider=provider,
        op_name="assert",
        match_names=lambda **kwargs: [name for name in kwargs["names"] if name.startswith("weight")],
        error_type=_ResolverError,
    )


def test_resolve_single_tensor_requires_single_match() -> None:
    provider = _Provider()

    with pytest.raises(_ResolverError, match="matched 2 tensors"):
        resolve_single_tensor(
            TensorRef(model="src", expr=r"weight\..*"),
            provider,
            op_name="assert",
            resolve_names=_resolve_names,
            error_type=_ResolverError,
        )


def test_resolve_tensors_applies_slice_and_concretizes_names() -> None:
    provider = _Provider()

    resolved = resolve_tensors(
        TensorRef(model="src", expr=r"weight\..*", slice_spec="[:, 1]"),
        provider,
        op_name="assert",
        resolve_names=_resolve_names,
    )

    assert [ref.expr for ref, _ in resolved] == ["weight.0", "weight.1"]
    assert torch.equal(resolved[0][1], torch.tensor([1, 4]))


def test_resolve_tensor_mappings_requires_existing_destination() -> None:
    provider = _Provider()

    with pytest.raises(_ResolverError, match="destination missing"):
        resolve_tensor_mappings(
            TensorRef(model="src", expr=r"weight\.1"),
            TensorRef(model="dst", expr="copy.1"),
            provider,
            op_name="equal",
            error_type=_ResolverError,
        )

    resolved = resolve_tensor_mappings(
        TensorRef(model="src", expr=r"weight\.0", slice_spec="[:, :2]"),
        TensorRef(model="dst", expr="copy.0", slice_spec="[:, :2]"),
        provider,
        op_name="equal",
        error_type=_ResolverError,
    )
    left_ref, left, right_ref, right = resolved[0]
    assert left_ref.expr == "weight.0"
    assert right_ref.expr == "copy.0"
    assert left.shape == right.shape == (2, 2)
