from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from brainsurgery.core import (
    BinaryMappingSpec,
    BinaryMappingTransform,
    DestinationPolicy,
    TensorRef,
    TernaryMappingSpec,
    TernaryMappingTransform,
    TransformError,
    UnarySpec,
    UnaryTransform,
)
from brainsurgery.engine.state_dicts import _InMemoryStateDict


class _Provider:
    def __init__(self) -> None:
        self.state_dicts = {
            "src": _InMemoryStateDict(),
            "other": _InMemoryStateDict(),
            "dst": _InMemoryStateDict(),
        }
        self.state_dicts["src"]["layer.0.weight"] = torch.ones(2, 2)
        self.state_dicts["src"]["layer.1.weight"] = torch.zeros(2, 2)
        self.state_dicts["other"]["peer.0.weight"] = torch.ones(2, 2)
        self.state_dicts["other"]["peer.1.weight"] = torch.zeros(2, 2)
        self.state_dicts["dst"]["copy.0.weight"] = torch.ones(2, 2)

    def get_state_dict(self, model: str) -> _InMemoryStateDict:
        return self.state_dicts[model]


class _Binary(BinaryMappingTransform[BinaryMappingSpec]):
    name = "binary"
    spec_type = BinaryMappingSpec
    destination_policy = DestinationPolicy.MUST_NOT_EXIST

    def validate_refs(self, from_ref: TensorRef, to_ref: TensorRef) -> None:
        del from_ref, to_ref

    def apply_mapping(self, spec, src_name, dst_name, provider) -> None:
        del spec, src_name, dst_name, provider


class _Unary(UnaryTransform[UnarySpec]):
    name = "unary"
    spec_type = UnarySpec
    slice_policy = "allow"

    def apply_to_target(self, spec: UnarySpec, name: str, provider) -> None:
        del spec, name, provider


@dataclass(frozen=True)
class _TernarySpec(TernaryMappingSpec):
    pass


class _Ternary(TernaryMappingTransform[_TernarySpec]):
    name = "ternary"
    spec_type = _TernarySpec
    destination_policy = DestinationPolicy.MUST_EXIST

    def validate_refs(
        self, from_a_ref: TensorRef, from_b_ref: TensorRef, to_ref: TensorRef
    ) -> None:
        del from_a_ref, from_b_ref, to_ref

    def apply_mapping(self, spec, src_a_name, src_b_name, dst_name, provider) -> None:
        del spec, src_a_name, src_b_name, dst_name, provider


def test_binary_mapping_transform_compile_and_destination_policy() -> None:
    transform = _Binary()
    spec = transform.compile(
        {"from": "src::layer\\.(\\d+)\\.weight", "to": "dst::copy.\\1.weight"}, None
    )
    assert spec.from_ref.model == "src"
    assert spec.to_ref.model == "dst"

    with pytest.raises(TransformError, match="destination already exists"):
        transform.validate_resolved_items(
            spec, transform.resolve_items(spec, _Provider()), _Provider()
        )


def test_unary_transform_compile_allows_slices_and_resolves_targets() -> None:
    transform = _Unary()
    spec = transform.compile({"target": "src::layer\\..*::[:1]"}, None)

    assert spec.target_ref == TensorRef(model="src", expr="layer\\..*", slice_spec="[:1]")
    assert transform.resolve_targets(spec, _Provider()) == ["layer.0.weight", "layer.1.weight"]


def test_ternary_mapping_transform_resolves_regex_triples_and_requires_existing_destination() -> (
    None
):
    transform = _Ternary()
    spec = transform.compile(
        {
            "from_a": "src::layer\\.(\\d+)\\.weight",
            "from_b": "other::peer.\\1.weight",
            "to": "dst::copy.\\1.weight",
        },
        None,
    )

    resolved = transform.resolve_items(spec, _Provider())
    assert resolved == [
        ("layer.0.weight", "peer.0.weight", "copy.0.weight"),
        ("layer.1.weight", "peer.1.weight", "copy.1.weight"),
    ]

    with pytest.raises(TransformError, match="destination missing"):
        transform.validate_resolved_items(spec, resolved, _Provider())
