from __future__ import annotations

import pytest
import torch

from brainsurgery.core import (
    match_expr_names,
    require_dest_missing,
    require_dest_present,
    resolve_name_mappings,
)
from brainsurgery.providers import InMemoryStateDict
from brainsurgery.core import TensorRef
from brainsurgery.core import TransformError


class _Provider:
    def __init__(self) -> None:
        self.state_dicts = {
            "src": InMemoryStateDict(),
            "dst": InMemoryStateDict(),
        }
        self.state_dicts["src"]["layer.0.weight"] = torch.ones(2)
        self.state_dicts["src"]["layer.1.weight"] = torch.zeros(2)

    def get_state_dict(self, model: str) -> InMemoryStateDict:
        return self.state_dicts[model]


def test_match_expr_names_supports_regex_and_structured_patterns() -> None:
    names = {"encoder.0.proj", "encoder.1.ffn", "decoder.0.proj"}

    assert match_expr_names(
        expr=r"encoder\..*",
        names=names,
        op_name="copy",
        role="source",
    ) == ["encoder.0.proj", "encoder.1.ffn"]
    assert match_expr_names(
        expr=["encoder", "$layer", "$name"],
        names=names,
        op_name="copy",
        role="source",
    ) == ["encoder.0.proj", "encoder.1.ffn"]


def test_resolve_name_mappings_regex_rewrites_and_parses_slices() -> None:
    provider = _Provider()

    resolved = resolve_name_mappings(
        from_ref=TensorRef(model="src", expr=r"layer\.(\d+)\.weight", slice_spec="[:1]"),
        to_ref=TensorRef(model="dst", expr=r"copy.\1.weight", slice_spec=None),
        provider=provider,
        op_name="copy",
    )

    assert [(item.src_name, item.dst_name, item.src_slice) for item in resolved] == [
        ("layer.0.weight", "copy.0.weight", (slice(None, 1, None),)),
        ("layer.1.weight", "copy.1.weight", (slice(None, 1, None),)),
    ]


def test_resolve_name_mappings_structured_rewrites() -> None:
    provider = _Provider()

    resolved = resolve_name_mappings(
        from_ref=TensorRef(model="src", expr=["layer", "$idx", "weight"]),
        to_ref=TensorRef(model="dst", expr=["block", "${idx}", "copy"]),
        provider=provider,
        op_name="copy",
    )

    assert [(item.src_name, item.dst_name) for item in resolved] == [
        ("layer.0.weight", "block.0.copy"),
        ("layer.1.weight", "block.1.copy"),
    ]


def test_require_dest_missing_and_present_validate_destination_state() -> None:
    provider = _Provider()
    provider.state_dicts["dst"]["copy.0.weight"] = torch.ones(2)

    mappings = resolve_name_mappings(
        from_ref=TensorRef(model="src", expr=r"layer\.0\.weight"),
        to_ref=TensorRef(model="dst", expr="copy.0.weight"),
        provider=provider,
        op_name="copy",
    )

    with pytest.raises(TransformError, match="destination already exists"):
        require_dest_missing(mappings=mappings, provider=provider, op_name="copy")

    require_dest_present(mappings=mappings, provider=provider, op_name="copy")


def test_resolve_name_mappings_rejects_mixed_expression_kinds() -> None:
    provider = _Provider()

    with pytest.raises(TransformError, match="same kind"):
        resolve_name_mappings(
            from_ref=TensorRef(model="src", expr=r"layer\..*"),
            to_ref=TensorRef(model="dst", expr=["block", "$x"]),
            provider=provider,
            op_name="copy",
        )
