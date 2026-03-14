from __future__ import annotations

import pytest
import torch

import brainsurgery.core.name_mapping as name_mapping_module
from brainsurgery.engine.state_dicts import _InMemoryStateDict
from brainsurgery.core import (
    match_expr_names,
    resolve_name_mappings,
)
from brainsurgery.core.name_mapping import _require_dest_missing, _require_dest_present

from brainsurgery.core import TensorRef
from brainsurgery.core import TransformError

class _Provider:
    def __init__(self) -> None:
        self.state_dicts = {
            "src": _InMemoryStateDict(),
            "dst": _InMemoryStateDict(),
        }
        self.state_dicts["src"]["layer.0.weight"] = torch.ones(2)
        self.state_dicts["src"]["layer.1.weight"] = torch.zeros(2)

    def get_state_dict(self, model: str) -> _InMemoryStateDict:
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
        _require_dest_missing(mappings=mappings, provider=provider, op_name="copy")

    _require_dest_present(mappings=mappings, provider=provider, op_name="copy")

def test_resolve_name_mappings_rejects_mixed_expression_kinds() -> None:
    provider = _Provider()

    with pytest.raises(TransformError, match="same kind"):
        resolve_name_mappings(
            from_ref=TensorRef(model="src", expr=r"layer\..*"),
            to_ref=TensorRef(model="dst", expr=["block", "$x"]),
            provider=provider,
            op_name="copy",
        )

def test_match_expr_names_and_structured_helpers_wrap_invalid_patterns() -> None:
    with pytest.raises(TransformError, match="invalid source regex"):
        match_expr_names(expr="(", names=["x"], op_name="copy", role="source")

    with pytest.raises(TransformError, match="invalid structured source pattern"):
        match_expr_names(expr=["$1bad"], names=["x"], op_name="copy", role="source")

    with pytest.raises(TransformError, match="invalid structured source pattern"):
        name_mapping_module._match_structured_expr(
            expr=["$1bad"],
            name="x",
            op_name="copy",
            role="source",
        )

    with pytest.raises(TransformError, match="invalid structured destination pattern"):
        name_mapping_module._rewrite_structured_expr(
            expr=["${missing}"],
            match=name_mapping_module._MATCHER.match(["x"], "x"),
            op_name="copy",
            role="destination",
        )

def test_resolve_name_mappings_regex_and_structured_error_paths() -> None:
    provider = _Provider()

    with pytest.raises(TransformError, match="internal error: regex resolver expected"):
        name_mapping_module._resolve_name_mappings_regex(  # type: ignore[arg-type]
            from_ref=TensorRef(model="src", expr=["layer", "$i"]),
            to_ref=TensorRef(model="dst", expr="copy"),
            provider=provider,
            op_name="copy",
        )

    with pytest.raises(TransformError, match="internal error: structured resolver expected"):
        name_mapping_module._resolve_name_mappings_structured(  # type: ignore[arg-type]
            from_ref=TensorRef(model="src", expr="layer.*"),
            to_ref=TensorRef(model="dst", expr=["x"]),
            provider=provider,
            op_name="copy",
        )

    with pytest.raises(TransformError, match="source matched zero tensors"):
        resolve_name_mappings(
            from_ref=TensorRef(model="src", expr=r"missing\..*"),
            to_ref=TensorRef(model="dst", expr=r"copy.\g<0>"),
            provider=provider,
            op_name="copy",
        )

    with pytest.raises(TransformError, match="invalid regex rewrite"):
        resolve_name_mappings(
            from_ref=TensorRef(model="src", expr=r"layer\.(\d+)\.weight"),
            to_ref=TensorRef(model="dst", expr=r"copy.\2.weight"),
            provider=provider,
            op_name="copy",
        )

    provider.state_dicts["src"]["dup.0"] = torch.ones(1)
    provider.state_dicts["src"]["dup.1"] = torch.ones(1)
    with pytest.raises(TransformError, match="destination collision"):
        resolve_name_mappings(
            from_ref=TensorRef(model="src", expr=r"dup\.(\d+)"),
            to_ref=TensorRef(model="dst", expr="same"),
            provider=provider,
            op_name="copy",
        )

def test_resolve_name_mappings_slice_type_and_internal_empty_results(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = _Provider()

    with pytest.raises(TransformError, match="source slice must be a string"):
        resolve_name_mappings(
            from_ref=TensorRef(model="src", expr="layer.0.weight", slice_spec=object()),  # type: ignore[arg-type]
            to_ref=TensorRef(model="dst", expr="copy.0.weight"),
            provider=provider,
            op_name="copy",
        )

    with pytest.raises(TransformError, match="destination slice must be a string"):
        resolve_name_mappings(
            from_ref=TensorRef(model="src", expr="layer.0.weight"),
            to_ref=TensorRef(model="dst", expr="copy.0.weight", slice_spec=object()),  # type: ignore[arg-type]
            provider=provider,
            op_name="copy",
        )

    monkeypatch.setattr(name_mapping_module, "_resolve_name_mappings_regex", lambda **kwargs: [])
    with pytest.raises(TransformError, match="resolved zero mappings"):
        resolve_name_mappings(
            from_ref=TensorRef(model="src", expr=r"layer\..*"),
            to_ref=TensorRef(model="dst", expr=r"copy.\g<0>"),
            provider=provider,
            op_name="copy",
        )

    monkeypatch.setattr(name_mapping_module, "_resolve_name_mappings_structured", lambda **kwargs: [])
    with pytest.raises(TransformError, match="resolved zero mappings"):
        resolve_name_mappings(
            from_ref=TensorRef(model="src", expr=["layer", "$i", "weight"]),
            to_ref=TensorRef(model="dst", expr=["copy", "${i}", "weight"]),
            provider=provider,
            op_name="copy",
        )

def test_require_dest_present_rejects_missing_destination() -> None:
    provider = _Provider()
    mappings = resolve_name_mappings(
        from_ref=TensorRef(model="src", expr=r"layer\.0\.weight"),
        to_ref=TensorRef(model="dst", expr="missing.0.weight"),
        provider=provider,
        op_name="copy",
    )
    with pytest.raises(TransformError, match="destination missing"):
        _require_dest_present(mappings=mappings, provider=provider, op_name="copy")
