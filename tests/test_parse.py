from __future__ import annotations

import pytest

from brainsurgery.cli.parse import (
    _normalize_single_transform_spec,
    normalize_transform_specs,
    parse_transform_block,
)


def test_normalize_single_transform_spec_accepts_mapping_and_string() -> None:
    assert _normalize_single_transform_spec({"copy": {"from": "a", "to": "b"}}) == {
        "copy": {"from": "a", "to": "b"}
    }
    assert _normalize_single_transform_spec({"help": None}) == {"help": {}}
    assert _normalize_single_transform_spec("  exit  ") == {"exit": {}}


def test_normalize_single_transform_spec_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="exactly one key"):
        _normalize_single_transform_spec({"a": {}, "b": {}})

    with pytest.raises(ValueError, match="non-empty string"):
        _normalize_single_transform_spec("   ")

    with pytest.raises(ValueError, match="YAML mapping or a bare transform name"):
        _normalize_single_transform_spec(123)


def test_normalize_transform_specs_handles_none_list_and_single_item() -> None:
    assert normalize_transform_specs(None) == []
    assert normalize_transform_specs([{"help": {}}, "exit"]) == [{"help": {}}, {"exit": {}}]
    assert normalize_transform_specs({"copy": {"from": "a", "to": "b"}}) == [
        {"copy": {"from": "a", "to": "b"}}
    ]


def test_parse_transform_block_parses_and_rejects_invalid_yaml() -> None:
    assert parse_transform_block("- help: {}\n- exit") == [{"help": {}}, {"exit": {}}]
    with pytest.raises(ValueError, match="invalid YAML"):
        parse_transform_block("help: [")
