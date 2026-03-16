from __future__ import annotations

import pytest

from brainsurgery.cli.parse import (
    _normalize_single_transform_spec,
    _parse_transform_block,
    normalize_transform_specs,
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
    assert _parse_transform_block("- help: {}\n- exit") == [{"help": {}}, {"exit": {}}]
    with pytest.raises(ValueError, match="invalid YAML"):
        _parse_transform_block("help: [")


def test_parse_transform_block_preserves_structured_output_template_tokens() -> None:
    parsed = _parse_transform_block(
        'copy: { from: ["layer", "$i", "attn"], to: ["layer", "${i}", "attention"] }'
    )
    assert parsed == [
        {
            "copy": {
                "from": ["layer", "$i", "attn"],
                "to": ["layer", "${i}", "attention"],
            }
        }
    ]


def test_parse_transform_block_falls_back_to_oly_when_yaml_fails() -> None:
    parsed = _parse_transform_block("copy: from: [*prefix, $i], to: [*prefix, $i]")
    assert parsed == [{"copy": {"from": ["*prefix", "$i"], "to": ["*prefix", "$i"]}}]


def test_parse_transform_block_reports_yaml_and_oly_errors_when_both_fail() -> None:
    with pytest.raises(ValueError, match=r"invalid YAML:[\s\S]*invalid OLY:"):
        _parse_transform_block("copy: from: [")


def test_parse_transform_block_empty_input_reports_yaml_error_only() -> None:
    with pytest.raises(ValueError, match="invalid YAML:"):
        _parse_transform_block("")
