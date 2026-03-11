from __future__ import annotations

from pathlib import Path

import pytest
import typer

from brainsurgery.engine.config import (
    apply_override,
    build_override_fragment,
    deep_merge_dicts,
    load_cli_config,
    parse_override_path,
    parse_override_value,
)


def test_parse_override_path_and_value_support_nested_structures() -> None:
    assert parse_override_path("a.b[2].c") == ["a", "b", 2, "c"]
    assert parse_override_value("3") == "3"
    assert parse_override_value("[1, 2]") == "[1, 2]"


def test_build_override_fragment_and_apply_override_merge_recursively() -> None:
    fragment = build_override_fragment(["a", "b", 1], 5)
    assert fragment == {"a": {"b": [None, 5]}}

    merged = apply_override({"a": {"b": [1, {"x": 1}]}, "keep": True}, "a.b[1].y=2")
    assert merged == {"a": {"b": [1, {"x": 1, "y": "2"}]}, "keep": True}

    assert deep_merge_dicts({"a": [1, {"x": 1}]}, {"a": [None, {"y": 2}]}) == {
        "a": [1, {"x": 1, "y": 2}]
    }


def test_load_cli_config_merges_yaml_and_overrides(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("a:\n  b: 1\nitems:\n  - x\n", encoding="utf-8")

    loaded = load_cli_config([str(config_path), "a.c=2", "items[1]=y"])
    assert loaded == {"a": {"b": 1, "c": "2"}, "items": ["x", "y"]}


def test_load_cli_config_rejects_missing_yaml_like_file() -> None:
    with pytest.raises(typer.BadParameter, match="does not exist"):
        load_cli_config(["missing.yaml"])
