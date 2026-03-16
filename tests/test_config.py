from __future__ import annotations

from pathlib import Path

import pytest
import typer
from omegaconf import OmegaConf

import brainsurgery.cli.config as config_module
from brainsurgery.cli.config import (
    _load_cli_config,
    apply_override,
    build_override_fragment,
    deep_merge_dicts,
    deep_merge_lists,
    is_yaml_file_arg,
    is_yaml_like_arg,
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
    assert deep_merge_dicts({"a": 1}, {"a": 2}) == {"a": 2}


def test_load_cli_config_merges_yaml_and_overrides(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("a:\n  b: 1\nitems:\n  - x\n", encoding="utf-8")

    loaded = _load_cli_config([str(config_path), "a.c=2", "items[1]=y"])
    assert loaded == {"a": {"b": 1, "c": "2"}, "items": ["x", "y"]}


def test_load_cli_config_rejects_missing_yaml_like_file() -> None:
    with pytest.raises(typer.BadParameter, match="does not exist"):
        _load_cli_config(["missing.yaml"])


def test_parse_override_path_rejects_invalid_paths() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        parse_override_path("")
    with pytest.raises(ValueError, match="invalid override path syntax"):
        parse_override_path("a..b")
    with pytest.raises(ValueError, match="invalid override path syntax"):
        parse_override_path("a b")


def test_apply_override_rejects_missing_equals_and_top_level_list_fragment() -> None:
    with pytest.raises(ValueError, match="key=value"):
        apply_override({}, "x")
    with pytest.raises(ValueError, match="top-level override must produce a mapping"):
        apply_override({}, "[0]=x")


def test_deep_merge_lists_covers_nested_list_and_none_fill() -> None:
    assert deep_merge_lists([1, [10, 11]], [None, [20]]) == [1, [20, 11]]
    assert deep_merge_lists([], [None, None]) == [None, None]


def test_parse_override_value_falls_back_on_parse_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        config_module,
        "OmegaConf",
        type(
            "_BrokenOC",
            (),
            {
                "create": staticmethod(lambda data: data),
                "to_container": staticmethod(
                    lambda data, resolve=True: (_ for _ in ()).throw(RuntimeError("bad"))
                ),
            },
        ),
    )
    assert parse_override_value("x") == "x"


def test_parse_override_value_returns_raw_when_container_not_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        config_module,
        "OmegaConf",
        type(
            "_BrokenOC2",
            (),
            {
                "create": staticmethod(lambda data: data),
                "to_container": staticmethod(lambda data, resolve=True: ["not", "dict"]),
            },
        ),
    )
    assert parse_override_value("x") == "x"


def test_yaml_arg_helpers(tmp_path: Path) -> None:
    file_path = tmp_path / "a.yaml"
    file_path.write_text("x: 1\n", encoding="utf-8")
    assert is_yaml_file_arg(str(file_path)) is True
    assert is_yaml_file_arg(str(tmp_path / "missing.yaml")) is False
    assert is_yaml_like_arg("x.yml") is True


def test_load_cli_config_handles_empty_and_non_mapping_yaml(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    empty_path = tmp_path / "empty.yaml"
    empty_path.write_text("", encoding="utf-8")

    bad_path = tmp_path / "bad.yaml"
    bad_path.write_text("[]\n", encoding="utf-8")

    original_load = OmegaConf.load
    original_to_container = OmegaConf.to_container

    def _fake_load(path: str):
        return original_load(path)

    def _fake_to_container(cfg, resolve=True):
        text = str(cfg)
        if "bad.yaml" in text:
            return []
        out = original_to_container(cfg, resolve=resolve)
        if out == {}:
            return None
        return out

    monkeypatch.setattr(config_module.OmegaConf, "load", _fake_load)
    monkeypatch.setattr(config_module.OmegaConf, "to_container", _fake_to_container)

    assert _load_cli_config([str(empty_path)]) == {}
    with pytest.raises(typer.BadParameter, match="must contain a mapping"):
        _load_cli_config([str(bad_path)])


def test_load_cli_config_wraps_invalid_override_errors() -> None:
    with pytest.raises(typer.BadParameter, match="Invalid override"):
        _load_cli_config(["a..b=1"])
