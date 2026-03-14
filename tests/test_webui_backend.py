from __future__ import annotations

import brainsurgery  # noqa: F401

from brainsurgery.webui.backend import render_execution_summary, transform_items


def test_transform_items_include_help_exit_dump() -> None:
    names = {item["name"] for item in transform_items()}
    assert {"help", "exit", "dump"}.issubset(names)


def test_help_transform_metadata_contains_dropdown_options() -> None:
    help_item = next(item for item in transform_items() if item["name"] == "help")
    assert "copy" in help_item["help_commands"]
    assert "assert" in help_item["help_subcommands"]
    assert help_item["help_subcommands"]["assert"]


def test_assert_transform_metadata_contains_expression_options() -> None:
    assert_item = next(item for item in transform_items() if item["name"] == "assert")
    assert "equal" in assert_item["assert_expressions"]
    assert "equal" in assert_item["assert_expression_meta"]
    equal_meta = assert_item["assert_expression_meta"]["equal"]
    assert "left" in equal_meta["allowed_keys"]
    assert "right" in equal_meta["allowed_keys"]


def test_render_execution_summary_renders_yaml_plan() -> None:
    class _Provider:
        def list_model_aliases(self) -> list[str]:
            return ["m2", "m1"]

    summary = render_execution_summary(
        provider=_Provider(),
        executed_transforms=[
            {"load": {"path": "/tmp/model.safetensors", "alias": "m1"}},
            {"copy": {"from": "m1::.*", "to": "m1::\\1"}},
            {"exit": None},
        ],
    )
    assert "transforms:" in summary
    assert "- load:" in summary
    assert "- copy:" in summary
    assert "- exit: {}" in summary
    assert "inputs:" not in summary
    assert "output:" not in summary
