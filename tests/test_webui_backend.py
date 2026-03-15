from __future__ import annotations

from pathlib import Path

import brainsurgery  # noqa: F401

from brainsurgery.web.ui.backend import _render_execution_summary, _transform_items
from brainsurgery.engine.plan import PlanStep, SurgeryPlan

def test_transform_items_include_help_exit_dump() -> None:
    names = {item["name"] for item in _transform_items()}
    assert {"help", "exit", "dump"}.issubset(names)

def test_help_transform_metadata_contains_dropdown_options() -> None:
    help_item = next(item for item in _transform_items() if item["name"] == "help")
    assert "copy" in help_item["help_commands"]
    assert "assert" in help_item["help_subcommands"]
    assert help_item["help_subcommands"]["assert"]

def test_assert_transform_metadata_contains_expression_options() -> None:
    assert_item = next(item for item in _transform_items() if item["name"] == "assert")
    assert "equal" in assert_item["assert_expressions"]
    assert "equal" in assert_item["assert_expression_meta"]
    equal_meta = assert_item["assert_expression_meta"]["equal"]
    assert "left" in equal_meta["allowed_keys"]
    assert "right" in equal_meta["allowed_keys"]

def test_set_transform_metadata_contains_boolean_keys() -> None:
    set_item = next(item for item in _transform_items() if item["name"] == "set")
    assert "dry-run" in set_item["boolean_keys"]
    assert "verbose" in set_item["boolean_keys"]

def test_transform_items_include_iterating_metadata() -> None:
    copy_item = next(item for item in _transform_items() if item["name"] == "copy")
    help_item = next(item for item in _transform_items() if item["name"] == "help")
    assert copy_item["iterating"] is True
    assert help_item["iterating"] is False

def test_render_execution_summary_renders_yaml_plan() -> None:
    plan = SurgeryPlan(
        inputs={},
        output=None,
        steps=[
            PlanStep(raw={"load": {"path": "/tmp/model.safetensors", "alias": "m1"}}, status="done"),
            PlanStep(raw={"copy": {"from": "m1::.*", "to": "m1::\\1"}}, status="done"),
            PlanStep(raw={"exit": None}, status="done"),
        ],
    )
    summary = _render_execution_summary(plan=plan)
    assert "transforms:" in summary
    assert "- load:" in summary
    assert "- copy:" in summary
    assert "- exit: {}" in summary
    assert "inputs:" not in summary
    assert "output:" not in summary

def test_render_execution_summary_resolve_mode_includes_inputs() -> None:
    plan = SurgeryPlan(
        inputs={"model": Path("/tmp/model.safetensors")},
        output=None,
        steps=[PlanStep(raw={"exit": {}}, status="done")],
    )
    summary = _render_execution_summary(plan=plan, mode="resolve")
    assert "inputs:" in summary
    assert "- model::/tmp/model.safetensors" in summary
