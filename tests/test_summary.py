from __future__ import annotations

import typer

from brainsurgery.cli.summary import build_raw_plan, _write_executed_plan_summary

def test_build_raw_plan_common_outputs() -> None:
    plan = build_raw_plan(inputs=["a"], output={"path": "out"}, transforms=[{"help": {}}])
    assert plan == {"inputs": ["a"], "output": {"path": "out"}, "transforms": [{"help": {}}]}

def test_write_executed_plan_summary_writes_yaml(tmp_path, monkeypatch) -> None:
    echoed: list[str] = []
    monkeypatch.setattr(typer, "echo", echoed.append)

    destination = tmp_path / "executed.yaml"
    _write_executed_plan_summary(
        inputs=["a"],
        output={"path": "out"},
        transforms=[{"help": {}}],
        destination=destination,
    )
    assert destination.exists()

    _write_executed_plan_summary(
        inputs=[],
        output=None,
        transforms=[],
        destination=None,
    )
    assert echoed
