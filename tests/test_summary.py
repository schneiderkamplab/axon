from __future__ import annotations

import typer

from brainsurgery.cli.summary import _write_executed_plan_summary
from brainsurgery.engine.plan import PlanStep, SurgeryPlan

def test_write_executed_plan_summary_writes_yaml(tmp_path, monkeypatch) -> None:
    echoed: list[str] = []
    monkeypatch.setattr(typer, "echo", echoed.append)

    plan = SurgeryPlan(
        inputs={},
        output=None,
        steps=[PlanStep(raw={"help": {}}, compiled=None, status="done")],
        raw_inputs=["a"],
        raw_output={"path": "out"},
    )

    destination = tmp_path / "executed.yaml"
    _write_executed_plan_summary(
        plan=plan,
        destination=destination,
    )
    assert destination.exists()

    _write_executed_plan_summary(
        plan=SurgeryPlan(inputs={}, output=None, raw_inputs=[]),
        destination=None,
    )
    assert echoed
