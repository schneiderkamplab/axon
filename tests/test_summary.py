from dataclasses import dataclass
from pathlib import Path

import typer

from brainsurgery.cli.summary import _write_executed_plan_summary
from brainsurgery.core import CompiledTransform, TensorRef
from brainsurgery.engine import compile_plan
from brainsurgery.engine.summary import executed_plan_summary_doc, parse_summary_mode
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

def test_parse_summary_mode_rejects_invalid_mode() -> None:
    assert parse_summary_mode("resolve") == "resolve"
    assert parse_summary_mode(" raw ") == "raw"
    try:
        parse_summary_mode("invalid")
    except ValueError as exc:
        assert "summary mode must be one of" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected invalid summary mode error")

def test_executed_plan_summary_doc_resolve_mode_reconstructs_compiled_step() -> None:
    @dataclass(frozen=True)
    class _FakeSpec:
        target_ref: TensorRef
        scale: float

    class _FakeTransform:
        name = "scale"

    plan = SurgeryPlan(
        inputs={"model": Path("/tmp/in.safetensors")},
        output=None,
        steps=[
            PlanStep(
                raw={"scale": {"target": "model::x", "scale": 2.0}},
                compiled=CompiledTransform(
                    transform=_FakeTransform(),
                    spec=_FakeSpec(
                        target_ref=TensorRef(model="model", expr="x"),
                        scale=2.0,
                    ),
                ),
                status="done",
            )
        ],
    )

    resolved = executed_plan_summary_doc(plan, mode="resolve")
    assert resolved == {
        "inputs": ["model::/tmp/in.safetensors"],
        "transforms": [{"scale": {"target": "model::x", "scale": 2.0}}],
    }

def test_executed_plan_summary_doc_resolve_mode_serializes_assert_and_fill_payloads() -> None:
    plan = compile_plan(
        {
            "inputs": ["model::/tmp/in.safetensors", "orig::/tmp/orig.safetensors"],
            "transforms": [
                {"assert": {"equal": {"left": "model::(.+)", "right": "orig::\\1"}}},
                {"fill": {"from": "model::demo.x", "to": "model::demo.filled", "mode": "constant", "value": 1.0}},
                {"fill_": {"target": "model::demo.filled", "mode": "tensor", "values": [0, 1, 2, 3]}},
            ],
        }
    )
    for step in plan.steps:
        step.status = "done"

    resolved = executed_plan_summary_doc(plan, mode="resolve")
    transforms = resolved["transforms"]
    assert transforms[0] == {
        "assert": {
            "equal": {
                "left": "model::(.+)",
                "right": "orig::\\1",
            }
        }
    }
    assert transforms[1] == {
        "fill": {
            "from": "model::demo.x",
            "to": "model::demo.filled",
            "mode": "constant",
            "value": 1.0,
        }
    }
    assert transforms[2] == {
        "fill_": {
            "target": "model::demo.filled",
            "mode": "tensor",
            "values": [0, 1, 2, 3],
        }
    }

def test_executed_plan_summary_doc_resolve_mode_is_replayable_for_help_prefixes_dump() -> None:
    source = {
        "inputs": ["model::/tmp/in.safetensors"],
        "transforms": [
            {"help": "diff"},
            {"prefixes": {"mode": "list"}},
            {"dump": {"target": ["*all"], "format": "compact"}},
        ],
    }
    plan = compile_plan(source)
    for step in plan.steps:
        step.status = "done"

    resolved = executed_plan_summary_doc(plan, mode="resolve")
    reparsed = compile_plan(resolved)
    assert len(reparsed.steps) == 3
