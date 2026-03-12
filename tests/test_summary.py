from __future__ import annotations

from pathlib import Path

import typer

from brainsurgery.cli.summary import _derive_summary_path, build_raw_plan, write_executed_plan_summary


def test_build_raw_plan_and_derive_summary_path_cover_common_outputs(tmp_path) -> None:
    plan = build_raw_plan(inputs=["a"], output={"path": "out"}, transforms=[{"help": {}}])
    assert plan == {"inputs": ["a"], "output": {"path": "out"}, "transforms": [{"help": {}}]}

    assert _derive_summary_path(None) == Path("brainsurgery-executed-plan.yaml")
    assert _derive_summary_path(tmp_path / "out.safetensors") == tmp_path / "out.safetensors.executed.yaml"

    directory = tmp_path / "dir"
    directory.mkdir()
    assert _derive_summary_path(directory) == directory / "executed-plan.yaml"
    assert _derive_summary_path(tmp_path / "outdir") == tmp_path / "outdir.executed.yaml"


def test_write_executed_plan_summary_writes_yaml(tmp_path, monkeypatch) -> None:
    echoed: list[str] = []
    monkeypatch.setattr(typer, "echo", echoed.append)

    destination = tmp_path / "executed.yaml"
    write_executed_plan_summary(
        inputs=["a"],
        output={"path": "out"},
        transforms=[{"help": {}}],
        destination=destination,
    )
    assert destination.exists()

    write_executed_plan_summary(
        inputs=[],
        output=None,
        transforms=[],
        destination=None,
    )
    assert echoed
