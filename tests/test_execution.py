from __future__ import annotations

import pytest

from brainsurgery.execution import execute_transform_pairs
from brainsurgery.transform import BaseTransform, CompiledTransform, TransformControl, TransformError, TransformResult


class _Transform(BaseTransform):
    name = "dummy"

    def __init__(self, *, control: TransformControl = TransformControl.CONTINUE, fail: bool = False):
        self.control = control
        self.fail = fail

    def compile(self, payload: dict, default_model: str | None) -> object:
        del payload, default_model
        return object()

    def apply(self, spec: object, provider: object) -> TransformResult:
        del spec, provider
        if self.fail:
            raise TransformError("boom")
        return TransformResult(name=self.name, count=1, control=self.control)

    def infer_output_model(self, spec: object) -> str:
        del spec
        return "model"


def test_execute_transform_pairs_stops_on_exit_control() -> None:
    pairs = [
        ({"first": {}}, CompiledTransform(_Transform(), object())),
        ({"exit": {}}, CompiledTransform(_Transform(control=TransformControl.EXIT), object())),
    ]

    should_continue, executed = execute_transform_pairs(pairs, object(), interactive=False)
    assert should_continue is False
    assert executed == [{"first": {}}, {"exit": {}}]


def test_execute_transform_pairs_returns_to_prompt_on_interactive_failure() -> None:
    pairs = [
        ({"ok": {}}, CompiledTransform(_Transform(), object())),
        ({"bad": {}}, CompiledTransform(_Transform(fail=True), object())),
    ]

    should_continue, executed = execute_transform_pairs(pairs, object(), interactive=True)
    assert should_continue is True
    assert executed == [{"ok": {}}]


def test_execute_transform_pairs_raises_in_non_interactive_mode() -> None:
    with pytest.raises(TransformError, match="boom"):
        execute_transform_pairs(
            [({"bad": {}}, CompiledTransform(_Transform(fail=True), object()))],
            object(),
            interactive=False,
        )
