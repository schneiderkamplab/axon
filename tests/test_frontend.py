from __future__ import annotations

from brainsurgery.engine.frontend import emit_line, set_output_emitter, use_output_emitter

def test_emit_line_uses_custom_emitter() -> None:
    lines: list[str] = []
    previous = set_output_emitter(lines.append)
    try:
        emit_line("hello")
    finally:
        set_output_emitter(previous)

    assert lines == ["hello"]

def test_use_output_emitter_restores_previous_emitter() -> None:
    outer: list[str] = []
    inner: list[str] = []
    previous = set_output_emitter(outer.append)
    try:
        with use_output_emitter(inner.append):
            emit_line("in")
        emit_line("out")
    finally:
        set_output_emitter(previous)

    assert inner == ["in"]
    assert outer == ["out"]
