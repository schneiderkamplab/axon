from __future__ import annotations

from contextlib import contextmanager

import pytest

import brainsurgery.cli.interactive as interactive


def test_prompt_interactive_transform_multiline_block_success(monkeypatch: pytest.MonkeyPatch) -> None:
    lines = iter(["this is not oly", ""])
    monkeypatch.setattr(interactive, "_collect_completion_candidates", lambda _: [])
    monkeypatch.setattr(interactive, "_readline_safe_prompt", lambda _: "... ")
    @contextmanager
    def _ctx(**_: object):
        yield

    monkeypatch.setattr(interactive, "_interactive_completion", _ctx)
    monkeypatch.setattr(interactive, "parse_transform_block", lambda block: [{"copy": {"from": "a", "to": "b"}}])
    history: list[str] = []
    monkeypatch.setattr(interactive, "_add_history_entry", history.append)
    monkeypatch.setattr("builtins.input", lambda _prompt: next(lines))
    result = interactive.prompt_interactive_transform(state_dict_provider=None)
    assert result == [{"copy": {"from": "a", "to": "b"}}]
    assert history
