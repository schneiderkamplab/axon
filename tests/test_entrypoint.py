from __future__ import annotations

import brainsurgery


def test_main_passes_explicit_subcommand_through(monkeypatch) -> None:
    calls: list[tuple[list[str], str]] = []

    def _fake_app(*, args, prog_name):  # type: ignore[no-untyped-def]
        calls.append((list(args), prog_name))

    monkeypatch.setattr(brainsurgery, "app", _fake_app)

    brainsurgery.main(["cli", "examples/gpt2.yaml"])

    assert calls == [(["cli", "examples/gpt2.yaml"], "brainsurgery")]


def test_main_passes_empty_args_without_defaulting(monkeypatch) -> None:
    calls: list[tuple[list[str], str]] = []

    def _fake_app(*, args, prog_name):  # type: ignore[no-untyped-def]
        calls.append((list(args), prog_name))

    monkeypatch.setattr(brainsurgery, "app", _fake_app)

    brainsurgery.main(["webcli", "--port", "9000"])

    assert calls == [(["webcli", "--port", "9000"], "brainsurgery")]


def test_main_defaults_to_cli_when_no_subcommand(monkeypatch) -> None:
    calls: list[tuple[list[str], str]] = []

    def _fake_app(*, args, prog_name):  # type: ignore[no-untyped-def]
        calls.append((list(args), prog_name))

    monkeypatch.setattr(brainsurgery, "app", _fake_app)

    brainsurgery.main(["examples/gpt2.yaml"])

    assert calls == [(["cli", "examples/gpt2.yaml"], "brainsurgery")]


def test_main_preserves_explicit_webui_subcommand(monkeypatch) -> None:
    calls: list[tuple[list[str], str]] = []

    def _fake_app(*, args, prog_name):  # type: ignore[no-untyped-def]
        calls.append((list(args), prog_name))

    monkeypatch.setattr(brainsurgery, "app", _fake_app)

    brainsurgery.main(["webui", "--port", "9010"])

    assert calls == [(["webui", "--port", "9010"], "brainsurgery")]
