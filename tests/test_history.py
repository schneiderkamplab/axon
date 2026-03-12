from __future__ import annotations

from types import SimpleNamespace

import brainsurgery.cli.history as history


class _Readline:
    def __init__(self) -> None:
        self.entries: list[str] = []
        self.history_length = None
        self.bound = None
        self.written = None

    def parse_and_bind(self, value: str) -> None:
        self.bound = value

    def read_history_file(self, path: str) -> None:
        self.written = path

    def set_history_length(self, value: int) -> None:
        self.history_length = value

    def write_history_file(self, path: str) -> None:
        self.written = path

    def get_current_history_length(self) -> int:
        return len(self.entries)

    def get_history_item(self, index: int) -> str:
        return self.entries[index - 1]

    def add_history(self, value: str) -> None:
        self.entries.append(value)


def test_add_history_entry_skips_blank_and_duplicate(monkeypatch) -> None:
    readline = _Readline()
    monkeypatch.setattr(history, "readline", readline)

    history._add_history_entry("  one  ")
    history._add_history_entry("one")
    history._add_history_entry("   ")

    assert readline.entries == ["one"]


def test_configure_history_sets_readline_options(monkeypatch, tmp_path) -> None:
    readline = _Readline()
    registered: list[object] = []

    monkeypatch.setattr(history, "readline", readline)
    monkeypatch.setattr(history, "_HISTORY_FILE", tmp_path / "history.txt")
    monkeypatch.setattr(history, "atexit", SimpleNamespace(register=registered.append))

    history.configure_history()

    assert readline.bound == "set editing-mode emacs"
    assert readline.history_length == history._HISTORY_LENGTH
    assert len(registered) == 1


def test_configure_history_no_readline_is_noop(monkeypatch) -> None:
    monkeypatch.setattr(history, "readline", None)
    history.configure_history()


def test_add_history_entry_no_readline_is_noop(monkeypatch) -> None:
    monkeypatch.setattr(history, "readline", None)
    history._add_history_entry("x")


class _BrokenReadline:
    def parse_and_bind(self, value: str) -> None:
        del value
        raise RuntimeError("bind failed")

    def set_history_length(self, value: int) -> None:
        del value
        raise RuntimeError("set history failed")


def test_configure_history_ignores_readline_binding_failures(monkeypatch) -> None:
    monkeypatch.setattr(history, "readline", _BrokenReadline())
    monkeypatch.setattr(history, "atexit", SimpleNamespace(register=lambda fn: None))
    history.configure_history()


def test_configure_history_write_failure_path(monkeypatch, tmp_path) -> None:
    class _WriteFails(_Readline):
        def write_history_file(self, path: str) -> None:
            del path
            raise RuntimeError("write failed")

    readline = _WriteFails()
    callbacks: list[object] = []
    monkeypatch.setattr(history, "readline", readline)
    monkeypatch.setattr(history, "_HISTORY_FILE", tmp_path / "history.txt")
    monkeypatch.setattr(history, "atexit", SimpleNamespace(register=callbacks.append))
    history.configure_history()
    assert len(callbacks) == 1
    callbacks[0]()


def test_add_history_entry_exception_path(monkeypatch) -> None:
    class _BrokenAdd(_Readline):
        def get_current_history_length(self) -> int:
            raise RuntimeError("broken")

    monkeypatch.setattr(history, "readline", _BrokenAdd())
    history._add_history_entry("x")
