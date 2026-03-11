from __future__ import annotations

from types import SimpleNamespace

import brainsurgery.utils.history as history


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

    history.add_history_entry("  one  ")
    history.add_history_entry("one")
    history.add_history_entry("   ")

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
