from __future__ import annotations

from pathlib import Path

import pytest

from brainsurgery.core.completion import complete_filesystem_paths


def test_complete_filesystem_paths_quotes_and_filters(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("x")
    (tmp_path / "adir").mkdir()
    out = complete_filesystem_paths(f"'{tmp_path.as_posix()}/a")
    assert any(item.startswith("'") for item in out)
    assert any(item.endswith("a.txt") for item in out)
    assert any(item.endswith("adir/") for item in out)


def test_complete_filesystem_paths_include_switches_and_limit(tmp_path: Path) -> None:
    for idx in range(5):
        (tmp_path / f"f{idx}.txt").write_text("x")
    only_files = complete_filesystem_paths(
        f"{tmp_path.as_posix()}/f",
        include_dirs=False,
        include_files=True,
        limit=2,
    )
    assert len(only_files) == 2
    assert all(not item.endswith("/") for item in only_files)

    missing = complete_filesystem_paths(f"{tmp_path.as_posix()}/does-not-exist/")
    assert missing == []


def test_complete_filesystem_paths_handles_os_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class _BadBase:
        def exists(self) -> bool:
            return True

        def is_dir(self) -> bool:
            return True

        def iterdir(self):
            raise OSError("boom")

    monkeypatch.setattr("brainsurgery.core.completion.Path", lambda *_args, **_kwargs: _BadBase())  # type: ignore[misc]
    assert complete_filesystem_paths("x") == []


def test_complete_filesystem_paths_entry_filtering_and_entry_is_dir_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _BadEntry:
        name = "bad"
        _calls = 0

        def is_dir(self) -> bool:
            self._calls += 1
            if self._calls == 1:
                return False
            raise OSError("boom")

    class _DirEntry:
        name = "dir"

        def is_dir(self) -> bool:
            return True

    class _FileEntry:
        name = "file"

        def is_dir(self) -> bool:
            return False

    class _Base:
        def exists(self) -> bool:
            return True

        def is_dir(self) -> bool:
            return True

        def iterdir(self):
            return [_BadEntry(), _DirEntry(), _FileEntry()]

    monkeypatch.setattr("brainsurgery.core.completion.Path", lambda *_args, **_kwargs: _Base())  # type: ignore[misc]
    only_files = complete_filesystem_paths("", include_files=True, include_dirs=False)
    assert only_files == ["file"]
    only_dirs = complete_filesystem_paths("", include_files=False, include_dirs=True)
    assert only_dirs == ["dir/"]
