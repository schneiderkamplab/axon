from __future__ import annotations

from dataclasses import dataclass

import pytest

from brainsurgery.engine.workers import _choose_num_io_workers, _run_threadpool_tasks_with_progress

def test_choose_num_io_workers_validates_inputs() -> None:
    with pytest.raises(ValueError, match="num_items"):
        _choose_num_io_workers(-1, 1)
    with pytest.raises(ValueError, match="max_io_workers"):
        _choose_num_io_workers(1, 0)

def test_choose_num_io_workers_bounds_to_items() -> None:
    assert _choose_num_io_workers(0, 8) == 1
    assert _choose_num_io_workers(1, 8) == 1
    assert _choose_num_io_workers(4, 8) == 4
    assert _choose_num_io_workers(8, 4) == 4

@dataclass
class _Progress:
    total: int
    desc: str
    unit: str
    leave: bool
    updates: int = 0
    closed: bool = False

    def update(self, amount: int) -> None:
        self.updates += amount

    def close(self) -> None:
        self.closed = True

def test_run_threadpool_tasks_with_progress_reports_results() -> None:
    progress_items: list[_Progress] = []
    seen_results: list[tuple[int, int]] = []

    def progress_factory(**kwargs: object) -> _Progress:
        progress = _Progress(**kwargs)
        progress_items.append(progress)
        return progress

    _run_threadpool_tasks_with_progress(
        items=[1, 2, 3],
        worker=lambda item: item * 2,
        num_workers=2,
        total=3,
        progress_desc="Load",
        progress_unit="file",
        progress_factory=progress_factory,
        on_result=lambda item, result: seen_results.append((item, result)),
    )

    assert sorted(seen_results) == [(1, 2), (2, 4), (3, 6)]
    assert progress_items == [
        _Progress(total=3, desc="Load", unit="file", leave=False, updates=3, closed=True)
    ]
