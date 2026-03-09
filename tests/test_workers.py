from __future__ import annotations

import pytest

from brainsurgery.workers import choose_num_io_workers


def test_choose_num_io_workers_validates_inputs() -> None:
    with pytest.raises(ValueError, match="num_items"):
        choose_num_io_workers(-1, 1)
    with pytest.raises(ValueError, match="max_io_workers"):
        choose_num_io_workers(1, 0)


def test_choose_num_io_workers_bounds_to_items() -> None:
    assert choose_num_io_workers(0, 8) == 1
    assert choose_num_io_workers(1, 8) == 1
    assert choose_num_io_workers(4, 8) == 4
    assert choose_num_io_workers(8, 4) == 4
