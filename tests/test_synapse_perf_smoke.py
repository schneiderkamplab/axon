from __future__ import annotations

import os
from pathlib import Path

import pytest

from brainsurgery.synapse import run_axon_test

_RUN_PERF_SMOKE = os.environ.get("BRAINSURGERY_RUN_PERF_SMOKE", "0") == "1"


@pytest.mark.skipif(
    not _RUN_PERF_SMOKE,
    reason="set BRAINSURGERY_RUN_PERF_SMOKE=1 to enable perf smoke tests",
)
def test_axon_test_olmoe_cpu_perf_smoke(repo_root: Path, olmoe_local_path: Path) -> None:
    result = run_axon_test(
        axon_file=repo_root / "examples" / "olmoe_1b_7b_0924.axon",
        weights=olmoe_local_path,
        device="cpu",
        text=["The future of AI is", "Hello world"],
        max_len=32,
    )
    assert bool(result["top1_eq"]) is True
    assert float(result["speed_ratio_axon_over_hf"]) < 3.0
