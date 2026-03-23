from __future__ import annotations

from pathlib import Path

import pytest

from brainsurgery.synapse import run_axon_test
from tests.test_flags import LONG_TEST_ENV, run_long_tests_enabled

_RUN_PERF_SMOKE = run_long_tests_enabled()


@pytest.mark.skipif(
    not _RUN_PERF_SMOKE,
    reason=f"set {LONG_TEST_ENV}=1 to enable perf smoke tests",
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
