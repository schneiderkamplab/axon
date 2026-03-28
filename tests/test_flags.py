from __future__ import annotations

import os

LONG_TEST_ENV = "BS_LONG"


def run_long_tests_enabled() -> bool:
    return os.environ.get(LONG_TEST_ENV, "0") == "1"
