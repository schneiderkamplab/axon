from __future__ import annotations

import brainsurgery

def test_normalize_cli_args_handles_mixed_flags_and_values() -> None:
    args = [
        "examples/gpt2.yaml",
        "--log-level",
        "debug",
        "-i",
        "output.path=models/out",
        "--provider=arena",
    ]
    normalized = brainsurgery._normalize_cli_args(args)
    assert normalized == [
        "--log-level",
        "debug",
        "-i",
        "--provider=arena",
        "examples/gpt2.yaml",
        "output.path=models/out",
    ]

def test_normalize_cli_args_keeps_dashdash_separator() -> None:
    args = ["plan.yaml", "--", "-i", "--log-level", "debug"]
    normalized = brainsurgery._normalize_cli_args(args)
    assert normalized == ["plan.yaml", "--", "-i", "--log-level", "debug"]

def test_normalize_cli_args_handles_summary_mode_option() -> None:
    args = [
        "examples/gpt2.yaml",
        "--summary-mode",
        "resolve",
        "--summarize-path",
        "/tmp/summary.yaml",
    ]
    normalized = brainsurgery._normalize_cli_args(args)
    assert normalized == [
        "--summary-mode",
        "resolve",
        "--summarize-path",
        "/tmp/summary.yaml",
        "examples/gpt2.yaml",
    ]
