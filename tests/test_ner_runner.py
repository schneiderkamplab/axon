import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import run_ner_benchmarks as runner


def arguments(monkeypatch, tmp_path, *extra):
    monkeypatch.setattr(
        runner.sys,
        "argv",
        [
            "run_ner_benchmarks.py",
            "--run-dir",
            str(tmp_path / "bundle"),
            "--results-dir",
            str(tmp_path / "results"),
            *extra,
        ],
    )


def test_mac_repetitions_are_independent_and_do_not_execute_in_dry_run(
    monkeypatch, tmp_path, capsys
):
    arguments(
        monkeypatch,
        tmp_path,
        "--suite",
        "mac-compare",
        "--stage",
        "performance",
        "--repetitions",
        "3",
        "--dry-run",
    )
    monkeypatch.setattr(runner.subprocess, "run", lambda *a, **k: pytest.fail("dry run executed"))
    runner.main()
    jobs = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert len(jobs) == 30
    outputs = [job["command"][job["command"].index("--output") + 1] for job in jobs]
    assert len(set(outputs)) == 30
    assert all(job["command"][-2:] == ["--rounds", "1"] for job in jobs)
    assert {job["repetition"] for job in jobs} == {0, 1, 2}
    configs = [{Path(p).name.rsplit("-r", 1)[0] for p in outputs[n : n + 10]} for n in (0, 10, 20)]
    assert configs[0] == configs[1] == configs[2]
    assert "hf-mps-fp32-performance" in configs[0]
    assert "axon-mps-fp16-performance" in configs[0]
    assert "mlx-metal-fp16-performance" in configs[0]
    assert "mlx-metal-fp32-compiled-performance" in configs[0]
    assert not (tmp_path / "results").exists()


def test_keep_going_retains_failure_and_resume_does_not_claim_success(monkeypatch, tmp_path):
    arguments(monkeypatch, tmp_path, "--suite", "mps", "--stage", "quality", "--keep-going")
    calls = []

    def execute(command, *, env, stdout, stderr):
        assert env["PYTORCH_ENABLE_MPS_FALLBACK"] == "0"
        calls.append(command)
        output = Path(command[command.index("--output") + 1])
        passed = len(calls) != 1
        output.write_text(json.dumps({"quality": {"quality_gate_passed": passed}}))
        stdout.write("retained evidence\n")
        return SimpleNamespace(returncode=0 if passed else 1)

    monkeypatch.setattr(runner.subprocess, "run", execute)
    with pytest.raises(SystemExit, match="1"):
        runner.main()
    assert len(calls) == 4  # A failure must not prevent trying the remaining configurations.
    artifacts = {p: p.read_bytes() for p in (tmp_path / "results").iterdir()}
    arguments(
        monkeypatch, tmp_path, "--suite", "mps", "--stage", "quality", "--keep-going", "--resume"
    )
    with pytest.raises(SystemExit, match="1"):
        runner.main()
    assert len(calls) == 4
    assert all(p.read_bytes() == content for p, content in artifacts.items())


def test_existing_log_without_json_is_preserved(monkeypatch, tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    evidence = results / "hf-mps-fp32-quality.log"
    evidence.write_text("original crash traceback")
    arguments(monkeypatch, tmp_path, "--suite", "mps", "--stage", "quality")
    monkeypatch.setattr(
        runner.subprocess, "run", lambda *a, **k: pytest.fail("existing log overwritten")
    )
    with pytest.raises(FileExistsError):
        runner.main()
    assert evidence.read_text() == "original crash traceback"
