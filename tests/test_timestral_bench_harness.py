"""Harness logic of the Timestral benchmark: workloads, cases, aggregation, CLI. No network, no runtimes."""

from __future__ import annotations

import json
import sys
import textwrap

import numpy as np
import pytest

from scripts.timestral_bench import __main__ as cli
from scripts.timestral_bench.build import COMMON_IMPORT, LOCAL_IMPORT, rewrite_common_import
from scripts.timestral_bench.cases import cases, compare, tolerance
from scripts.timestral_bench.loading import load_core_class
from scripts.timestral_bench.summarize import (
    BASELINE,
    geometric_mean,
    load_summary,
    render_markdown,
    summarize,
)
from scripts.timestral_bench.worker import (
    IMPLS,
    PARITY_IMPLS,
    REFERENCE_IMPL,
    make_adapter,
    rotated,
)
from scripts.timestral_bench.workloads import WORKLOADS, by_name, make_inputs, manifest


def test_workloads_are_nine_unique_and_deterministic():
    names = [workload.name for workload in WORKLOADS]
    assert len(names) == 9 == len(set(names))
    assert len({workload.seed for workload in WORKLOADS}) == 9
    first, second = manifest(), manifest()
    assert first == second
    for workload in WORKLOADS:
        inputs = make_inputs(workload)
        context = inputs["arrays"]["context"]
        assert context.dtype == np.float32
        assert context.shape == (workload.batch, workload.variates, workload.context)
        assert inputs["kwargs"] == {"prediction_length": workload.horizon}
        again = make_inputs(workload)["arrays"]["context"]
        assert np.array_equal(context, again, equal_nan=True)


def test_covariate_workload_has_missing_values_and_future_known_covariates():
    workload = by_name("covariates_missing_2000_1500")
    arrays = make_inputs(workload)["arrays"]
    context = arrays["context"]
    missing = np.isnan(context).mean()
    assert 0.05 < missing < 0.15
    assert arrays["past_covariates"].shape == (1, 2, 2000)
    assert arrays["known_covariates"].shape == (1, 2, 3500)
    assert set(np.unique(arrays["known_covariates"][:, -1])) <= {0.0, 1.0}
    assert not np.isnan(arrays["known_covariates"]).any()
    plain = make_inputs(by_name("short_256_96"))["arrays"]
    assert set(plain) == {"context"}
    assert not np.isnan(plain["context"]).any()


def test_parity_cases_are_deterministic_and_cover_covariates():
    first = list(cases())
    second = list(cases())
    assert [name for name, *_ in first] == [name for name, *_ in second]
    assert len(first) == 14
    assert len(list(cases(covariates=False))) == 9
    for (_, arrays, _, _), (_, again, _, _) in zip(first, second):
        for key in arrays:
            assert np.array_equal(arrays[key], again[key], equal_nan=True)
    by_case = {name: (arrays, kwargs, constructor) for name, arrays, kwargs, constructor in first}
    assert by_case["context_cap_mask"][2] == {"max_context": 32}
    assert "observation_mask" in by_case["context_cap_mask"][0]
    assert by_case["multi_block"][1] == {"prediction_length": 70, "symmetric_averaging": False, "clip_nonnegative": False}
    assert list(cases(long_horizon=1041))[7][2]["prediction_length"] == 1041
    assert "categorical_known_covariates" in by_case["categorical"][0]


def test_compare_applies_the_agreement_gate():
    expected = np.sort(np.random.default_rng(3).normal(size=(2, 1, 5, 4)).astype(np.float32), axis=-1)
    exact = compare(expected.copy(), expected)
    assert exact["passed"] and exact["max_abs"] == 0.0 and exact["quantiles_outside_tolerance"] == 0
    within = expected + 0.5 * tolerance(expected).astype(np.float32)
    assert compare(within, expected)["passed"]
    outside = expected.copy()
    outside[0, 0, 0, 0] -= 10 * tolerance(expected[0, 0, 0, 0])
    record = compare(outside, expected)
    assert not record["passed"] and record["quantiles_outside_tolerance"] == 1
    assert record["max_tolerance_fraction"] > 1
    unordered = expected.copy()
    unordered[..., 0], unordered[..., -1] = expected[..., -1], expected[..., 0]
    assert not compare(unordered, expected)["passed"]
    assert not compare(expected[..., :3], expected)["passed"]
    nan = expected.copy()
    nan[0, 0, 0, 0] = np.nan
    record = compare(nan, expected)
    assert not record["passed"] and not record["finite"]


def test_rewrite_common_import_requires_exactly_one_occurrence():
    code = f"import mlx.core as mx\n{COMMON_IMPORT}\n    render_path,\n)\n"
    rewritten = rewrite_common_import(code)
    assert LOCAL_IMPORT in rewritten and COMMON_IMPORT not in rewritten
    with pytest.raises(ValueError, match="helper import"):
        rewrite_common_import("import mlx.core as mx\n")
    with pytest.raises(ValueError, match="helper import"):
        rewrite_common_import(code + code)


def test_rotation_covers_every_position_and_adapters_resolve(tmp_path):
    seen = {impl: set() for impl in IMPLS}
    for offset in range(len(IMPLS)):
        for position, impl in enumerate(rotated(IMPLS, offset)):
            seen[impl].add(position)
    assert all(positions == set(range(len(IMPLS))) for positions in seen.values())
    assert rotated([], 3) == []
    assert REFERENCE_IMPL == "torch-2" and REFERENCE_IMPL not in PARITY_IMPLS
    for impl in IMPLS:
        adapter = make_adapter(impl, tmp_path)
        assert adapter.name == impl
        assert adapter.runtime == ("torch" if impl.startswith("torch") else "mlx")
    assert make_adapter("axon-fused-pruned", tmp_path).prune_final_block is True
    assert make_adapter("axon-fused", tmp_path).prune_final_block is False
    assert make_adapter("torch-8", tmp_path).threads == 8
    with pytest.raises(ValueError, match="unknown implementation"):
        make_adapter("onnx", tmp_path)


def test_load_core_class_imports_a_module_file_with_local_helpers(tmp_path):
    cores = tmp_path / "cores"
    cores.mkdir()
    (cores / "_axon_common.py").write_text("MARKER = 'local helpers'\n")
    (cores / "axon_core_fused.py").write_text(
        textwrap.dedent(
            """
            from _axon_common import MARKER

            class TimestralCore:
                marker = MARKER
            """
        )
    )
    core = load_core_class(cores / "axon_core_fused.py")
    assert core.marker == "local helpers"
    assert sys.path[0] == str(cores)
    with pytest.raises(FileNotFoundError, match="build step"):
        load_core_class(cores / "missing.py")
    other = tmp_path / "other"
    other.mkdir()
    (other / "_axon_common.py").write_text("MARKER = 'other helpers'\n")
    (other / "axon_core_fused.py").write_text((cores / "axon_core_fused.py").read_text())
    # A second core directory binds its own helper copy, even though the name repeats.
    assert load_core_class(other / "axon_core_fused.py").marker == "other helpers"
    (tmp_path / "bare").mkdir()
    (tmp_path / "bare" / "axon_core_fused.py").write_text("class TimestralCore: ...\n")
    with pytest.raises(FileNotFoundError, match="helper module missing"):
        load_core_class(tmp_path / "bare" / "axon_core_fused.py")


def _timing_record(impl, size, round_index, forecast, core=None, samples=3):
    workloads = []
    for name, value in forecast.items():
        entry = {
            "workload": name,
            "position": 0,
            "first_call_ms": value * 3,
            "first_call_sha256": "x",
            "forecast_ms": [value * factor for factor in (1.0, 1.1, 0.9)][:samples],
            "output_sha256": ["same"] * samples,
            "deterministic": True,
            "output_shape": [1, 1, 8, 5],
            "finite": True,
            "ordered": True,
            "memory": {},
            "max_rss_bytes": 1,
        }
        if core is not None:
            entry["core_ms"] = [core[name]] * samples
            entry["core_calls"] = 2
        workloads.append(entry)
    return {
        "kind": "timing", "impl": impl, "size": size, "round": round_index, "impl_index": 0,
        "checkpoint_dir": "ckpt", "samples": samples, "warmup": 2, "load_ms": 1.0, "load_info": {},
        "environment_before": {}, "environment_after": {},
        "torch_imported": impl.startswith("torch"), "mlx_imported": not impl.startswith("torch"),
        "workloads": workloads,
    }


def _parity_record(impl, size, passed_names, failing):
    cases_ = [
        {"name": name, "passed": True, "max_abs": 1e-6, "quantiles_outside_tolerance": 0} for name in passed_names
    ] + [
        {"name": name, "passed": False, "max_abs": value, "quantiles_outside_tolerance": 3} for name, value in failing.items()
    ]
    return {
        "kind": "parity", "impl": impl, "size": size, "reference_impl": "torch-2", "cases": cases_,
        "passed": len(passed_names), "total": len(cases_), "torch_imported": False, "mlx_imported": True,
    }


def test_summarize_aggregates_medians_ratios_and_parity(tmp_path):
    run_dir = tmp_path / "run"
    timing = run_dir / "timing"
    timing.mkdir(parents=True)
    base = {"short_256_96": 10.0, "batch32_2048_96": 100.0}
    core = {"short_256_96": 5.0, "batch32_2048_96": 80.0}
    for round_index, scale in enumerate((1.0, 1.2, 1.1)):
        record = _timing_record(BASELINE, "5M", round_index, {k: v * scale for k, v in base.items()}, core)
        (timing / f"{BASELINE}-5M-r{round_index}.json").write_text(json.dumps(record))
    for round_index, scale in enumerate((2.0, 2.0, 2.0)):
        values = {"short_256_96": 20.0 * scale / 2, "batch32_2048_96": 50.0}  # 2x slower, 2x faster
        (timing / f"vikiival-5M-r{round_index}.json").write_text(json.dumps(_timing_record("vikiival", "5M", round_index, values)))
    parity = run_dir / "parity"
    parity.mkdir()
    (parity / "axon-fused-5M.json").write_text(json.dumps(_parity_record(BASELINE, "5M", ["a", "b"], {"c": 0.01})))
    (parity / "vikiival-5M.json").write_text(json.dumps(_parity_record("vikiival", "5M", ["a", "b", "c"], {})))
    (parity / "reference-5M").mkdir()
    (parity / "reference-5M" / "manifest.json").write_text(json.dumps({"kind": "parity-reference"}))

    summary = summarize(run_dir)
    assert (run_dir / "summary.json").exists() and (run_dir / "comparison.csv").exists() and (run_dir / "samples.csv").exists()
    assert load_summary(run_dir)["counts"]["timing_processes"] == 6
    short = summary["latency"][BASELINE]["5M"]["short_256_96"]["forecast"]
    assert short["median"] == pytest.approx(11.0) and short["min"] == pytest.approx(10.0) and short["max"] == pytest.approx(12.0)
    assert short["rounds"] == 3
    ratios = summary["ratios"]["vikiival"]["5M"]
    assert ratios["per_workload"]["short_256_96"] == pytest.approx(20.0 / 11.0)
    assert ratios["per_workload"]["batch32_2048_96"] == pytest.approx(50.0 / 110.0)
    assert ratios["geomean"] == pytest.approx(geometric_mean([20.0 / 11.0, 50.0 / 110.0]))
    assert ratios["faster_than_baseline"] == ["batch32_2048_96"]
    assert ratios["min"] <= ratios["geomean"] <= ratios["max"]
    overhead = summary["host_overhead"][BASELINE]["5M"]
    assert overhead["per_workload"]["short_256_96"] == pytest.approx(11.0 / 5.0)
    assert summary["parity"][BASELINE]["5M"]["passed"] == 2
    assert summary["parity"][BASELINE]["5M"]["failing"][0]["name"] == "c"
    assert summary["isolation"] == {"mlx_workers_without_torch": True, "torch_workers_without_mlx": True}
    assert summary["counts"]["forecast_samples"] == 6 * 2 * 3
    assert summary["counts"]["core_samples"] == 3 * 2 * 3

    markdown = render_markdown(summary)
    assert "Latency ratio to Axon MLX (fused)" in markdown
    assert "vikiival MLX" in markdown and "| 5M |" in markdown
    assert "batch32_2048_96" in markdown and "`c`" in markdown
    csv_rows = (run_dir / "comparison.csv").read_text().splitlines()
    assert csv_rows[0].startswith("size,workload,axon-fused_ms")
    assert len(csv_rows) == 1 + 2
    sample_rows = (run_dir / "samples.csv").read_text().splitlines()
    assert len(sample_rows) == 1 + 6 * 2 * 3 + 3 * 2 * 3


def test_geometric_mean_handles_edge_cases():
    assert geometric_mean([2, 8]) == pytest.approx(4.0)
    assert geometric_mean([]) is None
    assert geometric_mean([1.0, 0.0]) is None


def test_cli_parses_subcommands_and_validates_choices():
    parser = cli.build_parser()
    args = parser.parse_args(["bench", "--run-id", "x", "--sizes", "5M,28M", "--impls", "axon-fused,vikiival", "--rounds", "1"])
    assert args.sizes == ["5M", "28M"] and args.impls == ["axon-fused", "vikiival"] and args.rounds == 1
    assert args.samples == 7 and args.warmup == 2 and not args.resume
    with pytest.raises(SystemExit):
        parser.parse_args(["bench", "--run-id", "x", "--sizes", "7B"])
    with pytest.raises(SystemExit):
        parser.parse_args(["parity", "--run-id", "x", "--impls", "torch-2"])
    worker = parser.parse_args(
        ["_worker", "time", "--impl", "axon-fused", "--size", "5M", "--round", "0", "--samples", "7",
         "--warmup", "2", "--output", "o.json", "--run-dir", "r", "--checkpoint-dir", "c"]
    )
    assert worker.worker == "time" and worker.impl == "axon-fused"
    with pytest.raises(SystemExit):
        parser.parse_args(["--help"])
    assert cli.run_directory("my-run") == cli.LOG_DIR / "my-run"
    assert cli.run_directory("/abs/run") .as_posix() == "/abs/run"
    assert cli.run_directory("log/other") .as_posix() == "log/other"


def test_init_records_provenance_and_refuses_existing_run(tmp_path):
    run_dir = tmp_path / "run"
    cli.cmd_init(cli.build_parser().parse_args(["init", "--run-id", str(run_dir)]))
    provenance = json.loads((run_dir / "provenance.json").read_text())
    assert provenance["checkpoint_revisions"] == cli.CHECKPOINT_REVISIONS
    assert len(provenance["workloads"]) == 9
    assert provenance["impls"] == list(IMPLS)
    assert "expected_hashes" in provenance["upstream"] and "expected_hashes" in provenance["vikiival"]
    assert provenance["git"]["head"]
    with pytest.raises(SystemExit, match="exists"):
        cli.cmd_init(cli.build_parser().parse_args(["init", "--run-id", str(run_dir)]))
    with pytest.raises(SystemExit, match="not fetched"):
        cli.checkpoint_directory(provenance, "5M")


def test_worker_env_sets_hygiene_variables():
    env = cli.worker_env(8)
    assert env["OMP_NUM_THREADS"] == "8" and env["MKL_NUM_THREADS"] == "8"
    assert env["HF_HUB_OFFLINE"] == "1" and env["CUDA_VISIBLE_DEVICES"] == ""
    assert cli.WORKER_THREADS.get("torch-8", cli.HOST_THREADS) == 8
    assert cli.WORKER_THREADS.get("torch-2", cli.HOST_THREADS) == 2
    assert cli.WORKER_THREADS.get("axon-fused", cli.HOST_THREADS) == 2


def _fake_run_worker(calls):
    from pathlib import Path

    def run_worker(worker_args, threads, log_path):
        args = list(map(str, worker_args))
        calls.append((args, threads, log_path))
        if args[0] == "reference":
            out = Path(args[args.index("--output-dir") + 1])
            assert not out.exists(), "the reference worker creates its own directory"
            out.mkdir(parents=True)
            (out / "manifest.json").write_text("{}")
        else:
            out = Path(args[args.index("--output") + 1])
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text("{}")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("log")
        return {"command": args, "returncode": 0, "seconds": 0.0}

    return run_worker


def _fake_run(tmp_path):
    run_dir = tmp_path / "run"
    (run_dir / "cores").mkdir(parents=True)
    (run_dir / "cores" / "build-info.json").write_text("{}")
    checkpoints = {}
    for size in ("5M", "28M"):
        ckpt = tmp_path / f"ckpt-{size}"
        ckpt.mkdir()
        (ckpt / "config.json").write_text("{}")
        (ckpt / "model.safetensors").write_bytes(b"")
        checkpoints[size] = {"path": str(ckpt)}
    cli.write_json(run_dir / "provenance.json", {"checkpoints": checkpoints})
    return run_dir


def test_parity_orchestration_keeps_the_reference_directory_for_the_worker(tmp_path, monkeypatch):
    run_dir = _fake_run(tmp_path)
    calls = []
    monkeypatch.setattr(cli, "run_worker", _fake_run_worker(calls))
    argv = ["parity", "--run-id", str(run_dir), "--sizes", "5M", "--impls", "axon-fused,torch-8", "--no-idle-wait"]
    cli.cmd_parity(cli.build_parser().parse_args(argv))
    assert (run_dir / "parity" / "reference-5M" / "manifest.json").exists()
    assert (run_dir / "parity" / "axon-fused-5M.json").exists()
    assert (run_dir / "parity" / "torch-8-5M.json").exists()
    assert [(args[0], threads) for args, threads, _ in calls] == [("reference", 2), ("compare", 2), ("compare", 8)]
    assert calls[0][2].parent == run_dir / "parity"
    stages = [entry["stage"] for entry in cli.read_json(run_dir / "manifest.json")]
    assert stages == ["parity-reference", "parity", "parity"]
    cli.cmd_parity(cli.build_parser().parse_args(argv + ["--resume"]))
    assert len(calls) == 3
    with pytest.raises(SystemExit, match="exists"):
        cli.cmd_parity(cli.build_parser().parse_args(argv))


def test_bench_orchestration_rotates_implementations_and_sets_threads(tmp_path, monkeypatch):
    run_dir = _fake_run(tmp_path)
    calls = []
    monkeypatch.setattr(cli, "run_worker", _fake_run_worker(calls))
    argv = ["bench", "--run-id", str(run_dir), "--sizes", "5M,28M", "--impls", "axon-fused,torch-8",
            "--rounds", "2", "--samples", "3", "--warmup", "1", "--no-idle-wait"]
    cli.cmd_bench(cli.build_parser().parse_args(argv))
    order = [(args[args.index("--impl") + 1], args[args.index("--size") + 1], args[args.index("--round") + 1]) for args, _, _ in calls]
    assert order == [
        ("axon-fused", "5M", "0"), ("torch-8", "5M", "0"), ("torch-8", "28M", "0"), ("axon-fused", "28M", "0"),
        ("torch-8", "5M", "1"), ("axon-fused", "5M", "1"), ("axon-fused", "28M", "1"), ("torch-8", "28M", "1"),
    ]
    assert {threads for args, threads, _ in calls if args[args.index("--impl") + 1] == "torch-8"} == {8}
    assert {threads for args, threads, _ in calls if args[args.index("--impl") + 1] == "axon-fused"} == {2}
    assert sorted(path.name for path in (run_dir / "timing").glob("*.json")) == sorted(
        f"{impl}-{size}-r{round_}.json" for impl in ("axon-fused", "torch-8") for size in ("5M", "28M") for round_ in (0, 1)
    )
    assert all(args[args.index("--samples") + 1] == "3" for args, _, _ in calls)
    cli.cmd_bench(cli.build_parser().parse_args(argv + ["--resume"]))
    assert len(calls) == 8
    monkeypatch.setattr(cli, "run_worker", lambda *a, **k: {"command": [], "returncode": 1, "seconds": 0.0})
    with pytest.raises(SystemExit, match="failed"):
        cli.cmd_bench(cli.build_parser().parse_args(["bench", "--run-id", str(run_dir), "--sizes", "5M",
                                                      "--impls", "vikiival", "--rounds", "1", "--no-idle-wait"]))


def test_warmup_and_samples_must_be_positive(tmp_path):
    from scripts.timestral_bench import worker

    parser = cli.build_parser()
    for flag in ("--warmup", "--samples"):
        with pytest.raises(SystemExit):
            parser.parse_args(["bench", "--run-id", "x", flag, "0"])
    assert parser.parse_args(["bench", "--run-id", "x", "--warmup", "1"]).warmup == 1
    with pytest.raises(ValueError, match="positive"):
        worker.time_worker(
            impl="axon-fused", size="5M", round_index=0, run_dir=tmp_path, checkpoint_dir=tmp_path,
            output=tmp_path / "o.json", samples=1, warmup=0,
        )
    assert not (tmp_path / "o.json").exists()
