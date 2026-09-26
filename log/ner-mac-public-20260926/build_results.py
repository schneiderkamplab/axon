#!/usr/bin/env python3
"""Audit two extracted Mac result archives and publish portable measurements.

Requires NumPy. Inputs are read only; no archive-supplied code is executed.
Run without Python's -O flag, since validation uses assertions.
"""

import argparse
import copy
import csv
import gzip
import hashlib
import json
import math
import random
import re
from pathlib import Path

import numpy as np

CHECKPOINT = "944c16a050788efd4ab111c165ea40edbc8759febdf4f0800f9c4fb1026f2120"
DATASET = "205f3e9c9f3577ea2561d43f2f62dc249ab92d5b"
EXPORT_COMMIT = "b9ffe322f7f1ec34c0b45176d9b35c6567f33f30"
RUNTIME_COMMIT = "9f799608b20a2d469795af58ebc3f1cd4f54ede6"
CONFIGS = [(f"{b}-cpu-fp32", b, "cpu", "fp32", False) for b in ("hf", "axon")]
CONFIGS += [(f"{b}-mps-{d}", b, "mps", d, False) for b in ("hf", "axon") for d in ("fp32", "fp16")]
CONFIGS += [
    (f"mlx-metal-{d}" + ("-compiled" if c else ""), "mlx", "metal", d, c)
    for d in ("fp32", "fp16")
    for c in (False, True)
]


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(actual, expected):
    assert math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12), (actual, expected)


def verify_manifest(root):
    hashes = {}
    for line in (root / "SHA256SUMS").read_text().splitlines():
        sha, name = line.split(None, 1)
        path = root / name
        assert path.resolve().is_relative_to(root.resolve())
        assert name not in hashes and digest(path) == sha, name
        hashes[name] = sha
    assert {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()} == set(hashes) | {
        "SHA256SUMS"
    }
    return hashes


def sanitized_result(data):
    result = copy.deepcopy(data)
    for key in ("output", "run_dir"):
        del result["arguments"][key]
    # Fail closed if unexpected machine-local paths enter portable results.
    assert not re.search(r"/Users/|/home/|[A-Za-z]:\\", json.dumps(result))
    return result


def conditions(root):
    result = {}
    for when in ("before", "after"):
        data = read(root / f"conditions-{when}-performance.json")
        assert all(
            data[k]["returncode"] == 0
            for k in ("power", "power_settings", "thermal", "process_snapshot_status")
        )
        settings = data["power_settings"]["stdout"]
        assert re.findall(r"lowpowermode\s+(\d+)", settings) == ["0", "0"]
        assert "AC Power" in data["power"]["stdout"]
        result[when] = {
            "captured_at": data["captured_at"],
            "ac_power": True,
            "low_power_mode": False,
            "battery_percent": int(re.search(r"(\d+)%;", data["power"]["stdout"])[1]),
            "thermal_status": data["thermal"]["stdout"].splitlines(),
            "top_cpu_processes": [
                {k: p[k] for k in ("command", "cpu_percent")} for p in data["top_cpu_processes"]
            ],
        }
    assert (
        read(root / "conditions-before-performance.json")["power_settings"]
        == read(root / "conditions-after-performance.json")["power_settings"]
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--quiet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert __debug__, "Run without -O"
    roots = {"previous": args.previous, "quiet": args.quiet}
    manifests = {key: verify_manifest(root) for key, root in roots.items()}
    continuation = read(args.previous / "continuation.json")
    execution = read(args.quiet / "execution.json")
    original_execution = read(args.previous / "execution.json")
    assert execution["packages"] == original_execution["packages"]
    assert execution["python"] == original_execution["python"]
    for record in (continuation, execution):
        assert record["artifact_source_commit"] == EXPORT_COMMIT
        assert record["runner_runtime_commit"] == RUNTIME_COMMIT
    assert execution["exit_code"] == 0 and execution["quality_repeated"] is False
    assert execution["previous_run_sha256"] == {
        **manifests["previous"],
        "SHA256SUMS": digest(args.previous / "SHA256SUMS"),
    }
    for name, sha in continuation["initial_mlx_quality_checksums"].items():
        assert digest(args.previous / name) == sha

    quality, runs, aggregates = {}, {}, {}
    versions, sample_hashes, count = set(), set(), 0
    order = []
    for rep in range(3):
        shuffled = list(CONFIGS)
        random.Random(20260926 + rep).shuffle(shuffled)
        order += [f"{c[0]}-performance-r{rep}.json" for c in shuffled]
    plans = [
        json.loads(line)
        for line in (args.previous / "performance-plan.jsonl").read_text().splitlines()
    ]
    assert [Path(p["command"][p["command"].index("--output") + 1]).name for p in plans] == order

    for condition, root in roots.items():
        expected_files = set(order)
        if condition == "previous":
            expected_files |= {f"{c[0]}-quality.json" for c in CONFIGS}
        assert {p.name for p in (root / "results").glob("*.json")} == expected_files
        events = []
        for line in (root / "performance-runner.log").read_text().splitlines():
            for prefix in ("Starting: ", "Completed: "):
                if line.startswith(prefix):
                    events.append(
                        (
                            prefix,
                            line.removeprefix(prefix).replace(" performance-r", "-performance-r")
                            + ".json",
                        )
                    )
        assert events == [
            (prefix, name) for name in order for prefix in ("Starting: ", "Completed: ")
        ]
        runs[condition], aggregates[condition] = {}, {}
        for name, backend, device, dtype, compiled in CONFIGS:
            stages = (["quality"] if condition == "previous" else []) + [
                f"performance-r{r}" for r in range(3)
            ]
            runs[condition][name] = []
            for stage in stages:
                data = read(root / "results" / f"{name}-{stage}.json")
                a = data["arguments"]
                assert (a["backend"], a["device"], a["dtype"], a["compile"]) == (
                    backend,
                    device,
                    dtype,
                    compiled,
                )
                assert a["threads"] == 4 and a["split"] == "test" and not a["write_reference"]
                assert (
                    data["checkpoint_sha256"] == CHECKPOINT and data["dataset_revision"] == DATASET
                )
                assert data["platform"] == "macOS-27.0-arm64-arm-64bit-Mach-O"
                assert device != "mps" or data["mps_cpu_fallback"] == "0"
                versions.add(json.dumps(data["versions"], sort_keys=True))
                if stage == "quality":
                    assert a["stage"] == "quality"
                    q = data["quality"]
                    assert (q["sentences"], q["words"]) == (37648, 921118)
                    for row in [q, *q["per_label"].values()]:
                        tp, predicted, gold = (
                            row[k] for k in ("true_positive_spans", "predicted_spans", "gold_spans")
                        )
                        close(row["precision"], tp / predicted if predicted else 0)
                        close(row["recall"], tp / gold if gold else 0)
                        close(row["f1"], 2 * tp / (predicted + gold) if predicted + gold else 0)
                    for key in ("true_positive_spans", "predicted_spans", "gold_spans"):
                        assert sum(row[key] for row in q["per_label"].values()) == q[key]
                    close(q["macro_f1"], np.mean([row["f1"] for row in q["per_label"].values()]))
                    close(q["f1_change"], q["f1"] - 0.6009153728798349)
                    close(q["macro_f1_change"], q["macro_f1"] - 0.511797373351898)
                    passed = (
                        q["fp32_allclose"]
                        if dtype == "fp32"
                        else q["f1_change"] >= -0.005 and q["macro_f1_change"] >= -0.005
                    )
                    assert passed and q["quality_gate_passed"]
                    quality[name] = sanitized_result(data)
                    continue
                assert (a["stage"], a["warmup"], a["repeat"], a["rounds"]) == (
                    "performance",
                    10,
                    50,
                    1,
                )
                if condition == "quiet":
                    old_args = read(args.previous / "results" / f"{name}-{stage}.json")["arguments"]
                    assert {k: v for k, v in a.items() if k != "output"} == {
                        k: v for k, v in old_args.items() if k != "output"
                    }
                perf = data["performance"]
                assert [(r["batch"], r["length"]) for r in perf["shapes"]] == [
                    (b, s) for b in (1, 8, 32) for s in (32, 128, 512)
                ]
                assert [r["batch"] for r in perf["end_to_end"]] == [1, 8, 32]
                for row in perf["shapes"] + perf["end_to_end"]:
                    values = np.asarray(row["samples_ms"])
                    assert (
                        values.shape == (50,) and np.isfinite(values).all() and (values > 0).all()
                    )
                    assert math.isfinite(row["first_call_ms"]) and row["first_call_ms"] > 0
                    close(row["p50_ms"], np.median(values))
                    close(row["p95_ms"], np.percentile(values, 95))
                    close(row["mean_ms"], np.mean(values))
                    for key, numerator in (
                        ("sequences_per_second", row["batch"]),
                        ("sentences_per_second", row["batch"]),
                        ("useful_tokens_per_second", row["batch"] * row.get("length", 1)),
                    ):
                        if key in row:
                            close(row[key], 1000 * numerator / np.mean(values))
                    sha = hashlib.sha256(values.tobytes()).hexdigest()
                    assert sha not in sample_hashes
                    sample_hashes.add(sha)
                    count += len(values)
                runs[condition][name].append(sanitized_result(data))
            aggregate = {}
            for group in ("shapes", "end_to_end"):
                aggregate[group] = []
                for rows in zip(
                    *(run["performance"][group] for run in runs[condition][name]), strict=True
                ):
                    row = {k: rows[0][k] for k in ("batch", "length") if k in rows[0]}
                    row.update(
                        {
                            k: float(np.median([r[k] for r in rows]))
                            for k in ("p50_ms", "p95_ms", "mean_ms", "first_call_ms")
                        }
                    )
                    row["process_p50_ms"] = [r["p50_ms"] for r in rows]
                    row["median_throughput_per_second"] = float(
                        np.median([1000 * r["batch"] / r["mean_ms"] for r in rows])
                    )
                    aggregate[group].append(row)
            aggregates[condition][name] = aggregate
        reported = read(root / "benchmark-validation.json")
        for name, aggregate in aggregates[condition].items():
            assert reported["quality"][name] == {
                k: v for k, v in quality[name]["quality"].items() if k != "per_label"
            }
            for shape in ((1, 128), (32, 128), (32, 512)):
                row = next(r for r in aggregate["shapes"] if (r["batch"], r["length"]) == shape)
                expected = reported["performance"][name][f"B{shape[0]}_S{shape[1]}"]
                for key in ("p50_ms", "p95_ms", "mean_ms"):
                    close(row[key], expected[key])
                assert row["process_p50_ms"] == expected["process_p50_ms"]
                close(row["median_throughput_per_second"], expected["sequences_per_second"])
            close(
                aggregate["end_to_end"][0]["p50_ms"], reported["performance"][name]["e2e_B1_p50_ms"]
            )
    assert len(versions) == 1 and count == 36000 and len(sample_hashes) == 720

    comparisons = []
    for name, *_ in CONFIGS:
        for group in ("shapes", "end_to_end"):
            for previous, quiet in zip(
                aggregates["previous"][name][group], aggregates["quiet"][name][group], strict=True
            ):
                row = dict(
                    configuration=name,
                    scope=group,
                    batch=previous["batch"],
                    length=previous.get("length"),
                )
                for label, data in (("previous", previous), ("quiet", quiet)):
                    values = data["process_p50_ms"]
                    row.update(
                        {
                            f"{label}_process_p50_ms": values,
                            f"{label}_median_p50_ms": data["p50_ms"],
                            f"{label}_min_p50_ms": min(values),
                            f"{label}_max_p50_ms": max(values),
                        }
                    )
                row["quiet_latency_change_percent"] = (
                    quiet["p50_ms"] / previous["p50_ms"] - 1
                ) * 100
                comparisons.append(row)
    supplied = read(args.quiet / "run-comparison.json")
    assert (
        supplied["same_execution_order"]
        and supplied["same_per_configuration_arguments_except_output"]
    )
    for actual, expected in zip(comparisons, supplied["cases"], strict=True):
        for key in actual:
            if isinstance(actual[key], float):
                close(actual[key], expected[key])
            else:
                assert actual[key] == expected[key], (key, actual[key], expected[key])
    with (args.quiet / "all-case-comparison.csv").open(newline="") as stream:
        for actual, expected in zip(csv.DictReader(stream), comparisons, strict=True):
            for key, value in expected.items():
                if isinstance(value, list):
                    assert json.loads(actual[key]) == value
                elif isinstance(value, (int, float)):
                    close(float(actual[key]), value)
                else:
                    assert actual[key] == ("" if value is None else value)
    preliminary = {
        digest(p) for p in (args.previous / "preliminary/results").glob("*performance*.json")
    }
    assert preliminary.isdisjoint(
        {digest(root / "results" / filename) for root in roots.values() for filename in order}
    )
    assert (
        read(args.previous / "conditions-before-performance.json")["power_settings"]
        == read(args.quiet / "conditions-before-performance.json")["power_settings"]
    )
    audit = dict(
        checksummed_files={k: len(v) for k, v in manifests.items()},
        quality_passed=10,
        performance_processes=60,
        timing_cases=720,
        raw_timing_samples=count,
        unique_sample_arrays=len(sample_hashes),
        previous_run_unchanged=True,
        preliminary_excluded=True,
        same_arguments_except_output=True,
        same_execution_order=True,
        supplied_statistics_and_comparison_match=True,
    )
    public = dict(
        schema_version=1,
        provenance=dict(
            checkpoint_sha256=CHECKPOINT,
            dataset_revision=DATASET,
            export_source_commit=EXPORT_COMMIT,
            runner_runtime_commit=RUNTIME_COMMIT,
            mlx_export_sha256="3eb48f7ac4602678fe113fc540ee4cbf6b84818521927e69f9f2fcb7d9125a11",
            input_verification_reported_on_mac=read(args.previous / "input-verification.json"),
        ),
        audit=audit,
        quality=quality,
        performance=runs,
        aggregates=aggregates,
        environment_reported_on_mac=dict(
            hardware="Apple M1 Pro, 10 CPU cores / 16 GPU cores, 32 GB unified memory (supplied report)",
            python=execution["python"],
            packages=execution["packages"],
            platform="macOS-27.0-arm64-arm-64bit-Mach-O",
        ),
        source_result_sha256={
            key: {
                name: sha
                for name, sha in hashes.items()
                if name.startswith("results/") and name.endswith(".json")
            }
            for key, hashes in manifests.items()
        },
        execution_order=order,
        versions=json.loads(next(iter(versions))),
        conditions={key: conditions(root) for key, root in roots.items()},
        sanitization="Removed result output/run_dir paths and condition snapshot process IDs, parent IDs, elapsed times and battery identifiers. Measurement values unchanged.",
        aggregation="Each condition separately: median of three process statistics; ranges are min/max process p50s, not confidence intervals. No sample pooling.",
        quality_limit="Offline audit recomputes scoring from counts and checks recorded gates; it does not rerun inference or independently recheck logits.",
    )
    args.output.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(public, indent=2, allow_nan=False) + "\n").encode()
    (args.output / "public-results.json.gz").write_bytes(gzip.compress(encoded, mtime=0))
    (args.output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    with (args.output / "all-case-comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparisons[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(comparisons)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
