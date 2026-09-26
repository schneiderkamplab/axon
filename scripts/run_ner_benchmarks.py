#!/usr/bin/env python3
"""Run isolated NER benchmark processes sequentially on Linux or a Mac."""

import argparse
import json
import os
import random
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--suite",
        choices=["cpu", "cuda", "linux", "mac", "mac-compare", "mps", "mlx", "coreml"],
        required=True,
    )
    parser.add_argument("--stage", choices=["all", "quality", "performance"], default="all")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--results-dir", type=Path, help="Separate output directory; defaults to RUN_DIR/results"
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=1,
        help="Fresh shuffled process repetitions (performance only)",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Retain failures and continue other configurations; exit nonzero if any fail",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print commands without executing or writing files"
    )
    args = parser.parse_args()
    if args.repetitions < 1 or (args.repetitions != 1 and args.stage != "performance"):
        parser.error(
            "--repetitions must be positive and multiple repetitions require --stage performance"
        )
    cpu = [
        ("hf-cpu-fp32", "hf", "cpu", "fp32", []),
        ("axon-cpu-fp32", "axon", "cpu", "fp32", []),
        ("ort-cpu-fp32", "ort", "cpu", "fp32", []),
        ("ort-cpu-int8", "ort", "cpu", "int8", []),
    ]
    cuda = [
        (f"{backend}-cuda-{dtype}", backend, "cuda:0", dtype, [])
        for backend in ("hf", "axon", "ort")
        for dtype in ("fp32", "fp16")
    ]
    mps = [
        (f"{backend}-mps-{dtype}", backend, "mps", dtype, [])
        for backend in ("hf", "axon")
        for dtype in ("fp32", "fp16")
    ]
    mlx = [
        (
            f"mlx-metal-{dtype}" + ("-compiled" if compiled else ""),
            "mlx",
            "metal",
            dtype,
            ["--compile"] if compiled else [],
        )
        for dtype in ("fp32", "fp16")
        for compiled in (False, True)
    ]
    coreml = [
        (
            f"ort-coreml-{units}-{shape}",
            "ort",
            "coreml",
            "fp32",
            ["--compute-units", units] + (["--coreml-static"] if shape == "static" else []),
        )
        for units in ("ALL", "CPUAndGPU", "CPUAndNeuralEngine")
        for shape in ("static", "dynamic")
    ]
    suites = {
        "cpu": cpu,
        "cuda": cuda,
        "linux": cpu + cuda,
        "mac": cpu + mps + mlx + coreml,
        "mac-compare": [row for row in cpu if row[1] in {"hf", "axon"}] + mps + mlx,
        "mps": mps,
        "mlx": mlx,
        "coreml": coreml,
    }
    environment = dict(
        os.environ,
        OMP_NUM_THREADS="4",
        MKL_NUM_THREADS="4",
        TOKENIZERS_PARALLELISM="false",
        PYTORCH_ENABLE_MPS_FALLBACK="0",
    )
    jobs = []
    for repetition in range(args.repetitions):
        configs = list(suites[args.suite])
        if args.repetitions > 1:
            random.Random(20260926 + repetition).shuffle(configs)
        jobs.extend((repetition, row) for row in configs)
    results_dir = args.results_dir or args.run_dir / "results"
    failures = []
    for repetition, (name, backend, device, dtype, extra) in jobs:
        suffix = f"-r{repetition}" if args.repetitions > 1 else ""
        output = results_dir / f"{name}-{args.stage}{suffix}.json"
        if args.resume and not output.exists() and output.with_suffix(".log").exists():
            if args.keep_going:
                failures.append(str(output.with_suffix(".log")))
                print(f"Retaining failed/interrupted log: {output.with_suffix('.log')}", flush=True)
                continue
            raise RuntimeError(
                f"Existing log has no completed result: {output}; use a fresh results directory"
            )
        if args.resume and output.exists():
            result = json.loads(output.read_text())
            required = [args.stage] if args.stage != "all" else ["quality", "performance"]
            if all(key in result for key in required) and result.get("quality", {}).get(
                "quality_gate_passed", True
            ):
                print(f"Already complete: {name} {args.stage}", flush=True)
                continue
            if args.keep_going:
                failures.append(str(output))
                print(f"Retaining failed/incomplete output: {output}", flush=True)
                continue
            raise RuntimeError(
                f"Failed/incomplete output exists: {output}; retain it and use a fresh run path"
            )
        command = [
            sys.executable,
            str(Path(__file__).with_name("bench_ner.py")),
            "--run-dir",
            str(args.run_dir),
            "--output",
            str(output),
            "--backend",
            backend,
            "--device",
            device,
            "--dtype",
            dtype,
            "--stage",
            args.stage,
            *extra,
        ]
        if args.repetitions > 1:
            command.extend(["--rounds", "1"])
        if (
            backend == "hf"
            and device == "cpu"
            and args.stage != "performance"
            and not (args.run_dir / "reference-test.npy").exists()
        ):
            command.append("--write-reference")
        if args.dry_run:
            print(json.dumps({"repetition": repetition, "command": command}), flush=True)
            continue
        if output.exists() or output.with_suffix(".log").exists():
            raise FileExistsError(
                f"Preserve existing artifacts; use --resume or a fresh results directory: {output}"
            )
        output.parent.mkdir(parents=True, exist_ok=True)
        print(f"Starting: {name} {args.stage}{suffix}", flush=True)
        with output.with_suffix(".log").open("w") as stream:
            process = subprocess.run(
                command,
                env=environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
            )
        if process.returncode:
            failures.append(str(output.with_suffix(".log")))
            print(f"FAILED: {name}; see {output.with_suffix('.log')}", flush=True)
            if not args.keep_going:
                raise SystemExit(process.returncode)
        else:
            print(f"Completed: {name} {args.stage}{suffix}", flush=True)
    if failures:
        print(json.dumps({"failures": failures}), flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
