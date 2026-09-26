#!/usr/bin/env python3
"""Run isolated NER benchmark processes sequentially on Linux or a Mac."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--suite", choices=["cpu", "cuda", "linux", "mac", "mlx", "coreml"], required=True
    )
    parser.add_argument("--stage", choices=["all", "quality", "performance"], default="all")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
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
    mlx = [
        ("mlx-metal-fp32", "mlx", "metal", "fp32", []),
        ("mlx-metal-fp16-compiled", "mlx", "metal", "fp16", ["--compile"]),
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
        "mac": cpu + mlx + coreml,
        "mlx": mlx,
        "coreml": coreml,
    }
    environment = dict(
        os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", TOKENIZERS_PARALLELISM="false"
    )
    for name, backend, device, dtype, extra in suites[args.suite]:
        output = args.run_dir / "results" / f"{name}-{args.stage}.json"
        if args.resume and output.exists():
            result = json.loads(output.read_text())
            required = [args.stage] if args.stage != "all" else ["quality", "performance"]
            if all(key in result for key in required):
                print(f"Already complete: {name} {args.stage}", flush=True)
                continue
            raise RuntimeError(
                f"Incomplete output exists: {output}; retain it and use a fresh run path"
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
        if (
            backend == "hf"
            and device == "cpu"
            and args.stage != "performance"
            and not (args.run_dir / "reference-test.npy").exists()
        ):
            command.append("--write-reference")
        output.parent.mkdir(parents=True, exist_ok=True)
        print(f"Starting: {name} {args.stage}", flush=True)
        with output.with_suffix(".log").open("w") as stream:
            subprocess.run(
                command, env=environment, stdout=stream, stderr=subprocess.STDOUT, check=True
            )
        print(f"Completed: {name} {args.stage}", flush=True)


if __name__ == "__main__":
    main()
