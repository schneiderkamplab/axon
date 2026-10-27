"""Benchmark Timestral inference: Axon-generated MLX vs vikiival MLX vs original PyTorch CPU.

Subcommands run in order: init, build, fetch, bench, parity, summarize, report. Every
artifact lands under ``log/<run-id>/``; existing outputs are never overwritten.
"""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

from .build import ROOT, _git_state
from .worker import IMPLS, PARITY_IMPLS, REFERENCE_IMPL, rotated
from .workloads import manifest as workload_manifest

LOG_DIR = ROOT / "log"
SIZES = ("5M", "28M", "123M", "209M")
CHECKPOINT_REVISIONS = {
    "5M": "bdd20cfc15a3fea0e4cb5ccef37f8d3ca7aa335f",
    "28M": "49b26409ac87b5935539a5d8ffe7f397b720db47",
    "123M": "5a147287addda85f6acd8da3fb8858fe9af7e8f7",
    "209M": "9874d252ab0aae16f7f7c999f98f98d6ad7d8365",
}
UPSTREAM = {
    "package": "timestral",
    "version": "1.0.0",
    "repository": "https://github.com/mistralai/timestral",
    "revision": "a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff",
    "hashes": {
        "__init__.py": "46777bcb15b903f7721a3d6d91a6cb93463fc454816da238799eb8d5360e5ab5",
        "config.py": "d46e37020875c536e7a75e1b6ace3f67419ba2f2c9d9e8465f4b07f837e92ed6",
        "covariate_encoding.py": "1c7f5c320fa4b7622d3705f765f53ab9676b829cb1d4fd68f839ec593b8404c0",
        "forecaster.py": "12a712f9be777d1e160513b7f8c81d545b991fcfe6cd4a46004a03533b7cb8fd",
        "layers.py": "480ba93e7e3681dea185e5fa08a96dcce327cf938931c715ae54e6b02934527d",
        "loading.py": "179257fa74e798c207887e2fed162e9a2fef9c37cadadb9437fabfeae68b3d6f",
        "masks.py": "1792898d809b99034bb70e2cc1ab74f897ce8478886f83140e705fc67e6762b3",
        "model.py": "988121c6a089e843c4941732cbc5c64b10cb0eccd9310e228abe38ecf2dfb668",
        "scaler.py": "838da836f70b1640bcaed7701a70a1a358e30442693e63522a93f878e157a8ca",
    },
}
VIKIIVAL = {
    "package": "timestral_mlx",
    "distribution": "timestral-mlx",
    "version": "0.1.0",
    "repository": "https://github.com/vikiival/timestral-mlx",
    "revision": "18a7d8692df850a4dc48286aa7f2c70a0808351a",
    "hashes": {
        "__init__.py": "4c20105b6a743c194fe3b57d1851387d49c1582abdaa5793506208dc5f8f2023",
        "config.py": "d46e37020875c536e7a75e1b6ace3f67419ba2f2c9d9e8465f4b07f837e92ed6",
        "covariate_encoding.py": "1c7f5c320fa4b7622d3705f765f53ab9676b829cb1d4fd68f839ec593b8404c0",
        "forecaster.py": "c4d3102c77902fe6970d47d2cd0875a2e9ba1b1e04d9fbc68e82037f22aa8112",
        "layers.py": "5442efdc2984598ffcd6b77083f20d667a18be0c2622a658eb0ba905ea13ac5f",
        "loading.py": "dca53a70df1ac1b78b984edb251a2b2fb50efc40c77d0b536fba1fe0b58a5875",
        "masks.py": "6a80e9cb7a7b4e13bef947ded000597acc45d41b82c2b60ceabaf04723e5516a",
        "model.py": "18528706dceefd6606b1facf4bc7bd4793ee3f5c1b4e4518ba47acf938fcbd47",
        "scaler.py": "4374d1f3d0f4a1888cb0f4a505fa8945a2e641b77bc61adb7c20ab1b15b669fe",
    },
}
HOST_THREADS = 2
WORKER_THREADS = {"torch-8": 8}
CHECKPOINT_FILES = ["config.json", "model.safetensors"]


def run_directory(run_id):
    path = Path(run_id)
    return path if path.is_absolute() or len(path.parts) > 1 else LOG_DIR / run_id


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _command_output(*command):
    try:
        return subprocess.run(command, capture_output=True, text=True, check=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _distribution_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def installed_package_hashes(package):
    spec = importlib.util.find_spec(package)
    if spec is None or spec.origin is None:
        return None
    directory = Path(spec.origin).parent
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(directory.glob("*.py"))}


def pinned_package_record(pin):
    installed = installed_package_hashes(pin["package"])
    return {
        "repository": pin["repository"],
        "revision": pin["revision"],
        "pinned_version": pin["version"],
        "installed_version": _distribution_version(pin.get("distribution", pin["package"])),
        "expected_hashes": pin["hashes"],
        "installed_hashes": installed,
        "matches_pin": installed == pin["hashes"],
    }


def provenance_snapshot():
    memory = _command_output("sysctl", "-n", "hw.memsize")
    return {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": {
            "cpu": _command_output("sysctl", "-n", "machdep.cpu.brand_string"),
            "memory_bytes": int(memory) if memory and memory.isdigit() else None,
            "cpu_count": os.cpu_count(),
            "macos": _command_output("sw_vers", "-productVersion"),
        },
        "git": _git_state(),
        "packages": {
            name: _distribution_version(name)
            for name in ("numpy", "mlx", "torch", "transformers", "safetensors", "huggingface_hub", "timestral", "timestral-mlx")
        },
        "distributions": sorted(
            f"{distribution.metadata['Name']}=={distribution.version}"
            for distribution in importlib.metadata.distributions()
            if distribution.metadata["Name"]
        ),
        "upstream": pinned_package_record(UPSTREAM),
        "vikiival": pinned_package_record(VIKIIVAL),
        "checkpoint_revisions": CHECKPOINT_REVISIONS,
        "impls": list(IMPLS),
        "sizes": list(SIZES),
        "workloads": workload_manifest(),
    }


def cmd_init(args):
    run_dir = run_directory(args.run_id)
    if run_dir.exists():
        raise SystemExit(f"{run_dir} exists; choose a fresh run id")
    run_dir.mkdir(parents=True)
    provenance = provenance_snapshot()
    write_json(run_dir / "provenance.json", provenance)
    for name in ("upstream", "vikiival"):
        print(f"{name}: installed {provenance[name]['installed_version']}, matches pin: {provenance[name]['matches_pin']}")
    print(run_dir)


def cmd_build(args):
    from .build import write_cores

    print(write_cores(run_directory(args.run_id)))


def cmd_fetch(args):
    from huggingface_hub import snapshot_download

    run_dir = run_directory(args.run_id)
    provenance = read_json(run_dir / "provenance.json")
    checkpoints = provenance.setdefault("checkpoints", {})
    for size in args.sizes:
        revision = CHECKPOINT_REVISIONS[size]
        repo_id = f"mistralai/Timestral-{size}"
        path = Path(snapshot_download(repo_id, revision=revision, allow_patterns=CHECKPOINT_FILES))
        files = {}
        for name in CHECKPOINT_FILES:
            data = (path / name).read_bytes()
            files[name] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
        checkpoints[size] = {"repo_id": repo_id, "revision": revision, "path": str(path), "files": files}
        print(f"{size}: {path} ({files['model.safetensors']['bytes']} bytes)")
    write_json(run_dir / "provenance.json", provenance)


def checkpoint_directory(provenance, size):
    entry = provenance.get("checkpoints", {}).get(size)
    if entry is None:
        raise SystemExit(f"checkpoint {size} not fetched; run `fetch --sizes {size}` first")
    path = Path(entry["path"])
    if not all((path / name).is_file() for name in CHECKPOINT_FILES):
        raise SystemExit(f"checkpoint files missing under {path}; rerun fetch")
    return path


def worker_env(threads):
    env = os.environ.copy()
    env.update(
        HF_HUB_OFFLINE="1",
        PYTHONDONTWRITEBYTECODE="1",
        CUDA_VISIBLE_DEVICES="",
        TOKENIZERS_PARALLELISM="false",
        OMP_NUM_THREADS=str(threads),
        MKL_NUM_THREADS=str(threads),
        VECLIB_MAXIMUM_THREADS=str(threads),
    )
    return env


def cpu_idle_fraction():
    """Second ``top`` sample's idle fraction, or None when unavailable."""
    try:
        text = subprocess.run(
            ["top", "-l", "2", "-n", "0", "-s", "1"], capture_output=True, text=True, timeout=15
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    matches = re.findall(r"([\d.]+)% idle", text)
    return float(matches[-1]) / 100 if matches else None


def wait_for_idle(threshold=0.85, quiet_samples=3, timeout=120):
    """Block until the CPU has been mostly idle for consecutive samples; record what was seen."""
    start = time.monotonic()
    consecutive, last = 0, None
    while True:
        last = cpu_idle_fraction()
        if last is None:
            load = os.getloadavg()[0]
            return {"waited_s": time.monotonic() - start, "idle_fraction": None, "loadavg": load, "timed_out": False}
        consecutive = consecutive + 1 if last >= threshold else 0
        if consecutive >= quiet_samples:
            return {"waited_s": time.monotonic() - start, "idle_fraction": last, "loadavg": os.getloadavg()[0], "timed_out": False}
        if time.monotonic() - start > timeout:
            return {"waited_s": time.monotonic() - start, "idle_fraction": last, "loadavg": os.getloadavg()[0], "timed_out": True}


def run_worker(worker_args, threads, log_path):
    command = [sys.executable, "-m", "scripts.timestral_bench", "_worker", *map(str, worker_args)]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with log_path.open("w") as log:
        log.write(" ".join(command) + "\n")
        log.flush()
        completed = subprocess.run(command, cwd=ROOT, env=worker_env(threads), stdout=log, stderr=subprocess.STDOUT)
    return {"command": command, "returncode": completed.returncode, "seconds": time.monotonic() - start}


def append_manifest(run_dir, entry):
    path = run_dir / "manifest.json"
    entries = read_json(path) if path.exists() else []
    entries.append(entry)
    write_json(path, entries)


def _log_path(output):
    attempts = len(list(output.parent.glob(f"{output.stem}.attempt*.log"))) if output.parent.exists() else 0
    return output.parent / f"{output.stem}.attempt{attempts + 1}.log"


def _dispatch(run_dir, stage, worker_args, threads, output, idle_options, resume, extra, log_for=None):
    """Run one worker unless ``output`` exists; ``log_for`` names the log stem when it must not sit beside ``output``."""
    if output.exists():
        if resume:
            print(f"skip existing {output.relative_to(run_dir)}", flush=True)
            return True
        raise SystemExit(f"{output} exists; pass --resume or choose a fresh run id")
    idle = wait_for_idle(**idle_options) if idle_options is not None else None
    log_path = _log_path(log_for or output)
    result = run_worker(worker_args, threads, log_path)
    accepted = result["returncode"] == 0 and output.exists()
    entry = {
        "stage": stage,
        **extra,
        "output": str(output.relative_to(run_dir)),
        "log": str(log_path.relative_to(run_dir)),
        "threads": threads,
        "idle": idle,
        "accepted": accepted,
        "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        **result,
    }
    append_manifest(run_dir, entry)
    status = "ok" if accepted else f"FAILED (exit {result['returncode']}, see {log_path.name})"
    print(f"{stage} {' '.join(str(value) for value in extra.values())}: {status} in {result['seconds']:.0f} s", flush=True)
    return accepted


def _idle_options(args):
    if args.no_idle_wait:
        return None
    return {"threshold": args.idle_threshold, "quiet_samples": args.idle_samples, "timeout": args.idle_timeout}


def cmd_bench(args):
    run_dir = run_directory(args.run_id)
    provenance = read_json(run_dir / "provenance.json")
    if not (run_dir / "cores" / "build-info.json").exists() and any(impl.startswith("axon") for impl in args.impls):
        raise SystemExit("generated cores missing; run `build` first")
    failures = []
    for round_index in range(args.rounds):
        for size_index, size in enumerate(args.sizes):
            checkpoint = checkpoint_directory(provenance, size)
            for impl in rotated(args.impls, round_index + size_index):
                output = run_dir / "timing" / f"{impl}-{size}-r{round_index}.json"
                worker_args = [
                    "time", "--impl", impl, "--size", size, "--round", round_index,
                    "--run-dir", run_dir, "--checkpoint-dir", checkpoint, "--output", output,
                    "--samples", args.samples, "--warmup", args.warmup,
                ]
                accepted = _dispatch(
                    run_dir, "timing", worker_args, WORKER_THREADS.get(impl, HOST_THREADS), output,
                    _idle_options(args), args.resume, {"impl": impl, "size": size, "round": round_index},
                )
                if not accepted:
                    failures.append((impl, size, round_index))
    if failures:
        raise SystemExit(f"{len(failures)} timing workers failed: {failures}")
    print("all timing workers accepted")


def cmd_parity(args):
    run_dir = run_directory(args.run_id)
    provenance = read_json(run_dir / "provenance.json")
    failures = []
    for size in args.sizes:
        checkpoint = checkpoint_directory(provenance, size)
        reference_dir = run_dir / "parity" / f"reference-{size}"
        manifest_path = reference_dir / "manifest.json"
        if reference_dir.exists() and not manifest_path.exists():
            raise SystemExit(f"incomplete reference {reference_dir}; remove it before rerunning")
        worker_args = [
            "reference", "--size", size, "--run-dir", run_dir, "--checkpoint-dir", checkpoint,
            "--output-dir", reference_dir,
        ]
        # The worker creates the reference directory itself, so its log lives beside it.
        if not _dispatch(
            run_dir, "parity-reference", worker_args, WORKER_THREADS.get(REFERENCE_IMPL, HOST_THREADS),
            manifest_path, _idle_options(args), args.resume, {"impl": REFERENCE_IMPL, "size": size},
            log_for=run_dir / "parity" / f"reference-{size}.json",
        ):
            failures.append((REFERENCE_IMPL, size))
            continue
        for impl in args.impls:
            output = run_dir / "parity" / f"{impl}-{size}.json"
            worker_args = [
                "compare", "--impl", impl, "--size", size, "--run-dir", run_dir,
                "--checkpoint-dir", checkpoint, "--reference-dir", reference_dir, "--output", output,
            ]
            if not _dispatch(
                run_dir, "parity", worker_args, WORKER_THREADS.get(impl, HOST_THREADS), output,
                _idle_options(args), args.resume, {"impl": impl, "size": size},
            ):
                failures.append((impl, size))
    if failures:
        raise SystemExit(f"{len(failures)} parity workers failed: {failures}")
    print("all parity workers accepted")


def cmd_summarize(args):
    from .summarize import summarize

    run_dir = run_directory(args.run_id)
    summary = summarize(run_dir)
    print(json.dumps(summary["counts"], indent=2))
    print(run_dir / "summary.json")


def cmd_report(args):
    from .summarize import load_summary, render_markdown

    run_dir = run_directory(args.run_id)
    output = Path(args.output) if args.output else run_dir / "report.md"
    output.write_text(render_markdown(load_summary(run_dir)))
    print(output)


def cmd_worker(args):
    from . import worker

    if args.worker == "time":
        worker.time_worker(
            impl=args.impl, size=args.size, round_index=args.round, run_dir=args.run_dir,
            checkpoint_dir=args.checkpoint_dir, output=args.output, samples=args.samples, warmup=args.warmup,
        )
    elif args.worker == "reference":
        worker.reference_worker(
            size=args.size, run_dir=args.run_dir, checkpoint_dir=args.checkpoint_dir, output_dir=args.output_dir,
        )
    else:
        worker.compare_worker(
            impl=args.impl, size=args.size, run_dir=args.run_dir, checkpoint_dir=args.checkpoint_dir,
            reference_dir=args.reference_dir, output=args.output,
        )


def _csv_choices(allowed, label):
    def parse(text):
        values = [item.strip() for item in text.split(",") if item.strip()]
        unknown = [value for value in values if value not in allowed]
        if unknown or not values:
            raise argparse.ArgumentTypeError(f"unknown {label} {unknown}; choose from {list(allowed)}")
        return values

    return parse


def _positive_int(text):
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def _add_idle_arguments(parser):
    parser.add_argument("--no-idle-wait", action="store_true", help="start workers without waiting for CPU idle")
    parser.add_argument("--idle-threshold", type=float, default=0.85, help="idle CPU fraction required (default 0.85)")
    parser.add_argument("--idle-samples", type=int, default=3, help="consecutive one-second samples above the threshold")
    parser.add_argument("--idle-timeout", type=float, default=120.0, help="seconds to wait before starting anyway")


def build_parser():
    parser = argparse.ArgumentParser(prog="python -m scripts.timestral_bench", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    def command(name, handler, help_text):
        sub = commands.add_parser(name, help=help_text)
        sub.add_argument("--run-id", required=True, help="run name below log/, or a path")
        sub.set_defaults(handler=handler)
        return sub

    command("init", cmd_init, "create log/<run-id> and record provenance")
    command("build", cmd_build, "generate the unfused and fused Axon MLX cores (imports the compiler)")
    fetch = command("fetch", cmd_fetch, "download the pinned checkpoints and record their hashes")
    fetch.add_argument("--sizes", type=_csv_choices(SIZES, "sizes"), default=list(SIZES))
    bench = command("bench", cmd_bench, "run timing workers sequentially in fresh processes")
    bench.add_argument("--sizes", type=_csv_choices(SIZES, "sizes"), default=list(SIZES))
    bench.add_argument("--impls", type=_csv_choices(IMPLS, "implementations"), default=list(IMPLS))
    bench.add_argument("--rounds", type=int, default=3)
    bench.add_argument("--samples", type=_positive_int, default=7, help="timed calls per workload (default 7)")
    bench.add_argument("--warmup", type=_positive_int, default=2, help="untimed calls per workload before timing; the first is recorded as the first call (default 2)")
    bench.add_argument("--resume", action="store_true", help="skip outputs that already exist")
    _add_idle_arguments(bench)
    parity = command("parity", cmd_parity, "generate the PyTorch CPU reference and compare every implementation")
    parity.add_argument("--sizes", type=_csv_choices(SIZES, "sizes"), default=list(SIZES))
    parity.add_argument("--impls", type=_csv_choices(PARITY_IMPLS, "implementations"), default=list(PARITY_IMPLS))
    parity.add_argument("--resume", action="store_true", help="skip outputs that already exist")
    _add_idle_arguments(parity)
    command("summarize", cmd_summarize, "aggregate worker JSON into summary.json and CSV tables")
    report = command("report", cmd_report, "render markdown tables from summary.json")
    report.add_argument("--output", help="markdown path (default log/<run-id>/report.md)")

    worker = commands.add_parser("_worker", help=argparse.SUPPRESS)
    worker.set_defaults(handler=cmd_worker)
    kinds = worker.add_subparsers(dest="worker", required=True)
    timing = kinds.add_parser("time")
    timing.add_argument("--impl", required=True, choices=IMPLS)
    timing.add_argument("--size", required=True, choices=SIZES)
    timing.add_argument("--round", type=int, required=True)
    timing.add_argument("--samples", type=_positive_int, required=True)
    timing.add_argument("--warmup", type=_positive_int, required=True)
    timing.add_argument("--output", type=Path, required=True)
    reference = kinds.add_parser("reference")
    reference.add_argument("--size", required=True, choices=SIZES)
    reference.add_argument("--output-dir", type=Path, required=True)
    compare = kinds.add_parser("compare")
    compare.add_argument("--impl", required=True, choices=IMPLS)
    compare.add_argument("--size", required=True, choices=SIZES)
    compare.add_argument("--reference-dir", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    for sub in (timing, reference, compare):
        sub.add_argument("--run-dir", type=Path, required=True)
        sub.add_argument("--checkpoint-dir", type=Path, required=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main()
