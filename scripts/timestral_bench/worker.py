"""Fresh-process timing and parity workers.

The orchestrator in ``__main__`` starts one process per implementation, checkpoint
size and round. Only the adapters import PyTorch or MLX, and each worker asserts that
the other runtime never entered the process.
"""

import hashlib
import importlib.metadata
import json
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from .adapters import AxonAdapter, TorchAdapter, VikiivalAdapter
from .cases import cases, compare
from .config import TimestralConfig
from .workloads import WORKLOADS, make_inputs

IMPLS = ("axon-unfused", "axon-fused", "axon-fused-pruned", "vikiival", "torch-2", "torch-8")
REFERENCE_IMPL = "torch-2"
PARITY_IMPLS = tuple(impl for impl in IMPLS if impl != REFERENCE_IMPL)


def make_adapter(impl, cores_dir):
    cores_dir = Path(cores_dir)
    if impl == "axon-unfused":
        return AxonAdapter("unfused", core_module=cores_dir / "axon_core_unfused.py")
    if impl == "axon-fused":
        return AxonAdapter("fused", core_module=cores_dir / "axon_core_fused.py")
    if impl == "axon-fused-pruned":
        return AxonAdapter(
            "fused", core_module=cores_dir / "axon_core_fused.py", prune_final_block=True
        )
    if impl == "vikiival":
        return VikiivalAdapter()
    if impl.startswith("torch-"):
        return TorchAdapter(int(impl.split("-", 1)[1]))
    raise ValueError(f"unknown implementation {impl!r}; expected one of {IMPLS}")


def _version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def environment_snapshot():
    snapshot = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "pid": os.getpid(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "loadavg": list(os.getloadavg()),
        "versions": {
            name: _version(name)
            for name in ("numpy", "mlx", "torch", "timestral", "timestral-mlx", "safetensors", "huggingface_hub")
        },
        "env": {
            key: os.environ.get(key)
            for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "HF_HUB_OFFLINE", "CUDA_VISIBLE_DEVICES")
        },
    }
    try:
        therm = subprocess.run(["pmset", "-g", "therm"], capture_output=True, text=True, timeout=5)
        snapshot["pmset_therm"] = therm.stdout.strip()
    except (OSError, subprocess.TimeoutExpired) as error:
        snapshot["pmset_therm"] = f"unavailable: {error!r}"
    return snapshot


def _sha256(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def write_json(path, payload):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n")


def rotated(items, offset):
    items = list(items)
    if not items:
        return items
    shift = offset % len(items)
    return items[shift:] + items[:shift]


def _timed(function, *args, **kwargs):
    start = time.perf_counter_ns()
    result = function(*args, **kwargs)
    return result, (time.perf_counter_ns() - start) / 1e6


def _output_checks(output):
    return {
        "output_shape": list(output.shape),
        "finite": bool(np.isfinite(output).all()),
        "ordered": bool((np.diff(output, axis=-1) >= 0).all()),
    }


def time_worker(*, impl, size, round_index, run_dir, checkpoint_dir, output, samples, warmup):
    if samples < 1 or warmup < 1:
        # The first call always runs untimed and is recorded as the first-call latency,
        # so the recorded warmup count must be at least one.
        raise ValueError("samples and warmup must be positive integers")
    run_dir = Path(run_dir)
    checkpoint_dir = Path(checkpoint_dir)
    impl_index = IMPLS.index(impl)
    adapter = make_adapter(impl, run_dir / "cores")
    before = environment_snapshot()
    load_info, load_ms = _timed(adapter.load, checkpoint_dir)
    order = rotated(WORKLOADS, round_index + impl_index)
    records = []
    for position, workload in enumerate(order):
        inputs = make_inputs(workload)
        adapter.memory_reset()
        first, first_ms = _timed(adapter.forecast, inputs)
        for _ in range(max(0, warmup - 1)):
            adapter.forecast(inputs)
        forecast_ms, hashes = [], []
        output_array = first
        for _ in range(samples):
            output_array, elapsed = _timed(adapter.forecast, inputs)
            forecast_ms.append(elapsed)
            hashes.append(_sha256(output_array))
        record = {
            "workload": workload.name,
            "position": position,
            "first_call_ms": first_ms,
            "first_call_sha256": _sha256(first),
            "forecast_ms": forecast_ms,
            "output_sha256": hashes,
            "deterministic": len(set(hashes)) == 1,
            **_output_checks(output_array),
        }
        if adapter.supports_core_replay:
            record["core_calls"] = adapter.record(inputs)
            adapter.replay()
            core_ms = []
            for _ in range(samples):
                _, elapsed = _timed(adapter.replay)
                core_ms.append(elapsed)
            record["core_ms"] = core_ms
            adapter.clear_recording()
        record["memory"] = adapter.memory_info()
        record["max_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        records.append(record)
        print(
            json.dumps(
                {
                    "impl": impl,
                    "size": size,
                    "workload": workload.name,
                    "median_ms": float(np.median(forecast_ms)),
                    "core_median_ms": float(np.median(record["core_ms"])) if "core_ms" in record else None,
                }
            ),
            flush=True,
        )
    adapter.assert_isolation()
    payload = {
        "kind": "timing",
        "impl": impl,
        "size": size,
        "round": round_index,
        "impl_index": impl_index,
        "checkpoint_dir": str(checkpoint_dir),
        "samples": samples,
        "warmup": warmup,
        "load_ms": load_ms,
        "load_info": load_info,
        "environment_before": before,
        "environment_after": environment_snapshot(),
        "torch_imported": "torch" in sys.modules,
        "mlx_imported": "mlx.core" in sys.modules,
        "workloads": records,
    }
    write_json(output, payload)
    return payload


def parity_cases(config):
    """The 14 synthetic cases on this checkpoint plus the nine timing workloads."""
    long_horizon = config.block_size * config.patch_size + 17
    items = []
    for name, arrays, kwargs, constructor in cases(config.covariate_enabled, long_horizon):
        items.append((f"case_{name}", arrays, kwargs, constructor))
    for workload in WORKLOADS:
        inputs = make_inputs(workload)
        items.append((f"workload_{workload.name}", inputs["arrays"], inputs["kwargs"], {}))
    return items


def reference_worker(*, size, run_dir, checkpoint_dir, output_dir, impl=REFERENCE_IMPL):
    run_dir = Path(run_dir)
    checkpoint_dir = Path(checkpoint_dir)
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    output_dir.mkdir(parents=True)
    adapter = make_adapter(impl, run_dir / "cores")
    load_info, load_ms = _timed(adapter.load, checkpoint_dir)
    config = TimestralConfig.load(checkpoint_dir / "config.json")
    manifest = {
        "kind": "parity-reference",
        "impl": impl,
        "size": size,
        "checkpoint_dir": str(checkpoint_dir),
        "load_ms": load_ms,
        "load_info": load_info,
        "environment": environment_snapshot(),
        "cases": [],
    }
    for name, arrays, kwargs, constructor in parity_cases(config):
        expected, elapsed = _timed(adapter.forecast, {"arrays": arrays, "kwargs": kwargs}, constructor or None)
        np.savez(output_dir / f"{name}.npz", expected=expected, **arrays)
        manifest["cases"].append(
            {
                "name": name,
                "file": f"{name}.npz",
                "kwargs": kwargs,
                "constructor": constructor,
                "milliseconds": elapsed,
                **_output_checks(expected),
            }
        )
        print(json.dumps({"reference": name, "shape": list(expected.shape)}), flush=True)
    adapter.assert_isolation()
    manifest["torch_imported"] = "torch" in sys.modules
    manifest["mlx_imported"] = "mlx.core" in sys.modules
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def compare_worker(*, impl, size, run_dir, checkpoint_dir, reference_dir, output):
    run_dir = Path(run_dir)
    reference_dir = Path(reference_dir)
    manifest = json.loads((reference_dir / "manifest.json").read_text())
    adapter = make_adapter(impl, run_dir / "cores")
    load_info, load_ms = _timed(adapter.load, Path(checkpoint_dir))
    records = []
    for entry in manifest["cases"]:
        with np.load(reference_dir / entry["file"], allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files if key != "expected"}
            expected = data["expected"]
        try:
            actual, elapsed = _timed(
                adapter.forecast, {"arrays": arrays, "kwargs": entry["kwargs"]}, entry["constructor"] or None
            )
            record = compare(actual, expected) | {"milliseconds": elapsed}
        except Exception as error:  # record the failure, keep comparing the other cases
            record = {"passed": False, "error": repr(error)}
        record = {"name": entry["name"]} | record
        records.append(record)
        print(json.dumps(record, default=_json_default), flush=True)
    adapter.assert_isolation()
    payload = {
        "kind": "parity",
        "impl": impl,
        "size": size,
        "reference_impl": manifest["impl"],
        "checkpoint_dir": str(checkpoint_dir),
        "load_ms": load_ms,
        "load_info": load_info,
        "environment": environment_snapshot(),
        "torch_imported": "torch" in sys.modules,
        "mlx_imported": "mlx.core" in sys.modules,
        "passed": sum(1 for record in records if record["passed"]),
        "total": len(records),
        "cases": records,
    }
    write_json(output, payload)
    return payload
