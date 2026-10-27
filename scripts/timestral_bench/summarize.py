"""Aggregate worker JSON into ``summary.json``, CSV tables and markdown. Stdlib and NumPy only.

Aggregation: median of the timed samples in each fresh process, then the median of those
process medians across rounds. Ratios compare each implementation with the baseline
``axon-fused`` per workload; the per-size figure is the geometric mean over workloads.
"""

import csv
import json
import math
from pathlib import Path
from statistics import median

from .workloads import REPRESENTATIVE, WORKLOAD_NAMES

BASELINE = "axon-fused"
IMPL_ORDER = ("axon-fused", "vikiival", "torch-2", "torch-8", "axon-unfused", "axon-fused-pruned")
COMPARED = tuple(impl for impl in IMPL_ORDER if impl != BASELINE)
SIZE_ORDER = ("5M", "28M", "123M", "209M")
LABELS = {
    "axon-fused": "Axon MLX (fused)",
    "axon-unfused": "Axon MLX (unfused)",
    "axon-fused-pruned": "Axon MLX (fused, pruned)",
    "vikiival": "vikiival MLX",
    "torch-2": "original PyTorch CPU, 2 threads",
    "torch-8": "original PyTorch CPU, 8 threads",
}


def geometric_mean(values):
    values = [float(value) for value in values]
    if not values or any(value <= 0 for value in values):
        return None
    return math.exp(sum(math.log(value) for value in values) / len(values))


def load_json_files(directory):
    directory = Path(directory)
    if not directory.is_dir():
        return []
    return [json.loads(path.read_text()) for path in sorted(directory.glob("*.json"))]


def process_medians(timing_records):
    """Return ``{(impl, size, workload, kind): {round: median_ms}}`` from worker records."""
    table = {}
    for record in timing_records:
        for entry in record["workloads"]:
            key = (record["impl"], record["size"], entry["workload"], "forecast")
            table.setdefault(key, {})[record["round"]] = median(entry["forecast_ms"])
            if entry.get("core_ms"):
                key = (record["impl"], record["size"], entry["workload"], "core")
                table.setdefault(key, {})[record["round"]] = median(entry["core_ms"])
    return table


def across_rounds(per_process):
    """Median, min and max of per-process medians for every key."""
    return {
        key: {
            "median": median(by_round.values()),
            "min": min(by_round.values()),
            "max": max(by_round.values()),
            "rounds": len(by_round),
            "by_round": dict(sorted(by_round.items())),
        }
        for key, by_round in per_process.items()
    }


def ratios(aggregated, baseline=BASELINE):
    """``{(impl, size, workload): {"ratio", "by_round"}}``; ratio = impl / baseline forecast latency."""
    table = {}
    for (impl, size, workload, kind), stats in aggregated.items():
        if kind != "forecast" or impl == baseline:
            continue
        base = aggregated.get((baseline, size, workload, "forecast"))
        if base is None:
            continue
        shared = sorted(set(stats["by_round"]) & set(base["by_round"]))
        table[(impl, size, workload)] = {
            "ratio": stats["median"] / base["median"],
            "by_round": {round_: stats["by_round"][round_] / base["by_round"][round_] for round_ in shared},
        }
    return table


def size_summary(ratio_table):
    """Geometric mean of workload ratios per (impl, size), with the per-round range."""
    grouped = {}
    for (impl, size, workload), entry in ratio_table.items():
        grouped.setdefault((impl, size), {})[workload] = entry
    summary = {}
    for (impl, size), workloads in grouped.items():
        rounds = set.intersection(*(set(entry["by_round"]) for entry in workloads.values()))
        by_round = {
            round_: geometric_mean(entry["by_round"][round_] for entry in workloads.values())
            for round_ in sorted(rounds)
        }
        summary[(impl, size)] = {
            "geomean": geometric_mean(entry["ratio"] for entry in workloads.values()),
            "workloads": len(workloads),
            "by_round": by_round,
            "min": min(by_round.values()) if by_round else None,
            "max": max(by_round.values()) if by_round else None,
            "faster_than_baseline": sorted(
                workload for workload, entry in workloads.items() if entry["ratio"] < 1
            ),
        }
    return summary


def host_overhead(aggregated):
    """Forecast / core latency ratio per (impl, size, workload) and its geomean per (impl, size)."""
    per_workload = {}
    for (impl, size, workload, kind), stats in aggregated.items():
        if kind != "core":
            continue
        forecast = aggregated.get((impl, size, workload, "forecast"))
        if forecast is not None:
            per_workload[(impl, size, workload)] = forecast["median"] / stats["median"]
    per_size = {}
    for (impl, size, workload), ratio in per_workload.items():
        per_size.setdefault((impl, size), []).append(ratio)
    return per_workload, {key: geometric_mean(values) for key, values in per_size.items()}


def parity_summary(parity_records):
    summary = {}
    for record in parity_records:
        if record.get("kind") != "parity":
            continue
        cases = record["cases"]
        failing = [case for case in cases if not case.get("passed")]
        summary[(record["impl"], record["size"])] = {
            "reference_impl": record.get("reference_impl"),
            "passed": sum(1 for case in cases if case.get("passed")),
            "total": len(cases),
            "max_abs": max((case["max_abs"] for case in cases if "max_abs" in case), default=None),
            "failing": [
                {
                    "name": case["name"],
                    "max_abs": case.get("max_abs"),
                    "outside": case.get("quantiles_outside_tolerance"),
                    "error": case.get("error"),
                }
                for case in failing
            ],
        }
    return summary


def _nest(table, depth):
    """Turn a dict keyed by tuples into nested dicts for JSON."""
    nested = {}
    for key, value in table.items():
        cursor = nested
        for part in key[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[key[-1]] = value
    return nested


def _order(keys, order):
    ranked = {name: index for index, name in enumerate(order)}
    return sorted(keys, key=lambda key: (ranked.get(key, len(order)), key))


def summarize(run_dir, output_dir=None):
    run_dir = Path(run_dir)
    output_dir = Path(output_dir) if output_dir is not None else run_dir
    timing = [record for record in load_json_files(run_dir / "timing") if record.get("kind") == "timing"]
    parity = load_json_files(run_dir / "parity")
    per_process = process_medians(timing)
    aggregated = across_rounds(per_process)
    ratio_table = ratios(aggregated)
    sizes = size_summary(ratio_table)
    overhead_workload, overhead_size = host_overhead(aggregated)
    agreement = parity_summary(parity)

    latency = {}
    for (impl, size, workload, kind), stats in aggregated.items():
        latency.setdefault(impl, {}).setdefault(size, {}).setdefault(workload, {})[kind] = {
            key: value for key, value in stats.items() if key != "by_round"
        } | {"by_round": {str(round_): value for round_, value in stats["by_round"].items()}}
    ratio_nested = {}
    for (impl, size), entry in sizes.items():
        ratio_nested.setdefault(impl, {})[size] = entry | {
            "by_round": {str(round_): value for round_, value in entry["by_round"].items()},
            "per_workload": {
                workload: ratio_table[(impl, size, workload)]["ratio"]
                for (impl_, size_, workload) in ratio_table
                if impl_ == impl and size_ == size
            },
        }
    overhead_nested = {}
    for (impl, size), value in overhead_size.items():
        overhead_nested.setdefault(impl, {})[size] = {
            "geomean": value,
            "per_workload": {
                workload: overhead_workload[(impl, size, workload)]
                for (impl_, size_, workload) in overhead_workload
                if impl_ == impl and size_ == size
            },
        }
    coverage = sorted(
        {(record["impl"], record["size"], record["round"]) for record in timing}
    )
    summary = {
        "baseline": BASELINE,
        "impls": _order({record["impl"] for record in timing} | {key[0] for key in agreement}, IMPL_ORDER),
        "sizes": _order({record["size"] for record in timing} | {key[1] for key in agreement}, SIZE_ORDER),
        "workloads": list(WORKLOAD_NAMES),
        "counts": {
            "timing_processes": len(timing),
            "forecast_samples": sum(len(entry["forecast_ms"]) for record in timing for entry in record["workloads"]),
            "core_samples": sum(len(entry.get("core_ms", [])) for record in timing for entry in record["workloads"]),
            "parity_processes": len(agreement),
            "samples_per_process": sorted({record["samples"] for record in timing}),
            "warmup_per_process": sorted({record["warmup"] for record in timing}),
        },
        "coverage": [{"impl": impl, "size": size, "round": round_} for impl, size, round_ in coverage],
        "isolation": {
            "mlx_workers_without_torch": all(
                not record["torch_imported"] for record in timing if not record["impl"].startswith("torch")
            ),
            "torch_workers_without_mlx": all(
                not record["mlx_imported"] for record in timing if record["impl"].startswith("torch")
            ),
        },
        "latency": latency,
        "ratios": ratio_nested,
        "host_overhead": overhead_nested,
        "parity": _nest(agreement, 2),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    _write_comparison_csv(output_dir / "comparison.csv", summary)
    _write_samples_csv(output_dir / "samples.csv", timing)
    return summary


def load_summary(run_dir):
    return json.loads((Path(run_dir) / "summary.json").read_text())


def _stat(summary, impl, size, workload, kind="forecast"):
    return summary["latency"].get(impl, {}).get(size, {}).get(workload, {}).get(kind)


def _write_comparison_csv(path, summary):
    impls = summary["impls"]
    columns = ["size", "workload"]
    for impl in impls:
        columns += [f"{impl}_ms", f"{impl}_min_ms", f"{impl}_max_ms", f"{impl}_rounds"]
        if impl != summary["baseline"]:
            columns.append(f"{impl}_ratio")
        if impl.startswith("axon"):
            columns.append(f"{impl}_core_ms")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for size in summary["sizes"]:
            for workload in summary["workloads"]:
                row = {"size": size, "workload": workload}
                for impl in impls:
                    stats = _stat(summary, impl, size, workload)
                    if stats:
                        row[f"{impl}_ms"] = f"{stats['median']:.4f}"
                        row[f"{impl}_min_ms"] = f"{stats['min']:.4f}"
                        row[f"{impl}_max_ms"] = f"{stats['max']:.4f}"
                        row[f"{impl}_rounds"] = stats["rounds"]
                    ratio = summary["ratios"].get(impl, {}).get(size, {}).get("per_workload", {}).get(workload)
                    if ratio is not None:
                        row[f"{impl}_ratio"] = f"{ratio:.4f}"
                    core = _stat(summary, impl, size, workload, "core")
                    if core:
                        row[f"{impl}_core_ms"] = f"{core['median']:.4f}"
                if any(key not in ("size", "workload") for key in row):
                    writer.writerow(row)


def _write_samples_csv(path, timing):
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["impl", "size", "round", "workload", "kind", "index", "ms"])
        for record in timing:
            for entry in record["workloads"]:
                for index, value in enumerate(entry["forecast_ms"]):
                    writer.writerow([record["impl"], record["size"], record["round"], entry["workload"], "forecast", index, f"{value:.4f}"])
                for index, value in enumerate(entry.get("core_ms", [])):
                    writer.writerow([record["impl"], record["size"], record["round"], entry["workload"], "core", index, f"{value:.4f}"])


def _ms(stats):
    if not stats:
        return "—"
    return f"{stats['median']:.2f} ({stats['min']:.2f}–{stats['max']:.2f})"


def _ratio(entry):
    if not entry or entry.get("geomean") is None:
        return "—"
    text = f"{entry['geomean']:.3f}×"
    if entry.get("min") is not None and entry.get("max") is not None:
        text += f" ({entry['min']:.3f}–{entry['max']:.3f})"
    return text


def _table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" if index == 0 else "---:" for index in range(len(header))) + "|"]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


def render_markdown(summary):
    impls = [impl for impl in summary["impls"] if impl in LABELS]
    compared = [impl for impl in impls if impl != summary["baseline"]]
    sizes = summary["sizes"]
    counts = summary["counts"]
    parts = []
    parts.append(
        f"{counts['timing_processes']} timing processes, {counts['forecast_samples']} timed complete "
        f"forecasts and {counts['core_samples']} generated-core replays; {counts['parity_processes']} "
        f"parity processes. Samples per workload per process: {counts['samples_per_process']}; warmup calls: "
        f"{counts['warmup_per_process']}. MLX workers without PyTorch: "
        f"{summary['isolation']['mlx_workers_without_torch']}; PyTorch workers without MLX: "
        f"{summary['isolation']['torch_workers_without_mlx']}."
    )
    parts.append("### Latency ratio to Axon MLX (fused)\n")
    parts.append(
        "Geometric mean over workloads of implementation median latency divided by the Axon fused "
        "median latency. Values above 1 mean the implementation is slower than Axon. Parentheses give "
        "the min–max of per-round geometric means, not confidence intervals.\n"
    )
    rows = []
    for size in sizes:
        row = [size]
        for impl in compared:
            row.append(_ratio(summary["ratios"].get(impl, {}).get(size)))
        rows.append(row)
    parts.append(_table(["Size"] + [LABELS[impl] for impl in compared], rows))
    faster = []
    for impl in compared:
        for size in sizes:
            entry = summary["ratios"].get(impl, {}).get(size)
            if entry and entry.get("faster_than_baseline"):
                faster.append(f"- {LABELS[impl]} at {size}: {', '.join(entry['faster_than_baseline'])}")
    parts.append("\nWorkloads where an implementation was faster than Axon fused:\n" + ("\n".join(faster) if faster else "- none"))

    parts.append("\n### Representative complete-forecast latencies\n")
    parts.append("Milliseconds; median of per-process medians with the min–max across processes.\n")
    rows = []
    for size in sizes:
        for workload in REPRESENTATIVE:
            rows.append([size, workload] + [_ms(_stat(summary, impl, size, workload)) for impl in impls])
    parts.append(_table(["Size", "Workload"] + [LABELS[impl] for impl in impls], rows))

    overhead_impls = [impl for impl in impls if impl in summary["host_overhead"]]
    if overhead_impls:
        parts.append("\n### Host share of Axon latency\n")
        parts.append(
            "Complete-forecast latency divided by generated-core replay latency, geometric mean over "
            "workloads. The replay reruns the recorded network calls on preloaded MLX arrays and excludes "
            "host normalization, mask construction, transfers and output conversion.\n"
        )
        rows = []
        for size in sizes:
            rows.append([size] + [
                (lambda v: f"{v:.3f}×" if v else "—")(summary["host_overhead"].get(impl, {}).get(size, {}).get("geomean"))
                for impl in overhead_impls
            ])
        parts.append(_table(["Size"] + [LABELS[impl] for impl in overhead_impls], rows))

    if summary["parity"]:
        parity_impls = [impl for impl in impls if impl in summary["parity"]]
        parts.append("\n### Agreement with the original PyTorch CPU reference\n")
        parts.append(
            "Cases passing `abs(actual - reference) <= 2e-4 + 2e-4 * abs(reference)` with finite, "
            "non-decreasing quantiles, out of all cases; parentheses give the maximum absolute difference.\n"
        )
        rows = []
        for size in sizes:
            row = [size]
            for impl in parity_impls:
                entry = summary["parity"].get(impl, {}).get(size)
                if entry:
                    max_abs = entry["max_abs"]
                    row.append(f"{entry['passed']}/{entry['total']}" + (f" ({max_abs:.3g})" if max_abs is not None else ""))
                else:
                    row.append("—")
            rows.append(row)
        parts.append(_table(["Size"] + [LABELS[impl] for impl in parity_impls], rows))
        failing = []
        for impl in parity_impls:
            for size in sizes:
                entry = summary["parity"].get(impl, {}).get(size)
                for case in (entry or {}).get("failing", []):
                    detail = case["error"] if case.get("error") else f"max abs {case['max_abs']:.3g}, {case['outside']} quantiles outside tolerance"
                    failing.append(f"- {size} {LABELS[impl]}: `{case['name']}` ({detail})")
        parts.append("\nFailing cases:\n" + ("\n".join(failing) if failing else "- none"))

    parts.append("\n### All workloads\n")
    parts.append("Median complete-forecast latency in milliseconds (median of per-process medians).\n")
    rows = []
    for size in sizes:
        for workload in summary["workloads"]:
            cells = [(lambda s: f"{s['median']:.2f}" if s else "—")(_stat(summary, impl, size, workload)) for impl in impls]
            if any(cell != "—" for cell in cells):
                rows.append([size, workload] + cells)
    parts.append(_table(["Size", "Workload"] + [LABELS[impl] for impl in impls], rows))
    return "\n".join(parts) + "\n"
