#!/usr/bin/env python3
"""Merge per-model ablation CSVs and generate cross-model comparison SVGs."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from synapse.axon_ablation import (
    merge_ablation_csvs,
    plot_cross_model_kernels,
    plot_cross_model_memory,
    plot_cross_model_nodes,
    plot_cross_model_speedup,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge per-model ablation CSVs and generate cross-model comparison SVGs.",
    )
    parser.add_argument(
        "log_dir",
        type=Path,
        help="Directory containing per-model subdirectories with ablation.csv files.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=None,
        help="Output directory for merged CSV and SVGs (default: <log_dir>).",
    )
    args = parser.parse_args()

    log_dir = args.log_dir
    output_dir = args.output_dir or log_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all ablation.csv files in subdirectories
    csv_paths = sorted(log_dir.glob("**/ablation.csv"))
    if not csv_paths:
        print(f"No ablation.csv files found under {log_dir}/", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(csv_paths)} ablation CSVs:")
    for p in csv_paths:
        print(f"  {p.relative_to(log_dir)}")

    # Merge CSVs
    combined_csv = output_dir / "combined.csv"
    merge_ablation_csvs(csv_paths, combined_csv)
    print(f"\nMerged CSV: {combined_csv}")

    # Generate cross-model visualizations
    nodes_svg = output_dir / "cross_model_nodes.svg"
    plot_cross_model_nodes(combined_csv, nodes_svg)
    print(f"Node count chart: {nodes_svg}")

    speedup_svg = output_dir / "cross_model_speedup.svg"
    plot_cross_model_speedup(combined_csv, speedup_svg)
    print(f"Speed ratio chart: {speedup_svg}")

    kernels_svg = output_dir / "cross_model_kernels.svg"
    plot_cross_model_kernels(combined_csv, kernels_svg)
    print(f"Kernel count chart: {kernels_svg}")

    memory_svg = output_dir / "cross_model_memory.svg"
    plot_cross_model_memory(combined_csv, memory_svg)
    print(f"Memory chart: {memory_svg}")

    # Print quick summary table
    print("\n=== T1 Summary (graph-full vs no-opt) ===")
    import csv

    models: dict[str, dict[str, dict[str, str]]] = {}
    with combined_csv.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("tier") != "t1-stages":
                continue
            model = row.get("model", "?")
            config = row.get("config_name", "?")
            if model not in models:
                models[model] = {}
            models[model][config] = row

    print(f"{'Model':30s} {'no-opt nodes':>12s} {'full nodes':>12s} {'reduction':>10s} {'no-opt kc':>10s} {'full kc':>10s} {'kc-red':>8s} {'no-opt MB':>10s} {'full MB':>10s} {'mem-red':>8s} {'speedup':>8s}")
    for model in sorted(models.keys()):
        configs = models[model]
        no_opt = configs.get("no-opt", {})
        full = configs.get("graph-full", {})
        no_opt_nodes = int(no_opt.get("graph_nodes", 0))
        full_nodes = int(full.get("graph_nodes", 0))
        reduction = f"{(1 - full_nodes / no_opt_nodes) * 100:.1f}%" if no_opt_nodes else "N/A"

        def _int(v): return int(v) if v else None
        def _float(v): return float(v) if v else None
        no_kc = _int(no_opt.get("kernel_count"))
        full_kc = _int(full.get("kernel_count"))
        no_kc_s = f"{no_kc}" if no_kc else "N/A"
        full_kc_s = f"{full_kc}" if full_kc else "N/A"
        kc_red = f"{(1 - full_kc / no_kc) * 100:.0f}%" if no_kc and full_kc else "N/A"

        no_mem = _float(no_opt.get("peak_memory_bytes"))
        full_mem = _float(full.get("peak_memory_bytes"))
        no_mem_s = f"{no_mem / 1048576:.1f}" if no_mem else "N/A"
        full_mem_s = f"{full_mem / 1048576:.1f}" if full_mem else "N/A"
        mem_red = f"{(1 - full_mem / no_mem) * 100:.0f}%" if no_mem and full_mem else "N/A"

        no_opt_ratio = no_opt.get("speed_ratio", "")
        full_ratio = full.get("speed_ratio", "")
        speedup = "N/A"
        try:
            if no_opt_ratio and full_ratio:
                no_r = float(no_opt_ratio)
                full_r = float(full_ratio)
                if full_r > 0 and no_r > 0:
                    speedup = f"{no_r / full_r:.2f}x"
        except ValueError:
            pass
        print(f"{model:30s} {no_opt_nodes:>12d} {full_nodes:>12d} {reduction:>10s} {no_kc_s:>10s} {full_kc_s:>10s} {kc_red:>8s} {no_mem_s:>10s} {full_mem_s:>10s} {mem_red:>8s} {speedup:>8s}")


if __name__ == "__main__":
    main()
