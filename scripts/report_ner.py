#!/usr/bin/env python3
"""Collect NER results, preserving every measured sample and failed quality gate."""

import argparse
import gzip
import json
import subprocess
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def collect(root):
    configurations = {}
    for path in sorted((root / "results").glob("*.json")):
        if path.name.endswith("-quality.json"):
            name = path.name.removesuffix("-quality.json")
        elif path.name.endswith("-performance.json"):
            name = path.name.removesuffix("-performance.json")
        elif path.name.endswith("-all.json"):
            name = path.name.removesuffix("-all.json")
        else:
            continue  # ORT profiling traces are separate raw artifacts.
        result = read(path)
        result["arguments"] = {
            k: v for k, v in result["arguments"].items() if k not in {"run_dir", "output"}
        }
        configurations.setdefault(name, {})[result["arguments"]["stage"]] = result
    if not configurations:
        raise ValueError("No completed benchmark outputs found")
    manifest = read(root / "manifest.json")
    frozen = read(root / "training-complete.json")
    for name, stages in configurations.items():
        for result in stages.values():
            if result["checkpoint_sha256"] != frozen["checkpoint_sha256"]:
                raise ValueError(f"Checkpoint changed for {name}")
            if result["dataset_revision"] != manifest["dataset_revision"]:
                raise ValueError(f"Dataset changed for {name}")
    return {
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "manifest": manifest,
        "training_recipe": read(root / "training-recipe.json"),
        "checkpoint_sha256": frozen["checkpoint_sha256"],
        "selection": read(root / "selection.json"),
        "export": read(root / "export/export.json"),
        "validation": read(root / "validation.json") if (root / "validation.json").exists() else {},
        "configurations": configurations,
    }


def render(data, hardware):
    lines = [
        "# MiniLM / Few-NERD runtime measurements",
        "",
        hardware,
        "",
        "Timings start with host token arrays and end with completed host logits; "
        "transfers are included. CPU uses four intra-op threads and one inter-op thread. "
        "HF and Axon Torch use eager execution with SDPA; TF32 is disabled.",
        "",
        "Each shape has 10 warmups and 150 measured calls in three contiguous groups. "
        "These are within-process samples, not independent process replicates. "
        "Throughput divides batch size by mean latency; latency columns show p50.",
        "",
        "The checkpoint was selected on validation data. Quality is exact-span IO "
        "scoring on the complete official Few-NERD supervised test split. "
        "This experiment is a runtime comparison, not a claim about NER state of the art.",
        "",
        "| Configuration | Micro F1 | Macro F1 | Word agreement | Quality gate |",
        "|---|---:|---:|---:|---|",
    ]
    performance = {}
    for name, stages in data["configurations"].items():
        q = stages.get("quality", stages.get("all", {})).get("quality")
        if q:
            gate = "PASS" if q["quality_gate_passed"] else "FAIL"
            lines.append(
                f"| {name} | {q['f1']:.6f} | {q['macro_f1']:.6f} | "
                f"{q['word_prediction_agreement']:.6f} | {gate} |"
            )
        else:
            lines.append(f"| {name} | pending | pending | pending | pending |")
        p = stages.get("performance", stages.get("all", {})).get("performance")
        if p:
            performance[name] = p
    lines += [
        "",
        "FP32 passes when evaluated word logits match HF CPU FP32 at atol=rtol=1e-4. "
        "FP16, INT8, and Core ML pass when both micro and macro F1 lose at most 0.005 "
        "absolute. Per-label results and logit differences are retained in the JSON.",
        "",
        "| Configuration | B1/S32 ms | B1/S128 ms | B1/S512 ms | B32/S128 sequences/s |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, p in performance.items():
        shapes = {(s["batch"], s["length"]): s for s in p["shapes"]}
        lines.append(
            f"| {name} | {shapes[1, 32]['p50_ms']:.3f} | {shapes[1, 128]['p50_ms']:.3f} | "
            f"{shapes[1, 512]['p50_ms']:.3f} | {shapes[32, 128]['sequences_per_second']:.1f} |"
        )
    lines += ["", "## Full shape grid", ""]
    for name, p in performance.items():
        lines += [
            f"### {name}",
            "",
            "| Batch | Tokens | p50 ms | p95 ms | First call ms | Sequences/s |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
        for s in p["shapes"]:
            lines.append(
                f"| {s['batch']} | {s['length']} | {s['p50_ms']:.3f} | {s['p95_ms']:.3f} | "
                f"{s['first_call_ms']:.3f} | {s['sequences_per_second']:.1f} |"
            )
        lines += [
            "",
            "End-to-end starts from presegmented words and includes tokenization, "
            "windowing, inference, and span decoding on 256 deterministic test sentences.",
            "",
            "| Batch | p50 ms | p95 ms | Sentences/s |",
            "|---:|---:|---:|---:|",
        ]
        for s in p["end_to_end"]:
            lines.append(
                f"| {s['batch']} | {s['p50_ms']:.3f} | {s['p95_ms']:.3f} | "
                f"{s['sentences_per_second']:.1f} |"
            )
        if p["provider_node_events"]:
            lines += [
                "",
                f"ORT placement events: `{json.dumps(p['provider_node_events'], sort_keys=True)}`.",
            ]
        lines.append("")
    lines += [
        "## Provenance and limits",
        "",
        f"Checkpoint SHA-256: `{data['checkpoint_sha256']}`.",
        "",
        f"Base revision: `{data['manifest']['base_revision']}`. "
        f"Dataset revision: `{data['manifest']['dataset_revision']}`.",
        "",
        "ORT export fusion counts: "
        + "`"
        + json.dumps(data["export"]["fusion_counts"], sort_keys=True)
        + "`.",
        "",
        "Check actual fusion counts before describing these as fully fused transformer graphs. "
        "Export and optimizer versions can change the matched patterns. The JSON preserves "
        "graph hashes, operators, software versions, raw timings, and per-label quality.",
        "",
        "Process RSS includes runtime and data allocations. Torch's allocator counter is not "
        "total GPU memory for ORT. No Apple Silicon result can be inferred from Linux results; "
        "Core ML provider placement alone does not establish Neural Engine execution.",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--hardware", required=True, help="Hardware description to include verbatim"
    )
    args = parser.parse_args()
    data = collect(args.run_dir)
    data["hardware"] = args.hardware
    payload = (json.dumps(data, indent=2) + "\n").encode()
    (args.run_dir / "public-results.json").write_bytes(payload)
    (args.run_dir / "public-results.json.gz").write_bytes(gzip.compress(payload, mtime=0))
    (args.run_dir / "public-report.md").write_text(render(data, args.hardware))
    print(args.run_dir / "public-report.md")


if __name__ == "__main__":
    main()
