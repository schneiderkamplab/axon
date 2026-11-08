"""Render tests/data/parity_report.json (written by the test suite) as markdown tables.

    uv run python tools/parity_table.py
"""

import json
from pathlib import Path

REPORT = Path(__file__).resolve().parent.parent / "tests" / "data" / "parity_report.json"
SIZES = ["17m", "32m", "68m", "150m", "400m", "1b"]
CASES = ["short", "medium", "long", "padded"]


def order(name: str) -> tuple:
    kind, size = name.split("-")[1:3]
    return SIZES.index(size), kind


def e(x: float) -> str:
    return f"{x:.1e}"


def main():
    data = json.loads(REPORT.read_text())
    models = sorted(data, key=order)

    print("FP32 vs HF transformers FP32 (CPU, eager). Logits max abs diff / max rel diff / top-1 agreement;")
    print("then max abs diff vs the float64 HF run (exact for the fp32 weights) for ours and for HF fp32.\n")
    print("| model | case | positions | max abs | max rel | top-1 | ours vs fp64 | HF fp32 vs fp64 |")
    print("|---|---|---:|---:|---:|---:|---:|---:|")
    for m in models:
        for c in CASES:
            r = data[m].get(f"fp32/{c}")
            if r:
                flag = " *" if r["max_abs"] >= 1e-3 else ""
                print(f"| {m} | {c} | {r['positions']} | {e(r['max_abs'])}{flag} | {e(r['max_rel'])} "
                      f"| {r['top1_agreement']:.2%} | {e(r['max_abs_vs_fp64'])} | {e(r['hf_fp32_vs_fp64'])} |")

    print("\nWindow edges (2048 tokens: first/last 66 positions and every multiple of 64 +- 1):\n")
    print("| model | positions | max abs | ours vs fp64 | top-1 |")
    print("|---|---:|---:|---:|---:|")
    for m in models:
        r = data[m].get("fp32/window_edges")
        if r:
            print(f"| {m} | {r['positions']} | {e(r['max_abs'])} | {e(r['max_abs_vs_fp64'])} | {r['top1_agreement']:.2%} |")

    print("\nBF16 compute vs the FP32 HF reference:\n")
    print("| model | " + " | ".join(f"{c} max abs | {c} top-1" for c in CASES) + " |")
    print("|---|" + "---:|---:|" * len(CASES))
    for m in models:
        cells = []
        for c in CASES:
            r = data[m].get(f"bf16/{c}")
            cells += [f"{r['max_abs']:.2f}", f"{r['top1_agreement']:.1%}"] if r else ["", ""]
        print(f"| {m} | " + " | ".join(cells) + " |")

    print("\nPadding invariance (row of a right-padded batch vs the row alone), KV cache and greedy generation:\n")
    print("| model | padded batch max abs | long padded batch max abs | KV cache worst max abs | greedy 64 tokens |")
    print("|---|---:|---:|---:|---|")
    for m in models:
        d = data[m]
        pad = d.get("padding_invariance/padded", {}).get("max_abs")
        lpad = d.get("padding_invariance/long_padded", {}).get("max_abs")
        kv = [v["max_abs"] for k, v in d.items() if k.startswith("kv_cache/")]
        greedy = [v["match"] for k, v in d.items() if k.startswith("greedy/")]
        print(f"| {m} | {e(pad) if pad is not None else '-'} | {e(lpad) if lpad is not None else '-'} "
              f"| {e(max(kv)) if kv else '-'} | {f'{sum(greedy)}/{len(greedy)} exact' if greedy else '-'} |")

    print("\nFill-mask top-5 for 'The capital of France is [MASK].':\n")
    for m in models:
        if "fill_mask" in data[m]:
            print(f"- {m}: {', '.join(data[m]['fill_mask']['top5'])}")


if __name__ == "__main__":
    main()
