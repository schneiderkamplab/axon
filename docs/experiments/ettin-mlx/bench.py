"""Latency / throughput benchmark for the MLX Ettin models.

    uv run python bench.py                      # ettin-encoder-150m + ettin-decoder-150m
    uv run python bench.py --encoder ettin-encoder-17m --decoder ettin-decoder-17m
    uv run python bench.py --quick              # smoke run: fewer cells and runs

Encoder: ``encode()`` (embeddings -> last hidden state) for batch {1, 8} x length
{128, 512, 2048, 8192}, FP32 and BF16, eager and ``mx.compile``. The MLM vocab
projection is left out: at 8 x 8192 tokens its logits alone are 13 GB in FP32.
Decoder (batch 1): prefill latency at {128, 512, 2048} (one pass that fills the KV cache
and returns last-token logits) and decode throughput over 128 new greedy tokens, both
synchronous (``mx.eval`` every step) and pipelined (``generate_step``'s ``async_eval``).

Every timed call ends in ``mx.eval``. Each cell: warmup runs, then timed runs; median,
p10, p90 and peak memory (``mx.get_peak_memory``, reset per cell; includes weights).
Results go to ``<out>/bench.json`` and ``<out>/bench.md``.
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten

from ettin_mlx import generate_step, load_model
from ettin_mlx.revisions import PINNED_REVISIONS

ENC_BATCHES = [1, 8]
ENC_LENGTHS = [128, 512, 2048, 8192]
PREFILL_LENGTHS = [128, 512, 2048]
DECODE_PROMPT = 32
DECODE_TOKENS = 128


def system_info() -> dict:
    def sysctl(key):
        return subprocess.run(["sysctl", "-n", key], capture_output=True, text=True).stdout.strip()

    return {
        "chip": sysctl("machdep.cpu.brand_string"),
        "memory_gb": int(sysctl("hw.memsize")) / 2**30,
        "macos": platform.mac_ver()[0],
        "python": platform.python_version(),
        "mlx": mx.__version__,
        "gpu": mx.device_info().get("architecture"),
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def measure(fn, warmup: int, runs: int) -> dict:
    mx.synchronize()
    mx.clear_cache()
    mx.reset_peak_memory()
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    t = np.array(times) * 1e3
    return {
        "median_ms": float(np.median(t)), "p10_ms": float(np.percentile(t, 10)), "p90_ms": float(np.percentile(t, 90)),
        "runs": runs, "peak_memory_mb": mx.get_peak_memory() / 2**20,
    }


def weights_mb(model) -> float:
    return sum(v.nbytes for _, v in tree_flatten(model.parameters())) / 2**20


def bench_encoder(name: str, warmup: int, runs: int, batches, lengths, revisions: dict) -> list[dict]:
    rows = []
    rng = np.random.default_rng(0)
    for dtype in ["float32", "bfloat16"]:
        model = load_model(name, dtype=dtype)
        revisions[name] = model.revision
        w_mb = weights_mb(model)
        for compiled in [False, True]:
            if compiled:
                model.compile()
            for B in batches:
                for L in lengths:
                    ids = mx.array(rng.integers(1000, 50000, (B, L)).astype(np.int32))
                    stats = measure(lambda: mx.eval(model.encode(ids)), warmup, runs)
                    rows.append({"model": name, "dtype": dtype, "compile": compiled, "batch": B, "length": L,
                                 "tokens_per_s": B * L / stats["median_ms"] * 1e3, "weights_mb": w_mb, **stats})
                    print(f"  enc {dtype:8s} compile={compiled!s:5s} B={B} L={L:5d}: "
                          f"{stats['median_ms']:9.2f} ms  peak {stats['peak_memory_mb']:8.0f} MB", flush=True)
        del model  # free the weights before the next model so they don't inflate later peaks
        gc.collect()
        mx.clear_cache()
    return rows


def bench_decoder(name: str, warmup: int, runs: int, lengths, revisions: dict) -> list[dict]:
    rows = []
    rng = np.random.default_rng(1)
    for dtype in ["float32", "bfloat16"]:
        model = load_model(name, dtype=dtype)
        revisions[name] = model.revision
        w_mb = weights_mb(model)
        for L in lengths:
            ids = mx.array(rng.integers(1000, 50000, (1, L)).astype(np.int32))

            def prefill():
                mx.eval(model(ids, cache=model.make_cache(), logits_to_keep=1))

            stats = measure(prefill, warmup, runs)
            rows.append({"model": name, "dtype": dtype, "kind": "prefill", "batch": 1, "length": L,
                         "tokens_per_s": L / stats["median_ms"] * 1e3, "weights_mb": w_mb, **stats})
            print(f"  dec {dtype:8s} prefill L={L:5d}: {stats['median_ms']:9.2f} ms  "
                  f"peak {stats['peak_memory_mb']:8.0f} MB", flush=True)

        prompt = mx.array(rng.integers(1000, 50000, (1, DECODE_PROMPT)).astype(np.int32))

        def decode_sync():
            cache = model.make_cache()
            token = mx.argmax(model(prompt, cache=cache, logits_to_keep=1)[:, -1], axis=-1)
            mx.eval(token)
            start = time.perf_counter()
            for _ in range(DECODE_TOKENS):
                token = mx.argmax(model(token[:, None], cache=cache)[:, -1], axis=-1)
                mx.eval(token)
            return time.perf_counter() - start

        def decode_pipelined():
            steps = generate_step(model, prompt, DECODE_TOKENS + 1)
            mx.eval(next(steps))  # prefill + first token, not timed
            start = time.perf_counter()
            for token in steps:
                mx.eval(token)
            return time.perf_counter() - start

        for kind, fn in [("decode_sync", decode_sync), ("decode_pipelined", decode_pipelined)]:
            mx.synchronize()
            mx.clear_cache()
            mx.reset_peak_memory()
            for _ in range(warmup):
                fn()
            tps = np.array([DECODE_TOKENS / fn() for _ in range(runs)])
            rows.append({"model": name, "dtype": dtype, "kind": kind, "batch": 1, "length": DECODE_PROMPT,
                         "new_tokens": DECODE_TOKENS, "tokens_per_s": float(np.median(tps)),
                         "p10_tokens_per_s": float(np.percentile(tps, 10)),
                         "p90_tokens_per_s": float(np.percentile(tps, 90)),
                         "median_ms_per_token": 1e3 / float(np.median(tps)), "runs": runs,
                         "peak_memory_mb": mx.get_peak_memory() / 2**20, "weights_mb": w_mb})
            print(f"  dec {dtype:8s} {kind}: {np.median(tps):8.1f} tok/s", flush=True)
        del model  # free the weights before the next model so they don't inflate later peaks
        gc.collect()
        mx.clear_cache()
    return rows


def markdown(result: dict) -> str:
    info = result["system"]
    lines = [
        f"System: {info['chip']}, {info['memory_gb']:.0f} GB, macOS {info['macos']}, Python {info['python']}, "
        f"MLX {info['mlx']} ({info['date']})",
        "",
        "Revisions: " + ", ".join(f"`{m}@{r[:12]}`" for m, r in result["revisions"].items()),
        "",
    ]
    enc = result["encoder"]
    if enc:
        lines += [f"Encoder `{enc[0]['model']}`, `encode()` latency in ms (median [p10, p90]), peak memory in MB:", "",
                  "| batch | length | dtype | eager | compiled | compile speedup | peak MB (eager) |",
                  "|---:|---:|---|---:|---:|---:|---:|"]
        key = {(r["dtype"], r["compile"], r["batch"], r["length"]): r for r in enc}
        for dtype in ["float32", "bfloat16"]:
            for B in sorted({r["batch"] for r in enc}):
                for L in sorted({r["length"] for r in enc}):
                    e, c = key.get((dtype, False, B, L)), key.get((dtype, True, B, L))
                    if not e or not c:
                        continue
                    lines.append(
                        f"| {B} | {L} | {dtype} | {e['median_ms']:.2f} [{e['p10_ms']:.2f}, {e['p90_ms']:.2f}] "
                        f"| {c['median_ms']:.2f} [{c['p10_ms']:.2f}, {c['p90_ms']:.2f}] "
                        f"| {e['median_ms'] / c['median_ms']:.2f}x | {e['peak_memory_mb']:.0f} |")
        lines.append("")
    dec = result["decoder"]
    if dec:
        lines += [f"Decoder `{dec[0]['model']}`, batch 1:", "",
                  "| measurement | dtype | median | p10 | p90 | peak MB |", "|---|---|---:|---:|---:|---:|"]
        for r in dec:
            if r["kind"] == "prefill":
                lines.append(f"| prefill {r['length']} tokens (ms) | {r['dtype']} | {r['median_ms']:.2f} "
                             f"| {r['p10_ms']:.2f} | {r['p90_ms']:.2f} | {r['peak_memory_mb']:.0f} |")
            else:
                lines.append(f"| {r['kind'].replace('_', ' ')}, {r['new_tokens']} new tokens (tok/s) | {r['dtype']} "
                             f"| {r['tokens_per_s']:.1f} | {r['p10_tokens_per_s']:.1f} | {r['p90_tokens_per_s']:.1f} "
                             f"| {r['peak_memory_mb']:.0f} |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--encoder", default="ettin-encoder-150m")
    parser.add_argument("--decoder", default="ettin-decoder-150m")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--runs", type=int, default=7)
    parser.add_argument("--quick", action="store_true", help="1 warmup, 3 runs, lengths <= 512")
    parser.add_argument("--out", type=Path, default=Path("bench_results"))
    args = parser.parse_args()

    warmup, runs = (1, 3) if args.quick else (args.warmup, args.runs)
    enc_lengths = [n for n in ENC_LENGTHS if not args.quick or n <= 512]
    pre_lengths = [n for n in PREFILL_LENGTHS if not args.quick or n <= 512]

    result = {"system": system_info(), "revisions": {}, "protocol": {"warmup": warmup, "runs": runs}}
    print(json.dumps(result["system"]))
    revs = result["revisions"]
    result["encoder"] = bench_encoder(args.encoder, warmup, runs, ENC_BATCHES, enc_lengths, revs) if args.encoder else []
    result["decoder"] = bench_decoder(args.decoder, warmup, runs, pre_lengths, revs) if args.decoder else []
    result["pinned"] = {n: PINNED_REVISIONS.get(n.split("/")[-1]) for n in revs}

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "bench.json").write_text(json.dumps(result, indent=1))
    (args.out / "bench.md").write_text(markdown(result))
    print(f"wrote {args.out / 'bench.json'} and {args.out / 'bench.md'}")


if __name__ == "__main__":
    main()
