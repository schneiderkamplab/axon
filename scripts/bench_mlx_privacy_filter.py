#!/usr/bin/env python3
"""Benchmark the privacy-filter model on MLX: original vs optimized codegen.

Generates MLX model files from the privacy-filter .axon source:
  - v0: original (gather-based MoE, manual banded attention)
  - v1: sort+concat MoE only (no SDPA)
  - v3: fully optimized (sort+concat + SDPA + all intrinsics)

Each variant is benchmarked in a **separate subprocess** so only one model
is in memory at a time. Throughput is measured from actual output shape[1].

Usage:
    python scripts/bench_mlx_privacy_filter.py [--seq-len 128 256 512] [--duration 3]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

MODEL_PATH = Path("synapse/models/privacy-filter/generic-privacy-filter.axon")

NUM_LAYERS = 8
D = 640
NUM_HEADS = 14
KVH = 2
HEAD_DIM = 64
VOCAB = 200064
NUM_LABELS = 33
E = 128
FFN = 640
QD = NUM_HEADS * HEAD_DIM
KVD = KVH * HEAD_DIM
QKV_DIM = QD + 2 * KVD

CONFIG = {
    "VOCAB": VOCAB,
    "NUM_LABELS": NUM_LABELS,
    "NUM_LAYERS": NUM_LAYERS,
    "NUM_HEADS": NUM_HEADS,
    "KVH": KVH,
    "D": D,
    "FFN": FFN,
    "E": E,
    "EPT": 4,
    "HEAD_DIM": HEAD_DIM,
    "WINDOW": 64,
    "BAND": 64,
    "LEFT_CTX": 128,
    "RIGHT_CTX": 128,
    "THETA": 150000.0,
    "ROPE_SCALE": 32.0,
    "ROPE_LOW_FREQ": 1.0,
    "ROPE_HIGH_FREQ": 32.0,
    "ROPE_CONTEXT": 4096,
    "ROPE_ATTN_FACTOR": 1.1070552256,
    "ATTN_SCALE": 1.0 / (HEAD_DIM**0.5),
    "EPS": 1e-5,
    "SWIGLU_ALPHA": 1.702,
    "SWIGLU_LIMIT": 7.0,
    "LN2": 0.6931471805599453,
    "QD": QD,
    "KVD": KVD,
    "MODEL_DIM": D,
}

# Commit hash for the parent of the sort+concat change (true original codegen)
ORIGINAL_CODEGEN_COMMIT = "b63b8d2~1"


# ---------------------------------------------------------------------------
# Phase 1: Generate model .py files (done in the parent process)
# ---------------------------------------------------------------------------

def _generate_model_code(out_path: Path, *, optimize: bool, original_codegen: bool = False) -> None:
    """Generate MLX model code and write to out_path.

    optimize:         if True, run graph optimizer (SDPA, SwiGLU, etc.)
    original_codegen: if True, temporarily revert core.py to the pre-sort+concat
                      version so we get the gather-based _expert_linear.
    """
    if original_codegen:
        import importlib
        core_path = Path("synapse/axon/codegen2_mlx/core.py")
        saved = core_path.read_text()
        subprocess.run(
            ["git", "checkout", ORIGINAL_CODEGEN_COMMIT, "--", str(core_path)],
            capture_output=True, check=True,
            cwd=REPO_ROOT,
        )
        import synapse.axon.codegen2_mlx.core as _mlx_core_mod
        importlib.reload(_mlx_core_mod)
        _emit = _mlx_core_mod.emit_model_code_from_graph_ir
    else:
        from synapse.axon.codegen2_mlx.core import emit_model_code_from_graph_ir as _emit

    try:
        from synapse.axon import (
            resolve_axon_program_from_path,
            normalize_closed_axon_file,
            elaborate_closed_axon_file,
            flatten_closed_axon_file,
            typecheck2_flat_axon_file,
            lower_axon_program_to_graph_ir,
            optimize_graph_program,
            GraphOptimizeConfig,
        )

        report = resolve_axon_program_from_path(MODEL_PATH, strict=False)
        program = report.ast
        program = normalize_closed_axon_file(program)
        program = elaborate_closed_axon_file(program)
        program = flatten_closed_axon_file(program)
        program = typecheck2_flat_axon_file(program)
        graph = lower_axon_program_to_graph_ir(program)
        if optimize:
            graph = optimize_graph_program(
                graph, config=GraphOptimizeConfig(backend_intrinsics="codegen2-mlx")
            )
        code = _emit(graph)
        out_path.write_text(code)
    finally:
        if original_codegen:
            core_path = Path("synapse/axon/codegen2_mlx/core.py")
            core_path.write_text(saved)
            importlib.reload(_mlx_core_mod)


# ---------------------------------------------------------------------------
# Phase 2: Benchmark a single model variant in a subprocess
# ---------------------------------------------------------------------------

_WORKER_SCRIPT = r'''
import importlib.util, json, sys, time
from pathlib import Path

import mlx.core as mx

model_path = Path(sys.argv[1])
seq_len = int(sys.argv[2])
duration = float(sys.argv[3])
config = json.loads(sys.argv[4])

# Patch nn.Module.__setattr__ bug
spec = importlib.util.spec_from_file_location("model", str(model_path))
mod = importlib.util.module_from_spec(spec)
sys.modules["model"] = mod
spec.loader.exec_module(mod)

model_cls = getattr(mod, "AxonMlxModel", None) or getattr(mod, "AxonGeneratedModel", None)
_orig = model_cls.__setattr__
def _patched(self, key, value):
    try: _orig(self, key, value)
    except AttributeError: object.__setattr__(self, key, value)
model_cls.__setattr__ = _patched

# Build random weights
L=8;D=640;H=14;KVH=2;HD=64;V=200064;NL=33;E=128;FFN=640;QD=H*HD;KVD=KVH*HD;QKV=QD+2*KVD
def rand(*s): return (mx.random.normal(s)*0.02).astype(mx.bfloat16)
sd={}
sd["embedding.weight"]=rand(V,D)
sd["norm.scale"]=rand(D)
sd["unembedding.weight"]=rand(NL,D)
for n in range(L):
    p=f"block.{n}"
    sd[f"{p}.attn.norm.scale"]=rand(D)
    sd[f"{p}.attn.qkv.weight"]=rand(QKV,D)
    sd[f"{p}.attn.qkv.bias"]=rand(QKV)
    sd[f"{p}.attn.out.weight"]=rand(D,QD)
    sd[f"{p}.attn.out.bias"]=rand(D)
    sd[f"{p}.attn.sinks"]=rand(H)
    sd[f"{p}.mlp.norm.scale"]=rand(D)
    sd[f"{p}.mlp.gate.weight"]=rand(E,D)
    sd[f"{p}.mlp.gate.bias"]=rand(E)
    sd[f"{p}.mlp.swiglu.weight"]=rand(E,D,2*FFN)
    sd[f"{p}.mlp.swiglu.bias"]=rand(E,2*FFN)
    sd[f"{p}.mlp.out.weight"]=rand(E,FFN,D)
    sd[f"{p}.mlp.out.bias"]=rand(E,D)

model = model_cls(sd, config=config, target_dtype="bfloat16")
input_ids = mx.array([[i % 1000 for i in range(seq_len)]], dtype=mx.int32)

# Verify output shape
out = model(input_ids)
mx.eval(out)
assert out.shape[1] == seq_len, f"Expected {seq_len}, got {out.shape[1]}"

# Warmup
for _ in range(3):
    out = model(input_ids)
    mx.eval(out)
mx.clear_cache()

# Sustained throughput
total_tokens = 0
iters = 0
t_start = time.perf_counter()
while time.perf_counter() - t_start < duration:
    out = model(input_ids)
    mx.eval(out)
    total_tokens += out.shape[1]
    iters += 1
elapsed = time.perf_counter() - t_start
tps = total_tokens / elapsed

result = {"tps": tps, "iters": iters, "total_tokens": total_tokens, "elapsed": elapsed,
          "out_shape": list(out.shape)}
print(json.dumps(result))
'''


def _run_subprocess(model_path: Path, seq_len: int, duration: float) -> dict:
    """Run a single benchmark variant in a subprocess."""
    proc = subprocess.run(
        [sys.executable, "-c", _WORKER_SCRIPT,
         str(model_path), str(seq_len), str(duration),
         json.dumps(CONFIG)],
        capture_output=True, text=True, timeout=300,
        cwd=REPO_ROOT,
    )
    if proc.returncode != 0:
        print(f"  ERROR: {proc.stderr.strip()[:200]}", file=sys.stderr)
        return {"tps": 0, "iters": 0, "total_tokens": 0, "elapsed": 0, "out_shape": []}
    return json.loads(proc.stdout.strip().splitlines()[-1])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seq-len", type=int, nargs="+", default=[128, 256, 512],
        help="Sequence lengths to benchmark",
    )
    parser.add_argument(
        "--duration", type=float, default=3.0,
        help="Wall-clock seconds per measurement",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("/tmp"),
        help="Where to write generated model .py files",
    )
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    v0_path = out_dir / "pf_mlx_v0.py"
    v1_path = out_dir / "pf_mlx_v1.py"
    v3_path = out_dir / "pf_mlx_v3.py"

    print("Generating v0 (original: gather MoE, manual attention)...")
    _generate_model_code(v0_path, optimize=False, original_codegen=True)
    print(f"  -> {v0_path}")

    print("Generating v1 (sort+concat MoE, no SDPA)...")
    _generate_model_code(v1_path, optimize=False)
    print(f"  -> {v1_path}")

    print("Generating v3 (optimized: sort+concat + SDPA + intrinsics)...")
    _generate_model_code(v3_path, optimize=True)
    print(f"  -> {v3_path}")

    # Header
    print(f"\n{'Seq':>5} | {'v0 orig':>10} | {'v1 s+c':>10} | {'v3 opt':>10} | {'v3/v0':>6} | {'v3/v1':>6}")
    print("-" * 68)

    results = {}
    for seq_len in args.seq_len:
        row = {}
        for label, path in [("v0", v0_path), ("v1", v1_path), ("v3", v3_path)]:
            print(f"  Benchmarking {label} S={seq_len}...", end=" ", flush=True)
            r = _run_subprocess(path, seq_len, args.duration)
            row[label] = r
            print(f"{r['tps']:.0f} tok/s ({r['iters']} iters, {r['total_tokens']} tokens)")
        results[seq_len] = row
        tps0 = row["v0"]["tps"]
        tps1 = row["v1"]["tps"]
        tps3 = row["v3"]["tps"]
        print(
            f"{seq_len:>5} | {tps0:>10.0f} | {tps1:>10.0f} | {tps3:>10.0f} | "
            f"{tps3/tps0:>5.1f}x | {tps3/tps1:>5.1f}x"
        )

    print("\nv0 = gather MoE + manual banded attention (original codegen)")
    print("v1 = sort+concat MoE + manual banded attention (codegen change only)")
    print("v3 = sort+concat MoE + SDPA + all intrinsics (full optimization)")
    print("\nTokens measured from output shape[1] per forward pass.")
    print(f"Each run lasted ~{args.duration:.0f}s (wall clock, each variant in separate process).")


REFERENCE_RESULTS = """
  Seq |    v0 orig |     v1 s+c |     v3 opt |  v3/v0 |  v3/v1
--------------------------------------------------------------------
  128 |       1328 |       5260 |       6809 |   5.1x |   1.3x
  512 |       1190 |       7576 |      13526 |  11.4x |   1.8x

v0 = gather MoE + manual banded attention (original codegen)
v1 = sort+concat MoE + manual banded attention (codegen change only)
v3 = sort+concat MoE + SDPA + all intrinsics (full optimization)

Tokens measured from output shape[1] per forward pass.
Each run lasted ~3s (wall clock, each variant in separate process).

Results:
- v0 -> v3: 5.1x at 128 tokens, 11.4x at 512 tokens
- Sort+concat (v0->v1) is the big win: 4.0x / 6.4x
- SDPA fusion (v1->v3) adds 1.3x / 1.8x on top

Environment: Apple M3 Max, MLX 0.32.2, Sep 2026.
"""


if __name__ == "__main__":
    main()
