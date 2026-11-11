# Agent run log

Links under `log/` refer to local, Git-ignored evidence. Topic pages keep the checked-in
results and limitations; raw run artifacts are not part of a PR.

## 2026-11-10 — QTensor weight-only quantization for torch and jax backends

Implemented design (b) (packed weights in device memory, unpack-in-kernel GEMV)
for the torch and jax backends, mirroring the existing MLX support; MLX stays
on its native `mx.quantized_matmul` path. Topic page: `quantized-checkpoints.md`
(layout contract, per-backend lowering, JAX dead-end evidence, venv recipes).

Surface: runtime load kwarg `quantization={'bits','group_size','kernel'?}` only
— no new graph intrinsics, no compiler-layer changes. Canonical in-core layout:
`<path>.int8` uint8 packed (`(rows,K)` q8, `(rows,K//2)` q4, byte j = value 2j
low nibble + 2j+1 high nibble) + fp16 `<path>.scale`/`<path>.qbias`
`(rows, K//group)`. MLX uint32 checkpoints are unpacked with explicit shifts at
load (LSB-first value order confirmed against `mx.quantize`); W8A16 pairs are
requantized straight to the packed layout.

- torch: eager dequant matmul default; `kernel='triton'` opt-in runs the fused
  1-D decode GEMV (autotuned, correctness-gated configs) — validated end-to-end
  on RTX 2080 Ti through the generated core (`test_triton_decode_gemv_matches_eager`).
- jax: pure-jnp group-partial-sum lowering, fully jittable; batched-bias einsum
  fix (`'...a,ra->...r'`) found via the multi-group test shapes.
- tests: `tests/test_codegen2_torch_quantized_checkpoint.py` (15 passed, 1
  CUDA-gated skip path), `tests/test_codegen2_jax_quantized_checkpoint.py`
  (12 passed), multi-group shapes (DIM=128; groups 64/32) to avoid the DIM=64
  single-group blind spot; shared fixtures in `tests/quantized_fixture.py`.
- jax test env: isolated `/tmp/opencode/jax-venv` now also carries CPU-only
  torch (the `synapse` package imports torch unconditionally) + lark +
  safetensors + transformers.
- Probes archived under `log/`: fused Triton q8 278 GB/s / q4 180 GB/s vs JAX
  pure-jnp 112–139 GB/s packed (2.5–3.2x gap lives below the abstraction);
  JAX route eliminations (Pallas Ampere gate, jax-triton ptxas segfault,
  libnvjitlink refusal) recorded so nobody re-walks them.

## 2026-11-10 — QTensor end-to-end sanity: Qwen2.5-0.5B (follow-up)

Ran real-model batch-1 decode on Qwen2.5-0.5B (weights via
`ensure_model_downloaded`, cores from `synapse/models/qwen2/Qwen2.5-0.5B.axon`,
8-bit/group-64, greedy 64 tokens, RTX 2080 Ti). Result: quantized is
**slower** end to end despite winning the per-GEMV microbenchmarks — torch
0.49× (22 vs 46 tok/s), jax 0.83× (107 vs 130 tok/s); memory 988→525 MB.
Cause: 0.5B decode is launch/overhead-bound (dense effective ~45 GB/s, far
below bandwidth), and torch Triton GEMV carries ~35–45 µs fixed per-call
overhead vs the ~17 µs cuBLAS floor across 168 small calls/token. Crossover
≈ 8 M params per matrix. Full numbers + regime analysis in
`quantized-checkpoints.md` ("End-to-end reality check").

The e2e run also exposed a real dispatch gap: torch loop-inlined dynamic-key
linears bypassed quantized dispatch (inline `F.linear(self._param(...))`).
Fixed `_emit_linear_node` to route dynamic keys through `self._linear` (same
runtime-key dispatch the static path uses); torch quantized suite still 15
passed / 1 CUDA-gated skip. Do not transfer these ratios to MLX — its native
kernel has different dispatch overhead.

## 2026-10-27 — Timestral MLX comparison benchmark for the Axon paper

Added `scripts/timestral_bench/`, a benchmark-only harness that restores the NumPy
forecasting host removed in `cedc83d` and compares Axon-generated MLX (fused, unfused,
fused with opt-in pruning) with `vikiival/timestral-mlx` 0.1.0 and the original
`mistralai/timestral` on CPU (2 and 8 threads) over the four FP32 checkpoints.

Benchmark semantics: one fresh process per implementation, size and round; three rounds;
implementation and workload order rotated; CPU idle guard before each worker; two warmup
and seven timed complete forecasts per workload on nine fixed workloads; Axon adds a
generated-core replay on preloaded MLX arrays. Aggregation is the median of samples per
process, the median across rounds, ratios to `axon-fused` per workload and the geometric
mean over workloads. Agreement uses the `torch-2` output as reference on 23 cases per
checkpoint with the gate `abs(actual - reference) <= 2e-4 + 2e-4 * abs(reference)`.
MLX workers must not import PyTorch or `synapse`; PyTorch workers must not import MLX.

Result (`log/timestral-mlx-comparison-20261027/`): 72 timing processes,
4536 timed forecasts, 2268 core replays, 20 parity processes, all
accepted; 42 workers re-dispatched once under the >10% round-deviation rule, first pass kept under `timing-superseded/pass1/`. Geometric-mean latency ratio of vikiival MLX to Axon fused: 1.23×, 1.23×, 1.04×, 1.10× for
5M/28M/123M/209M; original PyTorch CPU at 2 threads 3.25×, 4.07×, 3.88×, 4.00×, at 8 threads 3.68×, 4.05×, 3.71×, 3.62×.
Unfused Axon 1.09×, 1.14×, 1.06×, 1.05×; pruned Axon 0.95×, 0.96×, 0.95×, 0.95×. Agreement with the PyTorch CPU
reference: Axon fused 21/23, 22/23, 20/23, 21/23; vikiival 21/23, 23/23, 20/23, 21/23;
PyTorch 8 threads 23/23, 23/23, 21/23, 22/23. 5 idle-guard timeouts and 15/216 cells
with more than 10% round spread are recorded on the topic page.

Uses [Timestral MLX comparison](timestral-mlx-comparison.md) and
[maintained scripts](scripts.md); validated-by `log/timestral-mlx-comparison-20261027/`
(`summary.json`, `comparison.csv`, `samples.csv`, per-worker JSON and logs) and the three
`tests/test_timestral_bench_*.py` files. Confidence: high for the recorded measurements;
timing applies to the recorded M1 Pro and conditions only. Last confirmed: 2026-10-27.
