# Agent run log

Links under `log/` refer to local, Git-ignored evidence. Topic pages keep the checked-in
results and limitations; raw run artifacts are not part of a PR.

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
