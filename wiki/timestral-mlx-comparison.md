# Timestral on Apple Silicon: Axon-generated MLX vs vikiival MLX vs original PyTorch

Status: measured. Last confirmed: 2026-10-27. Owner: Axon contributors/agents.
Confidence: high for the recorded measurements and agreement results on this machine;
timing conclusions are limited to the recorded hardware, package versions and desktop
conditions. Evidence: `log/timestral-mlx-comparison-20261027/` (Git-ignored run
artifacts; the checked-in tables below are derived from its `summary.json`).

On the measured M1 Pro, the Axon-generated MLX forecaster (`axon-fused`) has lower
complete-forecast latency than the vikiival MLX port on **30 of 36** size/workload
cells, with geometric-mean ratios of **1.23× (5M), 1.23× (28M), 1.04× (123M) and
1.10× (209M)**; vikiival is faster on 6 cells: batch-32 at 5M/123M/209M, multivariate 8×4 at 5M/123M and max-context at 123M. Against the
original PyTorch package on CPU, Axon MLX is **3.3–4.1× faster at 2 threads and
3.6–4.0× at 8 threads**, a cross-backend comparison. Within Axon, leaving graph fusion off raises
latency by 5–14%, and opt-in final-block pruning lowers it by a further
4–5%; both effects are smaller than the round-to-round spread of individual cells and rest
on per-round geometric means, which keep the fusion direction in every round at every size and the pruning
direction in every round except round 2 at 5M (1.019×). Agreement failures
against the PyTorch CPU reference concentrate in two case classes, the multi-block workload and
the covariate workload with missing values: Axon fused misses 8 of 92 cases over the four
sizes and vikiival 7; the two differ: at 5M Axon fused misses the multi-block workload that vikiival passes, while vikiival misses `case_past_known` that Axon fused passes; at 28M Axon fused misses the covariate workload that vikiival passes. The PyTorch reference itself misses 3 of those cases when run with 8 instead of 2
threads at 123M and 209M.

## Scope and claims

Three implementations of Mistral's Timestral zero-shot forecaster run the four released
FP32 checkpoints (5M, 28M, 123M, 209M) on the same inputs:

- **Axon MLX**: the neural core in
  [`generic-timestral.axon`](../synapse/models/timestral/generic-timestral.axon) compiled
  by Axon to MLX, with the NumPy forecasting host in `scripts/timestral_bench/`.
  `axon-fused` (SDPA and RMSNorm-scale backend intrinsics) is the reported configuration;
  `axon-unfused` (no intrinsics) and `axon-fused-pruned` (`prune_final_block=True`) are
  ablations.
- **vikiival MLX**: the community port `timestral-mlx` 0.1.0, run with its shipped defaults.
- **Original PyTorch**: Mistral's `timestral` 1.0.0 package loading the Hub weights, on
  CPU with 2 and with 8 intra-op threads. There is no `transformers` integration for
  Timestral; "HF" in paper drafts means this package with Hub-hosted weights.

What each comparison isolates:

- `axon-fused` vs `axon-unfused` and `axon-fused-pruned`: compiler-level effects on the
  same backend, host and weights (graph fusion; opt-in final-block pruning).
- `axon-fused` vs `vikiival`: two implementations on the same backend (Metal through MLX)
  with the same weights; hosts and batching defaults differ (Axon 128 rows / 4,096 tokens,
  vikiival 64 rows / 65,536 tokens, each as shipped).
- `axon-fused` vs `torch-2`/`torch-8`: different backend and implementation. The original
  runs on CPU because its float64 scaling and refinement path is unsupported on MPS, and
  upstream was not patched. These ratios are not an Axon compiler gain.

Agreement is measured against the original PyTorch CPU output; it is not an evaluation of
forecast accuracy on real data.

## Evidence and protocol

Hardware and software: Apple M1 Pro (10 CPU cores, 16 GPU cores, 32 GB unified memory),
macOS 27.0.1, Python 3.13.9, MLX 0.32.2, PyTorch 2.14.0, NumPy 2.5.3, safetensors 0.8.0,
huggingface_hub 1.33.0, `timestral` 1.0.0, `timestral-mlx` 0.1.0. All checkpoints stay in
FP32. AC power, `caffeinate -i`, lid open; other foreground apps closed, the coding agent
and terminal remained active.

Workloads (float32 random walks with fixed seeds, identical arrays for every
implementation; `B×V, context → horizon`):

| Name | Shape | Purpose |
|---|---|---|
| `short_256_96` | 1×1, 256 → 96 | latency floor, host overhead |
| `batch32_2048_96` | 32×1, 2048 → 96 | batching and chunking path |
| `max_context_32768_512` | 1×1, 32768 → 512 | maximum context |
| `multiblock_2000_2500` | 1×1, 2000 → 2500 | three autoregressive blocks with refinement |
| `covariates_missing_2000_1500` | 1×1, 2000 → 1500; 10% missing; 2 past + 2 known covariates (one binary) | two-stream masks, known-covariate normalization |
| `multivariate_8x4_1024_192` | 8×4, 1024 → 192 | grouped variate attention |
| `short_64_24` | 1×1, 64 → 24 | shortest series |
| `long_8192_1024` | 1×1, 8192 → 1024 | long series, one full block |
| `batch8_512_336` | 8×1, 512 → 336 | mid batch, unaligned horizon |

Timing protocol:

- One fresh Python process per implementation, size and round; three rounds; workers run
  sequentially. Implementation order rotates per round and size, and each worker rotates
  its workload order, so drift is spread across conditions.
- Before each worker the orchestrator waits for the CPU to be at least 85% idle for three
  consecutive one-second samples (120 s cap) and records what it saw. This is a contention
  check, not proof of an idle machine.
- Per workload: two warmup calls (the first also recorded as the first-call latency, which
  includes `mx.compile` tracing for a new shape), then seven timed calls. A timed call starts
  with NumPy inputs and ends with completed float32 NumPy quantiles; the Axon adapter also
  calls `mx.synchronize()` before the timer stops.
- Axon only: the generated-core replay reruns the recorded network calls of one forecast
  on preloaded MLX arrays with `mx.eval` per call. It excludes host normalization, mask
  construction, transfers and output conversion, and separates network time from host time.
- MLX workers never import PyTorch or `synapse`; PyTorch workers never import MLX; every
  worker asserts this before writing its results. PyTorch workers set the intra-op thread
  count explicitly and one inter-op thread; `OMP_NUM_THREADS` is 2 for the NumPy hosts.
- Re-dispatch rule: after the first pass, any worker whose per-process median for some workload
  deviates by more than 10% from the mean of the other two rounds' medians is run again once
  with identical settings; its first result is moved to `timing-superseded/pass1/` and the
  flagged list is recorded in `rerun-pass1.json`. The replacement is accepted as measured.
- Aggregation: median of the seven samples per process, then the median of the three
  process medians. Ratios divide an implementation's median by the `axon-fused` median per
  workload; the per-size figure is the geometric mean over the nine workloads, with the
  min–max of per-round geometric means in parentheses. Those ranges are not confidence
  intervals. Loading and first calls are excluded from steady-state figures. Peak MLX memory
  and process RSS are recorded as information only.

Agreement protocol: the `torch-2` output is the reference. 23 cases per checkpoint: the
nine workload inputs plus 14 synthetic cases (univariate and multivariate series with
missing values, positive-only, all-missing, constant, large-offset, single-step, multi-block,
context cap with observation mask, past and known covariates, categorical covariates). A
case passes when shapes match, all quantiles are finite and non-decreasing, and
`abs(actual - reference) <= 2e-4 + 2e-4 * abs(reference)` everywhere. `torch-8` is compared
the same way to show the reference's own thread-dependent spread.

## Results

72 timing processes, 4536 timed complete forecasts and 2268 generated-core replays; 20 parity processes. Seven timed samples and two warmup calls per workload per process. Every MLX worker ran without PyTorch and every PyTorch worker without MLX.

### Latency relative to Axon MLX (fused)

Geometric mean over workloads of implementation median latency divided by the Axon fused median latency. Values above 1 mean the implementation is slower than Axon. Parentheses give the min–max of per-round geometric means, not confidence intervals.

| Size | vikiival MLX | original PyTorch CPU, 2 threads | original PyTorch CPU, 8 threads | Axon MLX (unfused) | Axon MLX (fused, pruned) |
|---|---:|---:|---:|---:|---:|
| 5M | 1.229× (1.151–1.223) | 3.252× (2.969–3.305) | 3.676× (3.345–3.856) | 1.088× (1.005–1.110) | 0.946× (0.868–1.019) |
| 28M | 1.229× (1.183–1.242) | 4.073× (4.065–4.178) | 4.045× (4.016–4.107) | 1.143× (1.103–1.162) | 0.957× (0.946–0.964) |
| 123M | 1.036× (1.024–1.114) | 3.885× (3.832–4.081) | 3.709× (3.566–3.770) | 1.057× (1.050–1.067) | 0.948× (0.920–0.966) |
| 209M | 1.098× (1.097–1.110) | 3.995× (3.993–4.011) | 3.620× (3.467–3.690) | 1.053× (1.006–1.075) | 0.947× (0.925–0.967) |

Workloads where an implementation was faster than Axon fused:
- vikiival MLX at 5M: batch32_2048_96, multivariate_8x4_1024_192
- vikiival MLX at 123M: batch32_2048_96, max_context_32768_512, multivariate_8x4_1024_192
- vikiival MLX at 209M: batch32_2048_96
- Axon MLX (unfused) at 5M: multivariate_8x4_1024_192
- Axon MLX (unfused) at 123M: multivariate_8x4_1024_192
- Axon MLX (unfused) at 209M: multivariate_8x4_1024_192
- Axon MLX (fused, pruned) at 5M: batch32_2048_96, batch8_512_336, covariates_missing_2000_1500, long_8192_1024, max_context_32768_512, multiblock_2000_2500, multivariate_8x4_1024_192, short_256_96, short_64_24
- Axon MLX (fused, pruned) at 28M: batch32_2048_96, batch8_512_336, covariates_missing_2000_1500, long_8192_1024, max_context_32768_512, multiblock_2000_2500, multivariate_8x4_1024_192, short_256_96, short_64_24
- Axon MLX (fused, pruned) at 123M: batch32_2048_96, batch8_512_336, covariates_missing_2000_1500, long_8192_1024, max_context_32768_512, multiblock_2000_2500, multivariate_8x4_1024_192, short_256_96, short_64_24
- Axon MLX (fused, pruned) at 209M: batch32_2048_96, batch8_512_336, covariates_missing_2000_1500, long_8192_1024, max_context_32768_512, multiblock_2000_2500, multivariate_8x4_1024_192, short_256_96, short_64_24

Reading the table: the vikiival ratios above 1 mean the Axon forecaster is faster on the
geometric mean; vikiival is faster on 6 cells: batch-32 at 5M/123M/209M, multivariate 8×4 at 5M/123M and max-context at 123M. Batch-32 is the recurring one, where vikiival's larger per-call token budget (65,536 row-tokens
against Axon's 4,096) needs fewer network calls.
The `axon-unfused` column is the cost of leaving the SDPA and RMSNorm-scale intrinsics off
(5–14% higher latency, so fused is 5–12% lower); the `axon-fused-pruned` column is the opt-in
final-block pruning gain (4–5% lower latency), which is not the default because it changes
floating-point results. Both within-Axon effects are smaller than the spread between rounds of
many individual cells (up to 1.69× between the fastest and slowest round median of one cell,
see Run stability), so they rest on the per-round geometric means rather than on any single cell.
The vikiival and PyTorch ratios stay above 1 in every round at every size; the
unfused ratio stays above 1 in every round at every size, and the pruned ratio stays below 1 in every round except round 2 at 5M (1.019×).
The headline ratio uses the median across rounds for each workload, so it can lie slightly
outside the range of the per-round geometric means.

### Representative complete-forecast latencies

Milliseconds; median of per-process medians with the min–max across processes.

| Size | Workload | Axon MLX (fused) | vikiival MLX | original PyTorch CPU, 2 threads | original PyTorch CPU, 8 threads | Axon MLX (unfused) | Axon MLX (fused, pruned) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 5M | short_256_96 | 2.80 (2.80–3.10) | 5.15 (4.79–5.64) | 6.98 (6.72–7.09) | 8.44 (8.43–8.92) | 3.02 (2.95–3.49) | 2.78 (2.77–2.91) |
| 5M | batch32_2048_96 | 34.32 (33.94–38.05) | 30.88 (30.45–31.18) | 117.90 (113.80–118.01) | 107.39 (106.52–110.75) | 36.17 (35.58–36.27) | 32.20 (32.03–33.46) |
| 5M | max_context_32768_512 | 24.16 (23.13–26.39) | 24.62 (24.21–24.69) | 108.29 (105.45–108.70) | 88.15 (86.95–90.19) | 30.69 (30.38–31.32) | 21.53 (21.46–21.81) |
| 28M | short_256_96 | 4.68 (4.68–4.79) | 7.02 (6.97–7.22) | 22.13 (22.10–22.88) | 24.18 (24.01–25.23) | 5.17 (5.11–5.34) | 4.61 (4.59–4.76) |
| 28M | batch32_2048_96 | 102.10 (101.95–106.68) | 105.47 (103.43–105.90) | 346.14 (343.34–354.43) | 262.82 (262.01–278.85) | 113.14 (112.87–115.77) | 96.09 (95.84–100.66) |
| 28M | max_context_32768_512 | 66.77 (66.48–69.26) | 74.41 (74.41–75.16) | 307.17 (307.04–331.73) | 200.56 (198.96–212.30) | 90.73 (90.36–90.95) | 62.14 (62.14–65.22) |
| 123M | short_256_96 | 11.10 (10.77–11.25) | 13.20 (13.20–13.91) | 75.45 (73.19–79.29) | 83.29 (78.68–85.53) | 11.23 (11.21–11.48) | 10.71 (10.60–10.81) |
| 123M | batch32_2048_96 | 393.24 (388.49–399.65) | 355.33 (355.22–377.64) | 1089.61 (1087.58–1118.62) | 840.56 (810.59–841.75) | 404.63 (400.67–415.01) | 365.86 (354.21–369.85) |
| 123M | max_context_32768_512 | 235.77 (235.60–236.21) | 229.01 (228.90–256.43) | 890.47 (889.37–965.76) | 594.14 (572.31–599.57) | 283.96 (282.97–288.66) | 223.43 (210.17–223.98) |
| 209M | short_256_96 | 15.42 (15.33–15.89) | 19.24 (18.95–19.56) | 115.10 (114.48–118.21) | 119.55 (114.50–120.64) | 15.65 (15.47–16.03) | 15.13 (15.08–15.35) |
| 209M | batch32_2048_96 | 603.06 (599.78–603.95) | 579.01 (576.32–586.70) | 1744.63 (1720.13–1747.58) | 1290.26 (1244.68–1304.21) | 642.80 (612.71–650.37) | 577.58 (574.94–580.85) |
| 209M | max_context_32768_512 | 372.40 (364.52–374.71) | 404.63 (401.60–404.93) | 1361.02 (1359.47–1364.10) | 903.84 (865.45–910.72) | 425.81 (424.10–428.00) | 352.25 (334.63–357.24) |

### Host share of Axon latency

Complete-forecast latency divided by generated-core replay latency, geometric mean over workloads. The replay reruns the recorded network calls on preloaded MLX arrays and excludes host normalization, mask construction, transfers and output conversion.

| Size | Axon MLX (fused) | Axon MLX (unfused) | Axon MLX (fused, pruned) |
|---|---:|---:|---:|
| 5M | 1.454× | 1.414× | 1.488× |
| 28M | 1.150× | 1.158× | 1.167× |
| 123M | 1.047× | 1.039× | 1.024× |
| 209M | 1.016× | 1.027× | 1.004× |

At 5M the NumPy host (scaling, masks, refinement, conversions) adds 45% on top of the
network time; at 209M it adds 1.6%, so the large-model comparisons measure the generated
network almost exclusively.

### Agreement with the original PyTorch CPU reference

Cases passing `abs(actual - reference) <= 2e-4 + 2e-4 * abs(reference)` with finite, non-decreasing quantiles, out of all cases; parentheses give the maximum absolute difference.

| Size | Axon MLX (fused) | vikiival MLX | original PyTorch CPU, 8 threads | Axon MLX (unfused) | Axon MLX (fused, pruned) |
|---|---:|---:|---:|---:|---:|
| 5M | 21/23 (0.463) | 21/23 (0.408) | 23/23 (0.184) | 21/23 (0.225) | 21/23 (0.352) |
| 28M | 22/23 (1.12) | 23/23 (1.73) | 23/23 (0.00116) | 22/23 (0.785) | 23/23 (1.16) |
| 123M | 20/23 (0.504) | 20/23 (1.43) | 21/23 (0.211) | 20/23 (0.828) | 20/23 (0.504) |
| 209M | 21/23 (0.238) | 21/23 (0.215) | 22/23 (0.102) | 22/23 (0.293) | 21/23 (0.23) |

Failing cases:
- 5M Axon MLX (fused): `workload_multiblock_2000_2500` (max abs 0.00191, 3 quantiles outside tolerance)
- 5M Axon MLX (fused): `workload_covariates_missing_2000_1500` (max abs 0.00569, 50 quantiles outside tolerance)
- 28M Axon MLX (fused): `workload_covariates_missing_2000_1500` (max abs 0.000702, 3 quantiles outside tolerance)
- 123M Axon MLX (fused): `case_multi_block` (max abs 0.449, 5 quantiles outside tolerance)
- 123M Axon MLX (fused): `workload_multiblock_2000_2500` (max abs 0.00853, 132 quantiles outside tolerance)
- 123M Axon MLX (fused): `workload_covariates_missing_2000_1500` (max abs 0.00116, 28 quantiles outside tolerance)
- 209M Axon MLX (fused): `workload_multiblock_2000_2500` (max abs 0.012, 314 quantiles outside tolerance)
- 209M Axon MLX (fused): `workload_covariates_missing_2000_1500` (max abs 0.00154, 40 quantiles outside tolerance)
- 5M vikiival MLX: `case_past_known` (max abs 0.0344, 4 quantiles outside tolerance)
- 5M vikiival MLX: `workload_covariates_missing_2000_1500` (max abs 0.0244, 133 quantiles outside tolerance)
- 123M vikiival MLX: `case_multi_block` (max abs 0.236, 3 quantiles outside tolerance)
- 123M vikiival MLX: `workload_multiblock_2000_2500` (max abs 0.00848, 423 quantiles outside tolerance)
- 123M vikiival MLX: `workload_covariates_missing_2000_1500` (max abs 0.00126, 59 quantiles outside tolerance)
- 209M vikiival MLX: `workload_multiblock_2000_2500` (max abs 0.00502, 21 quantiles outside tolerance)
- 209M vikiival MLX: `workload_covariates_missing_2000_1500` (max abs 0.00087, 3 quantiles outside tolerance)
- 123M original PyTorch CPU, 8 threads: `workload_multiblock_2000_2500` (max abs 0.00303, 124 quantiles outside tolerance)
- 123M original PyTorch CPU, 8 threads: `workload_covariates_missing_2000_1500` (max abs 0.000912, 31 quantiles outside tolerance)
- 209M original PyTorch CPU, 8 threads: `workload_multiblock_2000_2500` (max abs 0.00579, 187 quantiles outside tolerance)
- 5M Axon MLX (unfused): `case_past_known` (max abs 0.0441, 17 quantiles outside tolerance)
- 5M Axon MLX (unfused): `workload_covariates_missing_2000_1500` (max abs 0.0135, 70 quantiles outside tolerance)
- 28M Axon MLX (unfused): `workload_covariates_missing_2000_1500` (max abs 0.0009, 11 quantiles outside tolerance)
- 123M Axon MLX (unfused): `case_multi_block` (max abs 0.229, 4 quantiles outside tolerance)
- 123M Axon MLX (unfused): `workload_multiblock_2000_2500` (max abs 0.00793, 157 quantiles outside tolerance)
- 123M Axon MLX (unfused): `workload_covariates_missing_2000_1500` (max abs 0.00157, 61 quantiles outside tolerance)
- 209M Axon MLX (unfused): `workload_multiblock_2000_2500` (max abs 0.0119, 366 quantiles outside tolerance)
- 5M Axon MLX (fused, pruned): `workload_multiblock_2000_2500` (max abs 0.00197, 3 quantiles outside tolerance)
- 5M Axon MLX (fused, pruned): `workload_covariates_missing_2000_1500` (max abs 0.00562, 52 quantiles outside tolerance)
- 123M Axon MLX (fused, pruned): `case_multi_block` (max abs 0.482, 5 quantiles outside tolerance)
- 123M Axon MLX (fused, pruned): `workload_multiblock_2000_2500` (max abs 0.00853, 132 quantiles outside tolerance)
- 123M Axon MLX (fused, pruned): `workload_covariates_missing_2000_1500` (max abs 0.00117, 49 quantiles outside tolerance)
- 209M Axon MLX (fused, pruned): `workload_multiblock_2000_2500` (max abs 0.012, 314 quantiles outside tolerance)
- 209M Axon MLX (fused, pruned): `workload_covariates_missing_2000_1500` (max abs 0.00143, 28 quantiles outside tolerance)

The multi-block workload (2000 → 2500, three autoregressive blocks) and the covariate
workload with missing values are the sensitive cases for every implementation: small
float32 differences compound across blocks and through the covariate normalization. The
implementations do not fail identically. Axon fused against vikiival: at 5M Axon fused misses the multi-block workload that vikiival passes, while vikiival misses `case_past_known` that Axon fused passes; at 28M Axon fused misses the covariate workload that vikiival passes.
Fused against unfused Axon: at 5M fused Axon misses the multi-block workload that unfused Axon passes, while unfused Axon misses `case_past_known` that fused Axon passes; at 209M fused Axon misses the covariate workload that unfused Axon passes.
Fused against pruned Axon: at 28M fused Axon misses the covariate workload that pruned Axon passes. The
`torch-8` column shows that changing only the PyTorch thread count moves these cases past
the gate at 123M and 209M, so the gate is at the level of the reference's own arithmetic
spread there. By error magnitude, the largest single disagreement of any MLX implementation is
0.48 (axon-fused-pruned on `case_multi_block` at 123M); on that case Axon fused reaches 0.45 on 5 of
21,861 quantiles against 0.24 for vikiival, and it remains the open numerical
question carried over from the earlier investigation.

### Run stability

72 timing processes completed and were accepted; no output was
non-finite, unordered or non-deterministic within a process. 15 of 216 implementation/size/
workload cells have a spread above 10% between the fastest and slowest round median; the
per-round geometric means in the first table bound that effect at the size level.
The approved protocol re-dispatches any worker whose round medians deviate by more than 10%
from the other two rounds. After the first pass, 42 of 72 timing workers met that rule
(any workload with a per-process median more than 10% from the mean of the other two rounds'
medians) and were run again once with identical settings; the tables use the replacement results,
and the first-pass results, summary and report are kept under `timing-superseded/pass1/` together
with the flagged list in `rerun-pass1.json`. In the first pass 76 of 216 cells exceeded the 10%
spread; after the re-dispatch 15 of 216 do. The replacement pass reduced the spread substantially.
The idle guard timed out 5 times (started after 120 s without three consecutive samples at
85% idle): timing axon-fused-pruned 209M round 1; timing vikiival 123M round 2; parity axon-fused 123M; parity axon-fused-pruned 123M; parity vikiival 123M.
Waiting for idle took 33 minutes against 42 minutes of worker time over all dispatches; the coding
agent and terminal stayed active throughout. Peak MLX memory reached 2.93 GB for the
Axon variants and 2.85 GB for vikiival (recorded per workload, informational only).

### All workloads

Median complete-forecast latency in milliseconds (median of per-process medians).

| Size | Workload | Axon MLX (fused) | vikiival MLX | original PyTorch CPU, 2 threads | original PyTorch CPU, 8 threads | Axon MLX (unfused) | Axon MLX (fused, pruned) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 5M | short_256_96 | 2.80 | 5.15 | 6.98 | 8.44 | 3.02 | 2.78 |
| 5M | batch32_2048_96 | 34.32 | 30.88 | 117.90 | 107.39 | 36.17 | 32.20 |
| 5M | max_context_32768_512 | 24.16 | 24.62 | 108.29 | 88.15 | 30.69 | 21.53 |
| 5M | multiblock_2000_2500 | 17.29 | 23.86 | 48.34 | 55.60 | 18.30 | 16.62 |
| 5M | covariates_missing_2000_1500 | 37.37 | 44.38 | 145.73 | 152.75 | 39.95 | 34.52 |
| 5M | multivariate_8x4_1024_192 | 24.42 | 23.11 | 74.26 | 76.76 | 23.55 | 21.87 |
| 5M | short_64_24 | 2.20 | 3.69 | 5.42 | 7.33 | 2.53 | 2.19 |
| 5M | long_8192_1024 | 8.48 | 11.20 | 31.85 | 42.08 | 9.24 | 8.09 |
| 5M | batch8_512_336 | 7.68 | 8.51 | 26.62 | 40.17 | 8.33 | 7.48 |
| 28M | short_256_96 | 4.68 | 7.02 | 22.13 | 24.18 | 5.17 | 4.61 |
| 28M | batch32_2048_96 | 102.10 | 105.47 | 346.14 | 262.82 | 113.14 | 96.09 |
| 28M | max_context_32768_512 | 66.77 | 74.41 | 307.17 | 200.56 | 90.73 | 62.14 |
| 28M | multiblock_2000_2500 | 33.37 | 42.96 | 143.00 | 191.33 | 36.52 | 32.27 |
| 28M | covariates_missing_2000_1500 | 107.27 | 127.91 | 390.51 | 334.88 | 121.50 | 99.85 |
| 28M | multivariate_8x4_1024_192 | 63.16 | 70.01 | 202.46 | 177.17 | 66.52 | 59.44 |
| 28M | short_64_24 | 3.82 | 5.76 | 19.21 | 22.46 | 4.57 | 3.80 |
| 28M | long_8192_1024 | 18.76 | 23.66 | 77.26 | 88.17 | 21.56 | 17.88 |
| 28M | batch8_512_336 | 16.19 | 18.52 | 65.44 | 84.79 | 18.01 | 15.77 |
| 123M | short_256_96 | 11.10 | 13.20 | 75.45 | 83.29 | 11.23 | 10.71 |
| 123M | batch32_2048_96 | 393.24 | 355.33 | 1089.61 | 840.56 | 404.63 | 365.86 |
| 123M | max_context_32768_512 | 235.77 | 229.01 | 890.47 | 594.14 | 283.96 | 223.43 |
| 123M | multiblock_2000_2500 | 94.27 | 101.54 | 356.58 | 453.26 | 98.63 | 90.78 |
| 123M | covariates_missing_2000_1500 | 394.05 | 403.11 | 1157.75 | 965.55 | 417.56 | 372.17 |
| 123M | multivariate_8x4_1024_192 | 232.09 | 226.77 | 617.46 | 513.85 | 229.75 | 219.77 |
| 123M | short_64_24 | 9.16 | 10.86 | 73.92 | 80.35 | 9.88 | 9.11 |
| 123M | long_8192_1024 | 62.28 | 63.72 | 217.32 | 224.69 | 66.43 | 56.69 |
| 123M | batch8_512_336 | 49.81 | 50.23 | 169.44 | 198.45 | 52.03 | 46.60 |
| 209M | short_256_96 | 15.42 | 19.24 | 115.10 | 119.55 | 15.65 | 15.13 |
| 209M | batch32_2048_96 | 603.06 | 579.01 | 1744.63 | 1290.26 | 642.80 | 577.58 |
| 209M | max_context_32768_512 | 372.40 | 404.63 | 1361.02 | 903.84 | 425.81 | 352.25 |
| 209M | multiblock_2000_2500 | 142.94 | 160.63 | 572.80 | 663.06 | 152.67 | 134.47 |
| 209M | covariates_missing_2000_1500 | 636.46 | 700.06 | 1826.65 | 1482.73 | 663.78 | 603.44 |
| 209M | multivariate_8x4_1024_192 | 360.99 | 381.45 | 982.57 | 787.57 | 358.59 | 340.64 |
| 209M | short_64_24 | 13.19 | 15.08 | 118.40 | 121.37 | 13.71 | 12.70 |
| 209M | long_8192_1024 | 97.37 | 105.28 | 333.32 | 324.00 | 102.45 | 88.29 |
| 209M | batch8_512_336 | 76.42 | 84.11 | 261.00 | 280.95 | 81.46 | 71.45 |
## Conditions and limits

- The PyTorch rows are CPU measurements. Upstream defaults to CPU, its float64 scaling and
  refinement path is unsupported on MPS, and upstream was not modified. Ratios against them
  mix backend and implementation differences.
- `axon-*` denotes Axon-generated MLX with the NumPy host from this repository, not an MLX
  implementation by Hugging Face or Mistral.
- The two MLX hosts batch differently (Axon 128 rows / 4,096 transformer tokens per call,
  vikiival 64 rows / 65,536 row-tokens per call). Both are the shipped defaults; chunking can
  change floating-point results as well as latency.
- `axon-fused-pruned` changes the floating-point results; `axon-fused` is the default and
  the reported configuration.
- Measurements come from a laptop with residual desktop activity. The idle guard and the
  recorded thermal snapshots document conditions; they do not control clocks or
  temperature. Large differences and consistent directions across rounds are more
  informative than small percentage gaps.
- Agreement is measured against the original implementation on synthetic inputs. It is not
  numerical parity in every case and not forecast accuracy on GIFT-Eval, TIME or fev-bench.

## Provenance

| Component | Pin |
|---|---|
| Original PyTorch | `timestral` 1.0.0 from `https://github.com/mistralai/timestral` at `a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff`; the nine installed source files match the pinned SHA-256 hashes recorded in `provenance.json` |
| vikiival MLX | `timestral-mlx` 0.1.0 (PyPI) = `https://github.com/vikiival/timestral-mlx` at `18a7d8692df850a4dc48286aa7f2c70a0808351a`; installed source files hashed and recorded |
| Timestral-5M | `mistralai/Timestral-5M` revision `bdd20cfc15a3fea0e4cb5ccef37f8d3ca7aa335f` |
| Timestral-28M | `mistralai/Timestral-28M` revision `49b26409ac87b5935539a5d8ffe7f397b720db47` |
| Timestral-123M | `mistralai/Timestral-123M` revision `5a147287addda85f6acd8da3fb8858fe9af7e8f7` |
| Timestral-209M | `mistralai/Timestral-209M` revision `9874d252ab0aae16f7f7c999f98f98d6ad7d8365` |
| Axon | `generic-timestral.axon` SHA-256 `b3b9259a67a32330062ba0151e2a5ab75b7185ad06c72a1ba1636f9d0d227ff1`; generated-module hashes in `cores/build-info.json` |

Checkpoint `config.json`/`model.safetensors` SHA-256 values, the full list of installed
distributions, git state and machine facts are in `provenance.json` under the run directory.

## Reproduction

From the repository root with `.venv/bin/python` on an Apple Silicon Mac:

```bash
uv pip install --python .venv/bin/python \
  "timestral @ git+https://github.com/mistralai/timestral.git@a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff" \
  "timestral-mlx==0.1.0"
RUN=timestral-mlx-comparison-<date>
.venv/bin/python -m scripts.timestral_bench init  --run-id $RUN
.venv/bin/python -m scripts.timestral_bench build --run-id $RUN
.venv/bin/python -m scripts.timestral_bench fetch --run-id $RUN
caffeinate -i .venv/bin/python -m scripts.timestral_bench bench  --run-id $RUN
# Re-dispatch rule: move each flagged timing/<impl>-<size>-r<round>.json to
# timing-superseded/pass1/, record the flagged list in rerun-pass1.json, then
caffeinate -i .venv/bin/python -m scripts.timestral_bench bench  --run-id $RUN --resume
caffeinate -i .venv/bin/python -m scripts.timestral_bench parity --run-id $RUN
.venv/bin/python -m scripts.timestral_bench summarize --run-id $RUN
.venv/bin/python -m scripts.timestral_bench report --run-id $RUN
```

The harness, its CLI and outputs are documented in [`scripts.md`](scripts.md); the run
entry is in [`log.md`](log.md). Regression tests: `tests/test_timestral_bench_host.py`,
`tests/test_timestral_bench_harness.py`, `tests/test_timestral_bench_axon_end_to_end.py`.

## Relation to earlier work

Supersedes the deleted `wiki/timestral-long-loop.md` (see `git show 44dad6f:wiki/timestral-long-loop.md`),
whose 2026-10-20 measurements used the pruning-enabled build, raw artifacts that no longer
exist, and a run-local harness that was never checked in. The protocol here keeps its
fresh-process, median-of-medians, geometric-mean design with fewer samples per process,
adds the unfused/pruned ablation and the generated-core replay, and makes everything
reproducible from a checkout. The agreement set also changed: 23 cases per checkpoint (14
synthetic cases plus the nine timing workloads) instead of the earlier 31, so the pass counts
above are not comparable with the earlier 29–30/31 figures.
