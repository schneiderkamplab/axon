# Timestral inference and final-block pruning

Status: pruning experiment complete; pruning now off by default. Six original-reference
differences remain unresolved. Last confirmed: 2026-10-21.
Owner: repository agents. Evidence is in
[`log/timestral-long-loop-20261020/`](../log/timestral-long-loop-20261020/).
Confidence: high for the reported measurements and test outcomes on this machine.
Scope: nine timing workloads and four released FP32 checkpoints. The standalone
package is in [`final/`](../log/timestral-long-loop-20261020/final/), with source,
generated code, requirements, and license notices.

This page is the checked-in summary. Links below `log/` refer to local run
artifacts, which are excluded from Git and are not included in the PR. The
forecast package and regression tests can be run from a fresh checkout; the
historical 124-case and timing runs also need their local checkpoints, fixtures,
and run scripts.

## Current correctness default

The user's correctness-first preference supersedes the original decision to
enable pruning by default. `prune_final_block=False` now retains every position
through the final transformer block and selects forecast positions afterward.
The optional optimization remains available through `from_pretrained` and
`load_model` with `prune_final_block=True`, or `forecast --prune-final-block`.
`export --optimize` controls graph fusion separately; it does not enable pruning.
The verifier accepts `check --prune-final-block` and records the setting.

Validated-by [124 fresh cases](../log/timestral-parity-fix-20261021.MfVj0A/default-full124/results.json):
the default is bit-identical to the frozen pre-pruning baseline in every case.
This removes the two added 123M pruning deltas described below. It preserves the
existing **118/124** original-reference result; no tolerance changed. All outputs
remain finite and ordered. The regression suite passed
[390 tests](../log/timestral-parity-fix-20261021.MfVj0A/tests-final.log), including
both settings, all schedules, gates, biases, and the empty schedule.

Run the regression suite on a machine with Torch and native MLX/Metal access:

```bash
OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 python -m pytest -q \
  tests/test_axon_mlx_rmsnorm_fusion.py tests/test_timestral_port.py \
  tests/test_timestral_pruning.py tests/test_agents_policy_guards.py \
  tests/test_axon_graph_ir.py
```

A [fresh standalone export](../log/timestral-parity-fix-20261021.MfVj0A/export/)
passed [six real-checkpoint API checks](../log/timestral-parity-fix-20261021.MfVj0A/export-smoke.json):
three sensitive scenarios, each with the default and opt-in setting. Defaults
reproduce the old baseline exactly; opt-in reproduces the old pruned candidate
exactly. Neither Torch nor Synapse is imported by inference. The historical
timing gains below apply to **opt-in pruning**, not the current default. No new
timing benchmark was run for this correctness change.

### Remaining numerical investigation

Confidence: high for these measured differences; incomplete for their full cause.
The following default results still exceed
`2e-4 + 2e-4 * abs(reference)`:

| Checkpoint | Case | Maximum absolute difference | Quantiles outside tolerance |
|---|---|---:|---:|
| 5M | `extended_batch32_2048_96` | 0.0018959045 | 94 |
| 5M | `extended_cov_missing_2000_1500` | 0.0479812622 | 1,550 |
| 28M | `extended_multiblock_2000_2500` | 0.0031585693 | 6 |
| 123M | `multi_block` | 0.44921875 | 5 |
| 123M | `extended_cov_missing_2000_1500` | 0.0018768311 | 14 |
| 209M | `extended_multiblock_2000_2500` | 0.0017261505 | 2 |

Run-local probes isolate several sources of floating-point sensitivity:

- The original Torch core under NumPy forecast orchestration, using unbounded
  batches and separate positive/negative passes, reproduces both failing 5M
  references exactly. Changing only the grouping of that original core can
  exceed the tolerance on the covariate case. See
  [original core control](../log/timestral-parity-fix-20261021.MfVj0A/original-unbounded/results.json).
- On identical inputs, restoring the original Q/K/V memory layout makes all
  generated Torch transformer blocks match the original exactly in the 5M
  batch case. With reference host math and scheduling, its complete forecast
  passes with maximum difference 0.0000305176. The covariate forecast still
  fails. This is a diagnostic, not a maintained Torch backend fix. See
  [combined probe](../log/timestral-parity-fix-20261021.MfVj0A/torch-reference-unpruned/results.json).
- The covariate layer trace first diverges at variate attention; later time
  attention amplifies that difference. Dense variate attention with unbounded,
  separate passes produces the same failing MLX outputs as the current default.
  See [layer trace](../log/timestral-parity-fix-20261021.MfVj0A/trace-cov-layout.log)
  and [dense MLX probe](../log/timestral-parity-fix-20261021.MfVj0A/mlx-reference-dense/results.json).
- Float64 known-covariate instance statistics, eager MLX execution, and disabling
  graph fusion do not resolve both 5M failures; some increase the errors.
  More precision or less optimization alone is not an established parity fix.

No diagnostic monkeypatch, reference-output substitution, backend routing, or
model-specific compiler branch was retained. The next investigation should
capture per-primitive inputs and outputs around variate QKV projection and SDPA,
then normalization/head selection, on the pinned failing fixtures. Any proposed
layout or arithmetic change must pass all 124 cases and the synthetic schedule
suite before replacing the default. Keep host-statistics and RoPE differences
separate from learned-network differences so a partial improvement is not
mistaken for a complete correction.

Artifacts are in `log/timestral-parity-fix-20261021.MfVj0A/` and are owned by this
run. `probe.py --backend mlx|torch|original --sizes 5M,28M,123M,209M --cases all
--tag NEW-TAG` uses the existing local checkpoints and fixtures and emits a
generated module, forecast arrays, and `results.json` beneath the new tag.
Additional diagnostic flags are documented by `--help`; they are not production
features. `trace.py CASE [reference-layout]` compares original and generated
Torch layers. `export_smoke.py` loads the fresh standalone export and checks
both load options against frozen outputs. These are raw run scripts, not new
maintained benchmark APIs. Use repository `.venv/bin/python`; native MLX requires
Metal access. Fresh forecast probe tags refuse overwrite.

## Historical pruning change

[`generic-timestral.axon`](../synapse/models/timestral/generic-timestral.axon)
passes the requested output length into the final transformer block. Earlier
blocks retain the complete sequence. Time attention retains complete keys and
values but evaluates only needed queries. Variate attention is independent at
each time position, so its input is narrowed before projection. Output gates,
projections, and the feed-forward network process only the returned positions.
A `variate_time` block retains the full first-axis output because the subsequent
time attention still needs its keys and values.
An empty schedule retains the previous forecast-position selection before the
output head; this edge case was found and fixed during finalization.

This change uses the Axon model's data flow and explicit dimensions. It adds no
compiler name matching or model-family routing. Forecast scheduling, batch
limits, host statistics, refinement, weights, and precision are unchanged from
the pre-run workspace. The existing compiler optimizations are present in both
exports; their gains are not counted as gains from this experiment.

Reviewed other Axon attention implementations in Bloom, Falcon, GPT-2, and OLMo3
for packed projections and head layout, together with the vikiival implementation.
The selection loop rejected:

- Reordering packed QKV weights: no robust measured benefit.
- Fusing QKV and gate projection: about 0.5% core improvement in the scout,
  insufficient to retain the added machinery.
- A singleton-only V projection: no meaningful additional gain over feed-forward
  trimming.
- Feed-forward-only trimming: approximately 2.4% scout core improvement,
  superseded by final-block pruning.
- Inline pruning variants that failed shape/constraint validation. The retained
  variate helper binds its input and mask dimensions independently.

The initial 5M alternating scout measured 1.073× core speedup for the retained
structure. This is selection evidence, not the final benchmark. Rejected source
variants and exports remain only in ignored run artifacts for audit; maintained
code contains the selected change.

## Measurement contract

Hardware: Apple M1 Pro, 32 GB unified memory. Python 3.13.9, MLX 0.32.2,
PyTorch 2.14.0, NumPy 2.5.3. All four released checkpoints retain FP32 weights.

The improvement attributed to Axon is the ratio between two frozen exports from
the same workspace and compiler, differing only in Axon source. Three fresh
processes per model size alternate baseline and candidate order, with 15 timed
samples per variant and workload after warmup. Nine workloads cover short/long
contexts, batches, multivariate inputs, covariates, maximum context, and
multi-block prediction. Generated-core timing replays the complete sequence of
model calls on precreated MLX arrays and synchronizes completion. It excludes
host normalization, mask construction, transfers, and output conversion.
Full-forecast timing includes these costs and returns CPU NumPy output.

The external comparison uses three fresh processes per implementation and size,
seven samples per workload, two warmup calls, and rotating implementation and
workload order. Original PyTorch runs on CPU at both two and eight intra-op
threads, with one inter-op thread; it imports no MLX. Axon and vikiival run on
Metal and import no PyTorch. This comparison includes backend and implementation
differences; it cannot isolate an Axon compiler gain. Original MPS execution was
not used because the original float64 scaling path is unsupported there.

Final aggregation takes the median of each process's samples, then the median
across three processes for each workload. The model-size summary is the
geometric mean of the nine workload speedup ratios. Loading, compilation, and
first-call latency are excluded from steady-state timings. Raw first-call and
peak memory fields are retained, but the recorded peak includes first-call and
warmup allocations and is not an isolated steady-state measurement.

Workers run sequentially. The runner polls for Rust/C compiler activity every
three seconds, waits for 15 quiet seconds before starting, and excludes and
retries workers with observed overlap. This is a contention check, not proof
that every background process or sub-three-second build was absent. Rejected
attempts remain in the process manifest and are excluded from aggregation.

## Final timing results

<!-- timing-results:start -->
All 60 planned timing workers completed. The aggregate contains 6,264 timed
full forecasts and 3,240 generated-core replays. Three fresh-process rounds
per size were retained. Speedup is baseline latency divided by candidate
latency; a value above 1 favors the candidate. Latency reduction is
`100 * (1 - 1 / speedup)`.

| Size | Axon core latency reduction | Axon full-forecast latency reduction | Core speedup range across rounds |
|---|---:|---:|---:|
| 5M | 7.33% | 6.14% | 1.073–1.087× |
| 28M | 5.70% | 5.65% | 1.052–1.060× |
| 123M | 3.78% | 3.38% | 1.035–1.044× |
| 209M | 3.16% | 3.12% | 1.029–1.033× |

The before/after comparison above isolates the Axon source change. The
external comparison below includes all implementation and backend differences.
All ratios summarize the same nine workloads with equal weight.
Core latency improves in 35/36 cases; the short 123M case is 0.04% slower.
Complete-forecast latency improves in all 36 before/after cases.

| Size | Candidate vs vikiival MLX | Candidate vs original CPU, 2 threads | Candidate vs original CPU, 8 threads |
|---|---:|---:|---:|
| 5M | 1.315× | 3.534× | 3.613× |
| 28M | 1.202× | 4.097× | 3.656× |
| 123M | 1.118× | 3.718× | 3.322× |
| 209M | 1.093× | 3.623× | 3.257× |

Four individual workloads remain faster in vikiival: batch 32 at 5M, 123M,
and 209M, plus the eight-series/four-variate 209M workload.

Representative complete-forecast latencies in milliseconds:

| Size | Workload | Axon candidate | vikiival MLX | Original CPU 2 | Original CPU 8 |
|---|---|---:|---:|---:|---:|
| 5M | context 256, horizon 96 | 2.82 | 4.94 | 7.11 | 9.25 |
| 5M | batch 32, context 2048, horizon 96 | 33.22 | 29.78 | 118.40 | 112.37 |
| 5M | context 32768, horizon 512 | 21.73 | 24.22 | 110.48 | 90.54 |
| 28M | context 256, horizon 96 | 5.41 | 7.11 | 24.02 | 24.79 |
| 28M | batch 32, context 2048, horizon 96 | 100.82 | 105.58 | 359.92 | 282.85 |
| 28M | context 32768, horizon 512 | 66.63 | 76.57 | 318.26 | 218.32 |
| 123M | context 256, horizon 96 | 10.87 | 14.03 | 78.64 | 82.07 |
| 123M | batch 32, context 2048, horizon 96 | 392.20 | 390.07 | 1124.15 | 867.80 |
| 123M | context 32768, horizon 512 | 234.81 | 265.60 | 910.95 | 610.37 |
| 209M | context 256, horizon 96 | 15.38 | 18.58 | 114.96 | 120.50 |
| 209M | batch 32, context 2048, horizon 96 | 603.49 | 594.41 | 1728.95 | 1329.94 |
| 209M | context 32768, horizon 512 | 370.68 | 415.10 | 1374.07 | 925.78 |

Excluded 6 completed workers with observed compiler overlap.
The worker interrupted for the requested pause remains unaccepted and is retained
as a partial raw artifact. Finalization added a zero-layer guard. After removing
that inactive guard for nonempty schedules and normalizing local names, the
entire generated module AST matches the measured candidate. The 100 shared
methods are unchanged apart from the main forward containing that guard;
two added helpers are unreachable on the nonempty path. Thus the measured
tensor computation is unchanged for all released checkpoints. Generated file
bytes differ, and both versions are retained. Pre-existing workspace changes
are preserved. See [equivalence proof](../log/timestral-long-loop-20261020/nonempty-equivalence.json).

Validated-by [summary.json](../log/timestral-long-loop-20261020/summary.json),
[all 36 workload comparisons](../log/timestral-long-loop-20261020/comparison.csv),
and [individual samples](../log/timestral-long-loop-20261020/samples.csv).
These timing gains do not resolve the numerical differences described below.
<!-- timing-results:end -->

## Correctness and limits

The focused suite passed **86 tests**, including 34 pruning comparisons across
Torch and MLX, all four final-layer schedules, singleton/multivariate groups,
covariate roles, attention gates, projection biases, and an empty layer schedule.
These compare requested
outputs with a full-sequence execution sliced outside the network.

The first pruning candidate regressed the accepted empty-schedule configuration
with an output reshape error. A zero-layer guard restores the baseline result
bit-for-bit. The final tests and emitted-program equivalence proof validate the
fix without changing the measured nonempty computation. See
[`tests-after-empty-guard.log`](../log/timestral-long-loop-20261020/tests-after-empty-guard.log)
and [`nonempty-equivalence.json`](../log/timestral-long-loop-20261020/nonempty-equivalence.json).

The integration gate remains
`abs(actual - reference) <= 2e-4 + 2e-4 * abs(reference)`.
No tolerance was loosened. Fresh original PyTorch results pass all 124 original
fixtures. Finite and ordered FP32 outputs hold throughout.

| Checkpoint | Axon before | Axon after | vikiival MLX | Original CPU, 2 threads |
|---|---:|---:|---:|---:|
| 5M | 29/31 | 29/31 | 28/31 | 31/31 |
| 28M | 30/31 | 30/31 | 30/31 | 31/31 |
| 123M | 29/31 | 29/31 | 29/31 | 31/31 |
| 209M | 30/31 | 30/31 | 29/31 | 31/31 |

The separate 60-case synthetic/released-5M fixture suite remains **58/60** before
and after. Six real-checkpoint reference failures remain; there are no newly
failing reference cases. This is not full numerical parity or an evaluation of
forecast accuracy on a real dataset.

Two already-failing 123M cases also exceed the strict before/after tolerance:

- `multi_block`: one of 21,861 quantiles exceeds the delta gate, with maximum
  absolute before/after difference 0.0625. Maximum error against the original
  increases from 0.44921875 to 0.482421875; both runs have five failing quantiles.
- `extended_cov_missing_2000_1500`: three of 126,000 quantiles exceed the delta
  gate, with maximum difference 0.0005090237. Maximum original-reference error
  remains 0.001876831; failing quantiles decrease from 14 to 10.

The other 122/124 cases meet the same before/after gate. The change is not
bit-exact and does not resolve the pre-existing numerical failures.

## Provenance and reproduction

Uses original
[`mistralai/timestral@a534bdb`](https://github.com/mistralai/timestral/tree/a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff)
and the implementation linked by
[`vikiival/timestral-mlx`](https://huggingface.co/vikiival/timestral-mlx),
[`timestral-mlx@18a7d86`](https://github.com/vikiival/timestral-mlx/tree/18a7d8692df850a4dc48286aa7f2c70a0808351a),
package version 0.1.0. The latter loads original Mistral weights; it does not use
a separately quantized checkpoint. All nine installed competitor Python files
match the pinned repository source. Original source and checkpoint hashes are
verified in [`provenance.json`](../log/timestral-long-loop-20261020/provenance.json).

The frozen baseline is the user's workspace at the start of this run, including
pre-existing uncommitted changes, not Git HEAD. `before.patch`, `before-status.txt`,
and `before-hashes.json` preserve that boundary. `change.patch` contains only the
new model-source delta. Raw evidence is owned by this run, not by a new maintained
benchmark API:

| File | Purpose / inputs | Output |
|---|---|---|
| `execute.py PLAN --quiet [--resume]` | Sequential JSON plan execution; compiler overlap monitoring; optional continuation preserving accepted workers | `PLAN.processes.json`, per-worker logs |
| `paired.py --size SIZE --tag TAG --modes baseline,candidate --samples 15` | Frozen exports and nine benchmark fixtures | Alternating forecast/core samples in `TAG-SIZE-paired.json` |
| `worker.py --size SIZE --mode MODE --tag TAG --stage benchmark --samples 7` | Frozen export or pinned original/competitor; `MODE` is `candidate`, `vikiival`, `original-2`, or `original-8` | Full-forecast samples and quality in `TAG-SIZE-MODE.json` |
| `worker.py ... --stage quality` | The same implementations and 31 cases per checkpoint | Per-case strict reference and baseline comparisons |
| `synthetic.py --mode candidate --size 5M --tag TAG` | Existing 60-case fixtures | Synthetic/released-5M parity record |
| `summarize.py` | Completed final plans, quality records, matching final export | `summary.json`, `comparison.csv`, `samples.csv` |
| `report.py` | Verified `summary.json` | Timing tables in this page |
| `resume.sh` | Existing paired/external plans and accepted-worker manifests | Resumes the two timing plans sequentially |
| `progress.py` | Existing timing manifests and latest worker records | Read-only progress summary without importing either backend |
| `empty_schedule.py --mode baseline\|candidate\|final` | Frozen export and deterministic tiny weights | Empty-schedule result JSON and successful output arrays |
| `verify_nonempty.py` | Measured/final source, generated ASTs, host files, and empty outputs | Hash-bound equivalence proof in `nonempty-equivalence.json` |

Run these with repository `.venv/bin/python`, from the repository root. Paths
are relative to `log/timestral-long-loop-20261020/`. The wrapper sets
`OMP_NUM_THREADS=2`, `CUDA_VISIBLE_DEVICES=''`, `HF_HUB_OFFLINE=1`, and disables
bytecode writes. Inputs depend on the pinned checkpoints/fixtures under
`log/timestral-all-sizes-20261020/` and original/competitor source under
`log/timestral-mac-20261020-comparison/`. Outputs refuse overwrite; reruns require
fresh plan names and tags. Metal execution needs an environment with GPU access.
The generic language-model `synapse axon-benchmark` does not expose Timestral's
forecast/refinement API, so this run uses the forecast-specific harness.

The maintained package workflow remains
`python -m scripts.timestral_axon export --output log/NEW-EXPORT --optimize`.
It consumes the Axon model and shared runtime helpers, and emits a standalone
MLX package plus requirements and license notices. Its CLI and ownership are
documented in [`scripts/timestral_axon/README.md`](../scripts/timestral_axon/README.md).
The final retained diff and file hashes are recorded in
[`retained-changes.patch`](../log/timestral-long-loop-20261020/retained-changes.patch)
and [`artifact-sha256.json`](../log/timestral-long-loop-20261020/artifact-sha256.json).

## Existing policy debt outside this change

The repository already contains a name-based Torch RMSNorm matcher in
`synapse/axon/graph_ir/optimize.py` and HF weight-key handling in
`synapse/axon/codegen2_mlx/core.py` (`_emit_build_ff`). Neither is changed by this
run or used as a new Timestral optimization. Root policy requires removing such
special cases. Replace the matcher with validated primitive/provenance matching;
move HF key adaptation into the approved loading integration, passing canonical
parameters to generic emission. Risks are false-positive graph fusion and
changed state-dict binding. Validate renamed/unrelated graph negative cases,
checkpoint loading, generated-code parity, cache behavior, and affected-model
benchmarks before landing that separate refactor.
