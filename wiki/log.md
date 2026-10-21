# Agent run log

Links under `log/` refer to local, Git-ignored evidence. The topic pages retain
the checked-in results and limitations; raw run artifacts are not part of the PR.

## 2026-10-21 — Make Timestral pruning opt-in

The user's correctness-first preference supersedes the earlier default-on
pruning decision. The Axon model now keeps the complete final block unless
`prune_final_block=True` is explicitly selected through the API or CLI. The
verifier records the setting. This preserves the optimization for deliberate
use while removing its two added 123M numerical deltas from the default path.

Validated-by 390 passing tests, 124/124 bit-identical comparisons to the frozen
pre-pruning baseline, and six standalone-export checks covering both settings
without importing Torch or Synapse. The original-reference gate is unchanged;
six older mismatches remain (118/124 passing). Layout, host math, batching, dense
attention, precision, and fusion probes narrowed the investigation but did not
establish a complete MLX parity fix. Only the validated pruning option is retained.
No new timing benchmark was run. Confidence: high for measured outputs and tests;
partial for the remaining root cause. Last confirmed: 2026-10-21.

Uses [current Timestral run knowledge](timestral-long-loop.md#current-correctness-default);
validated-by [fresh results](../log/timestral-parity-fix-20261021.MfVj0A/default-full124/results.json)
and [test log](../log/timestral-parity-fix-20261021.MfVj0A/tests-final.log).

## 2026-10-21 — Preserve required arguments during MLX RMSNorm fusion

Fixed the review finding in `graph_ir/optimize.py`: fusion now rejects wrappers
with unused inputs remaining after dead-formal pruning. This preserves required
parameter reads passed directly or through nested wrappers when `inline_safe=False`.
Validated-by four new cases in
[`test_axon_mlx_rmsnorm_fusion.py`](../tests/test_axon_mlx_rmsnorm_fusion.py),
354 passing focused/graph/policy tests, and an unchanged count of eight emitted
Timestral RMSNorm fusion calls. Confidence: high; last confirmed 2026-10-21.
Evidence: [fix validation](../log/ocr-review-20261021.8WyJk5/fix-validation.json).

## 2026-10-21 — Timestral Axon optimization loop

Default-on decision superseded by the correctness-first entry above. The timing
measurements below remain valid for opt-in pruning.

Implemented final-block output pruning in the Axon model after testing packed
projection, layout, singleton, and feed-forward alternatives. Maintained host and
compiler Python are unchanged from the initial workspace. Focused tests: 86
passed, including the empty-schedule regression found and fixed during finalization.
The final nonempty generated program is identical to the measured candidate after
eliminating its inactive zero-layer guard and normalizing local variable names.
Real-checkpoint reference results remain 118/124; two already-failing 123M cases
differ beyond the strict before/after tolerance.

Completed 12 paired and 48 external timing workers, with three rounds across all
four model sizes. Generated-core latency fell by 7.33%, 5.70%, 3.78%, and 3.16%
for 5M, 28M, 123M, and 209M. Compared against original PyTorch CPU at two/eight
threads and vikiival MLX with the same FP32 checkpoints. Six compiler-contended
workers and one interrupted partial worker are excluded. The user-requested
pause/resume preserved accepted measurements. No further benchmark loop is active.

Uses [Timestral run knowledge](timestral-long-loop.md); validated-by raw records
under [`log/timestral-long-loop-20261020/`](../log/timestral-long-loop-20261020/).
Benchmark semantics: sequential fresh processes, separate synchronized generated
core and complete forecast timing, fixed FP32 weights and quality tolerance.
