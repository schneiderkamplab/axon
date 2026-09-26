# Production optimization validation, 2026-09-26

Uses the frozen checkpoint, raw/encoded datasets, and reference logits from
`../bert-ner-20260926` through local symlinks. The selected model is unchanged.
`export/` contains new production codegen exports, not AST-rewritten prototypes.
`export-recheck/` confirms that exports match the final compiler source byte for byte.
The portable Mac archive copies all required files and has no symlink dependency.

`run_validation.py` drives `scripts/bench_ner.py` in sequential fresh processes.
The initial sandbox run completed Axon CPU FP32 full-test quality, then failed to
initialize CUDA. This failed log is retained in `results/`. The host retry uses
`--skip-cpu-quality --results-dir log/ner-optimized-20260926/results-host`.
It runs four CUDA quality configurations and three shuffled process repetitions
of HF/Axon CPU FP32 and CUDA FP32/FP16 performance. Each case has 10 warmups and
50 timed calls; no heavy tests run concurrently with timing. GPU 0, four CPU
threads, TF32 disabled, completed host logits including transfers.

`check_compile.py` separately checks the generated public `forward` through
full-graph CUDA compilation in FP32/FP16 under no_grad and inference_mode. It is
a parity check, not a compiled performance result or full-test quality result.
MLX is generated and parsed only; all Mac execution remains pending.

Main regression evidence: `final-tests.log`; targeted path/weight sharing guard
evidence: `pack-guards-retry.log`. The earlier failed guard test used an invalid
unhashable list literal; the corrected fixture uses a typed graph input.

Owner: Axon contributors/agents. These scripts are run-local research tools;
maintained benchmark commands are documented in `wiki/scripts.md`.
