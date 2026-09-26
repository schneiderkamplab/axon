# Maintained scripts

## Public BERT token-classification experiment

Owner: Axon contributors/agents. Last confirmed: 2026-09-26. Confidence: high
for Linux training, export, quality, and timing runs; Mac execution remains pending.
Uses: [public runbook](../docs/bert-token-classification.md).
Validated-by: [measured report](../docs/bert-token-classification-results.md) and
`log/bert-ner-20260926/public-results.json.gz` (all ten configurations passed quality gates).
The [optimized comparison](../docs/bert-token-classification-optimized-results.md)
uses `log/ner-optimized-20260926/` with fresh production exports and three
process-level timing repetitions. Full CPU/CUDA quality passed; weights and
ONNX artifact hashes are unchanged. The run-local `run_validation.py`,
`check_compile.py`, and `summarize.py` document this follow-up's execution and
aggregation. `report_ner.py` covers the original single-process result naming;
do not treat individual `-r0/-r1/-r2` files as one contiguous sampling run.

| Script | Purpose / CLI | Inputs | Outputs |
|---|---|---|---|
| `scripts/ner_common.py` | Shared first-subtoken alignment, word-boundary windows, IO span scoring; imported module | Word lists, labels, tokenizer, logits | Full-coverage windows and exact-span metrics |
| `scripts/train_ner.py` | `prepare --run-dir log/<run>` then `train --run-dir log/<run>` | Public MiniLM/Few-NERD pinned by preparation; configurable seed/epochs/batch/lr | Raw/encoded datasets, manifest, recipe, validation history, selected checkpoint and hash |
| `scripts/export_ner.py` | `--checkpoint <dir> --output log/<run>/export` | HF BERT checkpoint, generic Axon definition | Torch/MLX Python and provider-specific ONNX graphs; hashes/fusion counts |
| `scripts/bench_ner.py` | `--run-dir <run> --output <json> --backend hf/axon/ort/mlx --device cpu/cuda:0/mps/metal/coreml` | Frozen checkpoint, graphs, dataset, HF reference logits | Quality and/or host-to-host plus word-to-span performance; raw samples and placement profile; MPS availability and no-fallback guard |
| `scripts/run_ner_benchmarks.py` | `--run-dir <run> --suite linux/cpu/cuda/mac/mac-compare/mps/mlx/coreml [--stage quality/performance/all] [--results-dir log/<run>/results] [--repetitions 3] [--keep-going] [--dry-run]` | Prepared experiment and available hardware | Sequential isolated subprocess results/logs; repeated performance runs shuffle configurations and use 50 samples per process; failed outputs are retained and never resumed as successful |
| `scripts/report_ner.py` | `--run-dir <run> --hardware '<description>'` | Completed benchmark JSON, manifest, selection, export metadata | Public report and portable raw results under the run directory; preserves failed gates |
| `scripts/package_ner.py` | `--run-dir <run> --output log/<bundle>` | Frozen run, reference logits, optional Linux report | Test-only Mac handoff, checksums, source commit, instructions, and tar.gz; no downloads or execution |

Raw artifacts belong in `log/`. Public reports must keep equal-precision and
reduced-precision comparisons identifiable, and distinguish Linux CUDA from
Apple Metal/Core ML. Preserve failed quality gates in reporting.
Regenerate Axon Python exports after compiler changes before packaging. A new
checkout does not replace code in an old archive. Use a fresh versioned bundle
directory and its recorded source revision; never replace a frozen handoff in place.
