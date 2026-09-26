# Maintained scripts

## Public BERT token-classification experiment

Owner: Axon contributors/agents. Last confirmed: 2026-09-26. Confidence: high
for Linux training, export, quality, and timing runs; Mac execution remains pending.
Uses: [public runbook](../docs/bert-token-classification.md).
Validated-by: [measured report](../docs/bert-token-classification-results.md) and
`log/bert-ner-20260926/public-results.json.gz` (all ten configurations passed quality gates).

| Script | Purpose / CLI | Inputs | Outputs |
|---|---|---|---|
| `scripts/ner_common.py` | Shared first-subtoken alignment, word-boundary windows, IO span scoring; imported module | Word lists, labels, tokenizer, logits | Full-coverage windows and exact-span metrics |
| `scripts/train_ner.py` | `prepare --run-dir log/<run>` then `train --run-dir log/<run>` | Public MiniLM/Few-NERD pinned by preparation; configurable seed/epochs/batch/lr | Raw/encoded datasets, manifest, recipe, validation history, selected checkpoint and hash |
| `scripts/export_ner.py` | `--checkpoint <dir> --output log/<run>/export` | HF BERT checkpoint, generic Axon definition | Torch/MLX Python and provider-specific ONNX graphs; hashes/fusion counts |
| `scripts/bench_ner.py` | `--run-dir <run> --output <json> --backend hf/axon/ort/mlx --device cpu/cuda:0/metal/coreml` | Frozen checkpoint, graphs, dataset, HF reference logits | Quality and/or host-to-host plus word-to-span performance; raw samples and placement profile |
| `scripts/run_ner_benchmarks.py` | `--run-dir <run> --suite linux/cpu/cuda/mac/mlx/coreml [--stage quality/performance/all]` | Prepared experiment and available hardware | Sequential isolated subprocess results/logs under the run directory |
| `scripts/report_ner.py` | `--run-dir <run> --hardware '<description>'` | Completed benchmark JSON, manifest, selection, export metadata | Public report and portable raw results under the run directory; preserves failed gates |
| `scripts/package_ner.py` | `--run-dir <run> --output log/<bundle>` | Frozen run, reference logits, optional Linux report | Test-only Mac handoff, checksums, source commit, instructions, and tar.gz; no downloads or execution |

Raw artifacts belong in `log/`. Public reports must keep equal-precision and
reduced-precision comparisons identifiable, and distinguish Linux CUDA from
Apple Metal/Core ML. Preserve failed quality gates in reporting.
