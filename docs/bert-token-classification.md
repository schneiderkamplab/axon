# BERT token classification

`synapse/models/bert/generic-bert-token-classification.axon` implements an
inference-only BERT encoder followed by a learned token-classification head.
The shared encoder also serves the existing masked-LM definition. Supported
architecture: bidirectional BERT with absolute positions and GELU activation.
The checkpoint determines layer count, hidden/intermediate dimensions, heads,
normalization epsilon, embeddings, and classifier output size. Dropout is
disabled for inference. DistilBERT and relative-position variants are separate
architectures and are not covered by this definition.

Inputs are `input_ids`, optional `attn_mask`, and optional `token_type_ids`
(default all zeros). Output is raw `[batch, sequence, labels]` logits. Tokenizer
alignment, label interpretation, and entity decoding stay outside the graph.

## Public checkpoint validation

The definition declares [`dslim/bert-base-NER`](https://huggingface.co/dslim/bert-base-NER)
as a public integration checkpoint. Download it to a local directory and run:

```bash
synapse axon-test synapse/models/bert/generic-bert-token-classification.axon /path/to/checkpoint \
  --device cpu --dtype float32 --optimize-graph --forward-warmup 3 --forward-repeat 10
```

Task selection follows `TASK "token_classification"`; it always uses forward
inference, with `AutoModelForTokenClassification` as the HF reference. The
canonical `synapse axon-benchmark` runner also accepts this task and definition.

Offline regression checks use small randomly initialized HF BERT models, mixed
padding, nonzero segment IDs, different classifier sizes, and the existing
masked-LM head. Generating MLX source is tested without executing MLX:

```bash
OMP_NUM_THREADS=4 pytest -q tests/test_bert_token_classification.py tests/test_ner_evaluation.py
```

## Public MiniLM / Few-NERD experiment

Install the experiment dependencies on Linux:

```bash
uv pip install -e '.[ner-benchmark]'
```

The recipe fine-tunes [`nreimers/MiniLM-L6-H384-uncased`](https://huggingface.co/nreimers/MiniLM-L6-H384-uncased)
on [`DFKI-SLT/few-nerd`](https://huggingface.co/datasets/DFKI-SLT/few-nerd),
configuration `supervised`, using its 66 fine-grained entity types plus O.
Few-NERD uses IO tags. Exact spans are maximal runs of the same non-O label;
word-index spans are reconstructed across inference windows before scoring.

Preparation pins both Hub revisions and a separate public reference checkpoint.
Every annotated word is retained. Only first subtokens receive loss/metrics;
windows split at word boundaries. If tokenizer normalization erases a word, an
explicit UNK represents it; counts are recorded in `manifest.json`.

```bash
OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false \
  python scripts/train_ner.py prepare --run-dir log/ner-run

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false \
  python scripts/train_ner.py train --run-dir log/ner-run --device cuda:0

OMP_NUM_THREADS=4 python scripts/export_ner.py \
  --checkpoint log/ner-run/checkpoint --output log/ner-run/export
```

Training defaults: seed 17, three epochs, batch 64, AdamW at 3e-5, weight decay
0.01, 10% linear warmup followed by linear decay, gradient clipping at 1.0,
BF16 autocast with FP32 parameters, and TF32 disabled. The highest validation
exact-span micro F1 selects the checkpoint, with earliest epoch winning ties.
Test data is excluded from selection and calibration. Dynamic INT8 uses no
calibration dataset. `training-complete.json` freezes the checkpoint SHA-256.

Export emits Axon Torch and MLX Python, an unfused ONNX graph for Core ML,
provider-specific optimized FP32 graphs, a CUDA FP16 graph, and a per-channel
dynamic QInt8 CPU graph. Symbolic shape inference precedes quantization.
`export.json` records file hashes, operator counts, and actual fusion counts.
Running this export does not execute MLX.

## Measurements

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 \
  python scripts/run_ner_benchmarks.py --run-dir log/ner-run --suite linux
```

Each configuration gets an isolated process. The runner executes configurations
sequentially and writes raw JSON and logs below `log/ner-run/results/`. Avoid
competing workloads during performance runs. Quality and performance may also
be run separately with `--stage quality` and `--stage performance`; `--resume`
skips completed outputs. A new run/output path is required for reruns.

Produce a report with raw samples preserved in a portable JSON file:

```bash
python scripts/report_ner.py --run-dir log/ner-run --hardware 'CPU/GPU model, memory, OS, power settings'
```

`public-report.md`, `public-results.json`, and its gzip copy stay below the run directory.

- Model-call timings include host-to-device inputs and completed host logits,
  excluding tokenization. This is a host-to-host API comparison, not kernel-only
  GPU timing. Synthetic shape sweeps cover batch 1/8/32 and sequence 32/128/512.
- Each shape has a separately reported first call, ten warmups, and three
  contiguous groups of fifty measured calls. These are repeated calls within
  one process, not independent process-level replicates. Raw samples accompany
  p50/p95 and throughput. Four varying input batches prevent constant reuse.
- End-to-end measurements start from Few-NERD's presegmented words and include
  tokenization, windowing, model execution, and IO span decoding. They cycle over
  256 deterministic test sentences. They do not represent a raw-document service
  or reproduce an external leaderboard's serving protocol.
- Full quality evaluation covers the official test split, with identical word
  alignment, masks, and decoding for every backend. HF CPU FP32 writes the
  reference word logits. Reports include exact-span micro/macro F1, per-type
  precision/recall, logit errors, and prediction agreement.
- The predeclared FP32 gate is `atol=rtol=1e-4` on evaluated word logits. The
  reduced-precision gate permits at most 0.005 absolute loss in micro and macro
  F1. Per-type recall is reported even when these aggregate gates pass. Core ML
  uses the reduced-precision quality gate because provider compute precision
  may differ from its FP32 source graph.
- ORT placement is profiled in a separate session, keeping profiling overhead
  out of timings. Accelerator rows require actual accelerator node events.
  Core ML provider placement alone does not prove Neural Engine execution.
- Peak process RSS includes model/data/runtime allocations. Torch's GPU memory
  counter measures only Torch allocations and is not an ORT VRAM measurement.
  MLX reports its own allocator peak. These memory fields have different scopes.

## Mac handoff

Install on Apple Silicon using the same checkout:

```bash
uv pip install -e '.[mlx,ner-benchmark]'
```

Copy the experiment bundle/run directory containing `checkpoint/`, `export/`,
`encoded/`, `raw/`, the manifest, training metadata, and reference test logits.
Use a fresh `results/` directory for Mac outputs; preserve Linux results
separately. CPU graphs were saved before hardware-specific session optimization.

The Linux run can produce a test-only archive with file checksums and exact
source commit, omitting training data, download caches, and CUDA graphs:

```bash
python scripts/package_ner.py --run-dir log/ner-run --output log/ner-mac
```

Extract `ner-mac.tar.gz` below `log/` in the matching checkout on the Mac.
The archive's `MAC_RUN.md` includes setup and execution instructions.

```bash
OMP_NUM_THREADS=4 python scripts/run_ner_benchmarks.py --run-dir log/ner-mac --suite mlx
OMP_NUM_THREADS=4 python scripts/run_ner_benchmarks.py --run-dir log/ner-mac --suite cpu
OMP_NUM_THREADS=4 python scripts/run_ner_benchmarks.py --run-dir log/ner-mac --suite coreml
```

MLX runs FP32 and compiled FP16 on Metal. The compiled function wraps the Axon
encoder graph directly with `mx.compile`; every call materializes host logits.
Core ML runs `ALL`, `CPUAndGPU`, and `CPUAndNeuralEngine`, each with dynamic
inputs and fixed 32/128/512-token buckets. The latter compiles distinct batch
sizes as needed. Compilation is excluded from warmed model-call timings;
unseen shapes can still appear in the separate end-to-end workload, so inspect
first-call costs and raw samples before interpreting those results.

Record Mac chip, GPU cores, memory, macOS, power mode, and thermal conditions
alongside the automatically recorded package versions. Keep Mac and Linux
results in separate tables. No Apple Silicon or Core ML result is implied by
successful source generation or CUDA parity.

## Sources

- [Few-NERD dataset and original benchmark](https://huggingface.co/datasets/DFKI-SLT/few-nerd)
- [ORT transformer optimization](https://onnxruntime.ai/docs/performance/transformers-optimization.html)
- [ORT Core ML provider](https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html)
- [MLX](https://github.com/ml-explore/mlx)
