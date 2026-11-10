# Maintained scripts

Raw artifacts belong in `log/<run-id>/`, which Git ignores. Every maintained script is
listed here with its purpose, CLI, inputs, outputs and owner, as `scripts/AGENTS.md`
requires. Benchmark semantics changes are noted in [`log.md`](log.md).

## Timestral MLX comparison benchmark (`scripts/timestral_bench/`)

Owner: Axon contributors/agents. Last confirmed: 2026-10-27. Confidence: high for the
harness mechanics on Apple Silicon with the recorded package versions; timing
conclusions are limited to the recorded hardware and desktop conditions.
Uses: [`synapse/models/timestral/generic-timestral.axon`](../synapse/models/timestral/generic-timestral.axon)
through the standard compiler pipeline, the pinned original `timestral` package and the
pinned `timestral-mlx` package. Validated-by: [measured report](timestral-mlx-comparison.md),
`tests/test_timestral_bench_host.py`, `tests/test_timestral_bench_harness.py` and
`tests/test_timestral_bench_axon_end_to_end.py`.

The package restores the NumPy forecasting host that commit `cedc83d` removed from the
Timestral contribution (scaling, masks, RoPE tables, denormalization, block-autoregressive
refinement, batching limits) as benchmark-only code, so Axon-generated MLX can be compared
end to end with the two external implementations. It is not a maintained inference
product; the maintained contribution is the neural core and its direct tests.

Rules: run one subcommand at a time, from the repository root, with `.venv/bin/python`;
never run `bench` concurrently with `parity`, tests or builds. Worker processes are fresh
Python processes: MLX workers (`axon-*`, `vikiival`) must not import PyTorch or `synapse`,
PyTorch workers must not import MLX, and each worker asserts this before writing results.
Existing outputs are never overwritten; `--resume` skips completed files only. To re-dispatch
workers (for example under the measured report's >10% round-deviation rule), move their
`timing/<impl>-<size>-r<round>.json` files to `timing-superseded/<pass>/`, record the flagged
list and rule in `rerun-<pass>.json`, and run `bench --resume`; `summarize` reads only
`timing/*.json`, and the attempt logs and `manifest.json` keep every dispatch.

| Script | Purpose / CLI | Inputs | Outputs |
|---|---|---|---|
| `python -m scripts.timestral_bench init --run-id <id>` | Create `log/<id>/` and record provenance | Installed packages, git state, machine facts, pinned hashes | `provenance.json` (versions, distributions, upstream/vikiival file hashes vs pins, workload manifest) |
| `... build --run-id <id>` | Generate the unfused and fused Axon MLX cores; the only step that imports the compiler | `generic-timestral.axon`, `synapse/axon/codegen2_common/core.py` | `cores/axon_core_unfused.py`, `cores/axon_core_fused.py`, `cores/_axon_common.py`, `cores/model.axon`, `cores/build-info.json` (hashes) |
| `... fetch --run-id <id> [--sizes 5M,28M,123M,209M]` | Download `config.json` and `model.safetensors` at the pinned Hub revisions | Network, Hugging Face cache | `provenance.json["checkpoints"]` with snapshot paths and SHA-256 |
| `... bench --run-id <id> [--sizes] [--impls] [--rounds 3] [--samples 7] [--warmup 2] [--resume] [--no-idle-wait]` | Sequential fresh-process timing workers, implementation order rotated per round and size, CPU idle guard before each worker | Cores, fetched checkpoints, nine deterministic workloads | `timing/<impl>-<size>-r<round>.json` with per-sample milliseconds, output hashes, core replays, memory, environment; `.attemptN.log`; `manifest.json` |
| `... parity --run-id <id> [--sizes] [--impls] [--resume]` | PyTorch CPU (2 threads) reference, then one comparison worker per implementation | Checkpoints; 14 synthetic cases plus the nine workload inputs | `parity/reference-<size>/*.npz` + `manifest.json`; `parity/<impl>-<size>.json` with pass/fail, max abs, outside-tolerance counts |
| `... summarize --run-id <id>` | Median of samples per process, median across rounds, ratios to `axon-fused`, geometric means, host overhead, parity counts | `timing/*.json`, `parity/*.json` | `summary.json`, `comparison.csv` (per size and workload), `samples.csv` (every sample) |
| `... report --run-id <id> [--output <path>]` | Render the markdown tables used by the wiki page | `summary.json` | `report.md` |

Implementations: `axon-fused` (SDPA and RMSNorm-scale intrinsics, the reported default),
`axon-unfused`, `axon-fused-pruned` (`prune_final_block=True`, changes numerics),
`vikiival` (`timestral-mlx` 0.1.0 with shipped defaults), `torch-2` and `torch-8`
(original PyTorch on CPU with 2 or 8 intra-op threads, one inter-op thread).

Reproduce the fused core by hand with
`synapse axon-codegen-dump synapse/models/timestral/generic-timestral.axon out.py --backend codegen2-mlx --class-name TimestralCore --optimize-graph --graph-backend-intrinsics "codegen2-mlx:__mlx_sdpa,__mlx_rmsnorm_scaled"`;
the harness additionally rewrites the `synapse.axon.codegen2_common` import to the local
`_axon_common.py` copy so that workers never import `synapse`.

## PP-OCR conversion and MLX harness (`scripts/convert_ppocr_onnx.py`, `scripts/ppocr_mlx.py`)

Owner: Axon contributors/agents. Last confirmed: 2026-11-10. Confidence: high for the
mechanics on Linux (MLX CUDA) and on an M1 Pro (MLX Metal). Experiment, not a maintained
inference product. Uses: `synapse/models/ppocr/generic-ppocr-{det,rec}.axon` through the
standard compiler pipeline. Background and results: [CNN support and PP-OCR](vision-cnn.md).

Run from the repository root with `.venv/bin/python`. Artifacts go under `log/<run-id>/`.

| Script | Purpose / CLI | Inputs | Outputs |
|---|---|---|---|
| `python scripts/convert_ppocr_onnx.py --det <det.onnx> --rec <rec.onnx> --out <dir>` | Recover layer structure from the anonymous ONNX graphs, fold batch norm, rename parameters; raises if the graph does not match the layout fixed in the model files | PP-OCR ONNX exports; needs `onnx` | `<dir>/{det,rec}/config.json` and `model.safetensors` |
| `python scripts/ppocr_mlx.py build --run log/<id> [--onnx-dir DIR \| --repo ID]` | Fetch (default `blackpen-sk/ppocrv6-small`, needs Hub access) or reuse the ONNX files, convert, and emit the optimized MLX modules | ONNX files and dictionary | `det/`, `rec/`, `ppocr_det_mlx.py`, `ppocr_rec_mlx.py`, `build.json` |
| `... parity --run log/<id> [--dtype float32\|float16] [--tolerance 1e-3]` | Generated MLX vs ONNX Runtime CPU on seven random inputs; exits non-zero above the tolerance | Build output; needs `onnxruntime` | stdout |
| `... bench --run log/<id> [--providers CPUExecutionProvider,CoreMLExecutionProvider] [--samples 15] [--warmup 3]` | Median and minimum milliseconds per forward for five fixed workloads; ONNX rows only for installed providers | Build output | `bench.json`, stdout table |
| `... ocr --run log/<id> --image <file> [--backend mlx\|onnx] [--repeat N] [--json out.json]` | Full pipeline with PaddleOCR-default pre/post-processing in NumPy/OpenCV; prints lines and per-stage timings | Build output; needs `opencv-python-headless` | stdout, optional JSON with text, scores, boxes, timings |

MLX variants (`--variants` for `parity`/`bench`, `--variant` for `ocr`, default `auto`):
`base`, `compiled` (`mx.compile` per input shape), `shift` (3x3 depthwise convolutions as a
shifted sum), `shift+compiled`, `shift49+compiled` (also the 7x7 ones). `... tune --run
log/<id> [--rounds 3] [--samples 10]` times them for float32 and float16 and writes
`tuned.json` (per-network float32 choice), which `ocr --variant auto` reads. Inside a
published kit `--run` defaults to the script's directory, and `parity` checks the detector
on the kit's `sample.png`.

`bench` times what the pipeline consumes: the detector map as NumPy, and the recognizer's
best class and probability per column (taken on the device for MLX). On CUDA builds of MLX
set `MLX_ENABLE_TF32=0` for `parity`. Box unclipping grows the minimum-area rectangle
directly instead of calling `pyclipper`; for rectangles the two agree.
