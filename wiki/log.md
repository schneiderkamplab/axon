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

## 2026-11-10 — CNN primitives and PP-OCRv6 small through Axon (experiment)

Added six channels-last 2D primitives, the `Vision` builtins, Torch and MLX lowerings, the
PP-OCR detector and recognizer model files, an ONNX-to-checkpoint converter and an MLX
harness, on branch `feat/cnn-ppocr`. Goal: find out whether Axon-generated MLX makes
`blackpen-sk/ppocrv6-small` faster on Apple Silicon than ONNX Runtime.

Observed: generated Torch and MLX agree with ONNX Runtime to 1e-5 or better on both models
and give identical end-to-end OCR text on a rendered page. Two compiler findings recorded on
the topic page: loop carries are typed shape-invariant, and `inline_safe` corrupted
`Tensor.size` on spatial dims (worked around, not fixed). Pre-existing on untouched `main`
in this environment and unrelated: 16 failures in `tests/test_axon_graph_ir.py`,
`tests/test_codegen2_mlx_sdpa_band.py` and `tests/test_pplx_embed_support.py` (tinygrad
`CUDA_ERROR_UNSUPPORTED_PTX_VERSION`, MLX CUDA numerics), and the per-model roundtrip sweeps.

Not done: any Apple Silicon measurement. Uses [CNN support and PP-OCR](vision-cnn.md) and
[maintained scripts](scripts.md). Validated-by `tests/test_vision_spatial2d.py`,
`tests/test_ppocr_axon.py` and `log/ppocr-mlx-dev/` (`bench.json`). Confidence: high for
agreement, none yet for Apple Silicon speed. Last confirmed: 2026-11-10.
Superseded the same day for the "not done" part by the entry below.

## 2026-11-10 — PP-OCR timed on Apple Silicon (M1 Pro)

Ran `scripts/ppocr_mlx.py build`, `parity`, `bench`, `tune` and `ocr` on an M1 Pro (MLX
0.32.2 Metal, ONNX Runtime 1.30.0), and timed the generated Torch on CPU and MPS.

Observed: Axon MLX is 2.9x to 5.7x faster than ONNX Runtime CPU (detector 1x736x960 22.3
against 100.7 ms. End-to-end sample page 65 against 324 ms, identical text). The ONNX
Runtime CoreML provider is slower than CPU (partitioned graphs). `tune` keeps `compiled`.
The shifted-sum depthwise lowering never wins on Metal. Generated Torch on CPU was 7x
slower than ONNX Runtime, caused-by grouped convolutions on a permuted view. Fixed-by a
contiguous copy in `synapse/axon/codegen2_torch/spatial2d.py` (4-5x faster on CPU,
unchanged on MPS). PyTorch's NNPACK path makes Torch CPU about 10x slower from batch 16
on. That is a host setting and was left alone.

Uses [CNN support and PP-OCR](vision-cnn.md). Validated-by
`log/ppocr-mlx-m1pro-20261110/` and the 28 tests in `tests/test_vision_spatial2d.py`,
`tests/test_ppocr_axon.py` and `tests/test_agents_policy_guards.py`. Confidence: high for
the recorded measurements. Timing applies to this M1 Pro only. Last confirmed: 2026-11-10.

## 2026-11-10 — PP-OCR: compiler fixes, and MLX against ONNX Runtime CPU with memory

Fixed two of the compiler findings from the PP-OCR work: keyword terminals now end at a
word boundary (`downsample`, `format`, `nullable` parse), and a call whose dim
substitution would read a callee symbol as a caller symbol of the same name is no longer
inlined. Not fixed: loop carries typed as shape-invariant (needs a typing decision) and
call results typed with callee names (types only). Added `scripts/ppocr_mlx.py compare`.

Observed on the M1 Pro: MLX is 3.0x to 5.8x faster than ONNX Runtime CPU and peaks at 1.2x
to 2.2x its process memory with MLX defaults. The weights are 30 MB. The peak is
caused-by Metal command buffers holding intermediate results and by the unbounded MLX
buffer cache. With `--ops-per-buffer 1 --cache-mb 64` MLX is 2.5x to 5.2x faster at 0.6x to
1.2x the memory. Float16 is 2.6x to 6.2x faster at 0.5x to 1.1x. Importing
`synapse.axon.codegen2_common` through the package cost 300 MB per MLX process (PyTorch).
The harness now loads a copy written by `build`.

Uses [CNN support and PP-OCR](vision-cnn.md) and [maintained scripts](scripts.md).
Validated-by `log/ppocr-mlx-m1pro-20261110-b/`, `tests/test_axon_cnn_compiler_regressions.py`
and the passing `tests/test_axon_graph_ir.py`, `tests/test_synapse_axon_parser_roundtrip.py`
and `tests/test_synapse_axon_optimize.py`. Confidence: high for the recorded measurements on
this M1 Pro. The best MLX memory settings are machine-specific. Last confirmed: 2026-11-10.

## 2026-11-11 — PP-OCR on real pages: MLX 6x faster than ONNX Runtime CPU, cache limit needed

Ran the full pipeline over 19 real page images (959 text lines, internal test set) with the
new `scripts/ppocr_mlx.py pages`.

Observed: steady pass 4.66 s on Axon MLX float32 against 27.77 s on ONNX Runtime CPU (6.0x),
all 959 lines identical, peak memory 1052 against 1024 MB. Float16 3.95 s (7.0x), 865 MB, 8
lines differ. The recognizer is 88% of MLX time. Pipeline overhead is under 4%. With MLX's
unbounded buffer cache the same run reached 24 GB and took up to 52 s per pass, caused-by 55
distinct batch shapes. Fixed-by a default cache limit of 256 MB and 2 operations per Metal
command buffer in the harness. Supersedes the 2026-11-10 statement that these options are
not harness defaults.

Uses [CNN support and PP-OCR](vision-cnn.md) and [maintained scripts](scripts.md).
Validated-by `log/ppocr-pages-m1pro-20261111/` (`pages.json`). Confidence: high for this
machine and page set. 19 pages from one source. Last confirmed: 2026-11-11.
