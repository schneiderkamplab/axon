# CNN support and PP-OCR (experiment)

Owner: Axon contributors/agents. Last confirmed: 2026-11-10. Confidence: high for
numerical agreement (measured on Linux, PyTorch CPU and MLX 0.32.2 CUDA, and on an M1 Pro
with MLX 0.32.2 Metal); medium for speed: timed on one M1 Pro only, see
[Apple Silicon timing](#apple-silicon-timing-2026-11-10-m1-pro). Status: experiment on branch `feat/cnn-ppocr`, not reviewed.

## What exists

- Six channels-last primitives in `synapse/ops/`: `conv2d`, `conv_transpose2d`,
  `max_pool2d`, `avg_pool2d`, `global_avg_pool2d`, `upsample_nearest2d_like`; shared
  output-size typing in `synapse/ops/_spatial2d_common.py`.
- Builtin surface `synapse/builtins/Vision.axon` (also `upsample_nearest2d` by integer
  factor and `hard_sigmoid`, both derived from existing primitives).
- Runtimes `synapse/axon/codegen2_torch/spatial2d.py` (used by the graph interpreter and
  embedded by source into emitted Torch modules) and `synapse/axon/codegen2_mlx/spatial2d.py`
  (embedded into emitted MLX modules). Neither is emitted unless the graph uses a spatial
  primitive, so existing generated code is unchanged. JAX, tinygrad, Triton and vLLM
  codegens do not lower these ops.
- Models `synapse/models/ppocr/generic-ppocr-det.axon` and `generic-ppocr-rec.axon`
  (PP-OCRv6 small detector and recognizer, checkpoint `blackpen-sk/ppocrv6-small`).
- Scripts: see [maintained scripts](scripts.md).

## Decisions

- **Layout is channels-last** (`Tensor[B,H,W,C]`), with weights in the PyTorch checkpoint
  layouts (`[O,I/g,KH,KW]`, transposed `[I,O,KH,KW]`). MLX convolutions are channels-last,
  so the MLX runtime never transposes activations; it moves each weight once and caches it
  by identity. Torch permutes around each call. Rationale: the experiment targets MLX speed.
- **Pointwise convolutions lower to a matmul** in the MLX runtime (`addmm` when biased).
- **Output sizes are dim expressions.** A window extent is typed as
  `((size + span - 1) / stride) + 1` (dim `/` floors), exact for static sizes and unchanged
  when stride is 1 and padding is "same". Kernel, stride and padding must be static for
  this; otherwise the extent falls back to an opaque dim named after its inputs
  (`__sph_<hash>`), which is only sound inside one definition.
- **Batch norm is folded at conversion time**, not modelled.

## Speed levers (added 2026-11-10, all exact up to rounding)

In the MLX runtime, always on:
- a transposed convolution whose kernel equals its stride (no padding) is a matmul to
  `kh * kw * out` channels plus a depth-to-space reshape, because its windows do not overlap;
- pointwise convolutions are matmuls (above).

In the MLX runtime, opt-in through the emitted module global `_SPATIAL_DEPTHWISE_SHIFT_TAPS`
(default 0 = off): depthwise convolutions with at most that many taps become a sum of
shifted, per-channel scaled input slices. Purely elementwise, so it sidesteps a slow
grouped-convolution path where one exists and fuses under `mx.compile`; slower where the
grouped convolution is good. On the M1 Pro it never wins: uncompiled it is 35-70% slower,
compiled it is within 3% (3x3) or 17% slower (7x7, detector), so the default stays off.

In `scripts/ppocr_mlx.py`: `mx.compile` per input shape (variant `compiled`), detector
normalization on the device from uint8 pixels, recognizer argmax on the device, and
recognizer batch widths rounded up to a multiple of 32 so compiled graphs are reused.

Second round, same day:
- reference-sized nearest upsampling is one broadcast and one copy instead of a repeat per
  axis (MLX runtime, always on);
- recognizer batches are queued with `mx.async_eval` and read back once, so the host
  prepares the next batch while the device runs (lesson from the Ettin trace study, which
  measured 1.2-1.4x for pipelined decoding);
- variant `shift49+compiled` extends the shifted sum to the 7x7 depthwise convolutions;
- `scripts/ppocr_mlx.py tune` times all variants in interleaved rounds, compares fastest
  calls, and saves a per-network choice that replaces `compiled` only when 5% faster.
  Its choice was not stable on the Linux box (both GPUs were saturated by other jobs).

**float16 bug found and fixed:** `global_avg_pool2d` summed in half precision and
overflowed on real pages (a 152x240 map), turning the whole detector output into NaN;
random-noise inputs did not show it. Both runtimes now accumulate in float32, `parity`
uses the real sample page for the detector, and a test covers the overflow. float16 then
gives the same OCR text as float32 (detector max abs diff 8e-2 at text edges).

Published 2026-11-10 as the private Hub repo `blackpen-sk/ppocrv6-small-mlx`: generated modules, weights, vendored
`codegen2_common`, the harness, dictionary, the ONNX reference graphs and a sample page.
Verified from a fresh download in an environment without `torch` or `synapse`.

Measured on the Linux CUDA box only (noisy, about 2 ms floor per evaluation): end-to-end
OCR of the 6-line test page 45.7 ms before these changes, 35.9 ms plain, 23.9 ms compiled;
host preprocessing 5.4 ms to 0.3 ms. The four variants are within noise of each other in
`bench` there except uncompiled `shift`, which is slower. Text stays identical to ONNX
Runtime and all variants agree with it to 1e-5.

## Things that bit, and why the models look the way they do

- **Loop carries are typed as shape-invariant.** A `for ... carry (x)` whose body changes
  the shape of `x` compiles and runs on unoptimized Torch, but every downstream type is
  wrong and the graph optimizer miscompiles (it rewrote a size ratio to `H // H`). The
  PP-OCR files therefore loop only over shape-preserving blocks and write each strided
  block explicitly. Not fixed. Smallest known case: a carry through
  `Vision.max_pool2d x 2 2 stride_h=2 stride_w=2`, then a helper that reads
  `Tensor.size` of the result; optimized Torch returns 4x6 where unoptimized returns 16x24.
  The return check accepts any tensor of the same rank for a carry, so a fix is a typing
  decision (reject the loop, or give changed dims fresh names after it) and needs review.
- **Inlining could read a callee dim as a caller dim of the same name.** Fixed 2026-11-10.
  Callee and caller dim symbols share one namespace in a call substitution. A formal `H`
  whose actual size is an expression over the caller's `H` (`((H - 2) / 2) + 1` after a
  pool) gets no binding, because the binding would be a cycle; inlining then turned the
  callee's `H` into the caller's `H` and folded a size ratio to 0. Fixed-by
  `_call_dim_subst_captures_formal_dim` in `synapse/axon/graph_ir/optimize.py`: such a
  call is no longer inlined. Validated-by `tests/test_axon_cnn_compiler_regressions.py`.
  Generated code is unchanged for 26 sampled language models; in the PP-OCR files 40
  (detector) and 32 (recognizer) inline attempts are now refused, with no measurable
  change in speed. Global average pooling and reference-sized upsampling stay primitives.
- **Call results are typed with the callee's unsubstituted names.** Same cause, not fixed:
  with the binding missing, `Vision.max_pool2d y` is typed like `y`'s producer, and a dim
  parameter passed a *runtime* value (`c <- Tensor.size x 3`, then `conv2d x c ...`) keeps
  the callee's names. The types are wrong but, after the fix above, no code is derived
  from them across a call. A real fix gives callee symbols their own namespace at each
  call. Use named config dims (`C2 = List.index STAGE_DIMS 0`) where a width must be tracked.
- Kernel arguments must be declared `Dim` in a builtin signature for the type rule to see
  literal values; `Int` parameters reach it as bare names.
- **A name that starts with a keyword failed to parse.** Fixed 2026-11-10. The lexer cut
  `downsample` into `do` and `wnsample`; `format`, `forward`, `typed`, `nullable` and
  similar names failed in some positions too. Keyword terminals in
  `synapse/axon/parse/grammar.py` now end at a word boundary. All 300 `.axon` files in the
  repository parse to identical trees before and after.
- Paths with a computed index need the quoted form on a scope:
  `scope@'backbone.blocks.{e2}' do`. `f@'...'` call sugar does not accept it.

## Validation (2026-11-10, Linux, RTX 4090, MLX 0.32.2 CUDA, `MLX_ENABLE_TF32=0`)

| Check | Result |
|---|---|
| Generated Torch vs ONNX Runtime, detector, 3 input sizes | max abs diff 2.9e-6 |
| Generated MLX vs ONNX Runtime, detector | max abs diff 4.5e-6 |
| Generated Torch vs ONNX Runtime, recognizer, 4 input sizes | max abs diff 7.0e-6, argmax agreement 1.0 |
| Generated MLX vs ONNX Runtime, recognizer | max abs diff 1.0e-5, argmax agreement 1.0 |
| End-to-end OCR on a rendered 6-line page, MLX vs ONNX Runtime | identical text, boxes equal, score diff 1.8e-7 |
| float16 MLX vs ONNX Runtime | max abs diff 2.0e-2, recognizer argmax agreement 1.0 |

Random-noise detector inputs give near-zero probability maps, so the rendered page is the
meaningful detector check. Without `MLX_ENABLE_TF32=0` the CUDA build of MLX differs by up
to about 5e-2 on synthetic weights.

Timing on this machine compares MLX on a CUDA GPU with ONNX Runtime on CPU and says nothing
about Apple Silicon: detector 1x736x960 about 17 ms vs 70-80 ms, recognizer 6x48x320
3.6-6.9 ms vs about 22 ms across two runs. Copying the recognizer's full probability tensor to NumPy cost more than the
network (41 ms for 16x60x18710), so the harness takes the argmax on the device.

## Apple Silicon timing (2026-11-10, M1 Pro)

M1 Pro, 32 GB, macOS 27.0.1, AC power, Python 3.13.9, MLX 0.32.2 (Metal), ONNX Runtime
1.30.0, PyTorch 2.14.0. Run directory `log/ppocr-mlx-m1pro-20261110/` (`bench.json`,
`tuned.json`, `ocr-*.json`, `bench-torch.json`). Median ms per call, float32, random
inputs; MLX and ONNX Runtime rows from `scripts/ppocr_mlx.py bench` (15 samples), Torch
rows from an ad hoc script over the same generated-code path as the tests (10 samples).

| Workload | Axon MLX `compiled` | Axon MLX `base` | ONNX Runtime CPU | ONNX Runtime CoreML | Axon Torch MPS | Axon Torch CPU |
|---|---|---|---|---|---|---|
| det 1x736x960 | 22.3 | 24.3 | 100.7 | 135.7 | 34.4 | 172.9 |
| det 1x960x960 | 28.3 | 30.8 | 127.7 | 174.0 | 43.2 | 214.1 |
| rec 1x48x320 | 3.3 | 4.5 | 9.7 | 17.4 | 10.0 | 34.9 |
| rec 6x48x320 | 10.2 | 10.9 | 57.9 | 79.5 | 19.1 | 72.3 |
| rec 16x48x480 | 34.5 | 35.7 | 190.3 | 288.5 | 50.9 | 1476 (150 without NNPACK) |

- **Axon MLX is 2.9x to 5.7x faster than ONNX Runtime CPU** (detector 4.5x, batched
  recognizer 5.5x) and 1.5x to 3x faster than the generated Torch on MPS.
- End-to-end OCR of the 6-line sample page (`ocr --repeat 5`, last run): 65.1 ms on MLX
  (detector 23.7 ms, recognizer 37.1 ms) against 323.8 ms on ONNX Runtime CPU; identical
  text and scores. float16 MLX: 43.6 ms, same text.
- Parity on Metal: every variant within 4.7e-5 of ONNX Runtime, recognizer argmax
  agreement 1.0; float16 within 4.4e-2 on the sample page.
- `tune` keeps `compiled` for both networks. float16 is about 18% faster than float32.
- The ONNX Runtime CoreML provider is slower than its CPU provider here: it splits the
  graphs into 19 (detector) and 27 (recognizer) partitions. Default provider options only;
  the MLProgram format and a native CoreML conversion were not tried.
- The always-on MLX lowerings hold on Metal: routing pointwise convolutions through
  `mx.conv2d` instead of a matmul is 5-14% slower; `mx.conv_transpose2d` instead of the
  matmul form is 6% slower on the detector.
- Depthwise convolutions, the risk named before this run, are not a problem on Metal.

**Torch CPU fix (same day).** The Torch runtime handed `F.conv2d` a permuted view, which
Torch reads as channels-last memory; its CPU grouped convolution is about 50x slower there
than on contiguous NCHW (59 ms against 1.1 ms for one depthwise 3x3 at 184x240x96). The
runtime now copies the input of grouped convolutions to contiguous NCHW: detector
715 -> 173 ms, recognizer 6x48x320 385 -> 72 ms on CPU, unchanged on MPS. Fixed-by
`synapse/axon/codegen2_torch/spatial2d.py`. Torch CPU stays slower than ONNX Runtime
because of its 7x7 depthwise convolutions (about 37 ms each when contiguous).

**PyTorch NNPACK cliff (host setting, not changed).** From batch 16 on, PyTorch's CPU
convolution selects NNPACK, which is about 10x slower on this machine: recognizer
16x48x480 takes 1415 ms, and 150 ms after `torch.backends.nnpack.set_flags(False)`.
Lowering pointwise convolutions to `F.linear` avoids part of it (570 ms) but costs about 5%
at smaller batches and on MPS, so it was not adopted.

## MLX against ONNX Runtime CPU: speed and memory (2026-11-10, M1 Pro)

Same machine and versions as above, generated code rebuilt after the compiler fixes. Run
directory `log/ppocr-mlx-m1pro-20261110-b/` (`compare.json`, `compare-cache64-ops1.json`).
`scripts/ppocr_mlx.py compare`: 30 samples, float32 unless noted, one fresh process per
backend and workload. Memory is the peak physical footprint of the process, which
includes Metal buffers; the resident set size does not (it reads 160-190 MB for MLX).

| Workload | MLX ms | MLX MB | MLX float16 ms | MLX float16 MB | ONNX CPU ms | ONNX CPU MB |
|---|---|---|---|---|---|---|
| det 1x736x960 | 22.3 | 639 | 18.5 | 471 | 96.6 | 337 |
| det 1x960x960 | 28.6 | 784 | 23.2 | 556 | 123.2 | 398 |
| rec 1x48x320 | 3.3 | 179 | 3.0 | 189 | 9.8 | 145 |
| rec 6x48x320 | 10.3 | 321 | 8.7 | 273 | 58.3 | 196 |
| rec 16x48x480 | 34.3 | 844 | 28.4 | 633 | 190.8 | 458 |
| OCR of the sample page | 52.5 | 1133 | 43.3 | 782 | 306.6 | 528 |

With MLX defaults, MLX is 3.0x to 5.8x faster and uses 1.2x to 2.2x the memory. The
weights are 30 MB (15 MB in float16) and a loaded MLX process is 75 MB, less than a loaded
ONNX Runtime process (82-92 MB). The rest has two causes, both settings of MLX and not of
the generated code:

- **Command buffers.** MLX commits its Metal command buffer after a fixed number of
  operations, and intermediate results stay allocated until then. The peak inside one
  detector call is 523 MB; with `MLX_MAX_OPS_PER_BUFFER` set to 8, 4, 2 and 1 it is 327,
  279, 237 and 188 MB (recognizer 16x48x480: 728 MB down to 193 MB), for up to 6% more time.
- **Buffer cache.** MLX keeps freed buffers for reuse without a limit. After a page with
  several input shapes the cache holds 1 GB. `mx.set_cache_limit` bounds it; a limit below
  the per-call peak costs time (0 MB: detector 22 -> 30 ms).

`compare --ops-per-buffer 1 --cache-mb 64` (the harness sets both):

| Workload | MLX ms | MLX MB | MLX float16 ms | MLX float16 MB | ONNX CPU ms | ONNX CPU MB |
|---|---|---|---|---|---|---|
| det 1x736x960 | 25.1 | 373 | 21.1 | 257 | 98.2 | 343 |
| det 1x960x960 | 32.1 | 419 | 26.8 | 280 | 123.3 | 397 |
| rec 1x48x320 | 4.0 | 164 | 3.7 | 163 | 9.8 | 146 |
| rec 6x48x320 | 11.5 | 233 | 10.0 | 200 | 57.6 | 194 |
| rec 16x48x480 | 38.0 | 367 | 32.7 | 332 | 188.2 | 452 |
| OCR of the sample page | 58.7 | 321 | 48.8 | 283 | 304.8 | 527 |

With these settings MLX is 2.5x to 5.2x faster at 0.6x to 1.2x the memory of ONNX Runtime
(float16: 2.6x to 6.2x faster at 0.5x to 1.1x). The OCR text is identical in every run.
The two options are not defaults of the harness: the best value depends on the machine.

**Importing the helper module cost 300 MB.** A generated MLX module imports
`synapse.axon.codegen2_common`, which needs only the standard library, but importing it
through the package also imports PyTorch and the compiler: a loaded MLX process was 376 MB
instead of 77 MB. `build` now writes `axon_codegen2_common.py` into the run directory and
the harness loads it under the helper's name. Other hosts of generated MLX code pay this
cost unless they do the same; the package `__init__` files are unchanged.

## Open

- Timing on other Apple Silicon machines, and a CoreML baseline that is not partitioned
  (MLProgram format or a native conversion).
- `synapse axon-benchmark` and the matrix runner cannot run these files: they are not
  language models and the Hub repo holds ONNX only. The declared `CHECKPOINTS` pragma is
  there for the policy guard.
- The two model files repeat the backbone; Axon model files cannot import each other.
- Validated-by: `tests/test_vision_spatial2d.py`, `tests/test_ppocr_axon.py`.
