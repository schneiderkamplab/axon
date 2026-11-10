# CNN support and PP-OCR (experiment)

Owner: Axon contributors/agents. Last confirmed: 2026-11-10. Confidence: high for
numerical agreement (measured on Linux, PyTorch CPU and MLX 0.32.2 CUDA); **no Apple
Silicon timing has been taken yet**, so the speed question this experiment exists to
answer is still open. Status: experiment on branch `feat/cnn-ppocr`, not reviewed.

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
grouped convolution is good. Not decided for Apple Silicon.

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
  block explicitly.
- **Inlining can corrupt `Tensor.size` on spatial dims.** `inline_safe` replaced
  `size ref 1` in an inlined builtin with a caller symbol of the same name. With
  `GraphOptimizeConfig(inline_safe=False)` the same graph was correct. Cause not isolated
  (suspect: the inlined loop output keeps callee dim names). Worked around by making
  global average pooling and reference-sized upsampling primitives, so nothing in
  `Vision.axon` reads spatial sizes. Unfixed in the optimizer.
- A dim parameter passed a *runtime* value (`c <- Tensor.size x 3`, then `conv2d x c ...`)
  types the result with the callee's unsubstituted names. Harmless where the result only
  meets tensors of the same origin; use named config dims (`C2 = List.index STAGE_DIMS 0`)
  where the width must be tracked.
- Kernel arguments must be declared `Dim` in a builtin signature for the type rule to see
  literal values; `Int` parameters reach it as bare names.
- A top-level definition whose name starts with `do` (for example `downsample`) is lexed
  as the `do` keyword and fails to parse.
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

## Open

- Apple Silicon timing against ONNX Runtime CPU and CoreML (`scripts/ppocr_mlx.py bench`).
  Depthwise convolutions dominate both backbones; their Metal performance in MLX is the
  main risk to a speedup.
- `synapse axon-benchmark` and the matrix runner cannot run these files: they are not
  language models and the Hub repo holds ONNX only. The declared `CHECKPOINTS` pragma is
  there for the policy guard.
- The two model files repeat the backbone; Axon model files cannot import each other.
- Validated-by: `tests/test_vision_spatial2d.py`, `tests/test_ppocr_axon.py`.
