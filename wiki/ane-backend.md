# ANE backend exploration (2026-11-22)

Tier-1 feasibility spike for serving the privacy-filter encoder on the Apple
Neural Engine. Topic page; episodic details live in this page until folded
into `log.md` by whichever session owns that entry.

## Hardware (this machine)

- Apple M3 Max, macOS 27.2 — 16-core ANE present. Confirmed via
  `ioreg` ANE service nodes (`ANE`, `ANE-FAST-AF-BW`, `ANEClientHints`, ...).
- MLX does not use the ANE; today's serving path (codegen2-mlx) is GPU-only.
  ANE access requires the Core ML stack.

## Access paths to the ANE

1. PyTorch → `torch.jit.trace` → `coremltools ct.convert(mlprogram)` → Core ML.
   ANE code generation happens on-device (ANECompilerService) at load time.
2. Direct emission of the documented Core ML format (`MILSpec.Program`,
   spec v6+) — container format fully documented
   (https://apple.github.io/coremltools/mlmodel/Format/Model.html); the MIL
   op vocabulary is NOT in the docs and must be sourced from coremltools'
   registry + round-trip validator.
3. onnxruntime CoreMLExecutionProvider — drives Core ML (and the ANE) from
   ONNX directly. Valid `MLComputeUnits` option values (read from the
   onnxruntime 1.24.3 binary): `CPUOnly`, `CPUAndGPU`, `CPUAndNeuralEngine`.
   `MLProgram` is not a valid option name in 1.24.

## Findings so far (all reproduced, log/probe-ane-spike.py + log/probe-ane-onnx.py)

- **onnxruntime's CoreML EP silently falls back to CPU.** When session init
  fails, ORT prints an `EP Error` and retries with CPUExecutionProvider —
  `session.get_providers()` must be checked after every init. Reproduced with
  `model_q4.onnx` (int4 MatMulNBits graph): `SystemError : 20` → CPU fallback
  → the "ANE run" was pure CPU int4 (minutes per parity set at 100% CPU).
  The q4/int4 graph is rejected by the EP; fp16/fp32 graphs are the supported
  ANE input. Valid `MLComputeUnits` values (from the 1.24.3 binary):
  `CPUOnly`, `CPUAndGPU`, `CPUAndNeuralEngine` (CamelCase; `MLProgram` is not
  an option name in 1.24).
- **ONNX exports in `models/privacy-filter/onnx/` were split-chunked; the
  missing fp16 chunks are now downloaded** (huggingface_hub snapshot_download,
  `onnx/model_fp16.onnx*` pattern). `model_fp16.onnx` + `model_fp16.onnx_data`
  is the complete ANE instrument for privacy-filter; probe:
  `python log/probe-ane-onnx.py --model model_fp16.onnx`.
- **load_torch_model cannot serve opf-keyed checkpoints for privacy-filter.**
  Its safetensors loader expects HF-style keys (`model.embed_tokens.weight`,
  `score.weight`) while the compiled graph's parameter paths are opf-style
  (`embedding.weight`, `unembedding.weight`). Workaround used by the spike:
  compile via `_compile_graph_program` + emit, then load
  `SAX._convert_hf_to_opf_state_dict(...)` (scripts layer, MLX arrays)
  converted to torch fp32 through `from_state_dict` directly. The torch
  artifact itself is correct: fp32 CPU vs HF bf16 (mps) argmax mismatches
  0/32, 0/60, 0/128, 2/483 (near-ties) on the standard case texts.
- **The axon torch artifact is not Core ML-convertible as emitted.**
  coremltools' torch frontend rejects it with bool-tensor gathers
  (`gather_along_axis` input `x` dtype bool, names like
  `..._pad_keep`). The emitted mask arithmetic (torch.where/mask paths,
  sdpa helper's dtype branches) needs dtype casts before this converts.
  Record for Tier 2: an ANE-convertible emission mode, gated as opt-in-only
  intrinsics if it changes the artifact contract.
- **The HF architecture (trust_remote_code) is not trace-convertible.** It
  uses the modern transformers mask machinery
  (`create_bidirectional_sliding_window_mask`, BlockMask). With
  `attn_implementation="eager"` and a `new_ones`→`ones` shim it still fails:
  a rank-0 fp32 scalar reaches `fill` as `shape` (data-dependent sizing).
- **coremltools 9.0 dropped ONNX conversion** (`source` accepts only
  tensorflow/pytorch/milinternal), so the checkpoint's shipped ONNX exports
  cannot go through coremltools.
- **ONNX exports in `models/privacy-filter/onnx/` are split-chunked; only
  `model_q4.onnx` is complete.** `model.onnx`, `model_fp16.onnx`,
  `model_q4f16.onnx` reference external-data chunks (`model.*.onnx_data_1`)
  that were not fetched. Re-download with a chunk-including pattern if the
  fp16 variant is wanted. model_q4.onnx: dynamic [batch, seq] int64
  inputs, logits [batch, seq, 33]; q4 quantization is also the classic ANE
  serving format, so it is a fair instrument.

## Environment deltas

- `pip install coremltools` → coremltools 9.0 (torch 2.8 untested-but-adjacent
  warning; 2.7 last tested).
- `pip install onnxruntime` → 1.24.3, CoreMLExecutionProvider available.

## fp16 CoreML EP results — privacy-filter (2026-11-22, log/ane-spike/run-fp16.log)

Instrument: `model_fp16.onnx` (~2.8 GB fp16 ≈ 1.4B params) via ORT
CoreMLExecutionProvider, `MLComputeUnits` forced per session; MLX loader
defaults (bf16) as the incumbent. Parity vs HF bf16 (mps) on identical
padded inputs, valid positions only.

Parity: **clean.** ANE/GPU/CPU sessions all 0 argmax mismatches except
long S=512: 3/483 (near-ties; MLX/torch showed 2/483 on the same case).
All three sessions have near-identical deltas → same Core ML program, only
the compute unit differs.

Latency (ms, batch 1, 'short' text; ORT per-run overhead included):

| backend        | S=32  | S=64  | S=128 | S=512  |
|----------------|-------|-------|-------|--------|
| onnx coreml-ane| 42.1  | 61.3  | 85.6  | 375.6  |
| onnx coreml-gpu| 49.1  | 67.8  | 92.3  | 309.3  |
| onnx cpu       | 40.9  | 59.7  | 80.4  | 267.4  |
| mlx-gpu (def)  | 65.2  | 65.2  | 73.7  | 86.1   |

Reading:
- MLX-GPU wins decisively at S=512 (86 vs 267–376 ms) and is flat in S
  (65→86 ms over 16× tokens) — everything stays on-GPU including the
  embedding lookup.
- The CoreML EP paths scale ~linearly in S and lose at large S; ANE is not
  even faster than plain CPU through ORT at any bucket. Suspicion: per-run
  CPU↔CoreML staging (Apple runs the embedding on CPU by design) + ORT
  session overhead dominate; the ANE session also was not verified to map
  the transformer blocks to the ANE (providers accepted, but no compute
  plan inspection yet).
- Small-S (32): CoreML paths beat MLX by ~20 ms — but that is the same
  fixed-overhead regime; treat as instrument overhead, not an ANE win.
- The energy question is still open: lower watts at equal latency is the
  ANE's pitch, and latency via this route is the *worst-case* integration
  (Python ORT). Powermetrics burn decides.

## Energy results (2026-11-22, pm-all.txt × burn-marks.log via log/parse-powermetrics.py)

Instrument note: all burns batch 1, S=128 (plus mimir S=512), 30 s each,
timestamped markers sliced against the powermetrics capture. Baseline
subtracted (cpu 1.07 W, ane 0.00, gpu 0.01).

**The ANE never ran. `ANE Power` = 0.00 W in every burn** — ORT route and
standalone `CPU_AND_NE` alike. Every "ANE" latency in this study was CPU
execution in disguise; even the standalone MLProgram with fp16 weights fell
back per-op to CPU (12,086-op graph: gathers/cumsum/mask arithmetic are
ANE-unsupported → silent fallback).

| burn (S=128)                 | tot_W | tok/s | J/tok |
|------------------------------|-------|-------|-------|
| pf onnx-coreml ane (=CPU)    | 27.4  | 1536  | 0.018 |
| pf onnx-coreml gpu           | 27.6  | 1434  | 0.019 |
| pf onnx-coreml cpu           | 28.3  | 1626  | 0.017 |
| pf mlx-gpu                   | 11.6  | 1997  | **0.006** |
| mimir standalone-coreml ane (=CPU) | 15.7 | 404 | **0.039** |
| mimir standalone-coreml gpu  | 24.8  | 293   | 0.085 |
| mimir standalone-coreml cpu  | 16.8  | 403   | 0.042 |
| mimir mlx-gpu prefill        | 27.5  | 436   | 0.063 |
| mimir standalone-coreml ane S=512 (=CPU) | 19.0 | 451 | 0.042 |
| mimir mlx-gpu prefill S=512  | 37.1  | 660   | 0.056 |

Fairness caveat: the standalone instrument emits last-token logits only,
while the MLX artifact materializes full [S, 262144] logits (lm_head ≈
206 GFLOP at S=512) that serving does not need — part of MLX's mimir-prefill
wattage is avoidable with a last-token-logits artifact option.

Conclusions:
1. **Tier-2 go/no-go gate: NO-GO on current evidence.** The ANE provides zero
   measured benefit (never engages) for these architectures as-is; revisit
   only with an ANE-friendly graph (fused mask construction, no dynamic
   gathers in the hot path) verified by ane_power before further investment.
2. **MLX-GPU is the energy winner where it is fast** (privacy-filter
   classifier: 3× better J/tok than any Core ML route).
3. Core ML's CPU-compiled path is a real but modest mimir-prefill lever
   (~35% better J/tok than MLX at S=128/512, at 7–31% lower throughput).
4. The dominant bifrost serving energy lever remains batched prefill in the
   server: today's token-by-token path runs 16.8 tok/s (RESULTS.md) vs
   436–660 tok/s eager — 26–40× on request energy.


- Energy/latency numbers: onnx-q4 CoreML EP (ANE vs GPU vs CPU) vs MLX-GPU
  forward, per bucket 32/64/128/512 — running via log/probe-ane-onnx.py.
- Power measurements need `sudo powermetrics --samplers
  ane_power,gpu_power,cpu_power -i 500` in a second terminal during the
  probe's `--burn` modes (sudo requires a password here).

## Bifrost v0.5 (dfm_mimir) instrument (2026-11-22, log/probe-ane-mimir-onnx.py)

- The checkpoint ships no ONNX export, so the probe exports its own per
  bucket (torch.onnx.export, fixed shapes [1, 128] / [1, 512],
  attention_mask=None, use_cache=False — unpadded batch-1 prefill;
  use_cache=False keeps DynamicCache bookkeeping out of the trace, matmuls
  identical).
- Architecture facts (transformers.models.hrm_text.modeling_hrm_text):
  `HrmTextForCausalLM`, `prefix_lm=True` (token_type_ids==1 marks a
  bidirectional block; None → plain causal mask), sigmoid-gated attention
  (Qwen3Next-style gate), frozen `z_L_init` state, and H/L recurrence:
  H_cycles=2 × L_cycles=3 → per forward: 2×(3+1) stack invocations ×
  num_layers_per_stack (=2) = 16 layer-passes. Compute-equivalent to a plain
  16-layer decoder per token, spread over 8 stack calls.
- The HF impl itself is mask-machinery-light (no nonzero/masked_fill/
  new_ones hits in the modeling file) — promising for conversion, unlike the
  privacy-filter HF arch.
- Serving convention for the MLX reference (from log/bifrost-mlx/
  bench_decode.py): `synapse/models/hrm_text/generic-hrm-text.axon` +
  `models/hrm_text/bifrost_v05` opf checkpoint.
- **Torch-emitter bool-gather fix landed** (synapse/axon/codegen2_torch/
  core.py): Core ML MIL has no gather over bool tensors; the banded-mask
  builtin gathers a bool-cast padding mask (`cast padding_mask dtype="bool"`
  → `gather`), which coremltools' frontend rejects. All gather emission now
  routes through a `_gather` helper that gathers bool tensors as uint8 and
  casts back (identical values). Regression check pending (torch parity
  rerun + fresh convert attempt) once the background probes release the
  machine.

## Bifrost v0.5 prefill results (2026-11-22, log/ane-spike/mimir-measure.log)

Instrument: our own fp16 ONNX export (buckets 128/512, fixed shapes,
attention_mask=None, use_cache=False) via ORT CoreMLExecutionProvider,
vs the parent thread's serving convention (generic-hrm-text.axon +
bifrost_v05, bf16, loader defaults). Parity vs HF bf16 (mps).

Parity: clean — max|Δ| 0.33–0.39, argmax mismatches 2/128 and 2/512
(near-ties), next-token prediction correct at both buckets; ANE/GPU/CPU
sessions near-identical deltas (same Core ML program).

Prefill latency (ms, batch 1):

| backend             | S=128  | S=512  |
|---------------------|--------|--------|
| onnx-coreml ane     | 2369   | 8662   |
| onnx-coreml gpu     | 2526   | 9107   |
| onnx-coreml cpu     | 1944   | 7297   |
| mlx-gpu (loader df) | 333    | 774    |

Reading:
- MLX-GPU is 7–11× faster than every ORT/CoreML path on bifrost prefill.
  The parent's eager prefill reference (832 ms @ 512, RESULTS.md) matches
  our fresh 774 ms measurement — same machine state class.
- The lm_head (262144×1536, ~0.8 GB fp16) plus 16 layer-passes make the
  ORT/CoreML per-op overhead brutal; "Context leak detected, CoreAnalytics
  returned false" messages appeared around ANE sessions (EP state issues —
  instrument caveat).
- Latency verdict on both models: the ORT-CoreML route is not competitive
  for serving. The open question is energy (watts) — and whether the ANE
  even mapped the blocks (ane_power during the burn answers both).
- The parent's server context: today's server rides prefill through the
  token-by-token decode step (0.47 s of a 1.47 s request at 28 prompt
  tokens) — so batched-prefill improvements matter for bifrost serving
  independent of the ANE question.

## Core ML direct-conversion blockers (2026-11-22, convert-regression.log)

Regression chain after the torch-emitter bool-gather fix
(`synapse/axon/codegen2_torch/core.py`, uncommitted): the privacy-filter
artifact was re-traced and pushed through `ct.convert` until each blocker
surfaced. Three fixed, one open:

1. **Bool gathers — FIXED (the original blocker).** The frontend got past
   every gather after the `_gather` helper (uint8 + cast-back). Torch-native
   parity vs HF bf16 is unchanged line-for-line
   (torch-parity-regression2.log; 11/12 cases argmax-clean, long S=512
   2/483 near-ties — same class as before the fix).
2. **topk sorted=False — FIXED.** The IR's topk carries sorted=false; the
   torch emitter passed it through and coremltools rejects it below iOS17.
   Now emits `sorted=True` — same selected value/index pairs, same
   rationale as codegen2_tinygrad's `_topk` (comment there). Consumer here
   is MoE expert routing (128 experts, top-4), order-irrelevant.
3. **`torch.mm(..., out=y_g[...])` — FIXED.** Traced `out=` records the
   destination buffer as a third mm input; the frontend requires exactly 2.
   Replaced with slice assignment (`y_g[start:end] = torch.mm(...)`) in the
   grouped-expert fallback loop (runtime helper + both generated templates).
   Verified numerically inert: parity rerun identical to pre-change.
4. **Rank-6 reshape — OPEN, platform limit.** Core ML supports rank ≤ 5;
   the emitted graph materializes rank 6 in the position/attention
   machinery (`_def_Positions_position_ids_nomask`): reshape
   (1,14,128,64) → (1,2,7,128,1,64) [head-group split 14→2×7 + singleton]
   and (1,2,7,128,257) → (1,2,7,128,1,257) [broadcast singleton, 257=2S+1
   position table]. Diagnosis tool: log/probe-rank6-diagnose.py
   (TorchDispatchMode shape capture). Avoiding these requires emitting
   rank-≤5 broadcast formulations (merge group dims, expand instead of
   singleton reshape) — graph restructuring, i.e. Tier-2 `__coreml_*`
   intrinsic territory, gated behind the NO-GO energy verdict above.

Note for other backends: MLX/JAX topk helpers accept `sorted_` and pass it
through; mlx.topk has no sorted flag at all, so those backends never had
this blocker. The `out=` pattern is torch-specific.
