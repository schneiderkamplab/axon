# Serving kernel order-contracts (batched ≡ unbatched, bitwise)

Canonical design page for execution-shape generality in the serving stack.
Decision context: `wiki/log.md` 2026-11-24 entry (root cause + user direction
"keep the compiler fully general"). This page states the contract once and
phases the work; the Bifrost batched prefill is the first instance, not a
special case.

## The contract

1. **Order is part of a kernel's spec.** Every reduction kernel in the
   serving stack (q4 gemv/gemm family, fattn, fused rms/residual variants)
   fixes, per output element, the exact sequence of floating-point adds:
   K-loop order, per-lane partial chain, and the simd reduction tree.
2. **Shape-variante siblings preserve it.** An M=S (or B>1) variant of a
   kernel must produce, per output element, the identical accumulation
   sequence as its M=1 sibling. Then batched and unbatched executions are
   bitwise identical *by construction*, and execution shape stops being a
   correctness question — only a performance question.
3. **Gates compare same shapes.** Parity/reference gates always run the same
   execution shape on both sides (the r4 rule, generalized). Cross-shape
   differences (e.g. the Python server's artifact-SDPA prefill vs the
   engine's kernel-set prefill) are documented op-set differences, not bugs.
4. **Generality lives below the compiler.** No shape- or model-conditional
   behavior in emitter/compiler/graph-builder policy layers. Kernels take
   M, S, window, group size as runtime parameters; graph builders take S as
   a build parameter and cache per shape. The artifact stays S/B-generic.

## Why this is cheap for the q4 gemv family

`qgemv_v6` (and v6r/v7b/v8a) structure: one simdgroup per pair of output
elements; per-lane strided K-loop (`b8 = lane*8; b8 += 256`), scalar fp32
FMA chain in fixed order, `simd_sum` at the end. **No part of that
per-element sequence depends on M.** The M=S sibling is the same body with a
2D grid — threadgroup `(pair, m)` computes the same two outputs for input
row `m` (fused variants: the fused ssq/residual are per-row too). Bandwidth
is unchanged vs stepwise (each threadgroup re-reads its q rows per input row
either way); the win is eliminating S−1 whole-graph dispatches and per-step
syncs (~0.47 s of a 1.47 s request at 28 prompt tokens).

Attention needs no new math: per-query attention given written KV is
independent, so the prefill graph reuses fattn r4 per query row — bitwise
identical to stepwise provided the per-query ring/window iteration order
matches (the wrap case S>W is the one to prove).

## Phases

### Phase 1 — Python prototype (prove the contract on real weights)
- `make_qgemm` family in `log/bifrost-mlx/kernel_qgemv.py` style: the
  v6/v7b/v6r/v8a bodies with an m-indexed 2D grid.
- Bitwise A/B vs the gemv kernels row-by-row on real Bifrost weights,
  M ∈ {1, 2, 28, 128, 512}.
- **Gate:** exact equality everywhere; batched not slower per row.

### Phase 2 — Rust kernels
- Port the m-indexed variants through the export flow
  (`export_msl.py` → `msl.rs`), wrappers in `kernels.rs` take M.
- **Gate:** same bitwise A/B as a Rust unit test (random q4 packs + real
  weights).

### Phase 3 — Rust prefill graph + engine wiring
- S-parameterized prefill graph in `graph.rs`: same 128-slot unroll, M=S
  matmuls, per-slot prefill attention (S queries, r4 per-query order),
  KV ring writes for S positions, lm_head only on the last row (M=1 gemv on
  the last hidden state — bitwise same as today's last prefill step),
  last-row argmax out.
- Engine: `prefill(ids)` API mirroring `Session.prefill` semantics;
  `generate()` consumes it.
- **Gates:** (a) bitwise self-consistency — batched-prefill KV + last
  logits == stepwise loop, for S ∈ {1, 2, 28, 129 (wrap), 512};
  (b) `tests/generate` / `gref/ids.bin` unchanged and passing;
  (c) workspace tests green.

### Phase 4 — bench + record
- Prefill wall time vs the 0.47 s stepwise baseline (28 tokens); end-to-end
  request time; `log/bifrost-mlx/RESULTS.md` + `wiki/log.md` entries.

### Phase 5 — later generalization (explicitly out of the first pass)
- Dense bf16 GEMM order-contract kernel (serving dense-artifact models
  through the Rust stack).
- Python server batched prefill via the same kernels, if server↔engine
  bitwise ever becomes a requirement.
- Single prefill-attention kernel looping queries internally (perf only,
  after bitwise correctness is proven with per-query execution).

## Rust generalization ladder (decision 2026-11-25)

Mechanism decision (user): **family templates first** — config-driven backend
with per-family graph templates; the artifact-IR interpreter stays a later
option until kernel coverage justifies it.

- **R0 = this page's Phases 1–4** (order-preserving M=S kernels +
  S-parameterized prefill graph). Target is Bifrost, but the kernels and
  builder come out shape-generic — this *is* the first generalization step.
- **R1 — trait seam + config-driven dense-LM backend.** Extract the
  announced `Decoder/Backend` trait (`axon-engine/src/lib.rs:6` names it);
  replace `consts.rs` with config.json-driven geometry (D/NH/HD/kv_heads/
  layers/vocab/EOS/dtype/quant); port the weight-name mapping from Python
  `server/loading.py`; parameterize `fattn`'s HD=128 and the gemv family.
  Proof target: **qwen3** (checkpoint on disk; serving + parity probes
  exist under `log/qwen3-*`). Every new model gets the Bifrost gate
  pattern: ktest/refstep/tokref/chatref/gref-style dumps, bit-exact.
- **R2 — MoE + quantization breadth.** Metal gather_mm-equivalent routing
  (the MLX gather_mm emitter path is the reference pattern); int8 path.
  Needs a downloaded MoE checkpoint (matrix machinery drives that).
- **R3 (parked) — artifact/graph-IR-driven interpreter**: compiler emits
  serialized IR + materialized constants; Rust interprets. Revisit when
  family-template count makes hand templates the bottleneck.
- **Parked in Rust entirely**: enc-dec, SSM/hybrid, mrope, embedding
  serving — the Python server covers them. Chat templating switches to a
  real Jinja engine (minijinja) at R1 — first proof target: the gemma-4
  template family used by Bifrost + HRM-Mimir (the current
  `chat_template.rs` is a hand-port of it). Parser-spec config keys ride
  the same metadata-driven load path (`wiki/response-parsers.md`).


## Non-goals / guards

- No emitter or compiler changes; no model-family branches anywhere
  (policy-guarded — see `tests/test_backend_intrinsic_grants.py` for the
  pattern).
- The MLX serving path (`server/engine.py`) is unchanged; its op-set
  difference is documented, not fixed.
