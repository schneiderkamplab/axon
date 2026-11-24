# Log

Episodic working notes. One entry per investigation; durable conclusions move to
topic pages and are linked from here.

## 2026-11-22 — `__mlx_rope` is now opt-in-only; torch-emitter Core ML blockers fixed

- `synapse/axon/graph_ir/optimize.py`: `__mlx_rope` joined
  `__torch_paged_attention` in `_OPT_IN_ONLY_BACKEND_INTRINSICS`. Rationale:
  it emits per-instance frequency-table caches and decode-offset machinery
  that are only correct under serving execution, so it must never ride an
  "all fast-paths" default grant. The remaining MLX intrinsics are stateless
  function rewrites and stay default-granted. (ANE spike context: this is
  the serving-specific-intrinsic flag-gating decision, recorded here as
  promised; see `wiki/ane-backend.md` for the Tier-2 NO-GO that followed.)
- `synapse/axon/codegen2_torch/core.py` — three Core ML conversion blockers
  addressed while regression-testing the bool-gather fix: bool gathers
  (fixed via `_gather` uint8+cast), `topk sorted=True` coercion (same
  rationale as tinygrad's `_topk`), and grouped-expert `mm(out=…)` replaced
  with slice assignment (traced `out=` becomes a third mm input the
  frontend rejects). Torch-native parity vs HF bf16 unchanged line-for-line
  after all three (`log/ane-spike/torch-parity-regression2.log`); the next
  blocker is a rank-6 reshape platform limit (open, Tier-2 territory,
  NO-GO-gated — details in `wiki/ane-backend.md`).
- `log/ane-spike/` evening addition — ANE residency diagnostics: the
  compute-plan tool verified against canary controls; mimir mlprogram
  0/12,141 ANE ops; classic `neuralnetwork` canary 20/21 layers on the
  ANE → the NN format is the ANE highway on this stack. Torch backend now
  emits `F.rms_norm` (ANE-aware converter decomposition; parity unchanged).
  Full details + next gates in `wiki/ane-backend.md`.

## 2026-11-20 — privacy-filter serving: `/v1/classify` end-to-end (python)

Goal: serve `openai/privacy-filter` (gpt-oss-style bidirectional token
classifier, 33 BIOES labels over 8 categories) through the Python server.
User-approved on the decision page: (1) land the codegen2-mlx rope fix +
regression gate, (2) classify route in the Python server with model-kind
dispatch, (3) constrained Viterbi decoding in the served path. Confidence:
high — every claim below is gated or probe-verified today.

- **Rope fix landed and gated** (`synapse/axon/codegen2_mlx/core.py`
  `_mlx_rope_apply`): inv_freq recovery must use stride-2 (`sin[0,0,1,::2]`)
  for interleaved/traditional tables (each frequency duplicated adjacently),
  stride-1 `:half` stays NeoX; same stride split in the S==1 offset path.
  Second bug caught before landing the gate: yarn mscale was recovered as
  `cos[0,0,0,0]`, only valid at position 0 — decode-time single-row tables
  are built at position p. Replaced with amplitude form
  `sqrt(sin²+cos²)` (yarn builders scale both sin and cos by
  attention_factor, tables fp32 ⇒ amplitude = mscale at any row;
  position-independent). Offset recovery verified truncation-safe (max
  ratio = p + tiny slack, p ∈ {1,5,37,100,255}). Gate:
  `tests/test_codegen2_mlx_rope_apply.py` (6 tests; synthetic
  `GraphOp("__mlx_rope")` — double underscore, normalization strips one for
  both the gate and the emitter branch — → emit → exec → forward vs manual
  rotation). End-to-end parity unchanged after the amplitude change: v0/v1/v3
  0/60 short, 4/483 long (same near-ties), perf-neutral (82.9→82.1 ms S=512).
- **Classify module** (`server/classify.py`): `is_token_classifier` dispatch
  is generic metadata reading — `id2label` with >1 entry marks a token
  classifier, `*ForTokenClassification` architecture as fallback; no
  model-family string checks anywhere. `constrained_viterbi_decode`: generic
  BIOES linear-chain DP (numpy) with start/transition/end terms; the six
  calibration biases apply to transitions only, not the start term
  (validated against the model card's wording): `background_stay`=O→O,
  `background_to_start`=O→B/S, `end_to_background`=E/S→O,
  `end_to_start`=E/S→B/S, `inside_to_continue`=B/I→I, `inside_to_end`=B/I→E.
  `group_entities`: char spans via tokenizer offset_mapping, stray I/E
  salvage. `ClassifySession`: same compiled artifact forward-only, warmup,
  per-text forward under a lock, `decoder`=`viterbi`|`argmax`, optional
  `max_length`; dtype follows the checkpoint's declared `dtype` (bfloat16),
  not the engine's float16 default. Server wiring (`server/server.py`):
  `kind` on `_ServeState`, `/v1/classify`, mutual 400s naming the right
  route, `/health`+`/v1/models` report `task`, fleet rejects classifiers
  with a clear error. Unit gates: `server/tests/test_classify_decoder.py` +
  `server/tests/test_serving_classify.py` (12/12, incl. an HTTP-routing test
  over fakes on an ephemeral port, mlx-free). En-route catch: the Viterbi DP
  gather was row/col-swapped (`candidates[np.arange(count), bp]` →
  `candidates[bp, np.arange(count)]`) — the all-O flattening symptom in the
  unit tests pinned it.
- **Loading bug 1 — MLX `__setattr__` trap in the emitted template** (this
  was the `/v1/classify` 500: `_embedding() missing 1 required positional
  argument 'ids'`): the emitted `load_state_dict` installs shadow modules
  (`self._embedding = nn.Embedding(...)`) that make one-argument call sites
  resolve. `nn.Module` subclasses `dict`, so a module value always takes the
  container branch of `Module.__setattr__`; when the name only exists as a
  class attribute (the always-emitted `_embedding` helper method), its
  stale-attr guard `delattr`s that name and raises AttributeError. The
  harnesses always drove these classes with a `__setattr__` fallback patch,
  so the trap only surfaced on the server's direct construction. Fixed in
  the emitter template: `_bind_static_module` helper (setattr →
  `object.__setattr__` on AttributeError); a container entry could not
  shadow a class method anyway (normal lookup finds methods before
  `__getattr__`), so the instance attribute is the only binding that
  satisfies the call sites. Gate:
  `tests/test_codegen2_mlx_static_module_binding.py` — synthetic static-
  embedding graph (constant `GraphPath("embedding")` input → collector
  registers the static op → emitter emits the one-arg call), unpatched
  `from_state_dict` + one-arg forward; plus the no-shadow case keeps the
  two-arg helper bound. Parity re-verified after the template change
  (v0/v1/v3 identical to pre-change).
- **Loading bug 2 — checkpoint key naming**: `generic-privacy-filter.axon`
  binds gpt-oss-derived opf paths (`embedding.weight`, `block.N.attn.qkv.*`,
  `norm.scale`, `unembedding.weight`) while the HF checkpoint ships raw HF
  names (`model.embed_tokens.weight`, separate q/k/v projections,
  `score.weight`). The rename+concat mapping lives (correctly) in the
  model-specific harness `scripts/stream_privacy_filter_axon.py`
  (`_convert_hf_to_opf_state_dict`). Resolution: materialize once —
  `scripts/materialize_privacy_filter_opf.py` writes
  `models/privacy-filter/opf/` (opf-keyed bf16 safetensors + copied
  config/tokenizer/calibration; the `bifrost_v05` self-contained variant-dir
  pattern) — and the server stays fully generic (no model-specific code in
  `server/`). Re-run the script after replacing the HF download.
- **Compile-settings parity** *(superseded 2026-11-21 — see the flag-correction
  addendum below: the ad-hoc `ClassifySession` pin was removed and the bare-grant
  gating fixed; the loader defaults now reproduce this artifact exactly)*: the
  loader defaults
  (`MLX_INTRINSICS_SPEC = "codegen2-mlx:__mlx_sdpa,__mlx_rope"` +
  `optimize_safe_flat_typed_axon_file`) emit a *different artifact* than the
  parity-tested harness pipeline (full `codegen2-mlx` intrinsics, no AST
  optimization): restricted spec leaves gegelu/MoE unfused with extra
  `_def_*` bodies, and was never parity-verified for this model.
  `ClassifySession.from_axon` now pins `optimize_ast=False,
  backend_intrinsics="codegen2-mlx"`; the resulting server artifact is
  **bit-identical** to the stream artifact (max|Δ|=0.0 on a probe input).
- **End-to-end** (`log/probe-serve-classify.py`, server + materialized
  checkpoint): `/health` ✓, `/v1/models` task=token-classification ✓,
  `/v1/completions` rejected 400 ✓; argmax vs HF bf16: 0/60 short, 4/483
  long (same near-ties as the parity probe), 0/12 clean; viterbi groups the
  expected entities and recovers the full `55102` account number where raw
  argmax broke the span at `551`; latency 227 ms viterbi / 584 ms argmax
  (first batch includes fast-path compile warmup) for 3 texts / 555 tokens.
  Probe gotchas that cost time: raw emitted-array outputs are `[1,S,V]`
  (index `[0]` twice), and a 1-token sequence can never take a transition
  bias (test expectation, not a bug).
- **Pre-existing failure flagged, out of scope**:
  `tests/test_axon_graph_ir.py::test_codegen2_mlx_materializes_final_logits_bias_flat_alias`
  fails at clean HEAD (verified in a throwaway worktree at `dc0077e`):
  a `_linear` node with a bare `GraphPath` as `inputs[0]` loses the base in
  `_compose_path` (`@weight` used raw → KeyError). Emit-pipeline operand
  rendering gap, unrelated to today's changes; needs its own investigation.

### Addendum 2026-11-21 — serving-intrinsics flag correction (supersedes the compile-settings bullet above)

- User directive: intrinsics that do not work for both training and inference
  must only be grantable through the named-spec flag, never the bare target
  grant. Evidence gathered: `__mlx_rope` is the only serving-only MLX
  intrinsic — its emitted code owns per-instance state
  (`_rope_freqs`/`_rope_inv_freqs`/`_rope_cached_sin`/`_rope_cached_off`,
  first-call table freeze) and the S==1 decode-offset machinery. The other
  seven (`__mlx_sdpa`, `__mlx_rmsnorm_scaled`, `__mlx_swiglu_activation`,
  `__mlx_gegelu_activation`, `__mlx_expert_swiglu_ffn`,
  `__mlx_expert_packed_swiglu_ffn`, `__mlx_weighted_topk_sum`) are stateless
  function rewrites (three share the torch-target rewrites verbatim) and stay
  in the bare grant. `__jax_rope` has no lowering at all in codegen2_jax, so
  there is no serving-only state to protect there; set membership unchanged.
- Change: `__mlx_rope` added to `_OPT_IN_ONLY_BACKEND_INTRINSICS`
  (`synapse/axon/graph_ir/optimize.py`) — a bare `codegen2-mlx` now grants
  seven, and rope is granted only when named. `server/loading.py::
  MLX_INTRINSICS_SPEC` names the full serving set (all eight) explicitly;
  `ClassifySession.from_axon`'s ad-hoc pin
  (`optimize_ast=False, backend_intrinsics="codegen2-mlx"`) removed — loader
  defaults now carry the parity artifact. `scripts/stream_privacy_filter_axon.py`,
  `bench_mlx_privacy_filter.py`, `viz_privacy_filter.py` moved off the bare
  spec onto the named one (imported from `server.loading`, so the spec has
  one home).
- Evidence: `log/probe-intrinsics-grant-parity.py` — named-full+noast sha
  `1d22937fe92f9b7a` matches the pre-change artifact byte-for-byte (the spec
  extension is artifact-preserving); `optimize_ast` has no effect on the
  emitted artifact for this checkpoint; post-change bare grant emits a
  different, rope-less artifact (163,634 B vs 115,749 B — generic rope
  lowering is far larger); the former loader two-name spec emitted 123,325 B
  (fusions left ungranted — recorded as history).
- Gates: new policy guard `tests/test_backend_intrinsic_grants.py` (10
  tests); rope gate 6/6; static-binding gate 2/2; server suite 93/93; mlx
  subset green except the pre-existing final_logits_bias failure above;
  parity probe re-run identical (v3 4/483, same four positions); serve probe
  end-to-end pass (argmax 0/60, 4/483, 0/12; viterbi recovers `55102`;
  303 ms for 3 texts / 555 tokens through the loader-default path).


## 2026-11-23 — classify fast path: gather_mm MoE routing (4.6× end-to-end)

- User picked (A) classify fast path, (B) Bifrost decode kernels, (C) batched
  prefill — sequenced A → B → C so GPU-heavy measurement campaigns don't
  contaminate each other's timings.
- **Profile** (`log/probe-classify-profile.py`): the classify forward was
  97% Python-side — 28 ms fixed at S=12 (GPU only 2.5 ms), 98 ms at S=483
  (GPU 10 ms). Root cause: the artifact is MoE (8 layers × **128 experts,
  top-4**, hidden 640) and `_expert_linear` routed on the host: sort →
  `mx.eval` + `.tolist()` → Python loop, i.e. **16 GPU syncs per forward**
  plus generic per-node Python dispatch. The emitted `_compiled_fn`
  fast-path hook was dead (never set), and `mx.compile` of the raw artifact
  fails on the in-graph eval.
- **Fix** (`synapse/axon/codegen2_mlx/core.py`, approved via direction
  choice): `_expert_linear` now emits `mx.gather_mm`-based grouped GEMM —
  pairs sorted by expert with `mx.argsort` (pure MLX), gathered as
  `[N,1,D]×[E,D,D']` with `sorted_indices=True`, unsorted, bias gathered
  per pair. Eval-free → compile-compatible; no host routing. Validated
  first on-machine: bf16 gather_mm supported; unsorted reads the expert
  weight per pair (3.2 GB ⇒ 8.1 ms at S=483) while `sorted_indices=True`
  dedupes (2.2 ms, bit-identical to unsorted).
- **Results** (steady, same machine): classify request totals
  31.4/56.0/111.8 ms → **6.3/11.1/48.9 ms** (S=12/60/483) — 2.3–5×; the
  serve probe end-to-end went 303 → **66 ms** for 3 texts/555 tokens
  (4.6×), decomposing exactly as the per-phase profile predicts.
  Forward dispatch is now ~1 ms; the path is GPU-bound.
- **Parity held**: probe-pf-variants re-run — short 0/60, long 4/483 at the
  same four near-tie positions; server suite 93/93; focused gates 18/18.
  The probe now re-emits the v3 stream artifact when missing and skips the
  historical v0/v1 variants whose /tmp artifacts were wiped.
- **mx.compile after the fix: dead end, recorded** — steady ≈ uncompiled
  (dispatch already ~1 ms), first call 0.5–3 s *per new sequence length*,
  `shapeless=True` crashes (`Slice cannot infer output shapes`), and it
  shifts logits by up to Δ=4.75 (near-tie parity would need re-establishing).
  The uncompiled gather_mm artifact is the endpoint.
- **New known bug (latent, unfixed)**: padded batch forward diverges from
  sequential — max|Δ|=4.125 **on the row with zero padding** (batch
  composition alone changes outputs) → the artifact's batched attention
  path is suspect; server only sends [1,S] today so it is not
  user-visible. Batching is also *slower* post-gather_mm (padded rows
  multiply gather work: 125 vs 59 ms) — do not batch this path; fix the
  attention bug only if batch classification becomes a requirement.


## 2026-11-23 (later) — B: fattn r4 adopted in Rust (−0.95 ms/step, 63.9 tok/s)

- The kernel generator already shipped alternative middle loops (`r4`
  flash-style 4-row merge, `t2` two-pass) that were never adopted — the
  active set everywhere was `base`, and no A/B between them was recorded.
- Python-runner A/B, adjacent same-window runs (`bench_fast --kern --mm2d
  --q4-nt --softmax32 --tokens 64`, 3 rounds each): base 18.1/18.1/18.3,
  **r4 17.4/17.3/17.5** (−0.7 ms/step), t2 18.2–18.4 (neutral-to-worse).
  Window scaling (2 rounds): w=512 base 20.7 vs r4 19.8; w=1024 base 24.4
  vs r4 23.0 — r4's edge grows with KV length, as expected for a wider
  per-iteration merge.
- Adopted: `export_msl.py` now exports `FATTN_LOOP_R4`;
  `axon-backend/src/kernels.rs` assembles `fattn_r4_so`. The bit-exact
  port gates (`refstep`, `gen_ref`) re-anchored to r4 references
  (`export_refstep.py`, `gen_ref.py` now build with `fattn_variant="r4"`):
  r4 vs base differs by fp16 accumulation order (max abs 0.0078125 = one
  ULP; greedy text unchanged on the bench prompt) so both sides must run
  the same variant for the bit-exact port check to stay meaningful.
- `bench_decode` no longer breaks on EOS mid-run — it keeps stepping past
  it (feeding the id back) so variant A/Bs compare equal step counts.
- Rust same-window adjacent A/B (128 steps, minutes apart):
  base 16.74/16.87 ms mean (59.7 tok/s) vs **r4 15.80/15.76 (63.3 tok/s)**
  ⇒ **−0.95 ms/step (~5.7%)**. Confirm run after restore: 15.65 ms median
  (**63.9 tok/s**). New decode headline: ~15.7–15.8 ms/step contended,
  vs llama.cpp same-window 19.6 ms (51.35 tok/s) ⇒ **~21% faster**; still
  below llama.cpp's quiet 17.9 ms baseline. All Rust gates green
  (kernels 5, refstep 1, gen 1, chatrender/http suites).
- The remaining "middle ~32 vs ~10 µs/slot" expectation came from
  bandwidth arithmetic at full window occupancy; r4 recovered ~5.5 µs/slot
  of it at w=128. Deeper gains would need occupancy work (grid is
  NH×128 threads, 4 warps/threadgroup) — parked until C is done.

## 2026-11-20 — Rust serving stack (engine + server) ported and gated

- Task: remove Python from the serving runtime while keeping the Python server
  working in parallel; user chose Rust + MLX via C API. New `rust/` workspace
  mirrors `server/` layer-for-layer (mlx-c-sys → mlx → axon-backend →
  axon-engine → axon-server); nothing in `synapse/` touched.
- Gates, all bit-exact vs the Python stack (`cargo test --release --workspace`):
  q4 packing + Metal kernels (5), full compiled decode step — logits, argmax,
  updated 50 MB KV buffers at pos=24 and pos=0 (`export_refstep.py`), greedy
  generation ids + streamed text (`gen_ref.py`), chat template byte-exact vs
  transformers over a 13-case matrix incl. tools/thinking/parts
  (`export_chatref.py`), HTTP endpoint shapes live (`tests/http_api`).
- Same-window A/B (contended desktop): Rust median **16.9 ms/step (59 tok/s)**
  vs Python `bench_fast --kern` 22.2 — Rust ≥5 % faster; median already below
  the 17.9 llama.cpp baseline, quiet-window absolute confirmation pending.
  Startup 4.7 s (Rust) vs ~46 s (Python pack + artifact compile).
- Fixed en route: fattn k/v view must be `[NH, WINDOW, HD]` (kernel reads NH/W
  from the view shape — the `[1, NH, W, HD]` slice would read garbage); safetensors
  Load nodes are CPU-eval-only → CPU-stream load+cast (`Stream::cpu_stream()`).
- Chat template ported by hand (Rust `tokenizers` has no template support);
  gotchas that cost time: transformers raises on empty conversations (Python's
  fallback must be mirrored), rendered prompts + `add_special_tokens=True` =
  **double BOS** (Python-server behavior, reproduced), content-part markers are
  the literal specials `<|image|>`/`<|video|>`/`<|audio|>` (byte-check them —
  some terminals garble the template source), tool-turn closer sets
  `tool_response` inside the per-tool scan.
- Durable: wiki/rust-server.md (gates, mlx-c API facts, template notes);
  numbers in `log/bifrost-mlx/RESULTS.md` § "Rust port"; setup in `rust/README.md`.
- Open: quiet-window headline vs llama.cpp; `codegen2_mlx` intrinsics-landing
  proposal after that.

## 2026-11-19 — Bifrost v0.5 on MLX: decode optimization campaign (M3 Max)

- Task: beat the llama.cpp fork (Q4_K_M, 56.0 tok/s tg128) with MLX for the
  Bifrost v0.5 HRM-Text checkpoint; request came from the Bifrost-app session.
- Built: checkpoint normalization (`log/bifrost-mlx/normalize_checkpoint.py` →
  `models/hrm_text/bifrost_v05/`, 131 tensors), eager benchmarks
  (`bench_decode.py`), compiled+quantized standalone runner (`bench_fast.py`),
  microbenches, bisect tool. Full numbers in `log/bifrost-mlx/RESULTS.md`.
- Outcome: MLX best = 23.5 tok/s fp16 compiled (eager path: 23.7 — no compile win);
  quantized variants slower still (7.6–15.1). llama.cpp stays 2.4× ahead.
- Root cause: 128 slot-executions × 4 matmuls = 512 serial M=1 matmuls/step at
  ~80 µs each in MLX (in-graph, size-independent); non-matmul graph ≈ 2 ms after
  mx.compile fusion. llama.cpp's Q4 gemv ≈ 28 µs avg + 2.2 GB/token traffic.
- Parity: compiled runner vs eager artifact fp32 diff 3.8e-06 (exact), fp16 7.8e-03.
- Details and validated gotchas: wiki/mlx-hrm-text-decode.md (validated-by:
  ablation battery in `log/bifrost-mlx/bench_fast.py`).
- Next lever if MLX must win: custom Metal kernels (Q4 gemv + fused ops) via an
  MLX extension; quantized prefill untested.

## 2026-11-19 (later) — custom Metal kernel era: o_proj bug, v7b, same-window A/B

- Custom Metal q4-gemv kernels landed earlier in the day (v6/v6r/v7rms/v8a + fattn)
  took the compiled runner 42–44 → 20.6 ms/step (quiet window). Afternoon session
  found and fixed three kernel-level losses; details in
  `log/bifrost-mlx/RESULTS.md` § "Kernel era", measured by the new slot probe
  `log/bifrost-mlx/bench_slotqmm.py`.
- **o_proj was never quantized**: `kern_pack_weights` gated on QUANT_PATHS
  substrings that don't match `o_proj`, so the v8g branch was dead and o ran as
  eager sigmoid + fp16 GEMV (4× weight traffic). Fixed (explicit `"o_proj"`).
- **v7b**: v7rms's separate rms prologue re-reads x and stalls the weight
  pipeline when chained (gqkv 42.5 → 25.0 µs/slot when ssq is folded into the
  gemv loop; exact, max abs 0.001953). Eager gate + v6r beats v8g for o
  (18.6 → 12.5) — per-row g re-reads, not sigmoid cost (exp→exp2 was a null
  result). v8a kept for down.
- Same-window A/B (old vs new config, adjacent runs): 30.5/30.6 vs 25.7/25.9
  ms/step ⇒ **−4.8 ms/step**. Quiet-window projection ~16–17 ms vs 17.9 target;
  absolute confirmation pending (desktop load inflated all afternoon numbers —
  load avg spiked to 7.6; single-run deltas across time were shown worthless:
  an earlier "diet −2 ms" claim was a noise artifact, the node diet is neutral).
- Shapes corrected: I = 4096 (gate_up [8192,1536], down [1536,4096]); supersedes
  the "I ≈ 2048" timing guess in the morning entry.
- Remaining known gaps: fattn middle ~32 µs/slot vs ~10 expected; down solo
  ~36 vs ~14 GPU; lm_head+argmax ~1–2 ms.

## 2026-11-20 — serving-layer A/B + tokenizer parity fix (rust)

- Full HTTP A/B (same window, interleaved 64-token greedy completions): Rust
  1.468 s/req (43.6 tok/s) vs `python -m server` 2.86 s (22.4 tok/s) — ~1.9×
  full-stack. Rust HTTP ≈ free (16.7 ms/step through the socket ≈ in-process
  bench); Python's server uses the generic artifact engine (~31.5 ms/step,
  no custom kernels) — the honest serving comparison, since `bench_fast
  --kern` has no HTTP server.
- Found+fixed a real tokenizer divergence by live A/B: the Python server
  encodes via `AutoTokenizer`, which applies the checkpoint's
  `fix_mistral_regex` patch (config.json lacks `transformers_version`,
  vocab > 100k, flag set in tokenizer_config.json) — a raw `tokenizers`
  crate load (what the Rust engine did, and what `gen_ref.py` did) skips it
  and encodes differently (24 vs 25 prompt tokens on the dragon prompt;
  `"dragon, at"` merges vs splits). Fix: `axon-engine::load_tokenizer`
  reproduces transformers' `_patch_mistral_regex` decision tree at the JSON
  level; `gen_ref.py` switched to `AutoTokenizer`; new gate
  `tests/encode_parity` vs `export_tokref.py` (14-prompt matrix). HTTP gate
  canned values re-derived from the Python server path (chat answer changed:
  "Det Danske Flag" → "Dannebrog" — the old pinned value was stepwise-twin
  output, not server output).
- New known divergence (resolution pending): Python `Session.prefill` is
  batched (one forward over the prompt); the Rust engine prefills stepwise
  (mirrors `bench_fast --verify`, and stepwise-Python reproduces the Rust
  outputs exactly). fp16 tie-flips change greedy outputs on some prompts.
  Evidence and facts in `wiki/rust-server.md`; 19/19 tests green post-fix.

## 2026-11-20 — same-window llama.cpp + memory work (rust)

- Same-window llama.cpp (fork build-ax-bench, `llama-bench -p 0 -n 128`):
  tg128 51.35 ± 0.16 tok/s (19.6 ms/step) vs Rust bench_decode 16.82 ms
  median (59.5 tok/s) minutes apart, both contended ⇒ Rust decode ~14%
  faster than llama.cpp; quiet-window projection now supported from both
  sides (llama.cpp 19.6→17.9 quiet).
- Memory: Rust server phys_footprint 5.7 GB → **1.9 GB** (RSS 2186 MB) by
  dropping the bf16 source map after packing + `mlx::clear_cache` — MLX
  retains freed buffers in its allocator cache, so the load-time peak
  (bf16 safetensors 3.33 GiB) stays resident unless released. Python server
  11.0 GB footprint; llama.cpp ~1.5 GB RSS (GGUF 1.07 GiB). Both stacks run
  4-bit matmul weights (ours Q4_0-style group-64 f16 scale+min incl.
  lm_head; theirs Q4_K_M) — speed comparison is same-precision; the residual
  memory gap is mostly our f16 embedding table (768 MiB gather, kept for
  parity).
- `mlx::clear_cache` wrapped (`mlx-c` exposes `mlx_clear_cache`); 20/20
  tests green (encode_parity gate added earlier today; HTTP gate chat-text
  assertions annotated as stepwise-prefill outputs pending the batched
  prefill decision — batched-Python answer is "Dannebrog").
- New probe: `log/bifrost-mlx/probe_prefill_parity.py` — quantifies
  batched-vs-stepwise prefill divergence: 11/24 prompts diverge at greedy,
  flips also mid-generation (batched prefill writes different fp16 KV).
