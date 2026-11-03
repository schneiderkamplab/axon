# Axon serving layer

OpenAI-compatible HTTP serving over compiled Axon artifacts. Three ways to
run the same artifact, one throughput story.

## The three paths

1. **Barebone artifact** — the compiled inference-profile artifact carries its
   own greedy `generate(input_ids, max_new_tokens=...)` loop. No harness at
   all: import the model class, load weights, call `generate`.
2. **Session shim** (`server.engine.Session`) — one decode step per forward
   per request, shared weights across concurrent requests through a
   concurrency cap. Adds streaming, sampling and detokenization.
3. **Fleet engine** (`server.fleet`, `--engine fleet`) — continuous batching
   over the paged serving artifact: one shared engine step carries every
   scheduled sequence through a paged KV pool with block tables and prefix
   cache.

The HTTP layer (`server.server`, `python -m server`) sits over either engine
and is deliberately thin — stdlib `ThreadingHTTPServer`, JSON in, text or SSE
out. It adds no decode logic; `POST /v1/completions`,
`POST /v1/chat/completions`, `GET /v1/models`, `GET /health`.

```bash
python -m server --axon synapse/models/qwen3/generic-qwen3.axon \
    --checkpoint models/qwen3 --engine fleet --device mps --dtype bfloat16
```

## Throughput: the online server is what makes it scale

All numbers from one measurement session on the qwen3 0.6B-class checkpoint,
torch MPS (Apple GPU), bfloat16, greedy, decode to a 56-token total length.
Barebone rows come from the canonical runner (`synapse axon-benchmark
--benchmark-mode generate`, default `__torch_sdpa` intrinsics, artifact's own
`generate()`), logged under `log/qwen3-barebone-bench/`; server rows from
`log/probe-http-throughput.py` against `python -m server`. The runner prints
per-sequence tok/s; aggregates below multiply the batch back in.

| path                                    | per step (ms) | aggregate tok/s |
|-----------------------------------------|--------------:|----------------:|
| barebone `generate()`, 1 sequence        | 29.5          | 33              |
| barebone `generate()`, static batch 2    | 31.7          | 63              |
| barebone `generate()`, static batch 4    | 34.1          | 117             |
| **fleet engine over HTTP, 1 stream**     | 28.3          | **35**          |
| **fleet engine over HTTP, 4 concurrent** | ~42 in-process| **88**          |

- If every client arrives at once and you statically batch them, the barebone
  loop is the ceiling: per-step cost is nearly flat in batch width on MPS
  (29.5 → 34.1 ms from batch 1 to 4), so a static batch 4 reaches 117 tok/s.
  That path has no server around it, though: the batch is fixed when the call
  starts, KV memory grows with the longest sequence in the batch, and request
  5 waits for the whole batch to drain.
- The online server turns asynchronous arrivals into the same kind of batch:
  four streams that connected independently finish at 88 tok/s aggregate over
  HTTP (93 in-process) — 2.5× the server's own solo rate. The solo rate
  itself (35) matches the barebone solo loop (33): the paged serving layer
  costs nothing measurable per step solo. Because per-step cost grows slowly
  with batch width (41.5 → 56.7 ms from 4 to 8 rows), the aggregate keeps
  climbing: 137 tok/s in-process at 8 concurrent (~130 over HTTP) — past the
  static batch-4 ceiling. Per-user decode rate is the shared step time:
  35 solo, 22 at four users, 17 at eight.
- The HTTP layer costs ~6% (88 over HTTP vs 93 in-process); the rest of the
  gap to the static ceiling is per-step engine work: at batch 4 the fleet's
  step is ~42 ms versus the static path's 34 ms (1.2×) — the paged kernel
  rebuilds layer-invariant state (the decode mask, position range, metadata
  casts) once per layer instead of once per step, and aligns rope-promoted
  q/k to the pool dtype. hoisting those to per-step work is the remaining
  lever, worth up to the 117 tok/s static ceiling.
- Step-end syncs are engineered down to exactly one (the sampled token's
  readback). An earlier build additionally synced once per layer during
  prefill because the "host-side" cu_seqlens were smuggled through the
  context as device tensors — `bool()`/`int()` on their elements bypass
  `.item()`-style instrumentation, so only aten-level profiling exposed it.

**Absolute numbers are session-bound**: the machine's clock/thermal state
shifts every row by tens of percent between sessions (the composed fleet @8
measured 137 tok/s in the session that produced the table above and 99–102 in
the kernel session below). Only back-to-back, same-session comparisons are
evidence; the rows still hold as a same-session snapshot of the three paths.

## Metal KV-cache decode kernel (`--kv-kernel metal`)

Opt-in fleet flag that replaces the composed decode-attention chain (block-table
gather → context mask → SDPA, ~6 GPU launches per layer per step over data the
GPU already holds) with ONE fused Metal kernel per layer per step: a
flash-style online softmax over the block-addressed pool, one simdgroup (32
lanes) per (sequence, query head), each lane owning `head_dim/32` output dims.
The library is compiled lazily per (dtype, head geometry, block size) via
`torch.mps.compile_shader` (torch ≥ 2.11, Apple silicon) and dispatches on
torch's own MPS stream, so ordering against the surrounding torch ops is
automatic.

```bash
python -m server --axon synapse/models/qwen3/generic-qwen3.axon \
    --checkpoint models/qwen3 --engine fleet --device mps --dtype bfloat16 \
    --port 8123 --max-concurrency 4 --kv-kernel metal
```

Back-to-back on the qwen3 0.6B-class checkpoint (one session, two runs per
cell, bfloat16, greedy, decode to 56 total length; isolated rows are sustained
per-call rates at batch 8 against the same pool):

| measurement                              | composed (`torch`)            | metal kernel                  |
|------------------------------------------|------------------------------:|------------------------------:|
| decode attention, ctx 56, batch 8         | 526 µs/call                   | 24–26 µs/call (~20×)          |
| decode attention, ctx 512, batch 8        | 2268 µs/call                  | 213 µs/call (~10×)            |
| in-process solo (8 reqs × 32 tok, serial) | 5768 ms; attention 429 µs/call| 4308 ms; attention 20 µs/call |
| in-process, 4 concurrent                  | 69 tok/s                      | 98 tok/s                      |
| in-process, 8 concurrent                  | 99–102 tok/s (78–81 ms/step)  | 141–145 tok/s (55–57 ms/step) |
| HTTP, 1 stream                            | 23–25 tok/s                   | 29–30 tok/s                   |
| HTTP, 4 concurrent                        | 57–58 tok/s                   | 87–95 tok/s                   |

- The win grows with batch width because the composed attention GPU time does
  (batch 4 → batch 8 doubles it while the per-step CPU floor stays): +25% solo,
  +42% at 4, +40% at 8, +55% over HTTP at 4. At HTTP @4 the metal solo step
  (33–34 ms) sits near the barebone static-1 floor (29.5 ms in the earlier
  session) — the serving layer plus decode attention now cost almost nothing.
- Contract: decode only — prefill keeps the composed varlen path (a varlen
  Metal kernel is deferred phase 2; this is a documented scope limit, not a
  fallback). Requires mps device, bfloat16 or float32 pool, a power-of-two
  `kvcache_block_size`, and `head_dim` divisible by 64; every unsupported
  configuration raises loudly (`ValueError`/`RuntimeError`), nothing falls
  back silently. The `metal` choice is validated at config parse, at runner
  construction (device check), and per call (shapes/dtypes/contiguity).
- Token parity vs the composed path on the real model (bf16, greedy, 32
  tokens, 8 prompts): 3/8 streams identical; the 5 divergences all start at
  token ≥ 20 — bf16 argmax flips from fp32 accumulation-order differences,
  the same class as the composed path's own flips against the fp32 CPU
  oracle (recorded: 1 of 4 prompts). Early tokens match everywhere; on the
  probed prompt the metal stream tracked the fp32 shim for 10 tokens while
  the composed stream degenerated into a repetition loop — the kernel is not
  systematically worse, bf16 greedy generation is simply chaotic past the
  first handful of tokens.
- `torch.mps.compile_shader` facts the implementation depends on (see
  `synapse/axon/codegen2_torch/paged_attention.py`, `_metal_decode_source`):
  arguments bind **positionally** against the MSL parameter order — the
  output buffer must be passed in its declared position, and passing it
  first when the MSL lists it later silently writes into the wrong buffer
  (the kernel "succeeds" and the output stays untouched); `threads=` is
  total threads (dispatchThreads semantics) and `group_size=` the threadgroup
  shape; scalars bind by value; int64 tensors bind to `device const int64_t*`.
- torch compiles MSL **without** the optimizer passes MLX applies: runtime-
  bound loops over per-thread arrays spill to threadgroup-local memory,
  runtime div/mod emit real (emulated) udiv, and precise `exp` is slow. The
  generated kernel therefore bakes all geometry as compile-time `#define`s,
  emits manually unrolled per-lane code (no register arrays), uses
  `fast::exp`, and addresses blocks with shift/mask — which is also why
  block sizes must be powers of two. bf16 rows load/store as 4-byte uint
  pairs unpacked with `as_type<float>(bits << 16)`; the fp32 variant uses
  plain float loads/stores from the same generator.
- Isolated per-call numbers need sustained repetitions (3×300 calls): one-shot
  timings are clock-ramp artifacts with a ~3× spread. Enqueue cost is ~8.5 µs
  per call, so 28 layers add ~0.24 ms of CPU per step — the win is GPU time.

Reproduce the barebone rows (the runner maps a declared checkpoint id to
`<repo-root>/models/<id>`; this checkout's layout puts `models/` one level
above the package, hence the pragma rewrite into a log-local copy):

```
sed '1s/.*/{-# CHECKPOINTS ["qwen3"] #-}/' synapse/models/qwen3/generic-qwen3.axon \
    > log/qwen3-barebone-bench/qwen3-bench.axon
synapse axon-benchmark log/qwen3-barebone-bench/qwen3-bench.axon \
    --checkpoint qwen3 --device mps --dtype bfloat16 \
    --benchmark-mode generate --max-len 56 --generate-warmup 1 --generate-repeat 3 \
    --skip-hf --tokenizer models/qwen3 --text "The future of AI is" \
    --log-dir log/qwen3-barebone-bench
```

Repeat `--text` for the static batches (2 prompts → 63 tok/s, 4 → 117).

Backend note: these numbers are torch-MPS. On MLX the shim reaches
75 tok/s solo / 113 tok/s at four streams — the fleet is torch-only until
the deferred `__mlx_paged_attention` kernel lands, at which point batching
and the backend win compound.
