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
