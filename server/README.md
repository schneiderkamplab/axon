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

Measured on the qwen3 0.6B-class checkpoint, torch MPS (Apple GPU),
bfloat16, greedy, four prompts of 24 tokens generating 32 new tokens each
(`log/qwen3-throughput-real.py` and `log/probe-mps-*.py`):

| path                                   | per token (solo) | aggregate, 4 concurrent |
|----------------------------------------|-----------------:|------------------------:|
| barebone artifact `generate()`          | 68 ms            | 15 tok/s — a hard cap   |
| Session shim (2-slot concurrency)       | 56 ms            | 20 tok/s                |
| fleet engine, solo                      | 60 ms            | —                       |
| **fleet engine, 4 concurrent streams**  | **20 ms/token**  | **51 tok/s**            |
| the same, over real HTTP (`/v1/completions`) | —           | 47 tok/s                |

The single-sequence paths cannot turn concurrent load into throughput: the
barebone loop and the shim spend one forward call per token of one sequence,
so four queued requests are four sequential runs at the solo rate. The
fleet's per-step cost is nearly flat in batch width — 33 forwards carried
4 × 32 tokens, where the shim needed 128 — so the same GPU delivers 51 tok/s
to four clients where any single-sequence path delivers 15–20.

The solo penalty is engineered out, not inherent: an early build paid
~1.5× per token over the barebone loop because the paged kernel re-extracted
scalars from device tensors once per layer — ~30 forced CPU↔GPU pipeline
syncs per step (each ~1 ms on MPS; 35–38% of step time). The engine now
derives those scalars once per step host-side and carries them on the
serving context, so a decode step syncs exactly once (the sampled token
coming back). After that, fleet solo (60 ms/token) beats the barebone loop
while scaling to 2.6× the shim's throughput at four streams.

Backend note: these numbers are torch-MPS. On MLX the shim reaches
75 tok/s solo / 113 tok/s at four streams — the fleet is torch-only until
the deferred `__mlx_paged_attention` kernel lands, at which point batching
and the backend win compound.
