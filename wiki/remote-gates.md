# Remote serving gates — SDU UCloud GPU box

*Status: active runbook. Last confirmed 2026-11-26 (all facts below exercised that day).*

## The box

- `ssh -p 6948 ucloud@ssh.cloud.sdu.dk` — RTX PRO 6000 Blackwell (96 GB), 1 TB RAM,
  181 TB free on `/work`, torch 2.13.0+CUDA, transformers 5.20.2, vllm 0.27.1
  (for native comparisons) in `/home/ucloud/miniforge3/bin/python3`.
- Repo: main checkout `/work/dfm/jacobwashere/axon` (other branches' WIP — do not touch);
  our aligned worktree `/work/dfm/jacobwashere/axon-gates` on `feat/axon-server`
  (`git worktree add ../axon-gates feat/axon-server`).
- `models/` in the worktree is a symlink to the main checkout's `models/`
  (gitignored, machine-local). The harness downloads checkpoints to
  `/work/dfm/jacobwashere/models/<org>/<name>` — a DIFFERENT root than the
  checkout's `models/`; pass explicit paths when they must not be confused.
- Invocation quirks (cost us time, keep them):
  - The `synapse` console script resolves the OLD checkout's package. Always pin
    `PYTHONPATH=$PWD` and run through the worktree.
  - `python3 -m synapse.cli.synapse` no-ops (typer app has no `__main__` guard);
    use `python3 -c "from synapse.cli.synapse import app; app()" ...`.
  - `pkill -f "python3 -m server"` kills the calling ssh shell too (the pattern
    matches the remote shell's own cmdline); use `pkill -f "[p]ython3 -m server"`.
  - Batch jobs must fully detach: `nohup ... > log 2>&1 < /dev/null &`.
  - Relaunch shells complete LATE: a lingering nohup ssh can start a server
    after you already moved on, leaving stale GPU processes. Before
    benchmarking, list `nvidia-smi --query-compute-apps` and READ it — the GPU
    is shared with other jobs (`study.py` training runs under
    /work/dfm/brainsurgery). Never combine `ps` and `kill` in one command:
    on 2026-11-26 a co-tenant training run was killed this way alongside a
    stale server; check first, then kill the exact PID in a separate step.
    (`pkill -f "python3 -m server"` also misses `-u` launches — pattern is
    `"[p]ython3 -u\? -m server"` — and matching variants bite; PID kill is
    the reliable form.)

## Per-family gate (the Bifrost gate pattern, generalized)

1. **Offline parity + throughput** (canonical runner, repo conventions from
   `scripts/AGENTS.md`):
   ```
   CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 PYTHONPATH=$PWD \
   python3 -c "from synapse.cli.synapse import app; app()" axon-benchmark \
     synapse/models/<family>/<Model>.axon --device cuda:0 --processes 1 \
     --generate-warmup 2 --generate-repeat 5 --forward-warmup 3 --forward-repeat 10 \
     --max-len 256 --log-dir log/<run-id> --stream-csv log/<run-id>/stream.csv --debug-errors
   ```
   Gate: `masked_top1_eq=True` for every declared checkpoint. The runner
   auto-downloads missing checkpoints into `/work/dfm/jacobwashere/models/`.
2. **Online serving** (fleet engine, paged artifact, same GPU):
   ```
   CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD nohup python3 -m server \
     --axon synapse/models/<family>/<Model>.axon \
     --checkpoint /work/dfm/jacobwashere/models/<org>/<name> \
     --engine fleet --device cuda:0 --port 8123 --max-concurrency 8 --dtype float32 \
     > serve.log 2>&1 < /dev/null &
   ```
   then `log/probe-http-throughput.py` (32-tok) and
   `log/probe-http-throughput-256.py` (256-tok, matches the offline leg)
   against `127.0.0.1:8123` — warmup, solo, 4-concurrent aggregate tok/s.
   Gate: online solo within noise of the offline generate tok/s (the serving
   layer must not tax the artifact).

## Results 2026-11-26 (float32, cuda:0, max_len 256)

Offline (axon-benchmark, generate leg, per checkpoint):

| checkpoint | HF tok/s | Axon tok/s | ratio | top1 eq | max abs diff |
|---|---|---|---|---|---|
| SmolLM2-135M | 36.79 | 34.57 | 0.94 | True | 1.0e-4 |
| SmolLM2-135M-Instruct | 36.75 | 34.92 | 0.95 | True | 4.3e-5 |
| Llama-3.2-1B | 64.56 | 58.31 | 0.90 | True | 1.5e-5 |
| Llama-3.2-1B-Instruct | 64.55 | 57.96 | 0.90 | True | 1.2e-5 |
| gemma-3-1b-pt | 27.80 | 26.55 | 0.96 | True | 3.2e-5 |
| gemma-3-1b-it | 27.81 | 26.82 | 0.96 | True | 6.7e-5 |
| OLMo-2-0425-1B | 58.72 | 53.38 | 0.91 | True | 2.2e-5 |
| Qwen2.5-0.5B | 45.12 | 40.95 | 0.91 | True | 2.9e-5 |

Online (fleet + HTTP, Qwen2.5-0.5B, same GPU/precision), clean server
(single process, backlog-fixed, pool from available memory):

| streams (256-tok completions) | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|---|---|---|
| aggregate tok/s | 40 | 78 | 152 | 289 | 542 | 906 | 1643 | 2507 |
| per-stream tok/s | 40 | 39 | 38 | 36 | 34 | 28 | 26 | 20 |

Solo 40 tok/s (offline standalone generate: 40.95); near-linear to 16
streams (542 tok/s, 13.6× solo), 2507 tok/s at 128 streams (63× solo),
zero connection errors. Pool autosized to 39.93 GiB from 69.6 GiB
available. Server suite on the box: 107 passed / 5 MPS-skips.
Single-run figures, same-day quiet window. Reporting preference (user):
plain tok/s + batching speedup only — no serving-tax / %-of-HF figures.

## Serving-generalization fixes landed (evidence-dated 2026-11-26)

- `e96a526` `server/classify.py` imported mlx.core at module top — Linux hosts
  could not import the server at all; now guarded, classify fails fast at load.
- `97ed238`+`0147d08` test portability: `accelerator="cuda"` assertion now
  guarded on `torch.cuda.is_available()`; accelerator-raise test pins an
  explicit decode context (an ambient prefill context made it vacuous).
- `50dc317` qwen2 artifact: dropped `window=MAX_POSITIONS` from
  `causal_mask_for_input` — a full-capacity window is semantically vacuous
  (identical to causal) and the paged runtime rejects bound windows.
- `90fc652` graph-ir: the paged KV trace followed GQA repeats only for exact
  `_repeat`/`Tensor.repeat` callees; inlined loop bodies specialize the repeat
  into `Tensor.repeat__spec_N` clones with constant args pruned (1-input
  signature). Now recognized by base name (same style as the cache-update
  predicate). qwen2 artifacts compile under `__torch_paged_attention`.
- `3b79c53` qwen2 artifact: hardcoded dim constants made constant-dim
  substitution fold the block body closed → single-callsite inline folded
  `qwen2_block` into the loop-step module → the serving lowering conflated
  block and step ('path template uses undefined value i'). Header now uses the
  qwen3-style `Config.dim@@/float@@/bool@@` lookups; the block module survives
  optimize and the paged serving emit succeeds. **This style is the general
  contract for new-family artifacts.**
- `bf73e44`+`0ff8504` fleet: KV pool took the checkpoint's declared
  torch_dtype while weights follow `--dtype` — float32 model vs bfloat16 pool
  failed SDPA at first prefill. Pool (and autosize math) now read the loaded
  model's dtype (parameters → state_dict_tensors → buffers; serving artifacts
  keep weights in state_dict_tensors, not nn.Parameters).
- `753d77a` server: the HTTP listener used socketserver's default listen
  backlog of 5 — bursts of >=64 simultaneous connections were reset by the
  kernel instead of queued (seen in the concurrency sweep). Now 256, as a
  class attribute (TCPServer consumes it inside `__init__`).
- `96f838c` fleet: KV-pool autosize budgeted the device's TOTAL memory; with
  a co-tenant holding most of the GPU the pool OOM'd at startup. Now budgets
  `torch.cuda.mem_get_info` free bytes (identical on an empty device).

## Known follow-ups

- 9 families still use the additive-mask attention path (`attention_gqa_with_additive_mask`):
  gemma2, gemma3, gpt-oss, mistral, mistral3, pleias, smollm, smollm3, t5gemma —
  these need the definition modernized to the plain `attention q k v mask` form
  (with GQA repeat) before the paged profile will take them.
- `window=MAX_POSITIONS` remains in 137 artifact files (~37 families); vacuous
  but each family's parity gate must rerun after the sweep. Do it family by
  family as checkpoints land, not as a blind sed.
- gemma3/gemma4 need the slide-window attention decision (real banded windows
  are unsupported by the paged runtime by design).
- Qwen3.5 (`model_type qwen3_5`) checkpoints exist on the box; the compiler does
  not define the family yet.
- bf16 serving on CUDA (pool + weights) is the obvious next throughput step;
  kept float32 for the first parity-matched comparison.
