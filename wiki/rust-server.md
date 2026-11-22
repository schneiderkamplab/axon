# Rust serving stack for Bifrost v0.5 (server/ mirror in `rust/`)

Confidence: high (built and gated 2026-11-20; all gates green on MLX 0.32.2 /
mlx-c v0.7.0, branch `feat/axon-server`). Evidence: `cargo test --release
--workspace` in `rust/`, same-window A/B in `log/bifrost-mlx/RESULTS.md`
§ "Rust port".

## Why / shape

Rust implementation of the serving stack to remove Python at runtime; the
Python server stays working in parallel (authoritative for `.axon` artifacts
and the fleet engine). Workspace mirrors `server/`'s layering:

- `mlx-c-sys` — raw FFI over mlx-c (C API).
- `mlx` — safe wrapper (Array/Stream/Dtype, `mx.compile` over a Rust closure
  via a C trampoline, custom `MetalKernel`, safetensors).
- `axon-backend` — q4 packing, the custom Metal kernels (v6/v6r/v7b/v8a +
  fattn), the single compiled decode step (128 slot executions unrolled).
- `axon-engine` — decode loop (step-wise prefill + streaming generate),
  sampling (`server/sampling.py` port), detokenization (`server/detokenize.py`
  port). Prefill rides the compiled decode step one token per step — identical
  semantics, zero parity risk vs `bench_fast --verify`.
- `axon-server` — OpenAI-compatible HTTP (stdlib-only HTTP, thread per
  connection, `Slots` = `BoundedSemaphore`), endpoints + chunk shapes mirrored
  1:1; chat template ported from `chat_template.jinja`.

## Parity gates (the contract)

Each layer is bit-exact against the Python stack; run
`cargo test --release --workspace`. Regenerate dumps with:

- `log/bifrost-mlx/export_refstep.py` → step parity (logits/argmax/KV, pos=24 & 0)
- `log/bifrost-mlx/gen_ref.py` → greedy generation ids + text
- `log/bifrost-mlx/export_chatref.py` → transformers chat-template renders
  (13 cases incl. tools/tool-calls/thinking/parts)

Kernel-vector and pack tests in `crates/axon-backend/tests/kernels.rs` skip
when `log/bifrost-mlx/ktest` is absent; all dumps are gitignored.

## mlx-c C API facts (pinned, verified)

- Opaque handles are **by-value structs** with a `void* ctx`; all-zero =
  empty (freeing is a no-op).
- Return codes: 0 ok / 1 error / **2 = "missing"** (map iterator end).
- `mlx_compile` copies the closure reference; the Rust `GraphFn` payload must
  outlive the compiled function.
- `mlx_array_set_data` **copies** the host buffer.
- An error handler must be installed or failures abort the process (Rust:
  thread-local capture + `mlx::take_error()`).
- `mlx_stream_new()` returns NULL in 0.7.0 — resolve the default stream via
  `mlx_stream_thread_local_new` (Rust caches per thread).
- Safetensors `Load` nodes have **no GPU eval**: load + cast on a CPU stream
  (`Stream::cpu_stream()`), mirroring the Python loader; release load+pack ≈ 4 s.
- `mlx_fast_metal_kernel_config_new()` must be called before kernel config use.
- `argmax` returns uint32, first-max (matches np.argmax).

## MLX op semantics pinned during the port

- `slice` requires index vectors of length == ndim (`mlx::slice_ranges` pads
  trailing dims); `mlx_take` is the *flattened* take — embedding row gather
  uses `take_along_axis(axis=0)` with indices reshaped [1,1].
- Python `mx.slice_update` with an **array** start dispatches to the dynamic
  variant → `mlx_slice_update_dynamic` with `start = stack([pos % WINDOW])`.
- fattn reads NH/W from the k view's shape — the view must be exactly
  `[NH, WINDOW, HD]` (Python's `k_buf[slot]`), not the `[1, NH, W, HD]` slice.
- Python's weak-scalar rule: `f16_array * python_float` binds the scalar as
  f16 → pre-round EMB_SCALE to f16 in Rust (bit-exact).
- f32 only in in-kernel accumulators (standard) and 1-element scalars (pos);
  the model path is fp16 end-to-end.

## Chat template notes

The Rust `tokenizers` crate has **no** chat-template support —
`crates/axon-server/src/chat_template.rs` is a hand port of the checkpoint's
`chat_template.jinja`, differential-tested byte-exact vs transformers. Facts
that cost time to learn:

- `transformers` **raises** on an empty conversation → the Python server's
  try/except falls back to the naive join; mirrored in `render_chat`.
- Rendered prompts start with the literal `<bos>`, and the engine's
  `encode(prompt, add_special_tokens=True)` adds **another** BOS → served
  chat prompts carry a double BOS (ids `[2, 2, ...]`). That is Python-server
  behavior; the Rust server reproduces the string.
- The encoders agree only after reproducing transformers' tokenizer patch:
  `tokenizer_config.json` sets `fix_mistral_regex: true` (and `config.json`
  has no `transformers_version`), so `AutoTokenizer` prepends its Mistral
  regex `Split` ahead of the tokenizer.json pre-tokenizer. A raw
  `tokenizers` crate load skips it and encodes differently
  (`"dragon, at"` → `"▁at"` merged vs `","`+`"▁"`+`"at"` split) — found by
  live A/B (24 vs 25 prompt tokens, different generations). `axon-engine`
  patches the JSON before deserialization (`load_tokenizer` in
  `engine.rs`), gated by `tests/encode_parity` vs `export_tokref.py`
  dumps. Engine-level twins must encode via `AutoTokenizer`, not the raw
  crate (fixed in `gen_ref.py` 2026-11-20).
- Python `Session.prefill` runs the whole prompt in ONE batched forward;
  the Rust engine prefills stepwise through the compiled decode step
  (mirroring `bench_fast --verify`). fp16 reduction-order differences flip
  near-tie argmaxes: same prompt ids, first sampled token can differ, whole
  generation diverges (proven on the flag-prompt chat case: batched-Python
  "Dannebrog" vs stepwise "Det Danske Flag" — stepwise-Python matches the
  Rust output exactly). Resolution pending; greedy-only surface,
  distribution-identical.
- Content-parts markers are the literal specials in the template (image and
  video markers are distinct specials; audio is `<|audio|>`) — byte-check
  them against `transformers`, some terminals garble the template source.
  The `prev_message_type='tool_response'` check runs inside the per-tool
  scan, not after the loop — an assistant tool-call turn with no following
  tool messages closes with `<|tool_response>` and no `<turn|>`.
- The Jinja `format_parameters` macro declares a `required` parameter it
  never reads, and emits commas via leading-comma guards on each optional
  section plus a final one before `type:`.

## Measurement discipline

Single-run timings on the loaded desktop are worthless (±5 ms drift). Use
same-window A/B pairs or median/min-of-3. Same-window numbers: Rust median
16.9 ms/step vs Python 22.2; quiet-window absolute vs llama.cpp 17.9 still
pending. Bench: `cargo run --release -p axon-engine --example bench_decode`
(from `rust/`), Python twin `bench_fast.py --kern`.

Full HTTP stacks, same window, interleaved 64-token greedy completions
(wall/request): Rust 1.468 s = 43.6 tok/s vs `python -m server` 2.86 s =
22.4 tok/s (~1.9×). The Rust HTTP layer is free — per-step through the
socket equals the in-process bench; the Python server's number is dominated
by its generic artifact engine (no custom kernels). Per-request gap vs
llama.cpp's tg128 (56 tok/s quiet, 51.4 same-window) is prefill: stepwise
prefill spends ~0.47 s of a 1.47 s request at 28 prompt tokens; batched
prefill is the pending fix (also the parity fix).

Memory (phys_footprint, load + one request): Rust 1.9 GB after
`mlx::clear_cache` post-pack (was 5.7 GB — MLX retains freed load buffers
in its allocator cache, so the load-time peak becomes resident unless
released; wrap: `mlx::clear_cache`, call it after dropping the bf16 source
map in `Model::load`). Python server 11.0 GB. llama.cpp ~1.5 GB RSS.
Both stacks run 4-bit matmul weights (ours: Q4_0-style, group-64 f16
scale+min, packed at load from the bf16 checkpoint, incl. lm_head; theirs:
Q4_K_M). The remaining gap is mostly our f16 embedding table (768 MiB,
a gather — quantizing saves memory, not speed; kept for parity).

Depends-on: wiki/mlx-hrm-text-decode.md (the kernel-era work this ports),
`rust/README.md` (setup + endpoint details).
