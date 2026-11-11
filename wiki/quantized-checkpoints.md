# Weight-Only Quantized Checkpoints (QTensor)

`confidence: high` `source: implementation + GPU probes 2026-11-10` `last-confirmed: 2026-11-10`

## Overview

Generated torch/jax cores load `mlx.nn.quantize`-style weight-only quantized
checkpoints (4/8-bit, group sizes 32/64/128) without dequantizing to fp16 at
rest. Opt-in is the runtime load kwarg `quantization={'bits', 'group_size',
'kernel'?}` (MLX pattern); there is no QTensor type/AST machinery and no new
`__<backend>_*` graph intrinsics — `quantization=` is the whole surface, so
compiler/parser/typechecker/lowering layers stay untouched.

Design (b): packed quantized weights live in device memory; dequantization
happens inside the matmul algebra (the dequantized matrix is never
materialized on the quantized fast path).

## Canonical in-core layout (torch/jax)

After load, each quantized weight path `<path>` (e.g. `embed.weight`) holds:

- `<path>.int8` — uint8 `(rows, K)` for 8-bit, `(rows, K//2)` for 4-bit;
  byte `j` holds value `2j` in its low nibble and value `2j + 1` in its high
  nibble (8-bit: one value per byte).
- `<path>.scale`, `<path>.qbias` — fp16 `(rows, K // group_size)`; dequant is
  `q * scale + bias`.

Load-time packing (`_materialize_quantized` / `_materialize_quantized_state`):

- MLX-layout input (uint32 `<base>.weight` + `<base>.scales` + `<base>.biases`):
  words are unpacked with explicit arithmetic — `view(torch.int32)` then
  `(w >> shift) & 0xFF` for shift in (0, 8, 16, 24). MLX packs values
  LSB-first inside uint32 words (verified against `mx.quantize`), so this byte
  stream *is* the canonical layout for both bit widths; endianness-proof
  because the byte order comes from shifts, not memory reinterpretation.
- W8A16 export input (`<path>.int8` int8 + `<path>.scale` fp32, symmetric
  per-output-channel absmax, as produced by `synapse/int8.py`): requantized
  straight to the canonical packed layout at the requested bits/group.
- One O(bytes) pass at load; the device tensor is never repacked again. A
  standalone converter could later make load a literal byte copy.

Loud-failure contract (mirrors the MLX emitter strings):

- quantized checkpoint + no `quantization=` → `'<base> is an mlx.nn.quantize
  weight: pass quantization='`
- `quantization=` + no quantized weights → `'quantization was given, but the
  state has no quantized weights'`
- unsupported bits/group/kernel, group/scale-shape mismatch, quantized +
  packed parameters → `ValueError`; `kernel='triton'` without CUDA →
  `RuntimeError`. Batched input under `kernel='triton'` is *documented
  dispatch* to the eager path, not a silent kernel swap.

Transpose semantics mirror MLX: weights stored `(out, in)`;
`_linear_matmul(transpose=False)` → `_quantized_matmul(transpose=True)`
computes `x @ w.T`.

## Per-backend lowering

| Backend | Lowering | Measured vs backend's own dense |
|---|---|---|
| MLX | `mx.quantized_matmul` / `mx.dequantize` (native) | 1.83x CPU (GPU path blocked upstream, see log) |
| torch | eager dequant matmul (default); fused Triton 1-D decode GEMV with `kernel='triton'` (lazy triton import, autotune over 8 configs with a correctness gate `err < 1e-3`) | Triton 1.77–2.83x (probe), eager-dequant slower than dense |
| JAX | pure-jnp group-partial-sum, fully jittable | 1.4–2.4x packed (probe) |

Triton kernel (torch): `_build_triton_gemv_kernels` in
`synapse/axon/codegen2_torch/core.py`; q8 ~510 GB/s and q4 ~452 GB/s against a
~540 GB/s dense GEMV on RTX 2080 Ti (sm_75). 1-D decode GEMV only
(`transpose=True`), fp32 accumulate, `store .to(y_ptr.dtype.element_ty)`.

JAX group-partial-sum (fully jittable, no triton dependency):

```
y[..., r] = sum_g s[r, g] * (sum_{k in g} q[r, g, k] * x[..., g, k])
          + sum_g b[r, g] * (sum_{k in g} x[..., g, k])
```

`jnp.einsum('...ag,rag->...ra', xg, qg)` for the group partials and
`jnp.einsum('...a,ra->...r', xg.sum(-1), bf)` for the bias term — the bias
einsum matters: a naive `(xg.sum(-1) * bf).sum(-1)` misaligns for batched `x`
(the 1-D probe masked this). Only valid when groups tile the contraction axis
(`transpose=True`); `transpose=False` falls back to eager dequant + dense
matmul because groups then tile the output axis.

## Why JAX has no custom kernel in v1 (dead-end evidence)

Probes live in `log/` (see `wiki/log.md` 2026-11-10 entry). Eliminated routes,
do not re-walk them:

- `jax.experimental.triton` — removed in JAX 0.10.
- Pallas-Triton on sm_75 — Ampere (sm_80+) gated; the compiler is baked into
  `xla_cuda_plugin.so`.
- Mosaic — Hopper+.
- jax-triton custom-call — plugin ptxas rejects ISA 8.0 vs 8.7 (segfault 139);
  `--xla_gpu_cuda_data_dir` ignored; libnvjitlink "not supported".
- `jax.pure_callback` around a direct launch — jittable syntactically but
  host-syncs every call; serializes a decode loop.
- A working direct-launch probe (ctypes-activated torch-free `CudaDriver`,
  CUDA-event timing) reached 278–417 GB/s (2.5–3.2x the best pure-JAX form),
  but it is not jittable and was archived as a future opt-in path, not part of
  the backend contract. The cubin/XLA-custom-call registration route is a
  documented future option (days of work, private-API fragility across
  jaxlib releases).

## Test venv recipes

- torch: `/tmp/opencode/torchao-venv` — venv over conda system-site-packages
  (torch 2.8.0+cu128, triton 3.4.0); run
  `tests/test_codegen2_torch_quantized_checkpoint.py` (triton case needs CUDA).
- jax: `/tmp/opencode/jax-venv` — fully isolated venv (jax 0.10.2, triton 3.7.0,
  CPU-only torch from `https://download.pytorch.org/whl/cpu` — the compiler
  imports torch unconditionally; CPU wheel ≈200 MB vs the CUDA wheel).
  Isolated installs use `PIP_CONFIG_FILE=/dev/null --index-url
  https://pypi.org/simple`.
- Shared fixtures: `tests/quantized_fixture.py` (numpy packer for both the
  canonical layout and the MLX uint32 on-disk layout).

## End-to-end reality check: Qwen2.5-0.5B (2026-11-10)

`confidence: high` for the recorded hardware (RTX 2080 Ti, batch-1 decode,
8-bit/group-64). Speedup conclusions are latency-regime claims, not kernel
quality claims.

Measured with `bench_qwen_e2e.py`-style drivers driving the generated cores
(`generate()` greedy loop, 64 new tokens, median of 3; prompt "The meaning of
life is"). Outputs token-identical to dense in all runs.

| Run | dense | q8 | ratio | weights on device |
|---|---|---|---|---|
| torch (Triton fused) | 45.7 tok/s | 22.3 tok/s | **0.49×** | 988 MB → 525 MB (−47%) |
| jax (`enable_jit`) | 129.5 tok/s | 107.4 tok/s | **0.83×** | same checkpoint |

Why quantized loses at 0.5B despite winning the per-GEMV microbenchmarks:
batch-1 decode at this scale is **overhead-bound, not bandwidth-bound** —
dense streams 988 MB/token at an effective ~45 GB/s, far below the card's
bandwidth, so shrinking weight bytes buys nothing. The quantized paths add
per-call cost: torch Triton GEMV carries a fixed ~35–45 µs launch/dispatch
overhead vs the ~17–20 µs cuBLAS launch floor (Qwen layer matrices are
0.1–4.5 M params; ×168 quantized calls/token ⇒ ~2× slower), and the JAX
group-partial einsum adds dequant compute that XLA cannot fold away.

Crossover rule of thumb on this class of GPU: the fused GEMV wins need
matrices of roughly ≥ 8 M params (4096×4096 breakeven; 11008×4096 ⇒ ~1.8×).
Small-layer models (0.5B-class) and batch-1 are the wrong regime for the
triton kernel; expect the memory saving to be the only win there. MLX native
`quantized_matmul` has C++ dispatch and different overhead structure — do not
transfer these ratios to the MLX machine.

This e2e test also exposed and fixed a dispatch gap: torch loop-inlined
(dynamic-key) linears emitted inline `F.linear(self._param(...))` and bypassed
quantized dispatch entirely (static-key linears and embeddings already routed
through `_linear`/`_embedding_path`). Fix: `_emit_linear_node` now routes
dynamic keys through `self._linear` too — the quantized tok/s numbers above
are only reachable with that fix. JAX had no gap (`linear` primitive emits
`_linear_matmul` with runtime paths unconditionally).

## Known limits (v1)

- Packed parameters (gate/up concat) + `quantization=` → loud `ValueError`;
  quantized expert banks out of scope.
- `kernel='triton'` covers 1-D decode GEMV against `(out, in)`-stored weights
  only; batched/`transpose=False` inputs use eager dequant.
- The direct-execution interpreter (`_execute_primitive_node`) is not
  quantization-aware — parity with MLX, whose interpreter is not either; the
  generated-code path is the contract.
- Eager torch dequant materializes the dequantized matrix per call (correct
  everywhere, slower than dense); use `kernel='triton'` for decode.
