# Bifrost v0.5 (HRM-Text) decode on MLX — why it loses to llama.cpp and where the floor is

Confidence: high (measured 2026-11-19, M3 Max 36 GB, MLX 0.32.2; evidence: in-graph
ablations in `log/bifrost-mlx/bench_fast.py`, summarized in `log/bifrost-mlx/RESULTS.md`).

## Model structure (drives everything)

Bifrost v0.5 is HRM-Text: 16 L-layers + 16 H-layers (32 unique layers, 1.79 B params
with 262k vocab), executed as H_cycles=2 × (3 L-stack passes + 1 H-stack pass) =
**8 stack passes = 128 layer-slot executions per token**, each slot with its own KV
cache entry (llama.cpp's `src/models/hrm-text.cpp` does exactly the same:
`n_layer == n_slot == 128` with aliased layer weights). Per slot: 4 matmuls
(gqkv 1536→6144, o 1536→1536, gate_up 1536→8192, down 4096→1536) ⇒ **512 serially
dependent matmuls per decode step at M=1**.

## Measured

| engine | decode |
|---|---:|
| llama.cpp fork Q4_K_M (tg128) | **56.0 tok/s** (17.9 ms/step) |
| MLX eager serving artifact fp16 (parity path) | 23.7 tok/s |
| MLX compiled standalone runner fp16 W128 | 23.5 tok/s |
| MLX same + q4/q8 `mx.quantized_matmul` | 7.6–15.1 tok/s (slower than fp16) |
| prefill 512: llama.cpp 876 tok/s vs MLX eager 615 | |

## Attribution (the load-bearing fact)

Ablating either attention (gqkv+o) or MLP (gate_up+down) removes 256 matmuls and
~21 ms each; everything else (norms, rope, attention math via sdpa or manual
softmax, silu, KV scatter, embed, lm_head) totals ~2 ms ⇒ **~80 µs per M=1 matmul
in-graph, nearly independent of weight size**. mx.compile fuses the non-matmul
graph almost perfectly (sdpa-vs-manual attention: neutral; 2D-vs-3D operands:
neutral; async no-sync loop: −2 ms). MLX's quantized kernels are dequant-bound at
M=1 (q4 ≈ 2× slower than fp16 per matmul), so the 4× weight-traffic win is
unreachable. llama.cpp's Q4_K_M gemv averages ~28 µs/matmul with ~2.2 GB/token
traffic — kernel quality, not graph structure, is the whole gap.

Supersedes: the earlier hypothesis that kernel-count/dispatch dominated (compile
and sdpa changes showed no effect).

## Gotchas (validated)

- Eager artifact appends K/V **before** attention ⇒ deferred KV writes need the
  self term added explicitly (concat into softmax) — fp32 parity 3.8e-06.
- `mx.scatter` doesn't exist in MLX 0.32; use `mx.slice_update`.
- `mx.fast.rope(scale=…)` scales *positions/frequencies*, not the output.
- Tokenizer auto-prepends `<bos>`; gemma4 turns are `<|turn>user\n…<turn|>\n<|turn>model\n`.
- Bifrost checkpoint needs fusion to the Axon layout (`log/bifrost-mlx/normalize_checkpoint.py`,
  output `models/hrm_text/bifrost_v05/`).

## If we ever want MLX ≥ 56 tok/s here

Custom Metal kernels via `mx.fast.metal_kernel` (no extension needed) got most of
the way on 2026-11-19: q4-gemv v6/v6r/v7b/v8a + single-kernel decode attention
(fattn) took the compiled runner 42–44 → ~20.6 ms/step (quiet window), and an
afternoon of kernel surgery (o_proj packing bug fix, v7b folded-rms, eager-gate
o_proj) showed −4.8 ms/step more in same-window A/B (quiet-window projection
~16–17 ms/step vs the 17.9 llama.cpp baseline; confirm on an idle machine).
Durable lessons: pack o_proj explicitly (substring path matching silently
skipped it → fp16 GEMV at 4× traffic); fold per-vector reductions (rms ssq) into
the gemv loop rather than a prologue pass; avoid per-output-row re-reads of
gate vectors (eager elementwise + plain gemv+residual beats that fusion);
single-run timings on a loaded desktop are worthless — interleave A/B pairs.
Full numbers: `log/bifrost-mlx/RESULTS.md` § "Kernel era".
