# Optimized MiniLM / Few-NERD runtime measurements

The same frozen public checkpoint is executed by freshly regenerated Axon code. No retraining or test-set selection was performed. These measurements supersede the earlier Axon execution timings; the [original report](bert-token-classification-results.md) retains the ONNX baselines and their protocol.

Measured 2026-09-26 on Linux, Threadripper PRO 7965WX / RTX 4090. Three sequential shuffled fresh processes per configuration; 10 warmups and 50 samples per case; host arrays to completed host logits; eager Torch SDPA; four threads; TF32 disabled. CPU affinity and clocks were not fixed. No MPS, MLX, or Core ML execution was performed.

[All 10,800 samples, per-process results, quality, versions, and export hashes](../log/ner-optimized-20260926/public-results.json.gz). Tables report medians of three process p50s; speedup is HF latency divided by Axon latency.

Changes: packed independent linear projections at load, single evaluation of Torch LayerNorm inputs, broadcast padding masks, and a compile-compatible public Torch forward. Packing recognizes primitives and typed graph structure.

## Quality

Full official test split: 37,648 sentences and 921,118 scored words. FP32 uses atol=rtol=1e-4; reduced precision permits at most 0.005 absolute loss in both micro and macro F1. The frozen HF CPU reference is reused; all other quality rows were rerun.

| Configuration | Micro F1 | Macro F1 | Word agreement | Gate |
|---|---:|---:|---:|---|
| axon-cpu-fp32 | 0.600915 | 0.511797 | 1.000000 | PASS |
| axon-cuda-fp16 | 0.600880 | 0.511734 | 0.999887 | PASS |
| axon-cuda-fp32 | 0.600915 | 0.511797 | 1.000000 | PASS |
| hf-cpu-fp32 | 0.600915 | 0.511797 | 1.000000 | PASS |
| hf-cuda-fp16 | 0.600853 | 0.511764 | 0.999885 | PASS |
| hf-cuda-fp32 | 0.600915 | 0.511797 | 1.000000 | PASS |

## Eager model-call latency

| Device / precision | Batch | Tokens | HF p50 ms | Axon p50 ms | HF / Axon |
|---|---:|---:|---:|---:|---:|
| cpu fp32 | 1 | 32 | 3.797 | 3.411 | 1.11x |
| cpu fp32 | 1 | 128 | 9.594 | 9.137 | 1.05x |
| cpu fp32 | 1 | 512 | 41.890 | 40.527 | 1.03x |
| cpu fp32 | 8 | 32 | 16.430 | 15.935 | 1.03x |
| cpu fp32 | 8 | 128 | 70.893 | 72.531 | 0.98x |
| cpu fp32 | 8 | 512 | 330.698 | 332.777 | 0.99x |
| cpu fp32 | 32 | 32 | 60.082 | 61.427 | 0.98x |
| cpu fp32 | 32 | 128 | 290.946 | 299.447 | 0.97x |
| cpu fp32 | 32 | 512 | 1447.082 | 1546.386 | 0.94x |
| cuda fp32 | 1 | 32 | 1.199 | 0.857 | 1.40x |
| cuda fp32 | 1 | 128 | 1.207 | 0.836 | 1.44x |
| cuda fp32 | 1 | 512 | 1.273 | 1.201 | 1.06x |
| cuda fp32 | 8 | 32 | 1.181 | 0.822 | 1.44x |
| cuda fp32 | 8 | 128 | 1.476 | 1.344 | 1.10x |
| cuda fp32 | 8 | 512 | 5.426 | 5.307 | 1.02x |
| cuda fp32 | 32 | 32 | 1.463 | 1.327 | 1.10x |
| cuda fp32 | 32 | 128 | 4.085 | 3.931 | 1.04x |
| cuda fp32 | 32 | 512 | 18.971 | 18.609 | 1.02x |
| cuda fp16 | 1 | 32 | 1.104 | 0.798 | 1.38x |
| cuda fp16 | 1 | 128 | 1.111 | 0.804 | 1.38x |
| cuda fp16 | 1 | 512 | 1.136 | 0.813 | 1.40x |
| cuda fp16 | 8 | 32 | 1.122 | 0.818 | 1.37x |
| cuda fp16 | 8 | 128 | 1.136 | 0.830 | 1.37x |
| cuda fp16 | 8 | 512 | 1.463 | 1.670 | 0.88x |
| cuda fp16 | 32 | 32 | 1.140 | 0.833 | 1.37x |
| cuda fp16 | 32 | 128 | 1.329 | 1.365 | 0.97x |
| cuda fp16 | 32 | 512 | 4.509 | 5.564 | 0.81x |

## End-to-end word-to-span latency

Includes tokenization, windowing, transfers, inference, and IO span decoding on 256 deterministic public test sentences. Inputs are presegmented words.

| Device / precision | Batch | HF p50 ms | Axon p50 ms | HF / Axon |
|---|---:|---:|---:|---:|
| cpu fp32 | 1 | 3.927 | 3.488 | 1.13x |
| cpu fp32 | 8 | 26.787 | 26.457 | 1.01x |
| cpu fp32 | 32 | 119.279 | 130.877 | 0.91x |
| cuda fp32 | 1 | 1.421 | 1.114 | 1.28x |
| cuda fp32 | 8 | 2.375 | 1.808 | 1.31x |
| cuda fp32 | 32 | 5.366 | 5.190 | 1.03x |
| cuda fp16 | 1 | 1.342 | 1.095 | 1.23x |
| cuda fp16 | 8 | 2.381 | 1.808 | 1.32x |
| cuda fp16 | 32 | 4.590 | 4.081 | 1.12x |

## Validation and limits

- 198 regression tests passed; five tinygrad cases skipped because clang was unavailable. Separate source-only MLX generation checks passed.
- Generated public forward passed full-graph CUDA compilation parity in FP32/FP16, under no_grad and inference_mode, with padded inputs. This is a separate smoke check; the performance tables use eager execution.
- The original CUDA attempt failed because the sandbox could not access the GPU. Its failed log is retained; successful CUDA runs used host GPU access.
- Mac results remain pending. MLX packing requires unquantized FP32/FP16 weights. Compilation/export success is not evidence of Apple Silicon speed or quality.

Frozen checkpoint SHA-256: `944c16a050788efd4ab111c165ea40edbc8759febdf4f0800f9c4fb1026f2120`.
