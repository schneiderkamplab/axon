# MiniLM / Few-NERD runtime measurements

This run fine-tuned public [MiniLM-L6-H384-uncased](https://huggingface.co/nreimers/MiniLM-L6-H384-uncased)
on [Few-NERD](https://huggingface.co/datasets/DFKI-SLT/few-nerd), using six layers,
384 hidden units, 12 attention heads, and 67 IO labels. Epoch 3 was selected on
validation F1 **0.602885**; the full test split has **37,648 sentences** and
**921,118 scored words**. See the [implementation and reproduction runbook](bert-token-classification.md).

On this machine, ORT CUDA FP16 had the lowest batch-1 latency across the three
measured lengths. CPU INT8 was faster than CPU FP32 with a small F1 reduction.
Axon CUDA FP16 reduced batch-1 latency relative to HF CUDA FP16, while HF had
higher throughput at batch 32 / sequence 128. These results do not establish
one fastest runtime for every workload. **MLX and Core ML remain unmeasured.**

[Raw results and all 18,000 timing samples (gzip JSON)](../log/bert-ner-20260926/public-results.json.gz)
include per-label quality, precision flags, provider placement, versions, hashes,
and validation evidence. The installed MLX version in package inventories is
not an MLX execution result.


Measured 2026-09-26 on Linux: AMD Ryzen Threadripper PRO 7965WX (24 cores/48 threads), one NVIDIA GeForce RTX 4090 (24 GB). Four CPU intra-op threads per process. Frequency boost enabled; CPU affinity and clock frequency were not fixed. Configurations ran sequentially.

Timings start with host token arrays and end with completed host logits; transfers are included. CPU uses four intra-op threads and one inter-op thread. HF and Axon Torch use eager execution with SDPA; TF32 is disabled.

Each shape has 10 warmups and 150 measured calls in three contiguous groups. These are within-process samples, not independent process replicates. Throughput divides batch size by mean latency; latency columns show p50.

The checkpoint was selected on validation data. Quality is exact-span IO scoring on the complete official Few-NERD supervised test split. This experiment is a runtime comparison, not a claim about NER state of the art.

| Configuration | Micro F1 | Macro F1 | Word agreement | Quality gate |
|---|---:|---:|---:|---|
| axon-cpu-fp32 | 0.600915 | 0.511797 | 1.000000 | PASS |
| axon-cuda-fp16 | 0.600891 | 0.511736 | 0.999884 | PASS |
| axon-cuda-fp32 | 0.600915 | 0.511797 | 1.000000 | PASS |
| hf-cpu-fp32 | 0.600915 | 0.511797 | 1.000000 | PASS |
| hf-cuda-fp16 | 0.600853 | 0.511764 | 0.999885 | PASS |
| hf-cuda-fp32 | 0.600915 | 0.511797 | 1.000000 | PASS |
| ort-cpu-fp32 | 0.600915 | 0.511797 | 1.000000 | PASS |
| ort-cpu-int8 | 0.600136 | 0.510530 | 0.992881 | PASS |
| ort-cuda-fp16 | 0.600833 | 0.511659 | 0.999868 | PASS |
| ort-cuda-fp32 | 0.600915 | 0.511797 | 1.000000 | PASS |

FP32 passes when evaluated word logits match HF CPU FP32 at atol=rtol=1e-4. FP16, INT8, and Core ML pass when both micro and macro F1 lose at most 0.005 absolute. Per-label results and logit differences are retained in the JSON.

| Configuration | B1/S32 ms | B1/S128 ms | B1/S512 ms | B32/S128 sequences/s |
|---|---:|---:|---:|---:|
| axon-cpu-fp32 | 3.531 | 9.712 | 41.513 | 106.4 |
| axon-cuda-fp16 | 0.904 | 0.906 | 0.916 | 21712.9 |
| axon-cuda-fp32 | 0.972 | 0.948 | 1.302 | 7506.5 |
| hf-cpu-fp32 | 3.764 | 9.585 | 41.915 | 108.8 |
| hf-cuda-fp16 | 1.103 | 1.114 | 1.133 | 24041.1 |
| hf-cuda-fp32 | 1.182 | 1.198 | 1.217 | 7775.3 |
| ort-cpu-fp32 | 1.971 | 6.343 | 28.216 | 158.2 |
| ort-cpu-int8 | 1.030 | 3.203 | 17.397 | 311.4 |
| ort-cuda-fp16 | 0.468 | 0.474 | 0.618 | 22445.7 |
| ort-cuda-fp32 | 0.559 | 0.684 | 1.174 | 9175.5 |

## Full shape grid

### axon-cpu-fp32

| Batch | Tokens | p50 ms | p95 ms | First call ms | Sequences/s |
|---:|---:|---:|---:|---:|---:|
| 1 | 32 | 3.531 | 4.660 | 8.817 | 265.2 |
| 1 | 128 | 9.712 | 12.852 | 10.468 | 97.2 |
| 1 | 512 | 41.513 | 46.813 | 44.728 | 23.8 |
| 8 | 32 | 16.333 | 16.557 | 16.566 | 485.5 |
| 8 | 128 | 71.103 | 76.402 | 73.405 | 111.8 |
| 8 | 512 | 329.462 | 340.958 | 365.931 | 24.2 |
| 32 | 32 | 61.600 | 62.008 | 65.571 | 519.3 |
| 32 | 128 | 300.682 | 303.538 | 298.085 | 106.4 |
| 32 | 512 | 1576.741 | 1617.138 | 1565.555 | 20.2 |

End-to-end starts from presegmented words and includes tokenization, windowing, inference, and span decoding on 256 deterministic test sentences.

| Batch | p50 ms | p95 ms | Sentences/s |
|---:|---:|---:|---:|
| 1 | 3.605 | 5.400 | 265.0 |
| 8 | 26.255 | 41.001 | 288.9 |
| 32 | 135.491 | 214.813 | 216.9 |

### axon-cuda-fp16

| Batch | Tokens | p50 ms | p95 ms | First call ms | Sequences/s |
|---:|---:|---:|---:|---:|---:|
| 1 | 32 | 0.904 | 0.930 | 199.248 | 1100.7 |
| 1 | 128 | 0.906 | 0.935 | 1.296 | 1098.5 |
| 1 | 512 | 0.916 | 0.940 | 1.519 | 1086.6 |
| 8 | 32 | 0.927 | 0.944 | 1.090 | 8607.6 |
| 8 | 128 | 0.945 | 0.973 | 2.251 | 8424.9 |
| 8 | 512 | 1.771 | 1.787 | 5.932 | 4515.4 |
| 32 | 32 | 0.941 | 0.980 | 1.018 | 33849.2 |
| 32 | 128 | 1.472 | 1.484 | 1.561 | 21712.9 |
| 32 | 512 | 5.859 | 5.869 | 8.478 | 5461.7 |

End-to-end starts from presegmented words and includes tokenization, windowing, inference, and span decoding on 256 deterministic test sentences.

| Batch | p50 ms | p95 ms | Sentences/s |
|---:|---:|---:|---:|
| 1 | 1.199 | 1.304 | 831.9 |
| 8 | 1.899 | 2.088 | 4185.3 |
| 32 | 4.195 | 4.331 | 7666.0 |

### axon-cuda-fp32

| Batch | Tokens | p50 ms | p95 ms | First call ms | Sequences/s |
|---:|---:|---:|---:|---:|---:|
| 1 | 32 | 0.972 | 1.046 | 133.711 | 1004.3 |
| 1 | 128 | 0.948 | 0.988 | 1.508 | 1050.1 |
| 1 | 512 | 1.302 | 1.317 | 2.251 | 767.3 |
| 8 | 32 | 0.942 | 0.961 | 1.103 | 8460.9 |
| 8 | 128 | 1.516 | 1.528 | 1.873 | 5275.7 |
| 8 | 512 | 5.635 | 5.654 | 6.177 | 1420.6 |
| 32 | 32 | 1.499 | 1.512 | 1.549 | 21347.8 |
| 32 | 128 | 4.241 | 4.374 | 4.410 | 7506.5 |
| 32 | 512 | 20.652 | 20.761 | 22.965 | 1549.0 |

End-to-end starts from presegmented words and includes tokenization, windowing, inference, and span decoding on 256 deterministic test sentences.

| Batch | p50 ms | p95 ms | Sentences/s |
|---:|---:|---:|---:|
| 1 | 1.230 | 1.329 | 813.2 |
| 8 | 1.908 | 2.219 | 4103.3 |
| 32 | 5.387 | 6.366 | 5859.6 |

### hf-cpu-fp32

| Batch | Tokens | p50 ms | p95 ms | First call ms | Sequences/s |
|---:|---:|---:|---:|---:|---:|
| 1 | 32 | 3.764 | 4.372 | 101.483 | 260.5 |
| 1 | 128 | 9.585 | 10.080 | 10.702 | 102.8 |
| 1 | 512 | 41.915 | 42.570 | 43.666 | 23.8 |
| 8 | 32 | 16.439 | 16.962 | 17.560 | 482.4 |
| 8 | 128 | 71.212 | 76.053 | 74.572 | 111.0 |
| 8 | 512 | 328.642 | 343.961 | 333.226 | 24.2 |
| 32 | 32 | 60.102 | 64.010 | 64.280 | 527.5 |
| 32 | 128 | 291.386 | 308.981 | 290.425 | 108.8 |
| 32 | 512 | 1450.989 | 1504.445 | 1460.687 | 21.9 |

End-to-end starts from presegmented words and includes tokenization, windowing, inference, and span decoding on 256 deterministic test sentences.

| Batch | p50 ms | p95 ms | Sentences/s |
|---:|---:|---:|---:|
| 1 | 4.091 | 6.207 | 235.1 |
| 8 | 27.390 | 41.479 | 280.1 |
| 32 | 121.301 | 200.040 | 234.8 |

### hf-cuda-fp16

| Batch | Tokens | p50 ms | p95 ms | First call ms | Sequences/s |
|---:|---:|---:|---:|---:|---:|
| 1 | 32 | 1.103 | 1.127 | 173.284 | 903.5 |
| 1 | 128 | 1.114 | 1.189 | 1.574 | 891.3 |
| 1 | 512 | 1.133 | 1.167 | 6.565 | 880.0 |
| 8 | 32 | 1.125 | 1.147 | 1.547 | 7092.9 |
| 8 | 128 | 1.137 | 1.172 | 2.023 | 7006.6 |
| 8 | 512 | 1.460 | 1.480 | 5.790 | 5518.6 |
| 32 | 32 | 1.133 | 1.162 | 1.202 | 28106.7 |
| 32 | 128 | 1.329 | 1.347 | 1.412 | 24041.1 |
| 32 | 512 | 4.515 | 4.529 | 6.610 | 7084.7 |

End-to-end starts from presegmented words and includes tokenization, windowing, inference, and span decoding on 256 deterministic test sentences.

| Batch | p50 ms | p95 ms | Sentences/s |
|---:|---:|---:|---:|
| 1 | 1.341 | 1.450 | 742.0 |
| 8 | 2.375 | 2.569 | 3348.2 |
| 32 | 4.610 | 4.784 | 6961.9 |

### hf-cuda-fp32

| Batch | Tokens | p50 ms | p95 ms | First call ms | Sequences/s |
|---:|---:|---:|---:|---:|---:|
| 1 | 32 | 1.182 | 1.216 | 100.936 | 841.3 |
| 1 | 128 | 1.198 | 1.223 | 1.775 | 832.1 |
| 1 | 512 | 1.217 | 1.229 | 1.865 | 821.2 |
| 8 | 32 | 1.175 | 1.200 | 1.331 | 6783.3 |
| 8 | 128 | 1.476 | 1.494 | 1.642 | 5416.4 |
| 8 | 512 | 5.448 | 5.469 | 6.054 | 1468.9 |
| 32 | 32 | 1.462 | 1.479 | 1.502 | 21863.7 |
| 32 | 128 | 4.107 | 4.203 | 4.241 | 7775.3 |
| 32 | 512 | 18.995 | 19.080 | 21.206 | 1684.5 |

End-to-end starts from presegmented words and includes tokenization, windowing, inference, and span decoding on 256 deterministic test sentences.

| Batch | p50 ms | p95 ms | Sentences/s |
|---:|---:|---:|---:|
| 1 | 1.399 | 1.499 | 716.4 |
| 8 | 2.361 | 2.568 | 3359.5 |
| 32 | 5.373 | 6.314 | 5831.4 |

### ort-cpu-fp32

| Batch | Tokens | p50 ms | p95 ms | First call ms | Sequences/s |
|---:|---:|---:|---:|---:|---:|
| 1 | 32 | 1.971 | 3.267 | 4.692 | 450.2 |
| 1 | 128 | 6.343 | 6.573 | 6.676 | 157.0 |
| 1 | 512 | 28.216 | 28.725 | 31.128 | 35.1 |
| 8 | 32 | 11.458 | 11.717 | 11.952 | 689.1 |
| 8 | 128 | 44.757 | 46.046 | 45.418 | 178.1 |
| 8 | 512 | 273.816 | 285.302 | 293.621 | 29.0 |
| 32 | 32 | 43.538 | 43.773 | 43.894 | 732.5 |
| 32 | 128 | 202.004 | 202.735 | 202.404 | 158.2 |
| 32 | 512 | 1172.255 | 1175.713 | 1261.975 | 27.3 |

End-to-end starts from presegmented words and includes tokenization, windowing, inference, and span decoding on 256 deterministic test sentences.

| Batch | p50 ms | p95 ms | Sentences/s |
|---:|---:|---:|---:|
| 1 | 1.975 | 3.314 | 483.5 |
| 8 | 19.218 | 32.053 | 397.9 |
| 32 | 84.926 | 138.028 | 341.6 |

ORT placement events: `{"CPUExecutionProvider": 231}`.

### ort-cpu-int8

| Batch | Tokens | p50 ms | p95 ms | First call ms | Sequences/s |
|---:|---:|---:|---:|---:|---:|
| 1 | 32 | 1.030 | 2.343 | 3.383 | 790.9 |
| 1 | 128 | 3.203 | 3.271 | 4.029 | 311.3 |
| 1 | 512 | 17.397 | 20.259 | 21.352 | 56.6 |
| 8 | 32 | 5.214 | 5.426 | 6.088 | 1525.2 |
| 8 | 128 | 22.833 | 23.050 | 23.172 | 350.2 |
| 8 | 512 | 169.578 | 170.471 | 197.298 | 47.1 |
| 32 | 32 | 19.723 | 19.909 | 20.328 | 1620.0 |
| 32 | 128 | 102.672 | 103.145 | 103.266 | 311.4 |
| 32 | 512 | 715.250 | 719.500 | 803.519 | 44.7 |

End-to-end starts from presegmented words and includes tokenization, windowing, inference, and span decoding on 256 deterministic test sentences.

| Batch | p50 ms | p95 ms | Sentences/s |
|---:|---:|---:|---:|
| 1 | 1.153 | 1.767 | 845.2 |
| 8 | 9.041 | 13.674 | 834.0 |
| 32 | 43.743 | 72.535 | 664.9 |

ORT placement events: `{"CPUExecutionProvider": 212}`.

### ort-cuda-fp16

| Batch | Tokens | p50 ms | p95 ms | First call ms | Sequences/s |
|---:|---:|---:|---:|---:|---:|
| 1 | 32 | 0.468 | 0.479 | 165.100 | 2127.1 |
| 1 | 128 | 0.474 | 0.483 | 0.700 | 2115.2 |
| 1 | 512 | 0.618 | 0.626 | 1.910 | 1617.4 |
| 8 | 32 | 0.482 | 0.495 | 4.060 | 16554.1 |
| 8 | 128 | 0.682 | 0.695 | 1.436 | 11695.0 |
| 8 | 512 | 2.830 | 2.844 | 5.696 | 2826.5 |
| 32 | 32 | 0.650 | 0.658 | 0.727 | 49218.1 |
| 32 | 128 | 1.425 | 1.434 | 1.487 | 22445.7 |
| 32 | 512 | 12.553 | 12.567 | 25.755 | 2549.1 |

End-to-end starts from presegmented words and includes tokenization, windowing, inference, and span decoding on 256 deterministic test sentences.

| Batch | p50 ms | p95 ms | Sentences/s |
|---:|---:|---:|---:|
| 1 | 0.685 | 0.834 | 1419.0 |
| 8 | 1.405 | 1.613 | 5648.9 |
| 32 | 4.038 | 4.246 | 8023.9 |

ORT placement events: `{"CPUExecutionProvider": 45, "CUDAExecutionProvider": 188}`.

### ort-cuda-fp32

| Batch | Tokens | p50 ms | p95 ms | First call ms | Sequences/s |
|---:|---:|---:|---:|---:|---:|
| 1 | 32 | 0.559 | 0.575 | 107.124 | 1767.3 |
| 1 | 128 | 0.684 | 0.695 | 0.816 | 1458.4 |
| 1 | 512 | 1.174 | 1.186 | 1.458 | 851.0 |
| 8 | 32 | 0.700 | 0.709 | 0.827 | 11412.8 |
| 8 | 128 | 1.355 | 1.364 | 1.567 | 5897.4 |
| 8 | 512 | 7.145 | 7.158 | 11.314 | 1119.5 |
| 32 | 32 | 1.363 | 1.374 | 1.422 | 23445.6 |
| 32 | 128 | 3.481 | 3.528 | 3.899 | 9175.5 |
| 32 | 512 | 30.446 | 30.674 | 51.852 | 1057.3 |

End-to-end starts from presegmented words and includes tokenization, windowing, inference, and span decoding on 256 deterministic test sentences.

| Batch | p50 ms | p95 ms | Sentences/s |
|---:|---:|---:|---:|
| 1 | 0.747 | 0.898 | 1320.9 |
| 8 | 1.689 | 1.960 | 4662.6 |
| 32 | 5.229 | 5.926 | 6122.5 |

ORT placement events: `{"CPUExecutionProvider": 45, "CUDAExecutionProvider": 187}`.

## Provenance and limits

Checkpoint SHA-256: `944c16a050788efd4ab111c165ea40edbc8759febdf4f0800f9c4fb1026f2120`.

Base revision: `3276f0fac9d818781d7a1327b3ff818fc4e643c0`. Dataset revision: `205f3e9c9f3577ea2561d43f2f62dc249ab92d5b`.

ORT export fusion counts: `{"cpu": {"Attention": 0, "BiasGelu": 6, "EmbedLayerNormalization": 0, "FastGelu": 0, "Gelu": 0, "GemmFastGelu": 0, "LayerNormalization": 1, "MultiHeadAttention": 0, "QOrderedAttention": 0, "QOrderedGelu": 0, "QOrderedLayerNormalization": 0, "QOrderedMatMul": 0, "RotaryEmbedding": 0, "SimplifiedLayerNormalization": 0, "SkipLayerNormalization": 12, "SkipSimplifiedLayerNormalization": 0}, "cuda": {"Attention": 0, "BiasGelu": 6, "EmbedLayerNormalization": 0, "FastGelu": 0, "Gelu": 0, "GemmFastGelu": 0, "LayerNormalization": 1, "MultiHeadAttention": 0, "QOrderedAttention": 0, "QOrderedGelu": 0, "QOrderedLayerNormalization": 0, "QOrderedMatMul": 0, "RotaryEmbedding": 0, "SimplifiedLayerNormalization": 0, "SkipLayerNormalization": 12, "SkipSimplifiedLayerNormalization": 0}}`.

Check actual fusion counts before describing these as fully fused transformer graphs. Export and optimizer versions can change the matched patterns. The JSON preserves graph hashes, operators, software versions, raw timings, and per-label quality.

Process RSS includes runtime and data allocations. Torch's allocator counter is not total GPU memory for ORT. No Apple Silicon result can be inferred from Linux results; Core ML provider placement alone does not establish Neural Engine execution.

## Reproduce the frozen experiment

Preparation accepts the exact revisions used here:

```bash
python scripts/train_ner.py prepare --run-dir log/ner-reproduction \
  --base-revision 3276f0fac9d818781d7a1327b3ff818fc4e643c0 \
  --dataset-revision 205f3e9c9f3577ea2561d43f2f62dc249ab92d5b
```

Continue with training, export, and benchmark commands in the runbook. The
recipe uses seed 17 and three epochs. Retraining may not produce bit-identical
weights across environments; exact cross-device comparisons use the frozen
checkpoint and reference-logit bundle produced by `scripts/package_ner.py`.
Weights and dataset copies are run artifacts rather than Git source files.

## Implementation validation

The final focused regression run passed **18 tests**, including padding, omitted
masks, segment IDs, configurable head sizes, shared masked-LM behavior, task
selection, word alignment/scoring, and repository policy guards. MLX source was
parsed without running MLX.

Canonical public `dslim/bert-base-NER` checks produced identical predictions on
CPU and CUDA with maximum logit differences **5.25e-6** and **2.15e-6**, respectively.
The local canonical-run wrapper disabled an unusable installed MLX dependency
probe; the Torch comparison and timing paths were retained.

The broader runner/CLI integration selection had **59 passed, 6 failed, 1 XPASS**.
The same six failures reproduced on unmodified base commit
`f0d34c471e675acea5404fd10c1feeac1e5d93f7`; they expect obsolete runner return/CSV
schemas and backend names. Their names are preserved in the raw results.
