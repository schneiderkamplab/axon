# Apple Silicon MiniLM / Few-NERD measurements

On the measured M1 Pro, Axon's Torch implementation has lower B1/S128 median latency than HF on CPU FP32, MPS FP32, and MPS FP16 in both runs. HF has lower B32/S512 median latency in all three matched comparisons. In the quieter confirmation, **uncompiled Axon MLX FP16 leads B1/S128 at 2.472 ms**, while **HF MPS FP16 leads B32/S512 at 119.034 ms**. These results depend on workload and precision; compiling MLX does not improve every case.

This completes the Apple Silicon comparison for the same frozen public MiniLM-L6-H384 / Few-NERD checkpoint used in the [Linux CPU/CUDA report](bert-token-classification-optimized-results.md). No retraining or test-set checkpoint selection was performed. The original [ONNX-inclusive Linux measurements](bert-token-classification-results.md) remain separate. ONNX Runtime CPU and Core ML were not measured in this Mac comparison.

## Evidence and protocol

Measured 2026-09-26 on an Apple M1 Pro with 10 CPU cores, 16 GPU cores, and 32 GB unified memory. Supplied environment records report macOS 27.0, Python 3.13.9, Torch 2.14.0, Transformers 5.17.0, MLX/MLX Metal 0.32.2, NumPy 2.5.3, safetensors 0.8.0, and datasets 5.0.1. Both runs used the same bundle, package versions, power settings, and arguments except the result output paths.

- Ten configurations: HF/Axon Torch CPU FP32, HF/Axon Torch MPS FP32/FP16, and Axon MLX Metal FP32/FP16, uncompiled and compiled. Torch uses eager SDPA. MPS CPU fallback is disabled.
- Three fresh, sequential processes per configuration per run, shuffled with seeds 20260926, 20260927, and 20260928. Four OMP/MKL and Torch intra-op threads, one Torch inter-op thread. Both runs use the same order; the earlier run precedes the quieter confirmation.
- Each process measures nine shapes (batch 1/8/32 × token length 32/128/512) and three end-to-end batch sizes (1/8/32): a separate first call, ten warmups, then fifty timed calls per case.
- Model calls start with host token arrays and finish with completed FP32 host logits, including transfers. Synthetic inputs cycle through four arrays per shape with full attention masks; padded and variable-length text is covered by quality and end-to-end evaluation.
- End-to-end calls start with presegmented words and include tokenization, windowing, inference, and IO span decoding. Calls draw deterministically from a pool of 256 evenly spaced test sentences. Each batch-size case measures fifty calls, not a full test-set pass.
- Tables show the median of three process p50s. Parentheses show min–max process p50s, **not confidence intervals**. Throughput is the median of three per-process batch-size / mean-latency values. Both runs are reported separately, without pooling or choosing the faster run per case.

[Portable raw results](../log/ner-mac-public-20260926/public-results.json.gz) contain all **36,000 samples**, 60 performance processes, shared quality records, per-label counts, first-call timings, memory counters, environment versions, and sanitized condition snapshots. [All 120 case comparisons](../log/ner-mac-public-20260926/all-case-comparison.csv) and the [audit and reproduction notes](../log/ner-mac-public-20260926/README.md) accompany them.

## B1/S128 model-call latency

Milliseconds; lower is better. `mlx-metal-*` denotes Axon-generated MLX, not an HF MLX implementation. Comparing it with HF MPS changes both implementation and runtime.

| Configuration | Earlier ms (range) | Quieter ms (range) |
|---|---:|---:|
| hf-cpu-fp32 | 10.107 (10.021–10.281) | 8.917 (8.855–8.964) |
| axon-cpu-fp32 | 9.255 (9.127–9.309) | 8.337 (8.336–8.345) |
| hf-mps-fp32 | 5.774 (5.645–5.814) | 4.889 (4.829–4.955) |
| hf-mps-fp16 | 5.655 (5.509–5.714) | 4.729 (4.725–4.740) |
| axon-mps-fp32 | 4.927 (4.854–4.980) | 4.258 (4.224–4.262) |
| axon-mps-fp16 | 4.488 (4.485–4.623) | 4.067 (4.055–4.071) |
| mlx-metal-fp32 | 3.148 (3.013–3.582) | 2.714 (2.711–2.714) |
| mlx-metal-fp32-compiled | 2.738 (2.673–2.765) | 2.580 (2.574–2.582) |
| mlx-metal-fp16 | 3.585 (2.930–3.855) | 2.472 (2.461–2.525) |
| mlx-metal-fp16-compiled | 3.154 (3.128–3.188) | 3.012 (3.010–3.026) |

Compiled MLX FP32 led this shape in the earlier run. The quieter confirmation changes that ranking to uncompiled MLX FP16. Report both observations; the data does not establish one universally fastest MLX mode.

## B32/S512 model-call latency and throughput

| Configuration | Earlier ms (range) | Quieter ms (range) | Quieter sequences/s |
|---|---:|---:|---:|
| hf-cpu-fp32 | 585.238 (577.478–585.509) | 545.260 (544.935–545.835) | 58.7 |
| axon-cpu-fp32 | 608.551 (607.464–616.572) | 576.800 (574.969–576.943) | 55.5 |
| hf-mps-fp32 | 134.732 (134.328–142.803) | 129.525 (129.392–133.666) | 247.0 |
| hf-mps-fp16 | 125.890 (123.569–132.962) | 119.034 (119.019–123.983) | 268.6 |
| axon-mps-fp32 | 143.577 (143.453–143.866) | 138.988 (138.665–142.226) | 229.9 |
| axon-mps-fp16 | 129.574 (126.338–129.839) | 125.351 (125.274–125.359) | 255.2 |
| mlx-metal-fp32 | 223.581 (223.183–225.311) | 207.790 (207.263–215.405) | 153.9 |
| mlx-metal-fp32-compiled | 222.733 (210.231–223.227) | 206.180 (205.439–206.985) | 155.2 |
| mlx-metal-fp16 | 164.887 (164.066–165.528) | 160.516 (160.291–160.819) | 199.0 |
| mlx-metal-fp16-compiled | 163.027 (161.897–164.587) | 159.784 (159.767–160.087) | 199.7 |

For MPS FP16, Axon has 2.93% higher median latency than HF in the earlier run and 5.31% higher latency in the quieter run. All three quieter repetitions favor HF: HF process p50s are 123.983, 119.019, and 119.034 ms; Axon's are 125.274, 125.351, and 125.359 ms. The observed ranges do not overlap. This supports the direction of the result, while three process repetitions and residual desktop activity limit precision.

## Matched HF/Axon Torch comparisons

HF latency / Axon latency; above 1 means lower Axon latency. Each pair uses the same Torch runtime, device, precision, weights, inputs, and thread settings.

| Device / precision | Earlier B1/S128 | Quieter B1/S128 | Earlier B32/S512 | Quieter B32/S512 |
|---|---:|---:|---:|---:|
| cpu-fp32 | 1.092x | 1.070x | 0.962x | 0.945x |
| mps-fp32 | 1.172x | 1.148x | 0.938x | 0.932x |
| mps-fp16 | 1.260x | 1.163x | 0.972x | 0.950x |

## End-to-end B1 latency

Milliseconds, including word tokenization and IO span extraction. Uncompiled MLX FP16 leads this case in both runs. Full B8/B32 results are in the CSV and raw JSON.

| Configuration | Earlier ms (range) | Quieter ms (range) |
|---|---:|---:|
| hf-cpu-fp32 | 6.056 (6.004–6.159) | 5.351 (5.275–5.380) |
| axon-cpu-fp32 | 5.583 (5.450–5.593) | 4.823 (4.758–4.840) |
| hf-mps-fp32 | 5.948 (5.855–8.623) | 4.845 (4.763–4.905) |
| hf-mps-fp16 | 5.734 (5.682–5.737) | 4.632 (4.609–4.875) |
| axon-mps-fp32 | 5.646 (5.281–5.788) | 4.686 (4.446–4.810) |
| axon-mps-fp16 | 5.462 (4.850–5.698) | 4.140 (4.131–4.175) |
| mlx-metal-fp32 | 2.517 (2.455–2.535) | 2.302 (2.236–2.312) |
| mlx-metal-fp32-compiled | 2.778 (2.548–2.792) | 2.296 (2.260–2.594) |
| mlx-metal-fp16 | 2.395 (2.290–2.472) | 2.143 (2.134–2.148) |
| mlx-metal-fp16-compiled | 3.438 (3.383–3.455) | 2.872 (2.870–2.929) |

## Quality

All ten configurations passed on all **37,648 sentences / 921,118 scored words** of the official Few-NERD supervised test split. The quieter run reuses these quality results because the checkpoint, exports, runtime, and environment are unchanged. It did not rerun quality or inference validation.

FP32 must match frozen HF CPU FP32 reference logits at atol=rtol=1e-4. FP16 may lose at most 0.005 absolute in both micro and macro F1. FP32 word predictions agree exactly; reduced-precision losses remain within the gates.

| Configuration | Micro F1 | Macro F1 | Word agreement | Gate |
|---|---:|---:|---:|---|
| hf-cpu-fp32 | 0.600915 | 0.511797 | 1.000000 | PASS |
| axon-cpu-fp32 | 0.600915 | 0.511797 | 1.000000 | PASS |
| hf-mps-fp32 | 0.600915 | 0.511797 | 1.000000 | PASS |
| hf-mps-fp16 | 0.600906 | 0.511804 | 0.999895 | PASS |
| axon-mps-fp32 | 0.600915 | 0.511797 | 1.000000 | PASS |
| axon-mps-fp16 | 0.600840 | 0.511703 | 0.999887 | PASS |
| mlx-metal-fp32 | 0.600915 | 0.511797 | 1.000000 | PASS |
| mlx-metal-fp32-compiled | 0.600915 | 0.511797 | 1.000000 | PASS |
| mlx-metal-fp16 | 0.600865 | 0.511814 | 0.999879 | PASS |
| mlx-metal-fp16-compiled | 0.600865 | 0.511814 | 0.999879 | PASS |

The offline audit recomputes F1 from recorded span counts and checks recorded quality gates. It does not rerun inference or independently compare logits; that evidence comes from the supplied Mac quality runs.

## Conditions and limits

Both runs used AC power with Low Power Mode off and `caffeinate`. Endpoint power settings match. Available `pmset` snapshots recorded no thermal or performance warning level and no CPU power status; temperatures, clocks, and CPU affinity were not controlled or continuously monitored.

The earlier run retained Chrome, terminal, window-server, and system activity. Before the confirmation, Chrome was quit and the operator reported closing other apps and hiding the terminal. Residual activity remained: endpoint snapshots show iTerm at 24.3–24.8%, WindowServer at 10.4–16.8%, the coding agent at 10.8–11.1%, and a monitoring process at 3.3–4.9% CPU, in one-core percentage units. These snapshots are not continuous measurements. The earlier run charged from 87% to 100%; confirmation snapshots show 100% charge.

The confirmation is quieter, not an isolated idle-machine experiment. Run order was not counterbalanced, so timing changes cannot be attributed solely to closing apps. Large differences and repeated directions are more informative than small percentage gaps. Cross-machine Linux/Mac timings also use different software versions and should not be interpreted as a controlled hardware comparison.

First-call values include initialization, possible compilation, and cache effects; they are not isolated compilation costs. The steady-state tables exclude them. Raw process RSS and MLX allocation counters have different scopes and are not directly comparable device-memory measurements. No Neural Engine placement, Core ML performance, power consumption, or energy efficiency was measured.

## Provenance and audit

The benchmark loads the supplied fixed MLX export generated from `b9ffe322f7f1ec34c0b45176d9b35c6567f33f30`, with the existing Mac runner/runtime at `9f799608b20a2d469795af58ebc3f1cd4f54ede6`. The compiler fix preserves relative linear weight/bias scopes using operand semantics; it changes generated source, not the checkpoint. All four MLX precision/compilation modes now pass full quality evaluation.

- Frozen checkpoint SHA-256: `944c16a050788efd4ab111c165ea40edbc8759febdf4f0800f9c4fb1026f2120`.
- Dataset revision: `205f3e9c9f3577ea2561d43f2f62dc249ab92d5b`.
- Fixed MLX export SHA-256: `3eb48f7ac4602678fe113fc540ee4cbf6b84818521927e69f9f2fcb7d9125a11`.

The independent audit verified 134 earlier and 72 confirmation file checksums, preservation of every earlier file, identical run arguments except output paths, deterministic sequential order, all 720 timing cases, 36,000 finite positive samples, recomputed p50/p95/mean/throughput statistics, and all 120 supplied case comparisons. There are 720 distinct sample arrays. No Mac benchmark was executed on the Linux review host.

An interrupted preliminary attempt (16 completed performance outputs and one interrupted log) remains preserved in the original archive and is excluded from both published condition sets. The published JSON removes local paths and process/battery identifiers while preserving measurements. See the [runbook](bert-token-classification.md) to reproduce the benchmark.
