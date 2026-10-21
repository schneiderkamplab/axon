# Timestral through Axon → MLX

The learned network is defined in
[`generic-timestral.axon`](../../synapse/models/timestral/generic-timestral.axon).
Axon generates grouped variate attention, separate context/forecast attention
queries, optional final-block computation of only the requested forecast positions, and
the monotone quantile head. NumPy implements forecast orchestration, float64 statistics,
denormalization, and refinement. Inference in an exported package does not need
PyTorch or Axon. This is an inference port, not a training implementation.

Supported behavior includes multivariate forecasts, missing values, observation
masks, numeric and categorical past/known-future covariates, causal scaling,
partial RoPE, all four layer schedules, attention gates, non-crossing quantiles,
advancing-frame refinement, multi-block feedback, symmetric averaging, and
nonnegative clipping. The API returns NumPy arrays. Weights remain in fp32.

The architecture is config-driven for the Timestral family. Released **5M,
28M, 123M, and 209M** FP32 checkpoints have been tested. All sizes execute the
comparison suite; strict numerical agreement with the original remains
incomplete. See the [all-size comparison](../../wiki/timestral-long-loop.md).

Final-block pruning is **off by default**. It changes two 123M scenarios beyond
the strict before/after tolerance. To opt in, pass `prune_final_block=True` to
`from_pretrained` or `load_model`, or use `forecast --prune-final-block`.
When enabled, the final block keeps complete keys and values for time attention
but computes only the requested output queries, variate positions, and feed-forward rows.
The [Axon optimization measurements](../../wiki/timestral-long-loop.md) compare
this source change against the previous Axon export, original PyTorch, and
`vikiival/timestral-mlx`, with unchanged host code and FP32 checkpoints. They
also record the remaining numerical differences.

## Quick test from a Mac checkout

From the repository root, use native Apple Silicon Python 3.13 or newer:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[mlx]'
python - <<'PY'
import numpy as np
from scripts.timestral_axon import TimestralForecaster

model = TimestralForecaster.from_pretrained(
    "mistralai/Timestral-5M",
    revision="bdd20cfc15a3fea0e4cb5ccef37f8d3ca7aa335f",
)
context = np.sin(np.arange(256, dtype=np.float32) / 12)
forecast = model.forecast(context, prediction_length=96)
print(forecast.quantiles.shape)  # (1, 1, 96, 21)
print("finite:", np.isfinite(forecast.quantiles).all())
print("ordered:", (np.diff(forecast.quantiles, axis=-1) >= 0).all())
PY
```

The first run downloads the approximately 21 MB checkpoint and generates the
MLX core with SDPA and RMSNorm-scale optimizations enabled. This checkout route
installs Axon's compiler dependencies, including Torch; standalone export
removes those runtime dependencies.

## Export

From this repository, with its dependencies installed:

```bash
python -m scripts.timestral_axon export \
  --output log/timestral-mlx --optimize
```

The output is a standalone `timestral_axon/` package and `requirements.txt`.
`--optimize` enables MLX SDPA and RMSNorm-scale graph optimizations; omitting it
generates the unfused graph. It does not enable final-block pruning; that remains
a separate load-time option. Export refuses to overwrite an existing package
and does not require importing MLX or an Apple GPU.

The package includes generated MLX code, shared runtime helpers, Axon source,
and license notices. The loader compiles the generated tensor forward directly
with `mx.compile`. Fusion matches primitive provenance and requires explicit
backend opt-in. Regression tests cover required parameter reads, epsilon,
tensor layouts, and dtype rounding.

## Run on Apple Silicon

Copy the exported directory to the Mac. Install and run from its parent directory:

```bash
python -m pip install -r log/timestral-mlx/requirements.txt
PYTHONPATH=log/timestral-mlx python -m timestral_axon forecast \
  mistralai/Timestral-5M \
  --revision bdd20cfc15a3fea0e4cb5ccef37f8d3ca7aa335f \
  --context context.npy --horizon 96 --output forecast.npz
```

`context.npy` holds `(T,)`, `(B,T)`, or `(B,V,T)` values; NaN denotes missing
data. The output contains `quantiles`, `quantile_levels`, and `median`.
Use `forecast --help` for the five optional mask/covariate array arguments.
A local checkpoint directory with `config.json` and `model.safetensors` also works.

For Python use, put the export on `PYTHONPATH` and use the checkout example
with `from timestral_axon import TimestralForecaster`. The median has shape
`(B,V,horizon)`; quantiles have shape `(B,V,horizon,num_quantiles)`.

`forecast()` also accepts `observation_mask`, `past_covariates`,
`known_covariates`, `categorical_past_covariates`,
`categorical_known_covariates`, `symmetric_averaging`, and `clip_nonnegative`.
Covariates use `(B,C,T)` or `(B,C,T+horizon)` as applicable.
Constructor options include `max_context`, `per_shot_horizon_patches`,
`quantile_real_cap_k`, `extreme_value_threshold`, `max_batch_rows`, and
`max_batch_tokens`.
Hub options include `revision`, `cache_dir`, `token`, and `local_files_only`.
Unknown/missing weights and incorrect shapes are rejected.

## Memory limits

The loader evaluates the FP32 checkpoint dictionary before compiling the forward
function. This lets different compiled input shapes share the loaded weights.

The default scheduler limits each model call to **128 packed rows** and
**4,096 transformer tokens**. A row is one target or covariate. The token count
includes the encoder and decoder, and accounts for context growth during
multi-block forecasts. Independent series are processed in chunks; all variables
and covariates of one series stay together. Positive and negative forecasts are
paired within each chunk, or evaluated separately when only one series fits.
An individual series can exceed a limit, so these are scheduling limits rather
than a hard memory cap.

For smaller batches, pass `max_batch_rows=64, max_batch_tokens=2048` to the
forecaster constructor or `from_pretrained`. The CLI exposes
`--max-batch-rows` and `--max-batch-tokens`. To retain unbounded scheduling while
keeping shared evaluated weights, set both constructor options to `None`.

Changing these limits can change floating-point results because batch shapes
affect kernel execution. Validate the chosen limits against your reference cases.

## Validation

Pinned upstream:
[`mistralai/timestral@a534bdb`](https://github.com/mistralai/timestral/tree/a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff).
The reference generator checks SHA-256 hashes of all nine upstream Python files.
Create synthetic fixtures using the upstream `src/` directory and optional local
5M checkpoint:

```bash
OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' python -m scripts.timestral_axon.verify reference \
  --upstream-src /path/to/timestral/src \
  --checkpoint /path/to/Timestral-5M \
  --output log/timestral-reference
```

Copy the fixture directory with the export to the Mac, then verify its MLX GPU:

```bash
PYTHONPATH=log/timestral-mlx python -m timestral_axon.verify check \
  --fixtures log/timestral-reference --backend mlx --device gpu \
  --output log/timestral-metal-parity.json
```

The fixture suite includes all four layer schedules,
gates disabled, projection biases enabled, asinh disabled, covariates disabled,
missing/constant inputs, large offsets, explicit observation masks, context
truncation, and long forecasts with and without covariates.
Tolerance is `abs(actual - expected) <= 2e-4 + 2e-4 * abs(expected)`.
Use `check --prune-final-block` to verify the optional pruning path. The report
records this setting. Disabling pruning removes its before/after regressions;
it does not resolve the six existing differences from the original implementation.

Recorded on 2026-10-21 with an Apple M1 Pro and MLX 0.32.2: the optimized
path passes **29/31 (5M), 30/31 (28M), 29/31 (123M), and 30/31 (209M)**
real-checkpoint scenarios. All outputs are finite and ordered, but full
numerical parity remains unresolved. The earlier 5M suite, including tiny
synthetic models, passes **58/60**, plus **8/8** batching-boundary cases.
The focused port, pruning, RMSNorm, graph, and policy suite passes **390 tests**,
including both pruning settings and an empty-schedule regression. The default
matches the frozen pre-pruning output bit for bit on all **124/124** cases.
Opt-in final-block pruning reduces measured generated-core
latency by **3.16–7.33%** across the four sizes; see the linked report for the
three-round protocol, workload differences, and numerical limits.
The optional unfused MLX path passes 56/60 in its recorded integration run;
use `--optimize` for the benchmarked path.

See [validation, known failures, and pinned checkpoints](../../wiki/timestral-long-loop.md)
for the complete evidence and reproduction notes. These checks measure
agreement with the original implementation, not forecasting accuracy on real data.

## License

The ported Timestral sources and Axon model are Apache-2.0; see `NOTICE` and
`LICENSE`. Exported Axon runtime helpers retain the repository's MIT license
(`AXON_LICENSE` in the export). Model weights retain their separate upstream license.
