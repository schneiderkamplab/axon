# Timestral through Axon → MLX

The learned network is defined in
[`generic-timestral.axon`](../../synapse/models/timestral/generic-timestral.axon).
Axon generates the MLX network; NumPy implements the forecast orchestration and
float64 statistics. Inference in an exported package does not need PyTorch or
Axon. This is an inference port, not a training implementation.

Supported behavior includes multivariate forecasts, missing values, observation
masks, numeric and categorical past/known-future covariates, causal scaling,
partial RoPE, all four layer schedules, attention gates, non-crossing quantiles,
advancing-frame refinement, multi-block feedback, symmetric averaging, and
nonnegative clipping. The API returns NumPy arrays. Weights remain in fp32.

The architecture is config-driven for the Timestral family. Released **5M**
weights have been tested; the 28M/123M/209M checkpoints have not been validated.

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
MLX core locally. This checkout route installs Axon's compiler dependencies,
including Torch; the standalone export below removes those runtime dependencies.

## Export

From this repository, with its dependencies installed:

```bash
python -m scripts.timestral_axon export \
  --output log/timestral-mlx --optimize
```

The output is a standalone `timestral_axon/` package and `requirements.txt`.
`--optimize` opts in to Axon's existing MLX SDPA graph optimization. Omitting it
generates the unfused graph; both have numerical validation. Export refuses to
overwrite an existing package. It does not require importing MLX or an Apple GPU.

The exported package contains its Axon source, license notices, and the shared
backend-neutral runtime helpers copied from Axon. No compiler source is modified.
The generic generated model's decoder-specific `.compile()` path is not used.

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

With the exported package on `PYTHONPATH`:

```python
import numpy as np
from timestral_axon import TimestralForecaster

model = TimestralForecaster.from_pretrained(
    "mistralai/Timestral-5M",
    revision="bdd20cfc15a3fea0e4cb5ccef37f8d3ca7aa335f",
)
context = np.sin(np.arange(256, dtype=np.float32) / 12)
forecast = model.forecast(context, prediction_length=96)
print(forecast.quantiles.shape)  # (1, 1, 96, 21)
print(forecast.median.shape)     # (1, 1, 96)
```

`forecast()` also accepts `observation_mask`, `past_covariates`,
`known_covariates`, `categorical_past_covariates`,
`categorical_known_covariates`, `symmetric_averaging`, and `clip_nonnegative`.
Covariates use `(B,C,T)` or `(B,C,T+horizon)` as applicable.
Constructor options include `max_context`, `per_shot_horizon_patches`,
`quantile_real_cap_k`, and `extreme_value_threshold`.
Hub options include `revision`, `cache_dir`, `token`, and `local_files_only`.
Unknown/missing weights and incorrect shapes are rejected.

For development within this checkout, import from `scripts.timestral_axon`;
it generates the core lazily on first load. Standalone export is the deployment
path that removes the compiler and PyTorch dependencies.

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

On 2026-10-20, **51/51 cases passed on MLX 0.32.4's Linux CPU backend**, both
with and without graph optimization. The runtime ran in an isolated environment
without Torch or Axon installed. Cases include all four layer schedules,
gates disabled, projection biases enabled, asinh disabled, covariates disabled,
missing/constant inputs, large offsets, explicit observation masks, context
truncation, and 1,041-step forecasts with and without covariates.
Tolerance is `abs(actual - expected) <= 2e-4 + 2e-4 * abs(expected)`.
Every output is finite and its quantiles are ordered.

Metal execution, speed, peak memory, and larger released checkpoints remain
unmeasured. Some long-horizon synthetic quantiles have large magnitudes, so
absolute differences alone are not a useful precision claim; the verifier
records absolute errors and the fraction of the allowed tolerance.

The diagnostic Torch backend passed 50/51 cases at the same tight tolerance:
one real-checkpoint mixed-covariate case exceeded it by 0.44% with fused SDPA
(47.5% without fusion). This does not affect the 51/51 MLX results. Reports retain
that failure rather than relaxing the threshold.

The first port uses the complete dense time-attention mask instead of upstream's
two-query-pass optimization. Forecast semantics are preserved, but peak memory
at long contexts or with many variates needs profiling on the target Mac.
NumPy preprocessing and returning NumPy forecasts synchronize each decode block.

## License

The ported Timestral sources and Axon model are Apache-2.0; see `NOTICE` and
`LICENSE`. Exported Axon runtime helpers retain the repository's MIT license
(`AXON_LICENSE` in the export). Model weights retain their separate upstream license.
