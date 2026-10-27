"""Deterministic parity cases and the agreement gate.

Synthetic inputs are run through the real checkpoints by every implementation;
the pinned original PyTorch package on CPU is the reference.
"""

import numpy as np

TOLERANCE_ABS = 2e-4
TOLERANCE_REL = 2e-4


def cases(covariates=True, long_horizon=70):
    """Yield ``(name, arrays, forecast_kwargs, constructor_kwargs)`` for each case."""
    rng = np.random.default_rng(71)
    specs = [
        ("univariate", 1, 1, 39, 12, {}),
        ("multivariate_missing", 2, 3, 37, 70, {}),
        ("positive", 1, 1, 41, 65, {"positive": True}),
        ("all_missing", 1, 1, 31, 20, {"blank": True}),
        ("constant", 2, 1, 31, 20, {"constant": True}),
        ("large_offset", 1, 1, 41, 40, {"offset": True}),
        ("single_step", 1, 1, 1, 1, {}),
        ("multi_block", 1, 1, 71, long_horizon, {"raw": True}),
        ("context_cap_mask", 2, 1, 137, 40, {"mask": True, "max_context": 32}),
    ]
    if covariates:
        specs += [
            ("past_known", 2, 1, 41, 40, {"cov": True}),
            ("past_only", 2, 1, 41, 40, {"past": True}),
            ("categorical", 2, 1, 41, 40, {"cat": True}),
            ("raw_categories", 1, 2, 41, 40, {"cat": True}),
            ("covariate_multi_block", 1, 1, 71, long_horizon, {"cov": True, "raw": True}),
        ]
    for name, batch, variates, length, horizon, options in specs:
        x = rng.standard_normal((batch, variates, length)).astype(np.float32).cumsum(-1)
        if options.get("positive"):
            x = np.abs(x) + 1
        if options.get("constant"):
            x.fill(2)
        if options.get("offset"):
            x += 1e5
        if length > 1:
            x[..., :3] = np.nan
        if length > 12:
            x[..., 7:11] = np.nan
        if options.get("blank"):
            x.fill(np.nan)
        arrays = {"context": x}
        if options.get("cov") or options.get("past"):
            arrays["past_covariates"] = rng.standard_normal((batch, 2, length)).astype(np.float32)
            arrays["past_covariates"][..., 1:5] = np.nan
        if options.get("cov"):
            known = rng.standard_normal((batch, 2, length + horizon)).astype(np.float32)
            known[:, 1] = rng.integers(0, 2, size=(batch, length + horizon))
            known[..., 6:9] = np.nan
            arrays["known_covariates"] = known
        if options.get("cat"):
            arrays["categorical_past_covariates"] = (
                rng.integers(0, 4, size=(batch, 1, length)).astype(np.float64) + 2**25
            )
            codes = rng.integers(0, 6, size=(batch, 1, length + horizon)).astype(np.float64)
            codes[..., 3:7] = np.nan
            arrays["categorical_known_covariates"] = codes
        if options.get("mask"):
            arrays["observation_mask"] = rng.random(x.shape) > 0.3
        kwargs = {"prediction_length": horizon}
        if options.get("raw"):
            kwargs.update(symmetric_averaging=False, clip_nonnegative=False)
        constructor = {"max_context": options["max_context"]} if "max_context" in options else {}
        yield name, arrays, kwargs, constructor


def tolerance(expected):
    return TOLERANCE_ABS + TOLERANCE_REL * np.abs(expected)


def compare(actual, expected):
    """Return the agreement record of ``actual`` against the reference ``expected``."""
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    record = {"shape": list(actual.shape), "shape_match": actual.shape == expected.shape}
    if not record["shape_match"]:
        record.update(passed=False, finite=False, ordered=False)
        return record
    exact_actual = actual.astype(np.float64)
    exact_expected = expected.astype(np.float64)
    difference = np.abs(exact_actual - exact_expected)
    limit = tolerance(exact_expected)
    finite = bool(np.isfinite(actual).all())
    ordered = finite and bool((np.diff(actual, axis=-1) >= 0).all())
    with np.errstate(invalid="ignore"):
        outside = int(np.count_nonzero(~(difference <= limit)))
        fraction = float(np.nanmax(difference / limit)) if difference.size else 0.0
    record.update(
        finite=finite,
        ordered=ordered,
        max_abs=float(np.nanmax(difference)) if difference.size else 0.0,
        mean_abs=float(np.nanmean(difference)) if difference.size else 0.0,
        max_tolerance_fraction=fraction,
        quantiles_outside_tolerance=outside,
        total_quantiles=int(difference.size),
        passed=bool(finite and ordered and outside == 0),
    )
    return record
