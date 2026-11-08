"""Nine deterministic timing workloads shared by every implementation."""

import hashlib
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class Workload:
    name: str
    batch: int
    variates: int
    context: int
    horizon: int
    seed: int
    missing_fraction: float = 0.0
    past_covariates: int = 0
    known_covariates: int = 0
    purpose: str = ""


WORKLOADS = (
    Workload("short_256_96", 1, 1, 256, 96, 1001, purpose="latency floor, host overhead"),
    Workload("batch32_2048_96", 32, 1, 2048, 96, 1002, purpose="batching and chunking path"),
    Workload("max_context_32768_512", 1, 1, 32768, 512, 1003, purpose="maximum context"),
    Workload("multiblock_2000_2500", 1, 1, 2000, 2500, 1004, purpose="three AR blocks, refinement"),
    Workload(
        "covariates_missing_2000_1500", 1, 1, 2000, 1500, 1005,
        missing_fraction=0.1, past_covariates=2, known_covariates=2,
        purpose="missing values, past and known covariates, two-stream masks",
    ),
    Workload("multivariate_8x4_1024_192", 8, 4, 1024, 192, 1006, purpose="grouped variate attention"),
    Workload("short_64_24", 1, 1, 64, 24, 1007, purpose="shortest single series"),
    Workload("long_8192_1024", 1, 1, 8192, 1024, 1008, purpose="long series, one full block"),
    Workload("batch8_512_336", 8, 1, 512, 336, 1009, purpose="mid batch, unaligned horizon"),
)
WORKLOAD_NAMES = tuple(workload.name for workload in WORKLOADS)
REPRESENTATIVE = ("short_256_96", "batch32_2048_96", "max_context_32768_512")


def by_name(name):
    for workload in WORKLOADS:
        if workload.name == name:
            return workload
    raise KeyError(name)


def make_inputs(workload):
    """Return ``{"arrays": {...}, "kwargs": {...}}`` for ``forecast(**arrays, **kwargs)``."""
    rng = np.random.default_rng(workload.seed)
    shape = (workload.batch, workload.variates, workload.context)
    context = rng.standard_normal(shape, dtype=np.float32).cumsum(-1)
    if workload.missing_fraction:
        context[rng.random(shape) < workload.missing_fraction] = np.nan
    arrays = {"context": context}
    if workload.past_covariates:
        past_shape = (workload.batch, workload.past_covariates, workload.context)
        arrays["past_covariates"] = rng.standard_normal(past_shape, dtype=np.float32).cumsum(-1)
    if workload.known_covariates:
        total = workload.context + workload.horizon
        known = rng.standard_normal((workload.batch, workload.known_covariates, total), dtype=np.float32)
        known = known.cumsum(-1)
        # The last known covariate is a binary indicator, as in the parity cases.
        known[:, -1] = rng.integers(0, 2, size=(workload.batch, total))
        arrays["known_covariates"] = known
    return {"arrays": arrays, "kwargs": {"prediction_length": workload.horizon}}


def array_sha256(array):
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(f"{array.dtype.str}:{array.shape}:".encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def manifest():
    entries = []
    for workload in WORKLOADS:
        inputs = make_inputs(workload)
        entries.append(
            asdict(workload)
            | {
                "shapes": {name: list(value.shape) for name, value in inputs["arrays"].items()},
                "sha256": {name: array_sha256(value) for name, value in inputs["arrays"].items()},
            }
        )
    return entries
