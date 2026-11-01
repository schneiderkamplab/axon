"""Tests for the backend-agnostic sampler chain."""

from __future__ import annotations

import numpy as np
import pytest

from server.sampling import SamplerConfig, sample_next_token


def test_greedy_selects_argmax_regardless_of_temperature() -> None:
    logits = np.array([0.1, 5.0, -1.0, 2.0])
    assert sample_next_token(logits, SamplerConfig(temperature=0.0)) == 1
    assert sample_next_token(logits, SamplerConfig(temperature=-1.0)) == 1


def test_top_k_one_is_deterministic() -> None:
    logits = np.array([0.1, 3.0, 2.9, -5.0])
    config = SamplerConfig(temperature=1.0, top_k=1, seed=0)
    assert sample_next_token(logits, config) == 1


def test_top_k_restricts_support() -> None:
    logits = np.array([10.0, 9.0, 0.0, 0.0, 0.0])
    config = SamplerConfig(temperature=1.0, top_k=2)
    rng = np.random.default_rng(1234)
    picks = {sample_next_token(logits, config, rng) for _ in range(200)}
    assert picks <= {0, 1}
    assert picks == {0, 1}


def test_top_p_small_restricts_to_head() -> None:
    logits = np.array([8.0, 0.1, 0.1, 0.1])
    config = SamplerConfig(temperature=1.0, top_p=0.5)
    rng = np.random.default_rng(7)
    picks = {sample_next_token(logits, config, rng) for _ in range(200)}
    assert picks == {0}


def test_temperature_flattens_distribution() -> None:
    logits = np.array([4.0, 3.9, 0.0, 0.0])
    flat = SamplerConfig(temperature=50.0)
    rng = np.random.default_rng(3)
    picks = {sample_next_token(logits, flat, rng) for _ in range(400)}
    assert picks <= {0, 1, 2, 3}
    # Very high temperature occasionally picks outside the top-2 ...
    assert picks & {2, 3}
    # ... while a sharp temperature never does (top-1 gap is e^10 at t=0.01).
    sharp = SamplerConfig(temperature=0.01)
    rng = np.random.default_rng(3)
    sharp_picks = {sample_next_token(logits, sharp, rng) for _ in range(200)}
    assert sharp_picks == {0}


def test_seed_reproducibility() -> None:
    logits = np.array([1.0, 1.2, 0.9, 1.1])
    a = [sample_next_token(logits, SamplerConfig(seed=99)) for _ in range(10)]
    b = [sample_next_token(logits, SamplerConfig(seed=99)) for _ in range(10)]
    assert a == b


def test_all_inf_logits_do_not_crash() -> None:
    logits = np.full(5, -np.inf)
    config = SamplerConfig(temperature=1.0, seed=1)
    assert 0 <= sample_next_token(logits, config) < 5


@pytest.mark.parametrize("temperature", [0.0, 0.7, 1.0])
def test_output_is_valid_token_id(temperature: float) -> None:
    logits = np.array([0.2, 0.8, -0.3])
    token = sample_next_token(logits, SamplerConfig(temperature=temperature, seed=5))
    assert 0 <= token < 3
