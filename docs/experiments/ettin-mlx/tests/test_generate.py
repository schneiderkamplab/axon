"""Test 6 (greedy generation vs HF generate) and sampling behaviour."""

import mlx.core as mx
import numpy as np
import pytest
import torch
from transformers.generation.logits_process import TopPLogitsWarper

from ettin_mlx import generate, sample
from ettin_mlx.generate import top_p_filter
from helpers import model, models, reference


@pytest.mark.parametrize("name", models("decoder"))
def test_greedy_matches_hf_generate(name, report):
    """Token-exact match with HF ``generate(do_sample=False)``, 64 new tokens, 3 prompts."""
    ref = reference(name)
    m = model(name)
    n_prompts = len({k.split("/")[1] for k in ref if k.startswith("gen/")})
    assert n_prompts == 3
    for i in range(n_prompts):
        prompt = ref[f"gen/{i}/prompt"].tolist()
        expected = ref[f"gen/{i}/tokens"].tolist()
        assert len(expected) == 64
        got = generate(m, prompt, max_new_tokens=64)[0]
        report.add(name, f"greedy/prompt{i}", {"prompt_tokens": len(prompt), "match": got == expected})
        assert got == expected, f"prompt {i}: first mismatch at {next(j for j, (a, b) in enumerate(zip(got, expected)) if a != b)}"


@pytest.mark.parametrize("top_p", [0.1, 0.5, 0.9, 0.95, 0.999])
def test_top_p_filter_matches_hf_warper(top_p):
    rng = np.random.default_rng(0)
    logits = rng.standard_normal((4, 1000)).astype(np.float32) * 3
    ours = np.isfinite(np.array(top_p_filter(mx.array(logits), top_p)))
    theirs = np.isfinite(TopPLogitsWarper(top_p)(None, torch.from_numpy(logits)).numpy())
    assert np.array_equal(ours, theirs)


def test_sampling_respects_top_p_and_seed():
    rng = np.random.default_rng(1)
    logits = mx.array(rng.standard_normal((1, 50)).astype(np.float32) * 4)
    allowed = set(np.flatnonzero(np.isfinite(np.array(top_p_filter(logits / 0.7, 0.8))[0])))
    key = mx.random.key(0)
    draws = set()
    for _ in range(300):
        key, sub = mx.random.split(key)
        draws.add(int(sample(logits, temperature=0.7, top_p=0.8, key=sub)[0]))
    assert draws <= allowed and len(draws) > 1
    assert int(sample(logits)[0]) == int(np.argmax(np.array(logits)))


def test_seeded_generation_is_reproducible():
    m = model("ettin-decoder-17m")
    prompt = reference("ettin-decoder-17m")["gen/0/prompt"].tolist()
    a = generate(m, prompt, 32, temperature=0.8, top_p=0.9, seed=123)
    b = generate(m, prompt, 32, temperature=0.8, top_p=0.9, seed=123)
    c = generate(m, prompt, 32, temperature=0.8, top_p=0.9, seed=124)
    assert a == b and a != c


def test_batched_greedy_equals_single():
    m = model("ettin-decoder-17m")
    ref = reference("ettin-decoder-17m")
    prompts = [ref[f"gen/{i}/prompt"].tolist() for i in range(3)]
    batched = generate(m, prompts, max_new_tokens=24)
    for p, got in zip(prompts, batched):
        assert got == generate(m, p, max_new_tokens=24)[0]


def test_generation_stops_at_eos():
    m = model("ettin-decoder-17m")
    prompt = reference("ettin-decoder-17m")["gen/0/prompt"].tolist()
    free = generate(m, prompt, max_new_tokens=64)[0]
    eos = free[5]  # pretend the 6th generated token is EOS
    assert generate(m, prompt, max_new_tokens=64, eos_token_id=eos)[0] == free[: free.index(eos) + 1]


def test_empty_prompt_is_rejected():
    with pytest.raises(ValueError, match="at least one token"):
        generate(model("ettin-decoder-17m"), [], max_new_tokens=4)
