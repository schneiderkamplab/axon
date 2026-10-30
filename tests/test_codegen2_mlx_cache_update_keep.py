"""``Cache.update_keep`` on codegen2-mlx, with and without the stateful KV cache.

Attention must see the same keys and values as ``Cache.update``, while the cache holds only the
last ``keep`` positions. A plain ``(k, v)`` pair, which a compiled prefill returns, seeds the
stateful buffers of the following eager steps.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.mlx_core_support import generate_core, load_core

mx = pytest.importorskip("mlx.core")

PROBE = """\
import Cache

cache_probe :: ?CacheLayer[B,H,P,C,D] -> Tensor[B,H,S,D] -> Tensor[B,H,S,D] -> ?Dim -> (Tensor[B,H,K,D], Tensor[B,H,K,D], ?CacheLayer[B,H,K,C,D])
cache_probe past k v keep = Cache.update_keep past k v keep=keep
"""
LITERAL = """\
import Cache

literal_probe :: ?CacheLayer[B,H,P,C,D] -> Tensor[B,H,S,D] -> Tensor[B,H,S,D] -> (Tensor[B,H,K,D], Tensor[B,H,K,D], ?CacheLayer[B,H,K,C,D])
literal_probe past k v = Cache.update_keep past k v keep=2
"""
STEPS = (3, 1, 1, 4, 1)


@pytest.fixture(scope="module")
def core(tmp_path_factory):
    directory = tmp_path_factory.mktemp("cache_probe")
    source = directory / "CacheProbe.axon"
    source.write_text(PROBE)
    code = generate_core(source, class_name="CacheProbe")
    return load_core(code, "CacheProbe")({"probe.weight": mx.zeros((1,), dtype=mx.float32)})


def _chunks():
    rng = np.random.default_rng(0)
    for step in STEPS:
        yield (
            mx.array(rng.standard_normal((1, 2, step, 4)).astype(np.float32)),
            mx.array(rng.standard_normal((1, 2, step, 4)).astype(np.float32)),
        )


def _update(core, past, k, v, keep):
    (keys, values, cache), = core._def_cache_probe(past, k, v, keep)
    return keys, values, cache


@pytest.mark.parametrize("stateful", [True, False], ids=["stateful", "plain"])
@pytest.mark.parametrize("keep", [None, 2, 5], ids=["all", "two", "five"])
def test_attention_sees_the_new_keys_after_the_kept_tail(core, stateful, keep):
    object.__setattr__(core, "_use_kv_cache", stateful)
    past, cached, all_k, all_v = None, 0, None, None
    for k, v in _chunks():
        all_k = k if all_k is None else mx.concatenate([all_k, k], axis=2)
        all_v = v if all_v is None else mx.concatenate([all_v, v], axis=2)
        keys, values, past = _update(core, past, k, v, keep)
        length = cached + k.shape[2]
        assert np.array_equal(np.array(keys), np.array(all_k[:, :, -length:]))
        assert np.array_equal(np.array(values), np.array(all_v[:, :, -length:]))
        cached = length if keep is None else min(keep, length)
        assert np.array_equal(np.array(past[0]), np.array(all_k[:, :, -cached:]))
        assert np.array_equal(np.array(past[1]), np.array(all_v[:, :, -cached:]))


@pytest.mark.parametrize("stateful", [True, False], ids=["stateful", "plain"])
def test_a_literal_keep_survives_specialization(tmp_path, stateful):
    """The optimizer folds a literal argument into the module and drops the formal."""
    source = tmp_path / "LiteralProbe.axon"
    source.write_text(LITERAL)
    code = generate_core(source, class_name="LiteralProbe")
    core = load_core(code, "LiteralProbe")({"probe.weight": mx.zeros((1,), dtype=mx.float32)})
    object.__setattr__(core, "_use_kv_cache", stateful)
    past = None
    for k, v in _chunks():
        (_, _, past), = core._def_literal_probe(past, k, v)
    assert past[0].shape[2] == 2


@pytest.mark.parametrize("keep", [None, 3], ids=["all", "three"])
def test_a_plain_pair_seeds_the_stateful_cache(core, keep):
    rng = np.random.default_rng(1)
    seed = (
        mx.array(rng.standard_normal((1, 2, 6, 4)).astype(np.float32)),
        mx.array(rng.standard_normal((1, 2, 6, 4)).astype(np.float32)),
    )
    k = mx.array(rng.standard_normal((1, 2, 1, 4)).astype(np.float32))
    v = mx.array(rng.standard_normal((1, 2, 1, 4)).astype(np.float32))
    object.__setattr__(core, "_use_kv_cache", True)
    keys, values, past = _update(core, seed, k, v, keep)
    tail = -3 if keep else 0
    assert np.array_equal(np.array(keys), np.array(mx.concatenate([seed[0][:, :, tail:], k], axis=2)))
    assert np.array_equal(np.array(values), np.array(mx.concatenate([seed[1][:, :, tail:], v], axis=2)))
    assert past[0].shape[2] == (3 if keep else 7)
