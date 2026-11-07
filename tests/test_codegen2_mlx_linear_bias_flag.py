"""Regression: codegen2-mlx linear honors a runtime (config-driven) bias flag.

The emitter used to look the bias up whenever the flag was not the literal ``false``, so a
false config flag still added a bias present in the state, and an absent bias was re-resolved
(a scan over all state keys) on every call.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.mlx_core_support import generate_core, load_core

mx = pytest.importorskip("mlx.core")

PROBE = """\
import Config
import NN

USE_BIAS <- Config.bool @@use_bias default=false

bias_probe :: Tensor[B,S,D] -> Tensor[B,S,E]
bias_probe x = NN.linear@proj x dim=3 bias=USE_BIAS
"""


@pytest.fixture(scope="module")
def probe_class(tmp_path_factory):
    directory = tmp_path_factory.mktemp("bias_probe")
    source = directory / "bias_probe.axon"
    source.write_text(PROBE)
    return load_core(generate_core(source, class_name="BiasProbe"), "BiasProbe")


@pytest.mark.parametrize("use_bias", [False, True])
def test_linear_bias_follows_runtime_flag(probe_class, use_bias):
    rng = np.random.default_rng(0)
    weight = rng.standard_normal((3, 4)).astype(np.float32)
    bias = rng.standard_normal(3).astype(np.float32)
    x = rng.standard_normal((1, 2, 4)).astype(np.float32)
    state = {"proj.weight": mx.array(weight), "proj.bias": mx.array(bias)}
    core = probe_class(state, config={"use_bias": use_bias})
    (out,) = core._def_bias_probe(mx.array(x))
    expected = x @ weight.T + (bias if use_bias else 0.0)
    assert np.allclose(np.array(out), expected, atol=1e-5)
