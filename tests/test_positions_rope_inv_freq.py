"""Regression: default-RoPE inverse frequencies follow HF's ``1 / theta ** (2i / d)``.

The Positions builtins used to compute ``exp(-log(theta) * 2i / d)``, which differs from HF
by up to ~6.5e-7 relative; the angle error grows with position and reached ~1e-3 in the
2048-token Ettin logits. The inverse-frequency threshold fails for that formula and passes
for ``pow``; the sin/cos tables are only bounded to two float32 ulps of a 2048-position angle,
since one-ulp angle rounding differences remain with either formula.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tests.mlx_core_support import generate_core, load_core

mx = pytest.importorskip("mlx.core")

PROBE = """\
import Positions (position_ids, rope_base_factors, rope_inv_freq_base)
import Config
import Tensor

THETA <- Config.float @@rope_theta default=10000.0

rope_probe :: TokenIds[B,S] -> Tensor[B,H,S,HD] -> (Tensor[B,1,S,HD], Tensor[B,1,S,HD], Tensor[HD / 2])
rope_probe input_ids q = do
  pos <- position_ids input_ids
  sin, cos <- rope_base_factors pos HD THETA
  inv <- rope_inv_freq_base q THETA
  return sin, cos, inv
"""
LENGTH = 2048
# Largest sin/cos difference from HF at LENGTH. The pow form reaches 1.3e-5 to 3.1e-5 at the higher
# thetas, where exp(-log(theta) * x) reaches 1.2e-4. At theta=10000 the two tie at 1.2e-4.
TABLE_LIMIT = {10000.0: 1.3e-4, 160000.0: 6e-5, 1000000.0: 6e-5}


@pytest.fixture(scope="module")
def probe_class(tmp_path_factory):
    directory = tmp_path_factory.mktemp("rope_probe")
    source = directory / "rope_probe.axon"
    source.write_text(PROBE)
    return load_core(generate_core(source, class_name="RopeProbe"), "RopeProbe")


def hf_tables(theta, dim, length):
    inv = 1.0 / (torch.tensor(theta) ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
    freqs = torch.arange(length, dtype=torch.float)[:, None] * inv[None]
    emb = torch.cat([freqs, freqs], dim=-1)
    return inv.numpy(), emb.sin().numpy(), emb.cos().numpy()


@pytest.mark.parametrize("dim", [64, 128])
@pytest.mark.parametrize("theta", [10000.0, 160000.0, 1000000.0])
def test_rope_tables_match_hf_pow_formula(probe_class, theta, dim):
    core = probe_class({}, config={"rope_theta": theta})
    sin, cos, inv = core._def_rope_probe(mx.zeros((1, LENGTH), dtype=mx.int64), mx.zeros((1, 1, 1, dim)))
    hf_inv, hf_sin, hf_cos = hf_tables(theta, dim, LENGTH)
    inv_rel = np.max(np.abs(np.array(inv) - hf_inv) / hf_inv)
    table = max(np.abs(np.array(sin)[0, 0] - hf_sin).max(), np.abs(np.array(cos)[0, 0] - hf_cos).max())
    assert inv_rel < 3e-7, inv_rel
    assert table <= TABLE_LIMIT[theta], table
