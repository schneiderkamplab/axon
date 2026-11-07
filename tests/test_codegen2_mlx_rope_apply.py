"""Regression: the codegen2-mlx ``__mlx_rope`` helper applies the given sin/cos tables as is.

The helper used to recover frequencies from the first table it saw and cache them per head
dim, so a second table with the same head dim (another theta: ModernBERT's sliding layers,
Gemma 3's local layers) reused the first table's frequencies; single-position calls also
cached their offset by ``id(sin)``. Each case below compares the ``__mlx_rope`` build with the
reference build (optimizer off) of the same program.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.mlx_core_support import generate_core, load_core

mx = pytest.importorskip("mlx.core")

PROBE = """\
import Positions (rope_base_factors, rope_pair_base_factors)

rope_probe :: Tensor[B,H,S,R] -> Tensor[B,H,S,R] -> Tensor[B,S] -> (Tensor[B,H,S,R], Tensor[B,H,S,R], Tensor[B,H,S,R], Tensor[B,H,S,R])
rope_probe q k pos = do
  sin_a, cos_a <- rope_base_factors pos R 160000.0 interleaved=INTERLEAVED
  sin_b, cos_b <- rope_base_factors pos R 10000.0 interleaved=INTERLEAVED
  qa, ka <- rope_pair_base_factors q k sin_a cos_a interleaved=INTERLEAVED
  qb, kb <- rope_pair_base_factors q k sin_b cos_b interleaved=INTERLEAVED
  return qa, ka, qb, kb
"""


def _build(directory, name, interleaved, **options):
    source = directory / f"{name}.axon"
    source.write_text(PROBE.replace("INTERLEAVED", "true" if interleaved else "false"))
    code = generate_core(source, class_name=name, **options)
    return load_core(code, name), code


@pytest.fixture(scope="module", params=[False, True], ids=["half", "interleaved"])
def cores(request, tmp_path_factory):
    directory = tmp_path_factory.mktemp("rope_probe")
    fused, fused_source = _build(directory, "RopeFused", request.param, backend_intrinsics="codegen2-mlx:__mlx_rope")
    reference, _ = _build(directory, "RopeReference", request.param, optimize=False)
    return fused({}), reference({}), fused_source


def _inputs(seq, start):
    rng = np.random.default_rng(seq + start)
    q = mx.array(rng.standard_normal((1, 2, seq, 16)).astype(np.float32))
    k = mx.array(rng.standard_normal((1, 2, seq, 16)).astype(np.float32))
    pos = mx.array(np.arange(start, start + seq)[None, :])
    return q, k, pos


def test_rope_intrinsic_is_used(cores):
    _, _, source = cores
    assert "self._mlx_rope_apply(" in source


def test_two_thetas_and_moving_offsets_match_reference(cores):
    fused, reference, _ = cores
    # A prefill, then single positions at moving offsets, in one sequence of calls on the
    # same core: any state carried between calls would show up in the later cases.
    for seq, start in [(12, 0), (1, 12), (1, 13), (1, 40), (5, 3)]:
        q, k, pos = _inputs(seq, start)
        got = fused._def_rope_probe(q, k, pos)
        want = reference._def_rope_probe(q, k, pos)
        for g, w in zip(got, want):
            assert g.shape == w.shape
            assert np.allclose(np.array(g), np.array(w), atol=1e-6), (seq, start)
