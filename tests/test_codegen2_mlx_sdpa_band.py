"""The codegen2-mlx ``__mlx_sdpa_band`` fusion and lowering of band attention.

The optimizer fuses attention over a keep mask only when primitive provenance proves that the
mask is a band: definition names are not evidence, so a renamed mask fuses and a changed mask
under a builtin's name does not. The lowering never builds the dense keep mask: it passes no
mask when every key is allowed, MLX's "causal" mask when that is exact, a band mask otherwise,
and for long windowed self-attention it attends in blocks over strided key spans. Each equality
case compares the ``__mlx_sdpa_band`` build with the reference build (optimizer off) of the same
program.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from synapse.axon import (
    GraphOptimizeConfig,
    elaborate_closed_axon_file,
    flatten_closed_axon_file,
    lower_axon_program_to_graph_ir,
    normalize_closed_axon_file,
    optimize_graph_program,
    resolve_axon_program_from_path,
    typecheck2_flat_axon_file,
)
from synapse.axon.graph_ir.core import GraphOp
from synapse.axon.graph_ir.optimize import _rewrite_mlx_sdpa_band_intrinsics
from synapse.axon.graph_ir.provenance import (
    _band_keep_fact,
    _make_provenance,
    _op_provenance,
)
from tests.mlx_core_support import generate_core, load_core

mx = pytest.importorskip("mlx.core")

PROBE = """\
import Attention (attention)
import Masking (bidirectional_mask, causal_mask)

band_probe :: Tensor[B,H,Q,D] -> Tensor[B,H,K,D] -> Tensor[B,H,K,D] -> ?Tensor[B,K] -> (Tensor[B,H,Q,D], Tensor[B,H,Q,D], Tensor[B,H,Q,D], Tensor[B,H,Q,D])
band_probe q k v pad = do
  causal_full <- attention q k v (causal_mask q k padding_mask=pad)
  causal_local <- attention q k v (causal_mask q k padding_mask=pad window=WINDOW)
  bidir_full <- attention q k v (bidirectional_mask q k padding_mask=pad)
  bidir_local <- attention q k v (bidirectional_mask q k padding_mask=pad window=WINDOW)
  return causal_full, causal_local, bidir_full, bidir_local
"""
WINDOW = 16
FUSED = "codegen2-mlx:__mlx_sdpa_band"

# The band masks written out with the same primitives as the builtins, under another name.
LOCAL_MASK = """\
import Attention (attention)
import Tensor (and, arange, expand, le, reshape)

MASK_NAME :: Tensor[B,HQ,Q,HDQ] -> Tensor[B,HK,K,HDK] -> ?Dim -> Tensor[B,1,Q,K]
MASK_NAME q k ?window=null = do
  row_ids <- arange q start=0 end=Q |> reshape shape=[Q, 1]
  col_ids <- arange k start=0 end=K |> reshape shape=[1, K]
  aligned <- row_ids + (K - Q)
  upper <- le col_ids aligned
  keep_window <- (window == null) ? upper : (and upper (le (aligned - window + 1) col_ids))
  keep <- reshape keep_window shape=[1, 1, Q, K]
  return expand keep shape=[B, 1, Q, K]

mask_probe :: Tensor[B,H,Q,D] -> Tensor[B,H,K,D] -> Tensor[B,H,K,D] -> (Tensor[B,H,Q,D], Tensor[B,H,Q,D])
mask_probe q k v = do
  full <- attention q k v (MASK_NAME q k)
  local <- attention q k v (MASK_NAME q k window=WINDOW)
  return full, local
"""

ATTENTION_OPTIONS = """\
import Attention (attention)
import Masking (causal_mask)

option_probe :: Tensor[B,H,Q,D] -> Tensor[B,H,K,D] -> Tensor[B,H,K,D] -> Tensor[B,H,Q,K] -> Tensor[B,H,Q,1] -> ?Tensor[B,K] -> Tensor[B,H,Q,D]
option_probe q k v rel sink pad = do
  keep <- causal_mask q k padding_mask=pad window=WINDOW
  return ATTENTION
"""

BRANCHES = """\
import Attention (attention)
import Masking (causal_mask)

branch_probe :: Tensor[B,H,Q,D] -> Tensor[B,H,K,D] -> Tensor[B,H,K,D] -> ?Tensor[B,K] -> Bool -> Tensor[B,H,Q,D]
branch_probe q k v pad full = full ? (attention q k v (causal_mask q k padding_mask=pad)) : (attention q k v (causal_mask q k padding_mask=pad window=WINDOW))
"""

# A select between a mask and the same mask with padding, the form a builtin mask inlines to.
SELECT = """\
import Attention (attention)
import Tensor (and, arange, cast, eq, expand, le, reshape, slice)

plain_band :: Tensor[B,HQ,Q,HDQ] -> Tensor[B,HK,K,HDK] -> ?Dim -> Tensor[B,1,Q,K]
plain_band q k ?window=null = do
  row_ids <- arange q start=0 end=Q |> reshape shape=[Q, 1]
  col_ids <- arange k start=0 end=K |> reshape shape=[1, K]
  aligned <- row_ids + (K - Q)
  upper <- le col_ids aligned
  keep_window <- (window == null) ? upper : (and upper (le (aligned - window + 1) col_ids))
  keep <- reshape keep_window shape=[1, 1, Q, K]
  return expand keep shape=[B, 1, Q, K]

other_band :: Tensor[B,HQ,Q,HDQ] -> Tensor[B,HK,K,HDK] -> ?Dim -> Tensor[B,1,Q,K]
other_band q k ?window=null = do
  row_ids <- arange q start=0 end=Q |> reshape shape=[Q, 1]
  col_ids <- arange k start=0 end=K |> reshape shape=[1, K]
  aligned <- row_ids + (K - Q)
  all_rows <- row_ids - row_ids
  all_cols <- col_ids - col_ids
  keep_window <- (window == null) ? (eq all_rows all_cols) : (and (le col_ids (aligned + window)) (le (aligned - window) col_ids))
  keep <- reshape keep_window shape=[1, 1, Q, K]
  return expand keep shape=[B, 1, Q, K]

padded_band :: Tensor[B,HQ,Q,HDQ] -> Tensor[B,HK,K,HDK] -> Tensor[B,SM] -> ?Dim -> Tensor[B,1,Q,K]
padded_band q k attn_mask ?window=null = do
  keep <- MASKED_BAND q k window=window
  pad_keep <- PAD_SOURCE |> cast dtype="bool" |> reshape shape=[B, 1, 1, K]
  return and keep pad_keep

select_probe :: Tensor[B,H,Q,D] -> Tensor[B,H,K,D] -> Tensor[B,H,K,D] -> ?Tensor[B,SM] -> ?Tensor[B,SM] -> (Tensor[B,H,Q,D], Tensor[B,H,Q,D])
select_probe q k v pad other = do
  full <- attention q k v ((CONDITION == null) ? (plain_band q k) : (padded_band q k pad))
  local <- attention q k v ((CONDITION == null) ? (plain_band q k window=WINDOW) : (padded_band q k pad window=PADDED_WINDOW))
  return full, local
"""
SELECT_DEFAULTS = {
    "MASKED_BAND": "plain_band",
    "PAD_SOURCE": "slice attn_mask dim=-1 start=(SM - K) end=SM",
    "CONDITION": "pad",
    "PADDED_WINDOW": "WINDOW",
}

LONG_PADDING = """\
import Attention (attention)
import Masking (bidirectional_mask)

padding_probe :: Tensor[B,H,Q,D] -> Tensor[B,H,K,D] -> Tensor[B,H,K,D] -> ?Tensor[B,SM] -> (Tensor[B,H,Q,D], Tensor[B,H,Q,D])
padding_probe q k v pad = do
  full <- attention q k v (bidirectional_mask q k padding_mask=pad)
  local <- attention q k v (bidirectional_mask q k padding_mask=pad window=WINDOW)
  return full, local
"""

OTHER_KEYS = """\
import Attention (attention)
import Masking (causal_mask)

keys_probe :: Tensor[B,H,Q,D] -> Tensor[B,H,K,D] -> Tensor[B,H,K,D] -> Tensor[B,H,K,D] -> Tensor[B,H,Q,D]
keys_probe q k v other = attention q k v (causal_mask q other)
"""


def _generated(directory, name, program, **options):
    source = directory / f"{name}.axon"
    source.write_text(program.replace("WINDOW", str(WINDOW)))
    return generate_core(source, class_name=name, **options)


def _build(directory, name, program=PROBE, **options):
    code = _generated(directory, name, program, **options)
    return load_core(code, name)({}), code


@pytest.fixture(scope="module")
def cores(tmp_path_factory):
    directory = tmp_path_factory.mktemp("band_probe")
    fused, source = _build(directory, "BandFused", backend_intrinsics=FUSED)
    reference, _ = _build(directory, "BandReference", optimize=False)
    assert source.count("self._sdpa_band(") == 4
    return fused, reference


def _case(batch, q_len, k_len, lengths):
    rng = np.random.default_rng(q_len * 1000 + k_len)
    q = mx.array(rng.standard_normal((batch, 2, q_len, 8)).astype(np.float32))
    k = mx.array(rng.standard_normal((batch, 2, k_len, 8)).astype(np.float32))
    v = mx.array(rng.standard_normal((batch, 2, k_len, 8)).astype(np.float32))
    pad = None if lengths is None else mx.array((np.arange(k_len)[None, :] < np.array(lengths)[:, None]).astype(np.int32))
    return q, k, v, pad


CASES = [
    # (batch, queries, keys, valid key count per row or None)
    (1, 9, 9, None),  # dense, band mask
    (1, 1, 12, None),  # one query over a short cache: no mask needed
    (1, 1, 40, None),  # one query over a long cache: window mask
    (1, 5, 30, None),  # a chunk after a cache
    (2, 33, 33, [33, 20]),  # padded keys
    (1, 4 * WINDOW + 1, 4 * WINDOW + 1, None),
    (1, 600, 600, None),  # blocked, length not a multiple of the window
    (2, 600, 600, [600, 517]),  # blocked with padded keys
]


@pytest.mark.parametrize("batch,q_len,k_len,lengths", CASES)
def test_band_lowering_matches_reference(cores, batch, q_len, k_len, lengths):
    fused, reference = cores
    q, k, v, pad = _case(batch, q_len, k_len, lengths)
    got = fused._def_band_probe(q, k, v, pad)
    want = reference._def_band_probe(q, k, v, pad)
    valid = np.ones((batch, q_len), dtype=bool)
    if lengths is not None and q_len == k_len:
        valid = np.arange(q_len)[None, :] < np.array(lengths)[:, None]  # pad query rows are unspecified
    for name, g, w in zip(("causal_full", "causal_local", "bidir_full", "bidir_local"), got, want):
        g, w = np.array(g), np.array(w)
        assert g.shape == w.shape, name
        assert np.allclose(g.transpose(0, 2, 1, 3)[valid], w.transpose(0, 2, 1, 3)[valid], atol=2e-5), name


def test_fusion_ignores_definition_names(tmp_path):
    """The same primitives fuse under any name, and match the reference."""
    program = LOCAL_MASK.replace("MASK_NAME", "my_window")
    fused, source = _build(tmp_path, "RenamedFused", program, backend_intrinsics=FUSED)
    reference, _ = _build(tmp_path, "RenamedReference", program, optimize=False)
    assert source.count("self._sdpa_band(") == 2
    q, k, v, _ = _case(1, 40, 40, None)
    for got, want in zip(fused._def_mask_probe(q, k, v), reference._def_mask_probe(q, k, v)):
        assert np.allclose(np.array(got), np.array(want), atol=2e-5)


def test_fusion_reaches_select_branches(tmp_path):
    """Models pick full or sliding attention per layer: each branch is a call inside a select."""
    assert _generated(tmp_path, "Branches", BRANCHES, backend_intrinsics=FUSED).count("self._sdpa_band(") == 2


@pytest.mark.parametrize(
    "old,new",
    [
        ("(aligned - window + 1)", "(aligned - window)"),  # one key more than the window
        ("upper <- le col_ids aligned", "upper <- le aligned col_ids"),  # the wrong triangle
        ("aligned <- row_ids + (K - Q)", "aligned <- row_ids"),  # queries not at the last keys
        ("start=0 end=Q", "start=1 end=Q"),  # not the query positions
    ],
    ids=["window", "bounds", "alignment", "positions"],
)
def test_changed_mask_under_a_builtin_name_does_not_fuse(tmp_path, old, new):
    program = LOCAL_MASK.replace("MASK_NAME", "causal_mask")
    assert program.count(old) == 1
    source = _generated(tmp_path, "Changed", program.replace(old, new), backend_intrinsics=FUSED)
    assert "self._sdpa_band(" not in source


@pytest.mark.parametrize(
    "attention",
    [
        "attention q k v keep scale=0.5",
        "attention q k v keep rel_bias=rel",
        "attention q k v keep sink_logits=sink",
        "attention q k v keep softcap=2.0",
        "attention q k v keep fp32_compute=true",
        "attention q k v keep zero_masked_after_softmax=false",
    ],
    ids=["scale", "rel_bias", "sink_logits", "softcap", "fp32_compute", "unzeroed"],
)
def test_attention_options_do_not_fuse(tmp_path, attention):
    """The kernel computes plain scaled dot-product attention, and nothing else."""
    program = ATTENTION_OPTIONS.replace("ATTENTION", attention)
    assert "self._sdpa_band(" not in _generated(tmp_path, "Option", program, backend_intrinsics=FUSED)
    assert "self._sdpa_band(" in _generated(
        tmp_path, "Plain", ATTENTION_OPTIONS.replace("ATTENTION", "attention q k v keep"), backend_intrinsics=FUSED,
    )


def test_mask_for_other_keys_does_not_fuse(tmp_path):
    assert "self._sdpa_band(" not in _generated(tmp_path, "Keys", OTHER_KEYS, backend_intrinsics=FUSED)


def test_mask_with_an_untyped_length_does_not_fuse(tmp_path):
    """The dim that a mask counts queries with must be the typed length of its q."""
    program = LOCAL_MASK.replace("MASK_NAME", "my_window")
    assert "self._sdpa_band(" in _generated(tmp_path, "Typed", program, backend_intrinsics=FUSED)
    program = (
        program.replace("my_window :: Tensor[B,HQ,Q,HDQ] -> Tensor[B,HK,K,HDK] -> ?Dim", "my_window :: Tensor[B,HQ,S,HDQ] -> Tensor[B,HK,K,HDK] -> Dim -> ?Dim")
        .replace("my_window q k ?window=null", "my_window q k Q ?window=null")
        .replace("(my_window q k)", "(my_window q k Q)")
        .replace("(my_window q k window=WINDOW)", "(my_window q k Q window=WINDOW)")
    )
    assert "self._sdpa_band(" not in _generated(tmp_path, "Untyped", program, backend_intrinsics=FUSED)


def _select_program(**changes):
    program = SELECT
    for name, value in {**SELECT_DEFAULTS, **changes}.items():
        program = program.replace(name, value)
    return program


def test_select_over_padding_fuses(tmp_path):
    assert _generated(tmp_path, "Select", _select_program(), backend_intrinsics=FUSED).count("self._sdpa_band(") == 2


@pytest.mark.parametrize(
    "changes,fused",
    [
        ({"CONDITION": "other"}, 0),  # the select tests another mask than the one applied
        ({"PADDED_WINDOW": "8"}, 1),  # the windowed branches differ; the full call still fuses
        ({"MASKED_BAND": "other_band"}, 0),  # the two branches are different bands
        ({"PAD_SOURCE": "slice attn_mask dim=-1 start=0 end=K"}, 0),  # the first K flags, not the last
    ],
    ids=["condition", "window", "band", "padding"],
)
def test_mismatched_select_does_not_fuse(tmp_path, changes, fused):
    source = _generated(tmp_path, "Mismatch", _select_program(**changes), backend_intrinsics=FUSED)
    assert source.count("self._sdpa_band(") == fused


def test_padding_longer_than_the_keys_uses_its_last_entries(tmp_path):
    """The band lowering must read the same flags as the mask: the last K of a longer padding."""
    fused, source = _build(tmp_path, "PaddingFused", LONG_PADDING, backend_intrinsics=FUSED)
    reference, _ = _build(tmp_path, "PaddingReference", LONG_PADDING, optimize=False)
    assert source.count("self._sdpa_band(") == 2
    q, k, v, _ = _case(2, 33, 33, None)
    flags = np.ones((2, 36), dtype=np.int32)
    flags[:, :3] = 0  # masks nothing when read correctly
    flags[1, 28:] = 0
    pad = mx.array(flags)
    valid = np.ones((2, 33), dtype=bool)
    valid[1, 25:] = False
    for got, want in zip(fused._def_padding_probe(q, k, v, pad), reference._def_padding_probe(q, k, v, pad)):
        got, want = np.array(got).transpose(0, 2, 1, 3), np.array(want).transpose(0, 2, 1, 3)
        assert np.allclose(got[valid], want[valid], atol=2e-5)


def _constant_window_band(width):
    """Provenance of an expanded causal band whose window was folded into the mask."""
    q, k, Q, K, B = (_make_provenance("input", name=name) for name in ("q", "k", "Q", "K", "B"))
    zero, one = _make_provenance("literal", value=0), _make_provenance("literal", value=1)
    row = _op_provenance("_reshape", _op_provenance("_arange", q, zero, Q), _op_provenance("core.list", Q, one))
    col = _op_provenance("_reshape", _op_provenance("_arange", k, zero, K), _op_provenance("core.list", one, K))
    aligned = _op_provenance("core.binary.+", row, _op_provenance("core.binary.-", K, Q))
    start = _op_provenance("core.binary.+", _op_provenance("core.binary.-", aligned, width), one)
    band = _op_provenance("_and", _op_provenance("_le", col, aligned), _op_provenance("_le", start, col))
    reshaped = _op_provenance("_reshape", band, _op_provenance("core.list", one, one, Q, K))
    return _op_provenance("_expand", reshaped, _op_provenance("core.list", B, one, Q, K))


@pytest.mark.parametrize(
    "width,fuses",
    [
        (_make_provenance("literal", value=16), True),
        (_make_provenance("input", name="window"), True),
        (_make_provenance("global", name="WINDOW"), False),  # a name could read a shadowing binding
        (_make_provenance("literal", value=1.5), False),
        (_make_provenance("literal", value=True), False),
    ],
    ids=["int", "formal", "global", "float", "bool"],
)
def test_band_width_must_reach_the_call_exactly(width, fuses):
    fact = _band_keep_fact(_constant_window_band(width))
    assert (fact is not None) == fuses
    if fuses:
        assert fact.causal and fact.window == width


def test_fusion_keeps_callees_with_other_work(tmp_path):
    """A callee with a node that feeds no output (a parameter read can fail) is not replaced."""
    source = tmp_path / "Dead.axon"
    source.write_text(ATTENTION_OPTIONS.replace("ATTENTION", "attention q k v keep").replace("WINDOW", str(WINDOW)))
    program = resolve_axon_program_from_path(source).ast
    for transform in (
        normalize_closed_axon_file,
        elaborate_closed_axon_file,
        flatten_closed_axon_file,
        typecheck2_flat_axon_file,
    ):
        program = transform(program)
    graph = optimize_graph_program(
        lower_axon_program_to_graph_ir(program), config=GraphOptimizeConfig(backend_intrinsics="codegen2-mlx:__mlx_rope"),
    )

    def fused_calls(candidate):
        return sum(node.op.name == "__mlx_sdpa_band" for module in candidate.modules for node in module.nodes)

    assert fused_calls(_rewrite_mlx_sdpa_band_intrinsics(graph)) == 1
    attention = next(module for module in graph.modules if module.name == "Attention.attention")
    template = attention.nodes[0]
    read = replace(
        template, op=GraphOp("Params.param"), inputs=(), attrs={}, id="unused_read",
        outputs=(replace(template.outputs[0], name="unused"),),
    )
    changed = replace(attention, nodes=(read, *attention.nodes))
    modules = tuple(changed if module is attention else module for module in graph.modules)
    assert fused_calls(_rewrite_mlx_sdpa_band_intrinsics(replace(graph, modules=modules))) == 0
