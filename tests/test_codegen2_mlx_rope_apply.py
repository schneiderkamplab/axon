"""Regression gate for the codegen2-mlx rope intrinsic (`_mlx_rope_apply`).

The optimized rope path feeds `mx.fast.rope` with frequencies and an amplitude
recovered from the model's sin/cos tables. Two defects shipped historically:

  - inv_freq was recovered with a stride-1 slice of the table's first half.
    Interleaved (GPT-J style) tables duplicate each frequency adjacently, so
    the recovery produced pairwise-duplicated frequencies and rotations that
    drift with position (parity broke on yarn/interleaved models only).
  - `mx.fast.rope` applies a pure rotation, while the tables carry an
    amplitude (the yarn mscale; 1.0 for plain rope) that must be reapplied to
    its output. It was dropped, so yarn models lost the attention_factor
    scaling on every rotated query/key.

The tests emit the model code for a synthetic `_mlx_rope` graph (no weights)
and compare the emitted `_mlx_rope_apply` against manual rotations for both
layouts, with and without a yarn mscale, for full-sequence (S > 1) tables and
for single-row (S == 1) tables built at a later position (the incremental
decode path, where the amplitude must be recovered from the row itself).
"""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import pytest

from synapse.axon.ast import TypeBool, TypeTensor
from synapse.axon.codegen2_mlx import (
    emit_model_code_from_graph_ir as emit_mlx_model_code_from_graph_ir,
)
from synapse.axon.graph_ir import (
    GraphLiteral,
    GraphModule,
    GraphNode,
    GraphOp,
    GraphProgram,
    GraphValue,
    GraphValueRef,
)

YARN_MSCALE = 0.1 * math.log(32.0) + 1.0
HEAD_DIM = 64
THETA = 10000.0
PREFILL_LEN = 48
DECODE_POS = 37
ATOL = 1e-4


def _tensor(*dims: str) -> TypeTensor:
    return TypeTensor("Tensor", dims)


def _rope_graph(traditional: bool) -> GraphProgram:
    x_type = _tensor("B", "H", "S", "D")
    table_type = _tensor("B", "HM", "S", "D")
    main = GraphModule(
        name="main",
        inputs=(
            GraphValue("x", x_type),
            GraphValue("sin", table_type),
            GraphValue("cos", table_type),
        ),
        outputs=(GraphValueRef("y", x_type),),
        output_names=("y",),
        nodes=(
            GraphNode(
                id="main:1",
                op=GraphOp("__mlx_rope"),
                inputs=(
                    GraphValueRef("x", x_type),
                    GraphLiteral(traditional, TypeBool()),
                    GraphValueRef("sin", table_type),
                    GraphValueRef("cos", table_type),
                ),
                attrs={},
                outputs=(GraphValue("y", x_type),),
                source_module="main",
                type_expr=x_type,
            ),
        ),
        return_type_expr=x_type,
    )
    return GraphProgram(modules=(main,), main_module="main", pragmas={"main": "main"})


def _model_cls(traditional: bool):
    pytest.importorskip("mlx")
    namespace: dict[str, object] = {}
    exec(emit_mlx_model_code_from_graph_ir(_rope_graph(traditional)), namespace)
    return namespace["AxonMlxModel"]


def _inv_freq() -> np.ndarray:
    return (THETA ** (-np.arange(0, HEAD_DIM, 2, dtype=np.float64) / HEAD_DIM)).astype(np.float64)


def _tables(
    inv_freq: np.ndarray,
    positions: np.ndarray,
    traditional: bool,
    mscale: float,
) -> tuple[mx.array, mx.array]:
    """Build sin/cos tables the way rope table builders emit them: full width,
    each frequency duplicated (adjacently for interleaved rope, in repeated
    halves for NeoX), both tables scaled by the amplitude (yarn mscale)."""
    angles = positions[:, None].astype(np.float64) * inv_freq[None, :]
    cos_half = (np.cos(angles) * mscale).astype(np.float32)
    sin_half = (np.sin(angles) * mscale).astype(np.float32)
    if traditional:
        cos = np.repeat(cos_half, 2, axis=-1)
        sin = np.repeat(sin_half, 2, axis=-1)
    else:
        cos = np.concatenate([cos_half, cos_half], axis=-1)
        sin = np.concatenate([sin_half, sin_half], axis=-1)
    return mx.array(sin[None, None]), mx.array(cos[None, None])


def _manual_rope(x: mx.array, sin: mx.array, cos: mx.array, traditional: bool) -> mx.array:
    """Reference rotation consuming the same full-width tables (which already
    carry the amplitude, so the reference applies no extra scale)."""
    half = x.shape[-1] // 2
    if traditional:
        cos_half, sin_half = cos[..., 0::2], sin[..., 0::2]
        pairs = x.reshape(*x.shape[:-1], half, 2)
        even, odd = pairs[..., 0], pairs[..., 1]
        rotated = mx.stack([even * cos_half - odd * sin_half, odd * cos_half + even * sin_half], axis=-1)
        return rotated.reshape(*x.shape)
    cos_half, sin_half = cos[..., :half], sin[..., :half]
    x1, x2 = x[..., :half], x[..., half:]
    return mx.concatenate([x1 * cos_half - x2 * sin_half, x2 * cos_half + x1 * sin_half], axis=-1)


def _random_x(*shape: int) -> mx.array:
    mx.random.seed(7)
    return mx.random.normal(shape)


def test_rope_apply_interleaved_with_yarn_mscale() -> None:
    model = _model_cls(traditional=True).from_state_dict({})
    sin, cos = _tables(_inv_freq(), np.arange(PREFILL_LEN), True, YARN_MSCALE)
    x = _random_x(1, 3, PREFILL_LEN, HEAD_DIM)
    out = model.forward(x=x, sin=sin, cos=cos)
    ref = _manual_rope(x, sin, cos, True)
    err = float(mx.abs(out.astype(mx.float32) - ref.astype(mx.float32)).max())
    assert err < ATOL, f"interleaved rope with mscale diverged: max|delta|={err}"


def test_rope_apply_interleaved_plain_rope_unchanged() -> None:
    model = _model_cls(traditional=True).from_state_dict({})
    sin, cos = _tables(_inv_freq(), np.arange(PREFILL_LEN), True, 1.0)
    x = _random_x(1, 3, PREFILL_LEN, HEAD_DIM)
    out = model.forward(x=x, sin=sin, cos=cos)
    ref = _manual_rope(x, sin, cos, True)
    err = float(mx.abs(out.astype(mx.float32) - ref.astype(mx.float32)).max())
    assert err < ATOL, f"plain interleaved rope diverged: max|delta|={err}"


def test_rope_apply_neox_plain_rope_unchanged() -> None:
    model = _model_cls(traditional=False).from_state_dict({})
    sin, cos = _tables(_inv_freq(), np.arange(PREFILL_LEN), False, 1.0)
    x = _random_x(1, 3, PREFILL_LEN, HEAD_DIM)
    out = model.forward(x=x, sin=sin, cos=cos)
    ref = _manual_rope(x, sin, cos, False)
    err = float(mx.abs(out.astype(mx.float32) - ref.astype(mx.float32)).max())
    assert err < ATOL, f"plain neox rope diverged: max|delta|={err}"


@pytest.mark.parametrize("traditional", [True, False], ids=["interleaved", "neox"])
def test_rope_apply_single_row_decode_recovers_position_and_mscale(traditional: bool) -> None:
    model = _model_cls(traditional=traditional).from_state_dict({})
    # Prefill first: populates the frequency caches from a full-sequence table.
    sin, cos = _tables(_inv_freq(), np.arange(PREFILL_LEN), traditional, YARN_MSCALE)
    prefill_x = _random_x(1, 3, PREFILL_LEN, HEAD_DIM)
    model.forward(x=prefill_x, sin=sin, cos=cos)
    # Decode step: a single-row table built at a later position. The emitted
    # path must recover both the position (for the rope offset) and the table
    # amplitude (for the mscale) from that row alone.
    sin_row, cos_row = _tables(_inv_freq(), np.array([DECODE_POS]), traditional, YARN_MSCALE)
    x = _random_x(1, 3, 1, HEAD_DIM)
    out = model.forward(x=x, sin=sin_row, cos=cos_row)
    ref = _manual_rope(x, sin_row, cos_row, traditional)
    err = float(mx.abs(out.astype(mx.float32) - ref.astype(mx.float32)).max())
    assert err < ATOL, f"single-row rope at pos {DECODE_POS} diverged: max|delta|={err}"


def test_rope_apply_single_row_first_call_uses_tables_directly() -> None:
    model = _model_cls(traditional=True).from_state_dict({})
    sin_row, cos_row = _tables(_inv_freq(), np.array([DECODE_POS]), True, YARN_MSCALE)
    x = _random_x(1, 3, 1, HEAD_DIM)
    out = model.forward(x=x, sin=sin_row, cos=cos_row)
    ref = _manual_rope(x, sin_row, cos_row, True)
    err = float(mx.abs(out.astype(mx.float32) - ref.astype(mx.float32)).max())
    assert err < ATOL, f"first-call single-row rope diverged: max|delta|={err}"
