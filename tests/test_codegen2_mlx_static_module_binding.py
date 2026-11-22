"""Regression gate: static shadow modules survive unpatched MLX construction.

The emitted ``load_state_dict`` installs ``nn.Embedding``/``nn.Linear``/
``nn.LayerNorm`` shadow modules under mangled parameter-path names so the
forward's one-argument call sites (``self._embedding(input_ids)``) resolve to
real modules. Those assignments go through ``nn.Module.__setattr__``, which —
because ``nn.Module`` subclasses ``dict`` — routes any module value into its
container branch and, when the name only exists as a class attribute (the
always-emitted ``_embedding`` helper method), tries to ``delattr`` it and
raises ``AttributeError``. The harnesses happened to drive these classes with
a ``__setattr__`` fallback patch, so the trap only surfaced when the server
constructed an emitted class directly.

``_bind_static_module`` binds through plain ``setattr`` and falls back to
``object.__setattr__`` on that AttributeError; a container entry could not
shadow the class method anyway (normal attribute lookup finds class methods
before ``__getattr__`` ever runs), so the instance attribute is the only
binding that satisfies the one-argument call sites.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import pytest

from synapse.axon.ast import TypeInt, TypeString, TypeTensor
from synapse.axon.codegen2_mlx import (
    emit_model_code_from_graph_ir as emit_mlx_model_code_from_graph_ir,
)
from synapse.axon.graph_ir import (
    GraphLiteral,
    GraphModule,
    GraphNode,
    GraphOp,
    GraphPath,
    GraphProgram,
    GraphValue,
    GraphValueRef,
)


def _tensor(*dims: str) -> TypeTensor:
    return TypeTensor("Tensor", dims)


def _embedding_graph() -> GraphProgram:
    """A single static embedding op: `y = embedding(@@embedding, x)`.

    The constant path makes the collector register a static param op and the
    emitter emit the one-argument `self._embedding(x)` call site backed by the
    `load_state_dict` shadow — the exact shape that broke under unpatched MLX
    construction.
    """
    ids_type = _tensor("B", "S")
    out_type = _tensor("B", "S", "V")
    main = GraphModule(
        name="main",
        inputs=(GraphValue("x", ids_type),),
        outputs=(GraphValueRef("y", out_type),),
        output_names=("y",),
        nodes=(
            GraphNode(
                id="main:1",
                op=GraphOp("embedding"),
                inputs=(
                    GraphPath(absolute=True, parts=("embedding",)),
                    GraphValueRef("x", ids_type),
                ),
                attrs={},
                outputs=(GraphValue("y", out_type),),
                source_module="main",
                type_expr=out_type,
            ),
        ),
        return_type_expr=out_type,
    )
    return GraphProgram(modules=(main,), main_module="main", pragmas={"main": "main"})


def _model_cls():
    pytest.importorskip("mlx")
    namespace: dict[str, object] = {}
    exec(emit_mlx_model_code_from_graph_ir(_embedding_graph()), namespace)
    return namespace["AxonMlxModel"]


def test_static_embedding_shadow_binds_unpatched() -> None:
    cls = _model_cls()
    weight = mx.arange(32, dtype=mx.float32).reshape(4, 8)
    # No __setattr__ patch: this is the exact construction the server runs.
    model = cls.from_state_dict({"embedding.weight": weight})
    shadow = model._embedding
    assert isinstance(shadow, nn.Embedding)
    assert shadow.weight.shape == (4, 8)
    # The one-argument call site resolves to the shadow, not the helper.
    ids = mx.array([[0, 3, 1]], dtype=mx.int64)
    out = model.forward(x=ids)
    assert mx.array_equal(mx.asarray(out), weight[ids])


def test_embedding_helper_method_stays_when_no_shadow() -> None:
    cls = _model_cls()
    model = cls.from_state_dict({})
    # Without 'embedding.weight' the two-argument path helper must remain
    # bound (it serves explicit-path embedding call sites).
    assert not isinstance(model._embedding, nn.Embedding)
    assert callable(model._embedding)
