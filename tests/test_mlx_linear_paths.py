"""Exercise emitted parameter lookups without importing or executing MLX."""

import ast

import pytest

from synapse.axon.ast import (
    TypeBool,
    TypeDim,
    TypeInt,
    TypeNull,
    TypePath,
    TypeString,
    TypeTensor,
    TypeTuple,
)
from synapse.axon.codegen2_common import compose_path
from synapse.axon.codegen2_mlx import emit_model_code_from_graph_ir
from synapse.axon.graph_ir import (
    GraphLiteral,
    GraphModule,
    GraphNode,
    GraphOp,
    GraphPath,
    GraphProgram,
    GraphValue,
    GraphValueRef,
    optimize_graph_program,
)


def linear_graph(leaves, *, pack=False, templated=False, dynamic=False):
    tensor = TypeTensor("Tensor", (2, 3, 4))
    inputs = [GraphValue("x", tensor, dims=tensor.dims)]
    if templated:
        inputs.append(GraphValue("layer", TypeInt()))
    if dynamic:
        inputs.extend(GraphValue(name, TypePath()) for name in ("weight_path", "bias_path"))
    nodes = []
    outputs = []
    for name in ("q", "k", "v") if pack else ("q",):
        base = GraphPath(True, ("encoder", "{layer}" if templated else "0", name))
        output = GraphValue(name, tensor, dims=tensor.dims)
        nodes.append(
            GraphNode(
                id=f"main:{name}",
                op=GraphOp("_linear"),
                inputs=(
                    base,
                    GraphValueRef("x", tensor, dims=tensor.dims),
                    GraphLiteral(4, TypeDim()),
                    GraphLiteral(True, TypeBool()),
                    GraphLiteral(False, TypeBool()),
                    GraphLiteral(None, TypeNull()),
                    *leaves,
                ),
                attrs={},
                outputs=(output,),
                source_module="main",
                type_expr=tensor,
                dims=tensor.dims,
            )
        )
        outputs.append(GraphValueRef(name, tensor, dims=tensor.dims))
    graph = GraphProgram(
        modules=(
            GraphModule(
                name="main",
                inputs=tuple(inputs),
                outputs=tuple(outputs),
                output_names=tuple(value.name for value in outputs),
                nodes=tuple(nodes),
                return_type_expr=TypeTuple(tuple(tensor for _ in outputs)) if pack else tensor,
            ),
        ),
        main_module="main",
        pragmas={"main": "main"},
    )
    return optimize_graph_program(graph) if pack else graph


def emitted_linear_call(graph):
    code = emit_model_code_from_graph_ir(graph)
    calls = [
        node
        for node in ast.walk(ast.parse(code))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_linear_matmul"
    ]
    assert len(calls) == 1
    return compile(ast.Expression(calls[0]), "<emitted-linear>", "eval")


class ParameterLookup:
    """Record actual emitted lookups; tensor execution remains a Mac check."""

    _compose_path = staticmethod(compose_path)
    _path_template_part = staticmethod(str)

    def __init__(self, weight_key, bias_key):
        self.weight_key = weight_key
        self.bias_key = bias_key

    def _optional_param(self, path):
        assert path.lstrip("@") == self.bias_key
        return "resolved-bias"

    def _linear_matmul(self, path, x, *, bias, transpose):
        assert path.lstrip("@") == self.weight_key
        assert bias == "resolved-bias"
        assert transpose is False
        return x


@pytest.mark.parametrize("templated", [False, True])
@pytest.mark.parametrize(
    "kind",
    ["relative", "absolute", "custom", "literal", "empty", "dynamic-relative", "dynamic-absolute"],
)
def test_emitted_linear_resolves_weight_and_bias_paths(kind, templated):
    dynamic = kind.startswith("dynamic-")
    prefix = f"encoder.{2 if templated else 0}.q"
    if dynamic:
        leaves = tuple(GraphValueRef(name, TypePath()) for name in ("weight_path", "bias_path"))
    elif kind == "literal":
        leaves = tuple(GraphLiteral(name, TypeString()) for name in ("weight", "bias"))
    else:
        parts = {
            "relative": (("weight",), ("bias",)),
            "absolute": (("shared", "weight"), ("shared", "bias")),
            "custom": (("custom", "kernel"), ("custom", "offset")),
            "empty": ((), ()),
        }[kind]
        leaves = tuple(GraphPath(kind == "absolute", item) for item in parts)
    if kind in ("absolute", "dynamic-absolute"):
        expected = ("shared.weight", "shared.bias")
    elif kind == "custom":
        expected = (prefix + ".custom.kernel", prefix + ".custom.offset")
    elif kind == "empty":
        expected = (prefix, prefix)
    else:
        expected = (prefix + ".weight", prefix + ".bias")
    lookup = ParameterLookup(*expected)
    graph = linear_graph(leaves, templated=templated, dynamic=dynamic)
    env = dict(
        self=lookup,
        x=object(),
        layer=2,
        weight_path="@@shared.weight" if kind == "dynamic-absolute" else "@weight",
        bias_path="@@shared.bias" if kind == "dynamic-absolute" else "@bias",
    )
    assert eval(emitted_linear_call(graph), env) is env["x"]


@pytest.mark.parametrize("templated", [False, True])
def test_packed_projection_lookup_matches_materialized_parameter_keys(templated):
    leaves = tuple(GraphPath(False, (name,)) for name in ("weight", "bias"))
    graph = linear_graph(leaves, pack=True, templated=templated)
    assert len(graph.packed_parameters) == 2
    keys = [".".join(spec.output.parts).replace("{layer}", "2") for spec in graph.packed_parameters]
    assert all(key.startswith("__packed.linear_pack.") for key in keys)
    env = dict(self=ParameterLookup(*keys), x=object(), layer=2)
    assert eval(emitted_linear_call(graph), env) is env["x"]
