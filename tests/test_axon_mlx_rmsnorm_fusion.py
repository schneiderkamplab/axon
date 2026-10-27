"""Structural fusion checks independent of any model family or definition name."""
from dataclasses import replace

import numpy as np
import pytest

from synapse.axon.ast import TypeBool, TypeDim, TypeFloat, TypeNull, TypePath, TypeTensor, TypeTuple
from synapse.axon.codegen2_mlx import emit_model_code_from_graph_ir
from synapse.axon.graph_ir import (
    GraphExpr,
    GraphLiteral,
    GraphModule,
    GraphNode,
    GraphOp,
    GraphOptimizeConfig,
    GraphPath,
    GraphProgram,
    GraphValue,
    GraphValueRef,
    optimize_graph_program,
    validate_graph_program,
)


def _graph(*, norm_name="arbitrary_normalizer", scale_name="arbitrary_scale",
           dim=None, cast_float=True, reverse=False, misleading=False,
           reuse_norm=False, scale_dims=("D",)):
    tensor = TypeTensor("Tensor", ("B", "D"))
    weight = TypeTensor("Tensor", scale_dims)
    x = GraphValueRef("x", tensor)
    eps = GraphValueRef("eps", TypeFloat())
    path = GraphValueRef("path", TypePath())

    def node(name, op, args, result, typ=tensor):
        return GraphNode(id=name, op=GraphOp(op), inputs=tuple(args), attrs={},
                         outputs=(GraphValue(result, typ),), type_expr=typ,
                         source_module="test")

    norm_args = (x, eps, GraphLiteral(dim, TypeNull() if dim is None else TypeDim()),
                 GraphLiteral(cast_float, TypeBool()))
    norm = GraphModule(
        name=norm_name, inputs=(GraphValue("x", tensor), GraphValue("eps", TypeFloat())),
        nodes=(node("normalize", "core.binary.+" if misleading else "_rmsnorm",
                    (x, x) if misleading else norm_args, "normalized"),),
        outputs=(GraphValueRef("normalized", tensor),), output_names=("normalized",),
        return_type_expr=tensor,
    )
    scale_args = (GraphValueRef("weight", weight), x) if reverse else (x, GraphValueRef("weight", weight))
    scale = GraphModule(
        name=scale_name, inputs=(GraphValue("x", tensor), GraphValue("path", TypePath())),
        nodes=(node("weight", "_params_param", (path,), "weight", weight),
               node("scale", "core.binary.*", scale_args, "scaled")),
        outputs=(GraphValueRef("scaled", tensor),), output_names=("scaled",),
        return_type_expr=tensor,
    )
    nodes = [
        node("norm_call", norm_name, (x, eps), "normalized"),
        node("scale_call", scale_name,
             (GraphValueRef("normalized", tensor), GraphPath(True, ("weight",))), "scaled"),
    ]
    if reuse_norm:
        nodes.append(node("reuse", "core.binary.+",
                          (GraphValueRef("normalized", tensor), GraphValueRef("scaled", tensor)), "combined"))
    output = "combined" if reuse_norm else "scaled"
    main = GraphModule(
        name="main", inputs=(GraphValue("x", tensor), GraphValue("eps", TypeFloat())),
        nodes=tuple(nodes), outputs=(GraphValueRef(output, tensor),), output_names=(output,),
        return_type_expr=tensor,
    )
    return GraphProgram(modules=(norm, scale, main), main_module="main", pragmas={"main": "main"})


def _optimize(graph, backend="codegen2-mlx:__mlx_rmsnorm_scaled", **settings):
    validate_graph_program(graph)
    result = optimize_graph_program(graph, config=GraphOptimizeConfig(backend_intrinsics=backend, **settings))
    validate_graph_program(result)
    return result


def _fused(graph):
    return [n for m in graph.modules for n in m.nodes if n.op.name == "__mlx_rmsnorm_scaled"]


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("inline_safe", [False, True])
def test_renamed_definitions_fuse_from_primitive_provenance(reverse, inline_safe):
    graph = _optimize(_graph(reverse=reverse), inline_safe=inline_safe, specialize_definitions="off")
    assert _fused(graph)
    # Epsilon remains a caller input; no guessed default is introduced.
    assert any(isinstance(n.inputs[2], GraphValueRef) and n.inputs[2].name == "eps"
               for n in _fused(graph))


def test_extra_parameter_read_in_a_wrapper_is_not_discarded():
    graph = _graph()
    scale = graph.modules[1]
    extra = replace(scale.nodes[0], id="unused_read",
                    inputs=(GraphPath(True, ("missing",)),),
                    outputs=(GraphValue("unused", TypeTensor("Tensor", ("D",))),))
    scale = replace(scale, nodes=(*scale.nodes, extra))
    graph = replace(graph, modules=(graph.modules[0], scale, graph.modules[2]))
    graph = _optimize(graph, inline_safe=False, specialize_definitions="off")
    assert not _fused(graph)


@pytest.mark.parametrize("nested", [False, True], ids=["direct", "nested"])
@pytest.mark.parametrize("backend", [None, "codegen2-mlx:__mlx_rmsnorm_scaled"])
def test_unused_wrapper_argument_preserves_required_parameter_read(nested, backend):
    graph = _graph()
    norm, scale, main = graph.modules
    weight = TypeTensor("Tensor", ("D",))
    scale = replace(scale, inputs=(*scale.inputs, GraphValue("unused", weight)))
    modules = [norm, scale]
    if nested:
        forwarded = replace(
            main.nodes[1], id="forward_scale",
            inputs=tuple(GraphValueRef(value.name, value.type_expr) for value in scale.inputs),
        )
        scale = replace(scale, name="outer_scale", nodes=(forwarded,))
        modules.append(scale)
    read = GraphExpr(
        GraphOp("_params_param"), (GraphPath(True, ("missing",)),), {}, weight,
    )
    call = replace(main.nodes[1], op=GraphOp(scale.name), inputs=(*main.nodes[1].inputs, read))
    main = replace(main, nodes=(main.nodes[0], call))
    graph = replace(graph, modules=(*modules, main))
    graph = _optimize(graph, backend, inline_safe=False, specialize_definitions="off")
    assert not _fused(graph)

    mx = pytest.importorskip("mlx.core")
    namespace = {}
    exec(emit_model_code_from_graph_ir(graph), namespace)
    model = namespace["AxonMlxModel"].from_state_dict({"weight": mx.ones((4,))})
    with pytest.raises(KeyError, match="missing"):
        mx.eval(model.forward(x=mx.ones((1, 4)), eps=1e-3))


@pytest.mark.parametrize("read_before", [False, True])
@pytest.mark.parametrize("inline_safe", [False, True])
def test_repeated_scale_read_cannot_introduce_a_forward_reference(read_before, inline_safe):
    mx = pytest.importorskip("mlx.core")
    graph = _graph()
    main = graph.modules[-1]
    tensor = main.return_type_expr
    weight = TypeTensor("Tensor", ("D",))
    read = GraphNode(
        id="read_again", op=GraphOp("_params_param"),
        inputs=(GraphPath(True, ("weight",)),), attrs={},
        outputs=(GraphValue("other_weight", weight),), type_expr=weight,
        source_module="test",
    )
    combine = GraphNode(
        id="combine", op=GraphOp("core.binary.+"),
        inputs=(GraphValueRef("scaled", tensor), GraphValueRef("other_weight", weight)),
        attrs={}, outputs=(GraphValue("combined", tensor),), type_expr=tensor,
        source_module="test",
    )
    nodes = (read, *main.nodes) if read_before else (*main.nodes, read)
    main = replace(main, nodes=(*nodes, combine),
                   outputs=(GraphValueRef("combined", tensor),), output_names=("combined",))
    graph = replace(graph, modules=(*graph.modules[:-1], main))
    graph = _optimize(graph, inline_safe=inline_safe, specialize_definitions="off")
    namespace = {}
    exec(emit_model_code_from_graph_ir(graph), namespace)
    x = mx.array([[0.5, -2.0, 3.0, 1.0]], dtype=mx.float32)
    scale = mx.array([1.2, -0.3, 2.0, 0.7], dtype=mx.float32)
    model = namespace["AxonMlxModel"].from_state_dict({"weight": scale})
    actual = model.forward(x=x, eps=1e-3)
    expected = mx.fast.rms_norm(x, None, 1e-3) * scale + scale
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), atol=1e-6, rtol=2e-6)


def test_shared_variadic_helpers_and_global_epsilon_fuse(tmp_path):
    from synapse.axon import (
        elaborate_closed_axon_file,
        flatten_closed_axon_file,
        lower_axon_program_to_graph_ir,
        normalize_closed_axon_file,
        resolve_axon_program_from_path,
        typecheck2_flat_axon_file,
    )
    path = tmp_path / "shared.axon"
    path.write_text("""
EPS <- _config_float @@eps 0.01
scale :: Tensor[..S] -> Path -> Tensor[..S]
scale x path = ((x * (_params_param path)) :: Tensor[..S])
normalize :: Tensor[..S] -> Tensor[..S]
normalize x = _rmsnorm x EPS null true
adjust :: Tensor[..S] -> Path -> Tensor[..S]
adjust x path = do
  n <- normalize x
  y <- scale n path
  return y
main :: Tensor[B,D] -> Tensor[B,D] -> Tensor[B,D]
main x z = do
  a <- adjust x @@first
  b <- adjust z @@second
  return a + b
""")
    program = resolve_axon_program_from_path(path).ast
    for transform in (normalize_closed_axon_file, elaborate_closed_axon_file,
                      flatten_closed_axon_file, typecheck2_flat_axon_file):
        program = transform(program)
    graph = _optimize(lower_axon_program_to_graph_ir(program))
    assert _fused(graph)
    assert any(isinstance(n.inputs[2], GraphValueRef) and n.inputs[2].name == "EPS"
               for n in _fused(graph))


def test_familiar_names_do_not_authorize_a_wrong_fusion():
    graph = _graph(norm_name="NN.rmsnorm_noscale", scale_name="Params.param_scale",
                   misleading=True)
    main = graph.modules[-1]
    graph = replace(graph, modules=(*graph.modules[:-1], replace(main, name="NN.rmsnorm")),
                    main_module="NN.rmsnorm", pragmas={"main": "NN.rmsnorm"})
    assert not _fused(_optimize(graph))


@pytest.mark.parametrize("dim,cast_float", [(0, True), (1, True), (None, False)])
def test_unproven_axis_or_cast_semantics_are_not_fused(dim, cast_float):
    assert not _fused(_optimize(_graph(dim=dim, cast_float=cast_float)))


@pytest.mark.parametrize("backend", [None, "codegen2-mlx:__mlx_sdpa"])
def test_scaled_rmsnorm_requires_explicit_backend_opt_in(backend):
    graph = _optimize(_graph(), backend)
    assert not _fused(graph)


def test_triton_still_uses_the_shared_structural_matcher():
    graph = _optimize(_graph(), "codegen2-triton:__triton_rmsnorm_scaled")
    assert any(n.op.name == "__triton_rmsnorm_scaled" for m in graph.modules for n in m.nodes)
    assert not _fused(graph)


@pytest.mark.parametrize("inline_safe", [False, True])
@pytest.mark.parametrize("alias", [False, True])
@pytest.mark.parametrize("reuse_norm", [False, True])
def test_fusion_requires_the_original_normalization_to_be_removable(inline_safe, alias, reuse_norm):
    graph = _graph(reuse_norm=reuse_norm)
    if alias:
        main = graph.modules[-1]
        tensor = main.return_type_expr
        renamed = GraphNode(
            id="alias", op=GraphOp("core.alias"),
            inputs=(GraphValueRef("normalized", tensor),), attrs={},
            outputs=(GraphValue("alias", tensor),), type_expr=tensor, source_module="test",
        )
        scale = replace(main.nodes[1], inputs=(GraphValueRef("alias", tensor), main.nodes[1].inputs[1]))
        main = replace(main, nodes=(main.nodes[0], renamed, scale, *main.nodes[2:]))
        graph = replace(graph, modules=(*graph.modules[:-1], main))
    graph = _optimize(graph, inline_safe=inline_safe, specialize_definitions="off")
    assert bool(_fused(graph)) is not reuse_norm


@pytest.mark.parametrize("inline_safe", [False, True])
def test_two_scale_consumers_share_one_normalization(inline_safe):
    graph = _graph()
    main = graph.modules[-1]
    tensor = main.return_type_expr
    second = replace(main.nodes[1], id="second_scale",
                     inputs=(GraphValueRef("normalized", tensor), GraphPath(True, ("other",))),
                     outputs=(GraphValue("second", tensor),))
    combine = GraphNode(
        id="combine", op=GraphOp("core.binary.+"),
        inputs=(GraphValueRef("scaled", tensor), GraphValueRef("second", tensor)),
        attrs={}, outputs=(GraphValue("combined", tensor),), type_expr=tensor, source_module="test",
    )
    main = replace(main, nodes=(*main.nodes, second, combine),
                   outputs=(GraphValueRef("combined", tensor),), output_names=("combined",))
    graph = replace(graph, modules=(*graph.modules[:-1], main))
    assert not _fused(_optimize(graph, inline_safe=inline_safe, specialize_definitions="off"))


@pytest.mark.parametrize("inline_safe", [False, True])
def test_returned_normalization_is_not_recomputed_for_scaling(inline_safe):
    graph = _graph()
    main = graph.modules[-1]
    tensor = main.return_type_expr
    main = replace(main, outputs=(GraphValueRef("normalized", tensor), *main.outputs),
                   output_names=("normalized", "scaled"), return_type_expr=TypeTuple((tensor, tensor)))
    graph = replace(graph, modules=(*graph.modules[:-1], main))
    assert not _fused(_optimize(graph, inline_safe=inline_safe, specialize_definitions="off"))


def test_normalization_with_a_required_parameter_read_is_not_duplicated():
    graph = _graph()
    norm = graph.modules[0]
    read = GraphNode(
        id="required_read", op=GraphOp("_params_param"), inputs=(GraphPath(True, ("required",)),),
        attrs={}, outputs=(GraphValue("required", TypeTensor("Tensor", ("D",))),),
        type_expr=TypeTensor("Tensor", ("D",)), source_module="test",
    )
    norm = replace(norm, nodes=(read, *norm.nodes))
    graph = replace(graph, modules=(norm, *graph.modules[1:]))
    assert not _fused(_optimize(graph, inline_safe=False, specialize_definitions="off"))


@pytest.mark.parametrize("dead_temp_elimination", [False, True])
def test_fusion_respects_disabled_dead_temp_elimination(dead_temp_elimination):
    graph = _optimize(_graph(), inline_safe=False, specialize_definitions="off",
                      dead_temp_elimination=dead_temp_elimination)
    assert bool(_fused(graph)) is dead_temp_elimination


@pytest.mark.parametrize("reuse_norm", [False, True])
@pytest.mark.parametrize("compiled", [False, True])
def test_generated_fp32_fusion_preserves_epsilon_and_other_norm_uses(reuse_norm, compiled):
    mx = pytest.importorskip("mlx.core")
    graph = _optimize(_graph(reuse_norm=reuse_norm))
    assert bool(_fused(graph)) is not reuse_norm
    namespace = {}
    exec(emit_model_code_from_graph_ir(graph), namespace)
    rng = np.random.default_rng(456)
    inputs = rng.normal(size=(3, 16)).astype(np.float32)
    scale = rng.normal(size=16).astype(np.float32)
    model = namespace["AxonMlxModel"].from_state_dict({"weight": mx.array(scale)})
    forward = mx.compile(model.forward) if compiled else model.forward
    for eps in [1e-2, 1e-7]:
        actual = np.asarray(forward(x=mx.array(inputs), eps=eps))
        normalized = inputs.astype(np.float64) / np.sqrt(
            np.mean(inputs.astype(np.float64) ** 2, axis=-1, keepdims=True) + eps)
        expected = normalized * scale
        if reuse_norm:
            expected += normalized
        np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=2e-6)


@pytest.mark.parametrize("scale_shape", [(), (1,), (1, 16), (3, 1), (16,)])
@pytest.mark.parametrize("dtypes", [("float32", "float32"), ("float16", "float16"),
                                  ("float16", "float32"), ("bfloat16", "bfloat16")])
@pytest.mark.parametrize("compiled", [False, True])
def test_other_broadcasts_and_dtypes_preserve_unfused_results(scale_shape, dtypes, compiled):
    mx = pytest.importorskip("mlx.core")
    graph = _graph(scale_dims=tuple(str(n) for n in scale_shape))
    graph = _optimize(graph)
    assert _fused(graph)
    namespace = {}
    exec(emit_model_code_from_graph_ir(graph), namespace)
    rng = np.random.default_rng(678)
    x = mx.array(rng.normal(size=(3, 16)), dtype=getattr(mx, dtypes[0]))
    scale = mx.array(rng.normal(size=scale_shape), dtype=getattr(mx, dtypes[1]))
    model = namespace["AxonMlxModel"].from_state_dict({"weight": scale})
    forward = mx.compile(model.forward) if compiled else model.forward
    actual = forward(x=x, eps=1e-3)
    expected = mx.fast.rms_norm(x, None, 1e-3) * scale
    mx.eval(actual, expected)
    assert actual.dtype == expected.dtype
    if dtypes == ("float32", "float32") and scale_shape == (16,):
        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), atol=1e-6, rtol=2e-6)
    else:
        np.testing.assert_array_equal(np.asarray(actual.astype(mx.float32)),
                                      np.asarray(expected.astype(mx.float32)))
