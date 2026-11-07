"""Compile Axon programs to generated cores for tests."""

from __future__ import annotations

from pathlib import Path

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


def generate_core(source, *, backend="mlx", backend_intrinsics=None, optimize=True, class_name="AxonCore"):
    """Generated code for an Axon file. ``optimize=False`` skips the graph optimizer, and
    ``backend_intrinsics`` is the explicit opt-in selector for ``GraphOptimizeConfig``."""
    if backend == "torch":
        from synapse.axon.codegen2_torch import emit_model_code_from_graph_ir
    else:
        from synapse.axon.codegen2_mlx import emit_model_code_from_graph_ir

    program = resolve_axon_program_from_path(Path(source)).ast
    for transform in (
        normalize_closed_axon_file,
        elaborate_closed_axon_file,
        flatten_closed_axon_file,
        typecheck2_flat_axon_file,
    ):
        program = transform(program)
    graph = lower_axon_program_to_graph_ir(program)
    if optimize:
        graph = optimize_graph_program(graph, config=GraphOptimizeConfig(backend_intrinsics=backend_intrinsics))
    return emit_model_code_from_graph_ir(graph, class_name=class_name)


def load_core(code, class_name):
    """The generated class of ``code``."""
    namespace = {}
    exec(code, namespace)
    return namespace[class_name]
