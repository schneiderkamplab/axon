"""Serving-profile lowering tests (2b contract).

Covers the pipeline pieces the fleet serving path stands on:

- the ``__torch_paged_attention`` intrinsic registers and conflicts loudly
  with ``__torch_sdpa`` (no silent precedence),
- ``lower_graph_for_serving`` strips the KV-cache threading, threads the
  layer index into the paged sites, and splits the head module out,
- the emitted serving artifact speaks the fleet-runner contract
  (``forward(input_ids, positions) -> hidden`` + ``compute_logits``, no
  decode loop) with ``PagedAttention`` modules registered,
- the loader pairs ``emit_profile='serving'`` with the intrinsic opt-in in
  both directions (never injected implicitly, never enabled silently).
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parent.parent.parent
QWEN3_AXON = REPO / "synapse" / "models" / "qwen3" / "generic-qwen3.axon"
PAGED_SPEC = "codegen2-torch:__torch_paged_attention"


@pytest.fixture(scope="module")
def paged_program():
    """The optimized qwen3 graph with the paged intrinsic applied."""
    from server.loading import _compile_graph_program

    graph, config = _compile_graph_program(
        QWEN3_AXON,
        checkpoint_dir=None,
        optimize_ast=True,
        optimize_graph=True,
        backend_intrinsics=PAGED_SPEC,
    )
    return graph, config


def test_paged_intrinsic_is_a_registered_torch_intrinsic() -> None:
    from synapse.axon.graph_ir.optimize import _parse_backend_intrinsics

    target, enabled = _parse_backend_intrinsics(PAGED_SPEC)
    assert target == "codegen2-torch"
    assert "__torch_paged_attention" in enabled


def test_paged_and_sdpa_intrinsics_conflict_loudly() -> None:
    from synapse.axon.graph_ir.optimize import (
        _check_conflicting_torch_intrinsics,
    )

    _check_conflicting_torch_intrinsics(frozenset({"__torch_sdpa"}))
    _check_conflicting_torch_intrinsics(frozenset({"__torch_paged_attention"}))
    with pytest.raises(ValueError, match="conflicting backend intrinsics"):
        _check_conflicting_torch_intrinsics(
            frozenset({"__torch_sdpa", "__torch_paged_attention"})
        )
    # And the full optimizer dispatch rejects the combination too, before any
    # phase runs (an empty pass-through program is enough).
    from synapse.axon import GraphOptimizeConfig, optimize_graph_program
    from synapse.axon.ast import TypeTensor
    from synapse.axon.graph_ir import (
        GraphModule,
        GraphProgram,
        GraphValue,
        GraphValueRef,
    )

    tensor_t = TypeTensor("Tensor", ("B",))
    program = GraphProgram(
        modules=(
            GraphModule(
                name="main",
                inputs=(GraphValue("x", tensor_t, tensor_t.dims),),
                outputs=(GraphValueRef("x", tensor_t, tensor_t.dims),),
                output_names=("out",),
                nodes=(),
                return_type_expr=tensor_t,
            ),
        ),
        main_module="main",
        pragmas={},
    )
    with pytest.raises(ValueError, match="conflicting backend intrinsics"):
        optimize_graph_program(
            program,
            config=GraphOptimizeConfig(
                backend_intrinsics="codegen2-torch:__torch_sdpa,__torch_paged_attention",
            ),
        )


def test_serving_transform_shape(paged_program) -> None:
    from synapse.axon.codegen2_torch.serving import lower_graph_for_serving

    graph, _ = paged_program
    new_graph, meta = lower_graph_for_serving(graph)

    assert meta["head_module"] == "qwen3__serving_head"
    assert meta["paged_site_count"] == 1

    by_name = {module.name: module for module in new_graph.modules}
    main = by_name["qwen3"]
    step = by_name["qwen3__loop_layers_step_1"]
    block = by_name["qwen3_block"]
    head = by_name["qwen3__serving_head"]

    # Artifact contract: input_ids + positions in, hidden out.
    assert [formal.name for formal in main.inputs] == ["input_ids", "positions"]
    assert [output.name for output in main.outputs] == ["x__g2"]

    # The static KV cache and the attention builtins are gone everywhere.
    for module in new_graph.modules:
        for node in module.nodes:
            assert not node.op.name.startswith("Cache."), (module.name, node.op.name)
            assert node.op.name != "Attention.attention", (module.name, node.op.name)

    # Block: layer index threaded into formals and the paged site.
    assert [formal.name for formal in block.inputs] == [
        "__scope",
        "x",
        "rope_sin",
        "rope_cos",
        "layer_idx",
    ]
    assert len(block.outputs) == 1
    paged_nodes = [node for node in block.nodes if node.op.name == "__torch_paged_attention"]
    assert len(paged_nodes) == 1
    paged_node = paged_nodes[0]
    assert len(paged_node.inputs) == 4
    assert paged_node.inputs[3].name == "layer_idx"

    # Step: one output, block call passes the loop index last.
    assert len(step.outputs) == 1
    block_calls = [node for node in step.nodes if node.op.name == "qwen3_block"]
    assert len(block_calls) == 1
    assert len(block_calls[0].inputs) == 5
    assert block_calls[0].inputs[-1].name == "i"

    # Loop rebuilt for the stripped step signature.
    repeat = next(node for node in main.nodes if node.op.name == "core.repeat")
    attrs = {key: value.value for key, value in repeat.attrs.items()}
    assert attrs["arg_count"] == len(step.inputs)
    assert attrs["carry_count"] == 1
    assert attrs["carry_0"] == "x"
    assert len(repeat.outputs) == 1

    # Head module split out for compute_logits.
    assert [formal.name for formal in head.inputs] == ["x__g2"]
    assert [output.name for output in head.outputs] == ["logits"]


def test_serving_artifact_contract(paged_program) -> None:
    from synapse.axon.codegen2_torch import emit_model_code_from_graph_ir

    graph, config = paged_program
    source = emit_model_code_from_graph_ir(graph, model_config=config, emit_profile="serving")
    namespace: dict[str, object] = {}
    exec(compile(source, "<serving-artifact>", "exec"), namespace)  # noqa: S102
    model_cls = namespace["GeneratedAxonModel"]

    # A shrunk vocab plus a fake tied weight keeps the compute_logits call
    # testable without the 151936x1024 embedding matrix; symbols resolve from
    # the runtime config, so only vocab_size changes.
    small_config = {**config, "vocab_size": 64}
    model = model_cls(
        {"model.embed_tokens.weight": torch.zeros(64, 1024)},
        small_config,
    )
    paged = {
        name: module
        for name, module in model.named_modules()
        if module.__class__.__name__ == "PagedAttention"
    }
    assert paged == {
        "_paged_attn_0": model._paged_attn_0,
    }
    assert (model._paged_attn_0.num_heads, model._paged_attn_0.num_kv_heads, model._paged_attn_0.head_dim) == (
        16,
        8,
        128,
    )
    assert hasattr(model, "compute_logits")
    assert not hasattr(model, "generate")

    # compute_logits keeps the artifact typed: [B, S, D] rows in, [B, S, V]
    # out (the runner flattens to packed [N, V] rows).
    assert model.compute_logits(torch.zeros(2, 5, 1024)).shape == (2, 5, 64)
    assert model.compute_logits(torch.zeros(2, 1, 1024)).shape == (2, 1, 64)

    # The forward contract requires positions.
    ids = torch.zeros(1, 3, dtype=torch.long)
    with pytest.raises(ValueError, match="positions"):
        model.forward(input_ids=ids)


def test_emitter_rejects_paged_nodes_outside_serving(paged_program) -> None:
    from synapse.axon.codegen2_torch import emit_model_code_from_graph_ir

    graph, config = paged_program
    with pytest.raises(ValueError, match="emit_profile='serving'"):
        emit_model_code_from_graph_ir(graph, model_config=config, emit_profile="inference")


def test_loader_pairs_serving_with_paged_intrinsic() -> None:
    from server.loading import load_torch_model

    missing = "paired opt-in"
    with pytest.raises(ValueError, match=missing):
        load_torch_model(QWEN3_AXON, emit_profile="serving", backend_intrinsics=None)
    with pytest.raises(ValueError, match=missing):
        load_torch_model(
            QWEN3_AXON,
            emit_profile="serving",
            backend_intrinsics="codegen2-torch:__torch_sdpa",
        )
    with pytest.raises(ValueError, match=missing):
        load_torch_model(
            QWEN3_AXON,
            emit_profile="inference",
            backend_intrinsics=PAGED_SPEC,
        )
