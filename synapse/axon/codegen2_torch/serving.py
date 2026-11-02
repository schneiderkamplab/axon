"""Serving-profile graph transform for codegen2-torch artifacts.

The opt-in ``__torch_paged_attention`` intrinsic (see
``synapse/axon/graph_ir/optimize.py``) rewrites each cache-threaded
attention site to a primitive carrying the fresh K/V.  This transform
completes the serving lowering: it removes the KV-cache threading the
paged pool replaces, reshapes the artifact contract to the fleet runner's
``forward(input_ids, positions) -> hidden`` plus ``compute_logits``, and
threads the layer-loop index into the paged sites so the single emitted
block module addresses the right pool slice per layer.

The result mirrors the nano-vllm artifact shape (https://github.com/
GeeeekExplorer/nano-vllm, MIT): no decode loop, no cache values in any
signature, per-site ``PagedAttention`` modules bound by the engine.

Builtin surfaces referenced structurally: ``Cache.update``-family calls
(the paged backing store) and the ``Positions.position_ids``-family calls
(the temporal-index computation the ``positions`` input replaces) — the
same supported-builtin recognition the backend SDPA rewrites use.  Every
shape decision is validated; the transform raises instead of silently
emitting a wrong artifact (no silent fallbacks).
"""

from __future__ import annotations

from dataclasses import replace as dc_replace
from typing import Any

from ..ast import TypeInt, TypePath, TypeTuple
from ..graph_ir import (
    GraphExpr,
    GraphLiteral,
    GraphModule,
    GraphNode,
    GraphOperand,
    GraphProgram,
    GraphValue,
    GraphValueRef,
    validate_graph_program,
)
from ..graph_ir.optimize import _sanitize_graph_constraints

__all__ = ["SERVING_HEAD_META_KEY", "lower_graph_for_serving"]

SERVING_HEAD_META_KEY = "head_module"

_PAGED_ATTENTION_OP = "__torch_paged_attention"
_POSITION_IDS_BUILTIN_PREFIX = "Positions.position_ids"
_LAYER_IDX_FORMAL = "layer_idx"


def _operand_value_ref_names(operand: GraphOperand) -> tuple[str, ...]:
    """All value names an operand references (through nested expressions)."""
    if isinstance(operand, GraphValueRef):
        return (operand.name,)
    if isinstance(operand, GraphExpr):
        names: list[str] = []
        for item in operand.inputs:
            names.extend(_operand_value_ref_names(item))
        for item in operand.attrs.values():
            names.extend(_operand_value_ref_names(item))
        return tuple(names)
    if isinstance(operand, tuple):
        names = []
        for item in operand:
            names.extend(_operand_value_ref_names(item))
        return tuple(names)
    return ()


def _operand_calls_module(operand: GraphOperand, prefix: str) -> bool:
    """Whether an operand tree contains a call to a module matching prefix."""
    if isinstance(operand, GraphExpr):
        op = operand.op.name
        if op == prefix or op.startswith(f"{prefix}__") or op.startswith(f"{prefix}_"):
            return True
        return any(_operand_calls_module(item, prefix) for item in operand.inputs) or any(
            _operand_calls_module(item, prefix) for item in operand.attrs.values()
        )
    if isinstance(operand, tuple):
        return any(_operand_calls_module(item, prefix) for item in operand)
    return False


def _rewrite_refs(operand: GraphOperand, mapping: dict[str, GraphValueRef]) -> GraphOperand:
    """Rewrite value refs by name throughout an operand tree."""
    if isinstance(operand, GraphValueRef):
        return mapping.get(operand.name, operand)
    if isinstance(operand, GraphExpr):
        return GraphExpr(
            op=operand.op,
            inputs=tuple(_rewrite_refs(item, mapping) for item in operand.inputs),
            attrs={key: _rewrite_refs(value, mapping) for key, value in operand.attrs.items()},
            type_expr=operand.type_expr,
            dims=operand.dims,
        )
    if isinstance(operand, tuple):
        return tuple(_rewrite_refs(item, mapping) for item in operand)
    return operand


def _reachable_nodes(module: GraphModule, outputs: tuple[GraphOperand, ...]) -> tuple[GraphNode, ...]:
    """Nodes reachable from ``outputs`` (through inputs and nested exprs)."""
    wanted: set[str] = set()
    stack: list[str] = []
    for operand in outputs:
        for name in _operand_value_ref_names(operand):
            if name not in wanted:
                wanted.add(name)
                stack.append(name)
    producers: dict[str, GraphNode] = {}
    for node in module.nodes:
        for output in node.outputs:
            producers[output.name] = node
    kept: set[int] = set()
    while stack:
        name = stack.pop()
        node = producers.get(name)
        if node is None or id(node) in kept:
            continue
        kept.add(id(node))
        for operand in (*node.inputs, *node.attrs.values()):
            for ref in _operand_value_ref_names(operand):
                if ref not in wanted:
                    wanted.add(ref)
                    stack.append(ref)
    return tuple(node for node in module.nodes if id(node) in kept)


def _used_formal_names(module: GraphModule) -> set[str]:
    used: set[str] = set()
    for node in module.nodes:
        for operand in (*node.inputs, *node.attrs.values()):
            used.update(_operand_value_ref_names(operand))
    for operand in module.outputs:
        used.update(_operand_value_ref_names(operand))
    return used


def _drop_unused_formals(
    module: GraphModule,
    *,
    extra_keep: tuple[str, ...] = (),
) -> tuple[GraphValue, ...]:
    """Drop formals no surviving node/output references; keep scope-path formals."""
    used = _used_formal_names(module)
    kept = []
    for formal in module.inputs:
        if isinstance(formal.type_expr, TypePath):
            kept.append(formal)
            continue
        if formal.name in used or formal.name in extra_keep:
            kept.append(formal)
    return tuple(kept)


def _strip_module(
    module: GraphModule,
    *,
    new_outputs: tuple[GraphOperand, ...],
    extra_keep_formals: tuple[str, ...] = (),
) -> GraphModule:
    """Prune to reachability from ``new_outputs`` and drop unused formals."""
    pruned = _reachable_nodes(module, new_outputs)
    staged = dc_replace(module, nodes=pruned, outputs=new_outputs)
    kept_inputs = _drop_unused_formals(staged, extra_keep=extra_keep_formals)
    return dc_replace(staged, inputs=kept_inputs)


def _repeat_attr(node: GraphNode, key: str) -> GraphOperand:
    try:
        return node.attrs[key]
    except KeyError as exc:  # pragma: no cover - malformed repeat node
        raise ValueError(f"core.repeat node {node.id!r} is missing the {key!r} attr") from exc


def _repeat_attr_text(node: GraphNode, key: str) -> str:
    value = _repeat_attr(node, key)
    if not isinstance(value, GraphLiteral) or not isinstance(value.value, str):
        raise ValueError(f"core.repeat node {node.id!r} has a non-literal {key!r} attr")
    return value.value


def _repeat_attr_int(node: GraphNode, key: str) -> int:
    value = _repeat_attr(node, key)
    if not isinstance(value, GraphLiteral) or not isinstance(value.value, int):
        raise ValueError(f"core.repeat node {node.id!r} has a non-literal {key!r} attr")
    return int(value.value)


def _paged_block_modules(program: GraphProgram) -> tuple[str, ...]:
    """Modules containing paged attention sites (the lowered blocks)."""
    return tuple(
        module.name
        for module in program.modules
        if any(node.op.name == _PAGED_ATTENTION_OP for node in module.nodes)
    )


def _is_cache_op_node(node: GraphNode) -> bool:
    return node.op.name == "Cache.update" or node.op.name.startswith("Cache.update__") or (
        node.op.name == "Cache.append" or node.op.name.startswith("Cache.append__")
    )


def _find_layer_loop(main: GraphModule) -> GraphNode:
    """The main module's single layer-loop ``core.repeat`` node."""
    repeats = [node for node in main.nodes if node.op.name == "core.repeat"]
    if len(repeats) != 1:
        raise ValueError(
            "serving profile expects exactly one layer loop (core.repeat) in the main "
            f"module; found {len(repeats)}"
        )
    return repeats[0]


def _transform_paged_block(module: GraphModule) -> tuple[GraphModule, tuple[int, ...]]:
    """Strip cache threading from a paged block and append the layer index.

    Returns the transformed module plus the positions (indexes into the old
    output list) of the surviving outputs, so call sites can rebuild their
    output bindings positionally.
    """
    layer_idx_type = TypeInt()
    new_nodes: list[GraphNode] = []
    removed_outputs: set[str] = set()
    site_count = 0
    for node in module.nodes:
        if node.op.name == _PAGED_ATTENTION_OP:
            site_count += 1
            new_nodes.append(
                dc_replace(
                    node,
                    inputs=(
                        *node.inputs,
                        GraphValueRef(name=_LAYER_IDX_FORMAL, type_expr=layer_idx_type),
                    ),
                )
            )
            continue
        if _is_cache_op_node(node):
            removed_outputs.update(output.name for output in node.outputs)
            continue
        new_nodes.append(node)
    if site_count == 0:
        raise ValueError(f"module {module.name!r} scheduled for the paged transform has no paged sites")
    staged = dc_replace(module, nodes=tuple(new_nodes))
    kept_output_positions = tuple(
        index
        for index, operand in enumerate(staged.outputs)
        if not any(ref in removed_outputs for ref in _operand_value_ref_names(operand))
    )
    kept_outputs = tuple(staged.outputs[index] for index in kept_output_positions)
    if len(kept_outputs) != 1:
        raise ValueError(
            f"serving profile expects the paged block {module.name!r} to keep exactly one "
            f"non-cache output; kept {len(kept_outputs)}"
        )
    pruned = _reachable_nodes(staged, kept_outputs)
    staged = dc_replace(staged, nodes=pruned, outputs=kept_outputs)
    kept_inputs = _drop_unused_formals(staged, extra_keep=(_LAYER_IDX_FORMAL,))
    kept_inputs = (*kept_inputs, GraphValue(name=_LAYER_IDX_FORMAL, type_expr=layer_idx_type))
    return dc_replace(staged, inputs=kept_inputs), kept_output_positions


def _rebuild_block_call(
    node: GraphNode,
    *,
    old_formal_names: tuple[str, ...],
    new_formal_names: tuple[str, ...],
    layer_idx_operand: GraphOperand,
    kept_output_positions: tuple[int, ...],
) -> GraphNode:
    """Rebuild a block call for the stripped block signature."""
    args = list(node.inputs)
    formal_positions = {name: index for index, name in enumerate(old_formal_names)}
    new_args: list[GraphOperand] = []
    for name in new_formal_names:
        if name in formal_positions:
            position = formal_positions[name]
            if position >= len(args):
                raise ValueError(
                    f"block call {node.id!r} passes {len(args)} args; formal {name!r} "
                    f"needs position {position}"
                )
            new_args.append(args[position])
        else:
            new_args.append(layer_idx_operand)
    new_outputs = tuple(node.outputs[index] for index in kept_output_positions)
    return dc_replace(
        node,
        inputs=tuple(new_args),
        outputs=new_outputs,
        type_expr=(
            new_outputs[0].type_expr
            if len(new_outputs) == 1
            else TypeTuple(items=tuple(output.type_expr for output in new_outputs))
        ),
        dims=None,
    )


def _transform_loop_step(
    module: GraphModule,
    *,
    block_names: set[str],
    block_old_formal_names: dict[str, tuple[str, ...]],
    block_new_formal_names: dict[str, tuple[str, ...]],
    block_kept_output_positions: dict[str, tuple[int, ...]],
    var_name: str,
) -> GraphModule:
    """Strip cache threading from the loop step; thread the loop index."""
    layer_idx_operand = GraphValueRef(name=var_name, type_expr=TypeInt())
    new_nodes: list[GraphNode] = []
    for node in module.nodes:
        if node.op.name in block_names:
            new_nodes.append(
                _rebuild_block_call(
                    node,
                    old_formal_names=block_old_formal_names[node.op.name],
                    new_formal_names=block_new_formal_names[node.op.name],
                    layer_idx_operand=layer_idx_operand,
                    kept_output_positions=block_kept_output_positions[node.op.name],
                )
            )
            continue
        new_nodes.append(node)
    staged = dc_replace(module, nodes=tuple(new_nodes))
    cache_outputs = _cache_output_names(staged)
    kept_outputs = tuple(
        operand
        for operand in staged.outputs
        if not any(ref in cache_outputs for ref in _operand_value_ref_names(operand))
    )
    if len(kept_outputs) != 1:
        raise ValueError(
            f"serving profile expects the loop step {module.name!r} to keep exactly one "
            f"non-cache output; kept {len(kept_outputs)}"
        )
    return _strip_module(staged, new_outputs=kept_outputs)


def _cache_output_names(module: GraphModule) -> set[str]:
    names: set[str] = set()
    for node in module.nodes:
        if _is_cache_op_node(node):
            names.update(output.name for output in node.outputs)
    return names


def _rebuild_layer_repeat(
    repeat: GraphNode,
    *,
    old_step_formals: tuple[GraphValue, ...],
    new_step_formal_names: tuple[str, ...],
) -> GraphNode:
    """Rebuild the layer loop for the stripped step signature.

    ``core.repeat`` layout: ``[start, count, step, *carried, *extra]`` with
    attrs ``var`` (index formal), ``arg_count`` (step formals incl. index),
    ``carry_count``/``carry_k`` (carried value names), ``arg_k`` bindings
    (``iter`` | ``carry:j`` | ``input:N``).
    """
    old_carry_count = _repeat_attr_int(repeat, "carry_count")
    old_args = list(repeat.inputs)
    trailing = old_args[3:]
    if len(trailing) < old_carry_count:
        raise ValueError(f"core.repeat node {repeat.id!r} carries fewer values than declared")
    carried = trailing[:old_carry_count]
    extras = trailing[old_carry_count:]

    old_bindings: dict[str, str] = {}
    for position, formal in enumerate(old_step_formals):
        old_bindings[formal.name] = _repeat_attr_text(repeat, f"arg_{position}")

    kept_formal_names = set(new_step_formal_names)

    # Carried values keep their identity through carry bindings: carry j binds
    # a step formal by name.
    carry_formal: dict[int, str] = {}
    for index in range(old_carry_count):
        carry_formal[index] = _repeat_attr_text(repeat, f"carry_{index}")
    kept_carries = [
        index for index in range(old_carry_count) if carry_formal[index] in kept_formal_names
    ]
    kept_carry_names = {carry_formal[index] for index in kept_carries}

    # Extras survive while at least one kept formal binds them.  Bindings
    # address repeat inputs by absolute index (input:N -> repeat.inputs[N]);
    # extras start after start/count/step and the carried values.
    extra_consumers: dict[int, set[str]] = {}
    for name, binding in old_bindings.items():
        if binding.startswith("input:"):
            extra_consumers.setdefault(int(binding.removeprefix("input:")), set()).add(name)
    kept_extra_indexes = sorted(
        index
        for index, consumers in extra_consumers.items()
        if consumers & kept_formal_names
    )

    new_carry_count = len(kept_carries)
    new_trailing: list[GraphOperand] = [carried[index] for index in kept_carries]
    new_trailing.extend(extras[index - 3 - old_carry_count] for index in kept_extra_indexes)

    new_bindings: dict[str, str] = {}
    for position, name in enumerate(new_step_formal_names):
        old_binding = old_bindings[name]
        if old_binding == "iter":
            new_bindings[str(position)] = "iter"
        elif old_binding.startswith("carry:"):
            carry_index = int(old_binding.removeprefix("carry:"))
            if carry_formal[carry_index] not in kept_carry_names:
                raise ValueError(
                    f"core.repeat node {repeat.id!r}: carried value for formal {name!r} "
                    "was dropped but the formal survives"
                )
            new_bindings[str(position)] = f"carry:{kept_carries.index(carry_index)}"
        elif old_binding.startswith("input:"):
            old_index = int(old_binding.removeprefix("input:"))
            extra_position = kept_extra_indexes.index(old_index)
            new_bindings[str(position)] = f"input:{3 + new_carry_count + extra_position}"
        else:  # pragma: no cover - unknown binding grammar
            raise ValueError(f"core.repeat node {repeat.id!r}: unknown arg binding {old_binding!r}")

    new_attrs = {
        key: value
        for key, value in repeat.attrs.items()
        if not (key.startswith("arg_") or key.startswith("carry_"))
    }
    new_attrs["arg_count"] = GraphLiteral(value=len(new_step_formal_names), type_expr=TypeInt())
    new_attrs["carry_count"] = GraphLiteral(value=new_carry_count, type_expr=TypeInt())
    for index in range(new_carry_count):
        new_attrs[f"carry_{index}"] = GraphLiteral(
            value=carry_formal[kept_carries[index]], type_expr=_repeat_attr(repeat, f"carry_{index}").type_expr
        )
    for position, binding in new_bindings.items():
        new_attrs[f"arg_{position}"] = GraphLiteral(
            value=binding, type_expr=_repeat_attr(repeat, "arg_0").type_expr
        )

    # Outputs correspond to carried values, in carry order; the node's
    # declared type must shrink with them (a multi-carry node declares the
    # tuple of carried types, a single carry declares the bare type).
    kept_outputs = tuple(repeat.outputs[index] for index in kept_carries)
    if len(kept_outputs) == 1:
        repeat_type = kept_outputs[0].type_expr
    else:
        repeat_type = TypeTuple(items=tuple(output.type_expr for output in kept_outputs))
    return dc_replace(
        repeat,
        inputs=tuple([*old_args[:3], *new_trailing]),
        attrs=new_attrs,
        outputs=kept_outputs,
        type_expr=repeat_type,
        dims=None,
    )


def _find_positions_select(main: GraphModule) -> GraphNode:
    """The select whose branches compute the temporal position ids."""
    matches = []
    for node in main.nodes:
        if node.op.name != "core.select" or len(node.inputs) != 3:
            continue
        branches = node.inputs[1:3]
        if any(_operand_calls_module(branch, _POSITION_IDS_BUILTIN_PREFIX) for branch in branches):
            matches.append(node)
    if len(matches) != 1:
        raise ValueError(
            "serving profile expects exactly one Positions.position_ids computation in the "
            f"main module; found {len(matches)}"
        )
    return matches[0]


def _find_head_node(main: GraphModule) -> tuple[GraphNode, str, Any]:
    """Extract (head select node, hidden value name, hidden type) from main."""
    first_output = main.outputs[0]
    if not isinstance(first_output, GraphValueRef):
        raise ValueError("serving profile expects the main module's first output to be a value ref")
    producers = {
        output.name: node for node in main.nodes for output in node.outputs
    }
    head = producers.get(first_output.name)
    if head is None or head.op.name != "core.select" or len(head.inputs) != 3:
        raise ValueError(
            "serving profile expects the main module's logits to come from a tied/untied "
            "select; adjust the head split for this artifact shape"
        )
    branch_refs: list[str] = []
    for branch in head.inputs[1:3]:
        refs = _operand_value_ref_names(branch)
        if not refs:
            raise ValueError(
                f"head select {head.id!r} has a branch without tensor operands; the head "
                "split cannot infer the hidden value"
            )
        branch_refs.extend(refs)
    common = set(branch_refs)
    if len(common) != 1:
        raise ValueError(
            f"head select {head.id!r} branches do not share exactly one hidden operand "
            f"({sorted(common)})"
        )
    hidden_name = common.pop()
    hidden_type = _value_ref_type_in_module(main, hidden_name)
    return head, hidden_name, hidden_type


def _value_ref_type_in_module(module: GraphModule, name: str) -> Any:
    for formal in module.inputs:
        if formal.name == name:
            return formal.type_expr
    for node in module.nodes:
        for output in node.outputs:
            if output.name == name:
                return output.type_expr
    raise ValueError(f"value {name!r} not found in module {module.name!r}")


def lower_graph_for_serving(program: GraphProgram) -> tuple[GraphProgram, dict[str, Any]]:
    """Lower an optimized graph to the serving-profile artifact shape.

    Returns the transformed program plus metadata; ``head_module`` names the
    module the emitter turns into ``compute_logits``.
    """
    modules_by_name = {module.name: module for module in program.modules}
    main = modules_by_name[program.main_module]

    block_names = set(_paged_block_modules(program))
    if not block_names:
        raise ValueError(
            "serving profile requires the __torch_paged_attention intrinsic: no paged "
            "attention sites were found; the pairing is enforced by the loader"
        )
    repeat = _find_layer_loop(main)
    var_name = _repeat_attr_text(repeat, "var")
    step_name = _repeat_attr_text(repeat, "callee")
    if step_name not in modules_by_name:
        raise ValueError(f"layer loop callee {step_name!r} is not a module of this program")

    # 1. Paged blocks: strip cache threading, append the layer index.
    block_old_formal_names: dict[str, tuple[str, ...]] = {}
    block_new_formal_names: dict[str, tuple[str, ...]] = {}
    block_kept_output_positions: dict[str, tuple[int, ...]] = {}
    new_blocks: dict[str, GraphModule] = {}
    for name in block_names:
        block = modules_by_name[name]
        block_old_formal_names[name] = tuple(formal.name for formal in block.inputs)
        new_blocks[name], block_kept_output_positions[name] = _transform_paged_block(block)
        block_new_formal_names[name] = tuple(formal.name for formal in new_blocks[name].inputs)

    # 2. Loop step: drop the cache plumbing, thread the loop index.
    step = modules_by_name[step_name]
    new_step = _transform_loop_step(
        step,
        block_names=block_names,
        block_old_formal_names=block_old_formal_names,
        block_new_formal_names=block_new_formal_names,
        block_kept_output_positions=block_kept_output_positions,
        var_name=var_name,
    )
    new_step_formal_names = tuple(formal.name for formal in new_step.inputs)
    new_repeat = _rebuild_layer_repeat(
        repeat,
        old_step_formals=step.inputs,
        new_step_formal_names=new_step_formal_names,
    )

    # 3. Main module: positions input, loop rebuild, head split.
    positions_select = _find_positions_select(main)
    pos_output = positions_select.outputs[0]
    positions_ref = GraphValueRef(name="positions", type_expr=pos_output.type_expr)
    ref_map = {pos_output.name: positions_ref}
    head_node, hidden_name, hidden_type = _find_head_node(main)

    new_main_nodes: list[GraphNode] = []
    for node in main.nodes:
        if node.id == positions_select.id or node.id == head_node.id:
            continue
        if node.op.name == "core.repeat":
            new_main_nodes.append(new_repeat)
            continue
        new_main_nodes.append(
            dc_replace(
                node,
                inputs=tuple(_rewrite_refs(item, ref_map) for item in node.inputs),
                attrs={
                    key: _rewrite_refs(value, ref_map) for key, value in node.attrs.items()
                },
            )
        )
    staged_main = dc_replace(
        main,
        nodes=tuple(new_main_nodes),
        outputs=(GraphValueRef(name=hidden_name, type_expr=hidden_type),),
    )
    new_main = _strip_module(staged_main, new_outputs=staged_main.outputs)
    new_main = dc_replace(
        new_main,
        inputs=(
            *new_main.inputs[:1],
            GraphValue(name="positions", type_expr=pos_output.type_expr),
            *new_main.inputs[1:],
        ),
    )

    # 4. Head module: the tied/untied logits select as its own module.
    head_name = f"{main.name}__serving_head"
    head_output = head_node.outputs[0]
    head_module = GraphModule(
        name=head_name,
        inputs=(GraphValue(name=hidden_name, type_expr=hidden_type),),
        outputs=(GraphValueRef(name=head_output.name, type_expr=head_output.type_expr),),
        output_names=(),
        nodes=(head_node,),
    )

    # 5. Assemble and validate.
    order = []
    for module in program.modules:
        if module.name == main.name:
            order.append(new_main)
        elif module.name == step_name:
            order.append(new_step)
        elif module.name in new_blocks:
            order.append(new_blocks[module.name])
        else:
            order.append(module)
    order.append(head_module)
    new_program = dc_replace(program, modules=tuple(order))
    # Constraints may reference removed plumbing values (past_kv, mask
    # capacity dims); reuse the optimizer's sanitizer rather than a copy.
    new_program = _sanitize_graph_constraints(new_program)
    validate_graph_program(new_program)
    meta = {
        SERVING_HEAD_META_KEY: head_name,
        "paged_site_count": sum(
            1
            for module in new_program.modules
            for node in module.nodes
            if node.op.name == _PAGED_ATTENTION_OP
        ),
    }
    return new_program, meta
