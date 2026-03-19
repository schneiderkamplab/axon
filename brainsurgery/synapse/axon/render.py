from __future__ import annotations

import ast
import re
from typing import Any


def _format_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, str):
        return value
    return repr(value)


def _try_eval_numeric(text: str) -> int | float | bool | None:
    try:
        parsed = ast.parse(text, mode="eval")
    except SyntaxError:
        return None
    allowed = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.FloorDiv,
        ast.Mod,
        ast.Pow,
        ast.USub,
        ast.UAdd,
        ast.Constant,
        ast.Compare,
        ast.Eq,
        ast.NotEq,
        ast.Lt,
        ast.LtE,
        ast.Gt,
        ast.GtE,
        ast.BoolOp,
        ast.And,
        ast.Or,
        ast.Not,
        ast.UnaryOp,
    )
    for node in ast.walk(parsed):
        if not isinstance(node, allowed):
            return None
        if isinstance(node, ast.Name):
            return None
    try:
        value = eval(compile(parsed, "<axon-render>", "eval"), {"__builtins__": {}}, {})
    except Exception:
        return None
    if isinstance(value, (int, float, bool)):
        return value
    return None


def _resolve_value(value: Any, symbols: dict[str, Any]) -> Any:
    if isinstance(value, dict):
        return {k: _resolve_value(v, symbols) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_value(v, symbols) for v in value]
    if not isinstance(value, str):
        return value

    token = value.strip()
    if token in symbols and isinstance(symbols[token], (int, float, bool)):
        return symbols[token]

    substituted = value
    for name, sym_val in sorted(symbols.items(), key=lambda kv: len(kv[0]), reverse=True):
        if not isinstance(sym_val, (int, float, bool)):
            continue
        substituted = re.sub(rf"\b{re.escape(name)}\b", repr(sym_val), substituted)
    evaluated = _try_eval_numeric(substituted)
    if evaluated is not None:
        return evaluated
    return substituted


def _axon_expr_from_node(node_spec: dict[str, Any], *, node_path: str | None = None) -> str:
    op = str(node_spec.get("op"))
    in_value = node_spec.get("in")
    in_args: list[str]
    if isinstance(in_value, list):
        in_args = [str(item) for item in in_value]
    elif isinstance(in_value, str):
        in_args = [in_value]
    else:
        in_args = []

    kwargs: list[str] = []
    for key, value in node_spec.items():
        if key in {"op", "in", "out", "params"}:
            continue
        kwargs.append(f"{key}={_format_scalar(value)}")

    params = node_spec.get("params")
    if isinstance(params, dict) and isinstance(params.get("weight"), str):
        weight = params["weight"]
        if weight.endswith(".weight"):
            path = weight[: -len(".weight")]
            callee = f"{op}@{path}"
        else:
            callee = op
    elif (
        node_path
        and op in {"linear", "embedding", "layernorm", "rmsnorm"}
        and not isinstance(node_spec.get("weight"), str)
        and not isinstance(node_spec.get("tie_weight"), str)
        and not isinstance(node_spec.get("share"), str)
    ):
        callee = f"{op}@{node_path}"
    elif op == "activation" and isinstance(node_spec.get("kind"), str):
        callee = f"act::{node_spec['kind']}"
        kwargs = [item for item in kwargs if not item.startswith("kind=")]
    elif op == "kv_cache_update":
        callee = "cache::update"
    elif op == "kv_seq_len":
        callee = "cache::seq_len"
    elif op == "coalesce":
        callee = "cache::coalesce"
    elif op == "split_last":
        callee = "split_last"
    else:
        callee = op

    if op == "add" and len(in_args) == 2 and not kwargs:
        return f"{in_args[0]} + {in_args[1]}"
    if op == "mul" and len(in_args) == 2 and not kwargs:
        return f"{in_args[0]} * {in_args[1]}"

    all_args = [*in_args, *kwargs]
    return f"{callee}({', '.join(all_args)})"


def _can_render_as_bind(node_spec: dict[str, Any]) -> bool:
    if "use" in node_spec:
        return False
    if "graph" in node_spec and "op" not in node_spec:
        return False
    out_ref = node_spec.get("out")
    if not isinstance(out_ref, (str, list)):
        return False
    return isinstance(node_spec.get("op"), str)


def _render_module(
    *,
    module_name: str,
    inputs: dict[str, Any],
    outputs: dict[str, Any],
    graph: list[Any],
    symbols: dict[str, Any],
    signatures: dict[str, tuple[list[str], list[str]]],
) -> list[str]:
    params: list[str] = []
    for name, input_spec in inputs.items():
        optional = isinstance(input_spec, dict) and bool(input_spec.get("optional", False))
        params.append(f"{name}?" if optional else str(name))

    return_names = list(outputs.keys()) if outputs else ["out"]
    arg_types = ["?Tensor" if p.endswith("?") else "Tensor" for p in params]
    if len(return_names) == 1:
        ret_type = "Tensor"
    else:
        ret_type = "(" + ", ".join("Tensor" for _ in return_names) + ")"
    sig = " -> ".join([*arg_types, ret_type]) if arg_types else ret_type
    def_params = [p[:-1] if p.endswith("?") else p for p in params]
    def_head = f"{module_name} {' '.join(def_params)}".rstrip()
    lines = [f"{module_name} :: {sig}", f"{def_head} = do"]

    def render_graph(items: list[Any], *, scope: str, indent: str) -> None:
        for item in items:
            if not isinstance(item, dict) or len(item) != 1:
                raise ValueError(f"invalid graph item: {item!r}")
            node_name, node_spec = next(iter(item.items()))
            if not isinstance(node_spec, dict):
                raise ValueError(f"invalid node spec: {node_spec!r}")
            node_spec = _resolve_value(node_spec, symbols)

            node_path = f"{scope}.{node_name}" if scope else str(node_name)
            if node_spec.get("op") == "repeat":
                var = str(node_spec.get("var"))
                range_expr = _format_scalar(node_spec.get("range"))
                start_expr = _format_scalar(node_spec.get("start", 0))
                end_expr = f"({start_expr}) + ({range_expr})"
                body = node_spec.get("body")
                if isinstance(body, list):
                    repeat_name = node_path if scope else str(node_name)
                    lines.append(
                        f"{indent}for@{repeat_name} {var} <- [{start_expr}..{end_expr}) do"
                    )
                    render_graph(body, scope=node_path, indent=indent + "  ")
                    continue

            if isinstance(node_spec.get("use"), str):
                callee = str(node_spec["use"])
                in_map = node_spec.get("in", {})
                out_map = node_spec.get("out", {})
                if not isinstance(in_map, dict) or not isinstance(out_map, dict):
                    raise ValueError(f"invalid use-node maps: {node_spec!r}")
                in_order = list(in_map.keys())
                out_order = list(out_map.keys())
                if callee in signatures:
                    in_order = [name for name in signatures[callee][0] if name in in_map]
                    out_order = [name for name in signatures[callee][1] if name in out_map]
                lhs = ", ".join(str(out_map[name]) for name in out_order)
                args = ", ".join(str(in_map[name]) for name in in_order)
                lines.append(f"{indent}{lhs} <- {callee}({args})")
                continue

            if "graph" in node_spec and "op" not in node_spec:
                nested = node_spec.get("graph")
                if not isinstance(nested, list):
                    raise ValueError(f"invalid nested graph node: {node_spec!r}")
                render_graph(nested, scope=node_path, indent=indent)
                continue

            if _can_render_as_bind(node_spec):
                out_ref = node_spec.get("out")
                lhs = (
                    ", ".join(str(name) for name in out_ref)
                    if isinstance(out_ref, list)
                    else str(out_ref)
                )
                rhs = _axon_expr_from_node(node_spec, node_path=node_path)
                lines.append(f"{indent}{lhs} <- {rhs}")
                continue

            raise ValueError(f"node {node_path!r} cannot be rendered in strict Axon syntax")

    render_graph(graph, scope="", indent="  ")

    if outputs:
        ordered = [str(value) for value in outputs.values() if isinstance(value, str)]
        if len(ordered) == len(outputs):
            lines.append(f"  return {', '.join(ordered)}")

    return lines


def synapse_spec_to_axon_module_text(spec: dict[str, Any], *, module_name: str = "main") -> str:
    model = spec.get("model", {})
    if not isinstance(model, dict):
        raise ValueError("spec.model must be a mapping")

    inputs = model.get("inputs", {})
    if not isinstance(inputs, dict):
        raise ValueError("model.inputs must be a mapping")
    outputs = model.get("outputs", {})
    if not isinstance(outputs, dict):
        raise ValueError("model.outputs must be a mapping")
    graph = model.get("graph", [])
    if not isinstance(graph, list):
        raise ValueError("model.graph must be a list")

    lines: list[str] = []
    symbols = model.get("symbols", {})
    if not isinstance(symbols, dict):
        symbols = {}
    signatures: dict[str, tuple[list[str], list[str]]] = {}
    signatures[module_name] = (list(inputs.keys()), list(outputs.keys()))
    blocks = model.get("blocks")
    if isinstance(blocks, dict):
        for block_name, block_spec in blocks.items():
            if isinstance(block_spec, dict):
                b_inputs = block_spec.get("inputs", {})
                b_outputs = block_spec.get("outputs", {})
                if isinstance(b_inputs, dict) and isinstance(b_outputs, dict):
                    signatures[str(block_name)] = (list(b_inputs.keys()), list(b_outputs.keys()))
        for block_name, block_spec in blocks.items():
            if not isinstance(block_spec, dict):
                raise ValueError(f"invalid block spec: {block_name!r}")
            block_inputs = block_spec.get("inputs", {})
            block_outputs = block_spec.get("outputs", {})
            block_graph = block_spec.get("graph", [])
            if (
                not isinstance(block_inputs, dict)
                or not isinstance(block_outputs, dict)
                or not isinstance(block_graph, list)
            ):
                raise ValueError(f"invalid block structure: {block_name!r}")
            lines.extend(
                _render_module(
                    module_name=str(block_name),
                    inputs=block_inputs,
                    outputs=block_outputs,
                    graph=block_graph,
                    symbols=symbols,
                    signatures=signatures,
                )
            )
            lines.append("")
    lines.extend(
        _render_module(
            module_name=module_name,
            inputs=inputs,
            outputs=outputs,
            graph=graph,
            symbols=symbols,
            signatures=signatures,
        )
    )
    return "\n".join(lines) + "\n"


__all__ = ["synapse_spec_to_axon_module_text"]
