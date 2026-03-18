from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from typing import Any

from .types import AxonBind, AxonMeta, AxonModule, AxonRawNode, AxonRepeat, AxonReturn

_CALL_PAREN_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_:.@]*)\((.*)\)$")
_CALLEE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_:.@]*$")
_LAMBDA_RE = re.compile(r"^\\([A-Za-z_][A-Za-z0-9_]*)\s*->\s*(.+)$")
_ZERO_ARG_CALLS = {"init_list"}
_INVALID_POSITIONAL_TOKENS = {"+", "-", "*", "/", "%", "==", "!=", "<", ">", "<=", ">="}


def _split_top_level(text: str, sep: str) -> list[str]:
    parts: list[str] = []
    depth = 0
    start = 0
    i = 0
    seplen = len(sep)
    while i < len(text):
        ch = text[i]
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        if depth == 0 and text.startswith(sep, i):
            parts.append(text[start:i].strip())
            i += seplen
            start = i
            continue
        i += 1
    tail = text[start:].strip()
    if tail:
        parts.append(tail)
    return parts


def _split_csv(text: str) -> list[str]:
    return _split_top_level(text, ",")


def _parse_scalar(token: str) -> Any:
    value = token.strip()
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    if value.lower() == "null":
        return None
    if value and ((value[0] == value[-1] == '"') or (value[0] == value[-1] == "'")):
        return value[1:-1]
    if re.fullmatch(r"-?[0-9]+", value):
        return int(value)
    if re.fullmatch(r"-?[0-9]+\.[0-9]+([eE][+-]?[0-9]+)?", value):
        return float(value)
    return value


def _parse_call(expr: str) -> tuple[str, list[str], dict[str, Any]]:
    text = expr.strip()
    match = _CALL_PAREN_RE.match(text)
    if match is not None:
        callee = match.group(1).strip()
        raw_args = match.group(2).strip()
        tokens = _split_csv(raw_args) if raw_args else []
    else:
        callee_match = re.match(r"^([A-Za-z_][A-Za-z0-9_:.@]*)\b(.*)$", text)
        if callee_match is None:
            raise ValueError(f"expected call expression, got: {expr!r}")
        callee = callee_match.group(1).strip()
        rest = callee_match.group(2).strip()
        if not rest and "@" not in callee and "::" not in callee and callee not in _ZERO_ARG_CALLS:
            raise ValueError(f"expected call expression, got: {expr!r}")
        if not rest:
            tokens = []
        else:
            key_spans: list[tuple[int, int, str]] = []
            depth = 0
            i = 0
            while i < len(rest):
                ch = rest[i]
                if ch in "([":
                    depth += 1
                    i += 1
                    continue
                if ch in ")]":
                    depth -= 1
                    i += 1
                    continue
                if depth == 0 and (i == 0 or rest[i - 1].isspace()):
                    key_match = re.match(r"([A-Za-z_][A-Za-z0-9_]*)\s*=", rest[i:])
                    if key_match is not None:
                        key = key_match.group(1)
                        key_end = i + key_match.end()
                        key_spans.append((i, key_end, key))
                        i = key_end
                        continue
                i += 1

            tokens = []
            if not key_spans:
                tokens.extend(part for part in _split_top_level(rest, " ") if part)
            else:
                first_key_start = key_spans[0][0]
                pos_prefix = rest[:first_key_start].strip()
                if pos_prefix:
                    tokens.extend(part for part in _split_top_level(pos_prefix, " ") if part)
                for idx, (_, key_end, key_name) in enumerate(key_spans):
                    next_start = key_spans[idx + 1][0] if idx + 1 < len(key_spans) else len(rest)
                    value_text = rest[key_end:next_start].strip()
                    tokens.append(f"{key_name}={value_text}")
    args: list[str] = []
    kwargs: dict[str, Any] = {}
    for token in tokens:
        if "=" in token:
            key, value = token.split("=", 1)
            kwargs[key.strip()] = _parse_scalar(value)
        else:
            stripped = token.strip()
            if stripped in _INVALID_POSITIONAL_TOKENS:
                raise ValueError(f"expected call expression, got: {expr!r}")
            args.append(stripped)
    return callee, args, kwargs


def _looks_like_call(expr: str) -> bool:
    try:
        _parse_call(expr)
        return True
    except ValueError:
        return False


def _render_call(callee: str, args: list[str], kwargs: dict[str, Any]) -> str:
    items = [*args, *[f"{k}={v}" for k, v in kwargs.items()]]
    return f"{callee}({', '.join(items)})"


def _to_synapse_op(
    callee: str,
    args: list[str],
    kwargs: dict[str, Any],
    out: str | list[str],
) -> dict[str, Any]:
    if callee == "split_last":
        split_op: dict[str, Any] = {"op": "split_last", "out": out}
        if args:
            split_op["in"] = args[0] if len(args) == 1 else args
        for key, value in kwargs.items():
            split_op[key] = value
        return split_op

    if "@" in callee:
        op_name, _ = callee.split("@", 1)
        at_op: dict[str, Any] = {"op": op_name, "out": out}
        if op_name == "embed":
            at_op["op"] = "embedding"
        if args:
            at_op["in"] = args[0] if len(args) == 1 else args
        for key, value in kwargs.items():
            at_op[key] = value
        return at_op

    if "::" in callee:
        ns, name = callee.split("::", 1)
        if ns == "act":
            return {"op": "activation", "in": args[0], "out": out, "kind": name}
        if ns == "cache" and name == "update":
            return {"op": "kv_cache_update", "in": args, "out": out}
        if ns == "cache" and name == "seq_len":
            return {"op": "kv_seq_len", "in": args[0], "out": out}
        if ns == "cache" and name == "coalesce":
            return {"op": "coalesce_triplet", "in": args, "out": out}

    default_op: dict[str, Any] = {"op": callee, "out": out}
    if args:
        default_op["in"] = args[0] if len(args) == 1 else args
    for key, value in kwargs.items():
        default_op[key] = value
    return default_op


@dataclass
class _LowerCtx:
    counter: int = 0
    block_signatures: dict[str, tuple[list[str], list[str]]] | None = None

    def fresh(self, base: str = "t") -> str:
        self.counter += 1
        return f"{base}_{self.counter}"


def _with_when(nodes: list[dict[str, Any]], when: str | None) -> list[dict[str, Any]]:
    if when is None:
        return nodes
    out: list[dict[str, Any]] = []
    for item in nodes:
        name, node_spec = next(iter(item.items()))
        spec = dict(node_spec)
        spec["when"] = when
        out.append({name: spec})
    return out


def _lower_simple_call(
    expr: str, out: str | list[str], ctx: _LowerCtx, *, when: str | None = None
) -> list[dict[str, Any]]:
    callee, args, kwargs = _parse_call(expr)
    if ctx.block_signatures and callee in ctx.block_signatures:
        input_names, output_names = ctx.block_signatures[callee]
        provided: dict[str, str] = {}
        for idx, value in enumerate(args):
            if idx >= len(input_names):
                raise ValueError(f"too many positional args for block call {callee!r}")
            provided[input_names[idx]] = value
        for key, value in kwargs.items():
            if key not in input_names:
                raise ValueError(f"unknown block input {key!r} for call {callee!r}")
            provided[key] = str(value)
        in_map = {name: provided[name] for name in input_names if name in provided}

        out_values = [out] if isinstance(out, str) else list(out)
        if len(out_values) != len(output_names):
            raise ValueError(
                f"block call {callee!r} expects {len(output_names)} outputs, got {len(out_values)}"
            )
        out_map = {name: out_values[idx] for idx, name in enumerate(output_names)}

        node_name = f"n_{ctx.fresh('use')}"
        node_spec: dict[str, Any] = {"use": callee, "in": in_map, "out": out_map}
        return _with_when([{node_name: node_spec}], when)

    node_spec = _to_synapse_op(callee, args, kwargs, out)
    if "@" in callee:
        _, param_path = callee.split("@", 1)
        segments = [part.strip() for part in param_path.split(".") if part.strip()]
        if not segments:
            raise ValueError(f"invalid @ path in Axon call: {expr!r}")
        item: dict[str, Any] = {segments[-1]: node_spec}
        for segment in reversed(segments[:-1]):
            item = {segment: {"graph": [item]}}
        return _with_when([item], when)
    node_name = f"n_{ctx.fresh('op')}"
    return _with_when([{node_name: node_spec}], when)


def _lower_alias_or_const(
    expr: str, out: str | list[str], ctx: _LowerCtx, *, when: str | None = None
) -> list[dict[str, Any]]:
    if isinstance(out, list):
        raise ValueError("alias/const lowering expects scalar out")
    token = expr.strip()
    node_name = f"n_{ctx.fresh('op')}"
    scalar = _parse_scalar(token)
    if (
        isinstance(scalar, str)
        and scalar == token
        and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", token)
    ):
        node = {"op": "alias", "in": token, "out": out}
    else:
        node = {"op": "const", "value": scalar, "out": out}
    return _with_when([{node_name: node}], when)


def _tuple_items(expr: str) -> list[str]:
    text = expr.strip()
    if text.startswith("(") and text.endswith(")"):
        inner = text[1:-1].strip()
        return _split_csv(inner)
    return _split_csv(text)


def _split_ternary(expr: str) -> tuple[str, str, str] | None:
    depth = 0
    qpos = -1
    cpos = -1
    for i, ch in enumerate(expr):
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        elif ch == "?" and depth == 0 and qpos < 0:
            qpos = i
        elif ch == ":" and depth == 0 and qpos >= 0:
            if (i > 0 and expr[i - 1] == ":") or (i + 1 < len(expr) and expr[i + 1] == ":"):
                continue
            cpos = i
            break
    if qpos < 0 or cpos < 0:
        return None
    return expr[:qpos].strip(), expr[qpos + 1 : cpos].strip(), expr[cpos + 1 :].strip()


def _substitute_var(expr: str, name: str, value: str) -> str:
    return re.sub(rf"\b{re.escape(name)}\b", value, expr)


def _split_binary(expr: str, operator: str) -> tuple[str, str] | None:
    depth = 0
    for i in range(len(expr) - 1, -1, -1):
        ch = expr[i]
        if ch in ")]":
            depth += 1
        elif ch in "([":
            depth -= 1
        elif ch == operator and depth == 0:
            left = expr[:i].strip()
            right = expr[i + 1 :].strip()
            if left and right:
                return left, right
    return None


def _is_name_token(expr: str) -> bool:
    token = expr.strip()
    return re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", token) is not None


def _known_output_arity(callee: str, ctx: _LowerCtx) -> int | None:
    if ctx.block_signatures and callee in ctx.block_signatures:
        _, output_names = ctx.block_signatures[callee]
        return len(output_names)

    known: dict[str, int] = {
        "reshape_heads_triplet": 3,
        "topk": 2,
        "apply_rope_pair": 2,
        "cache::update": 3,
        "cache::coalesce": 2,
        "moe_select_tokens": 4,
    }
    return known.get(callee)


def _infer_split_last_arity(kwargs: dict[str, Any]) -> int | None:
    sizes = kwargs.get("sizes")
    if isinstance(sizes, list):
        return len(sizes)
    if isinstance(sizes, str):
        text = sizes.strip()
        if text.startswith("[") and text.endswith("]"):
            inner = text[1:-1].strip()
            if not inner:
                return 0
            return len(_split_csv(inner))
    parts = kwargs.get("parts")
    if isinstance(parts, int):
        return parts
    if isinstance(parts, str):
        token = parts.strip()
        if re.fullmatch(r"-?[0-9]+", token):
            return int(token)
        try:
            parsed = ast.literal_eval(token)
            if isinstance(parsed, int):
                return parsed
        except (ValueError, SyntaxError):
            return None
    return None


def _pipeline_temp_out(stage: str, ctx: _LowerCtx) -> str | list[str]:
    if not _looks_like_call(stage):
        return ctx.fresh("pipe")
    callee, _, kwargs = _parse_call(stage)
    if callee == "split_last":
        arity = _infer_split_last_arity(kwargs)
    else:
        arity = _known_output_arity(callee, ctx)
    if arity is None or arity <= 1:
        return ctx.fresh("pipe")
    return [ctx.fresh("pipe") for _ in range(arity)]


def _lower_expr(
    expr: str,
    out: str | list[str],
    ctx: _LowerCtx,
    *,
    when: str | None = None,
) -> list[dict[str, Any]]:
    expr = expr.strip()

    bind_parts = _split_top_level(expr, ">>=")
    if len(bind_parts) > 1:
        bind_graph: list[dict[str, Any]] = []
        first_bind = bind_parts[0].strip()
        if _is_name_token(first_bind):
            bind_ref = first_bind
        else:
            bind_ref = ctx.fresh("bind")
            bind_graph.extend(_lower_expr(first_bind, bind_ref, ctx, when=when))
        for idx, part in enumerate(bind_parts[1:], start=1):
            match = _LAMBDA_RE.match(part.strip())
            if match is None:
                raise ValueError(f"expected lambda after >>=, got: {part!r}")
            var_name = match.group(1)
            body = _substitute_var(match.group(2), var_name, bind_ref)
            bind_next_out: str | list[str] = (
                out if idx == len(bind_parts) - 1 else ctx.fresh("bind")
            )
            bind_graph.extend(_lower_expr(body, bind_next_out, ctx, when=when))
            if isinstance(bind_next_out, str):
                bind_ref = bind_next_out
        return bind_graph

    ternary = _split_ternary(expr)
    if ternary is not None:
        cond, true_expr, false_expr = ternary
        if isinstance(out, list):
            true_items = _tuple_items(true_expr)
            false_items = _tuple_items(false_expr)
            ternary_graph: list[dict[str, Any]] = []
            if len(true_items) == 1 and _looks_like_call(true_items[0]):
                ternary_graph.extend(_lower_expr(true_items[0], out, ctx, when=cond))
            elif len(true_items) == len(out):
                for name, item in zip(out, true_items, strict=True):
                    ternary_graph.extend(_lower_expr(item, name, ctx, when=cond))
            else:
                raise ValueError("ternary true-branch arity must match binding targets")

            if len(false_items) == 1 and _looks_like_call(false_items[0]):
                ternary_graph.extend(_lower_expr(false_items[0], out, ctx, when=f"not ({cond})"))
            elif len(false_items) == len(out):
                for name, item in zip(out, false_items, strict=True):
                    ternary_graph.extend(_lower_expr(item, name, ctx, when=f"not ({cond})"))
            else:
                raise ValueError("ternary tuple arity must match binding targets")
            return ternary_graph

        cond_graph: list[dict[str, Any]] = []
        cond_graph.extend(_lower_expr(true_expr, out, ctx, when=cond))
        cond_graph.extend(_lower_expr(false_expr, out, ctx, when=f"not ({cond})"))
        return cond_graph

    if "|>" in expr:
        stages = _split_top_level(expr, "|>")
        if not stages:
            raise ValueError("empty pipeline")
        pipe_graph: list[dict[str, Any]] = []

        first = stages[0]
        if _looks_like_call(first):
            first_out: str | list[str] = out if len(stages) == 1 else _pipeline_temp_out(first, ctx)
            pipe_graph.extend(_lower_simple_call(first, first_out, ctx, when=when))
            pipe_ref = first_out
        else:
            pipe_ref = first.strip()

        for idx, stage in enumerate(stages[1:], start=1):
            stage = stage.strip()
            if _looks_like_call(stage):
                callee, args, kwargs = _parse_call(stage)
            else:
                callee = stage
                args, kwargs = [], {}
            next_out: str | list[str] = (
                out if idx == len(stages) - 1 else _pipeline_temp_out(stage, ctx)
            )
            piped_args = [pipe_ref] if isinstance(pipe_ref, str) else list(pipe_ref)
            call_expr = _render_call(callee, [*piped_args, *args], kwargs)
            pipe_graph.extend(_lower_simple_call(call_expr, next_out, ctx, when=when))
            pipe_ref = next_out
        return pipe_graph

    if _looks_like_call(expr):
        return _lower_simple_call(expr, out, ctx, when=when)

    plus = _split_binary(expr, "+")
    if plus is not None:
        left_expr, right_expr = plus
        add_graph: list[dict[str, Any]] = []
        left_ref = left_expr.strip() if _is_name_token(left_expr) else ctx.fresh("bin")
        right_ref = right_expr.strip() if _is_name_token(right_expr) else ctx.fresh("bin")
        if not _is_name_token(left_expr):
            add_graph.extend(_lower_expr(left_expr, left_ref, ctx, when=when))
        if not _is_name_token(right_expr):
            add_graph.extend(_lower_expr(right_expr, right_ref, ctx, when=when))
        add_graph.extend(_lower_simple_call(f"add({left_ref}, {right_ref})", out, ctx, when=when))
        return add_graph

    mul = _split_binary(expr, "*")
    if mul is not None:
        left_expr, right_expr = mul
        left_ref = left_expr.strip() if _is_name_token(left_expr) else ctx.fresh("bin")
        right_ref = right_expr.strip() if _is_name_token(right_expr) else ctx.fresh("bin")
        mul_graph: list[dict[str, Any]] = []
        if not _is_name_token(left_expr):
            mul_graph.extend(_lower_expr(left_expr, left_ref, ctx, when=when))
        if not _is_name_token(right_expr):
            mul_graph.extend(_lower_expr(right_expr, right_ref, ctx, when=when))
        mul_graph.extend(_lower_simple_call(f"mul({left_ref}, {right_ref})", out, ctx, when=when))
        return mul_graph

    return _lower_alias_or_const(expr, out, ctx, when=when)


def _module_return_names(module: AxonModule) -> tuple[str, ...]:
    if module.returns:
        return module.returns
    for stmt in reversed(module.statements):
        if isinstance(stmt, AxonReturn):
            inferred: list[str] = []
            for idx, value in enumerate(stmt.values):
                inferred.append(value if _is_name_token(value) else f"out_{idx}")
            if inferred:
                return tuple(inferred)
    return ()


def lower_axon_module_to_synapse_block(module: AxonModule) -> dict[str, Any]:
    inputs = {param.name: {"optional": param.optional} for param in module.params}
    graph: list[dict[str, Any]] = []
    outputs: dict[str, str] = {}
    ctx = _LowerCtx(block_signatures={})
    returns = _module_return_names(module)

    _lower_statements(
        statements=module.statements,
        graph=graph,
        outputs=outputs,
        returns=returns,
        ctx=ctx,
    )

    if not outputs:
        for name in returns:
            outputs[name] = name

    return {"inputs": inputs, "graph": graph, "outputs": outputs}


def lower_axon_module_to_synapse_spec(module: AxonModule) -> dict[str, Any]:
    block = lower_axon_module_to_synapse_block(module)
    model: dict[str, Any] = {
        "inputs": block["inputs"],
        "graph": block["graph"],
        "outputs": block["outputs"],
    }
    for stmt in module.statements:
        if isinstance(stmt, AxonMeta):
            model[stmt.key] = stmt.value
    return {
        "synapse": 1,
        "model": model,
    }


def _lower_statements(
    *,
    statements: tuple[AxonBind | AxonReturn | AxonRawNode | AxonMeta | AxonRepeat, ...],
    graph: list[dict[str, Any]],
    outputs: dict[str, str],
    returns: tuple[str, ...],
    ctx: _LowerCtx,
) -> None:
    for stmt in statements:
        if isinstance(stmt, AxonRawNode):
            graph.append({stmt.name: dict(stmt.node_spec)})
            continue

        if isinstance(stmt, AxonMeta):
            continue

        if isinstance(stmt, AxonRepeat):
            body_graph: list[dict[str, Any]] = []
            _lower_statements(
                statements=stmt.body,
                graph=body_graph,
                outputs={},
                returns=(),
                ctx=ctx,
            )
            repeat_name = (
                stmt.name
                if isinstance(stmt.name, str) and stmt.name
                else f"n_{ctx.fresh('repeat')}"
            )
            repeat_item: dict[str, Any] = {
                repeat_name.split(".")[-1]: {
                    "op": "repeat",
                    "var": stmt.var,
                    "range": stmt.range_expr,
                    "body": body_graph,
                }
            }
            segments = [part for part in repeat_name.split(".") if part]
            for segment in reversed(segments[:-1]):
                repeat_item = {segment: {"graph": [repeat_item]}}
            graph.append(repeat_item)
            continue

        if isinstance(stmt, AxonBind):
            out: str | list[str] = stmt.targets[0] if len(stmt.targets) == 1 else list(stmt.targets)
            graph.extend(_lower_expr(stmt.expr, out, ctx))
            continue

        if isinstance(stmt, AxonReturn):
            for idx, value in enumerate(stmt.values):
                output_name = returns[idx] if idx < len(returns) else f"out_{idx}"
                if _is_name_token(value):
                    outputs[output_name] = value
                    continue
                graph.extend(_lower_expr(value, output_name, ctx))
                outputs[output_name] = output_name
            continue


def lower_axon_program_to_synapse_spec(
    modules: tuple[AxonModule, ...], *, main_module: str | None = None
) -> dict[str, Any]:
    if not modules:
        raise ValueError("Axon program must contain at least one module")

    by_name = {module.name: module for module in modules}
    if len(by_name) != len(modules):
        raise ValueError("Axon program contains duplicate module names")

    main_name = modules[-1].name if main_module is None else main_module
    if main_name not in by_name:
        raise ValueError(f"Unknown main module: {main_name!r}")

    signatures: dict[str, tuple[list[str], list[str]]] = {}
    for module in modules:
        signatures[module.name] = (
            [param.name for param in module.params],
            list(_module_return_names(module)),
        )

    main = by_name[main_name]
    main_returns = _module_return_names(main)
    main_inputs = {param.name: {"optional": param.optional} for param in main.params}
    main_graph: list[dict[str, Any]] = []
    main_outputs: dict[str, str] = {}
    _lower_statements(
        statements=main.statements,
        graph=main_graph,
        outputs=main_outputs,
        returns=main_returns,
        ctx=_LowerCtx(block_signatures=signatures),
    )
    if not main_outputs:
        for name in main_returns:
            main_outputs[name] = name
    model: dict[str, Any] = {"inputs": main_inputs, "graph": main_graph, "outputs": main_outputs}
    for stmt in main.statements:
        if isinstance(stmt, AxonMeta):
            model[stmt.key] = stmt.value
    spec: dict[str, Any] = {"synapse": 1, "model": model}

    blocks: dict[str, Any] = {}
    for module in modules:
        if module.name == main_name:
            continue
        block_inputs = {param.name: {"optional": param.optional} for param in module.params}
        block_returns = _module_return_names(module)
        block_graph: list[dict[str, Any]] = []
        block_outputs: dict[str, str] = {}
        _lower_statements(
            statements=module.statements,
            graph=block_graph,
            outputs=block_outputs,
            returns=block_returns,
            ctx=_LowerCtx(block_signatures=signatures),
        )
        if not block_outputs:
            for name in block_returns:
                block_outputs[name] = name
        blocks[module.name] = {
            "inputs": block_inputs,
            "graph": block_graph,
            "outputs": block_outputs,
        }
    if blocks:
        spec["model"]["blocks"] = blocks

    return spec


__all__ = [
    "lower_axon_module_to_synapse_block",
    "lower_axon_module_to_synapse_spec",
    "lower_axon_program_to_synapse_spec",
]
