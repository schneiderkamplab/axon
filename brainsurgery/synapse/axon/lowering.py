from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import Any

from .types import (
    AxonBind,
    AxonModule,
    AxonRepeat,
    AxonReturn,
    AxonScope,
    AxonScopeBind,
    AxonStatement,
)

_CALL_PAREN_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_:.@]*)\((.*)\)$")
_CALLEE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_:.@]*$")
_LAMBDA_RE = re.compile(r"^\\([A-Za-z_][A-Za-z0-9_]*)\s*->\s*(.+)$")
_ZERO_ARG_CALLS = {"init_list"}
_INVALID_POSITIONAL_TOKENS = {"+", "-", "*", "/", "%", "==", "!=", "<", ">", "<=", ">="}

_OP_ARITY: dict[str, tuple[int, int]] = {
    "embedding": (1, 1),
    "linear": (1, 1),
    "layernorm": (1, 1),
    "rmsnorm": (1, 1),
    "attention": (3, 3),
    "causal_mask": (1, 1),
    "reshape_heads": (1, 1),
    "split_last": (1, 1),
    "apply_rope_pair": (2, 2),
    "repeat_kv": (1, 1),
    "arange_positions": (1, 1),
    "kv_cache_update": (3, 3),
    "coalesce": (2, 8),
    "topk": (1, 1),
    "softmax": (1, 1),
    "zeros_like": (1, 1),
    "moe_select_tokens": (3, 3),
    "moe_scatter_add": (4, 4),
    "index": (2, 2),
    "init_list": (0, 0),
    "append": (2, 2),
    "add": (2, 2),
    "mul": (2, 2),
    "merge_heads": (1, 1),
    "activation": (1, 1),
    "kv_seq_len": (1, 1),
}

_OP_ALLOWED_KWARGS: dict[str, set[str]] = {
    "embedding": {"dim", "scale"},
    "linear": {"dim", "bias", "transpose"},
    "layernorm": {"dim", "eps"},
    "rmsnorm": {"dim", "eps", "cast_float", "unit_offset"},
    "attention": {
        "backend",
        "causal",
        "mask",
        "scale",
        "rope_theta",
        "sliding_window",
        "causal_mask_buffer",
    },
    "causal_mask": {"key", "window", "padding_mask"},
    "reshape_heads": {"heads", "head_dim"},
    "split_last": {"parts", "sizes"},
    "apply_rope_pair": {"position_ids", "theta"},
    "repeat_kv": {"heads", "kv_heads"},
    "arange_positions": {"attention_mask"},
    "kv_cache_update": {"when"},
    "coalesce": set(),
    "topk": {"k", "dim"},
    "softmax": {"dim", "dtype"},
    "zeros_like": set(),
    "moe_select_tokens": {"expert"},
    "moe_scatter_add": set(),
    "index": set(),
    "init_list": set(),
    "append": {"when"},
    "add": set(),
    "mul": set(),
    "merge_heads": set(),
    "activation": {"kind"},
    "kv_seq_len": set(),
}

_OP_KWARG_KINDS: dict[str, dict[str, str]] = {
    "embedding": {"dim": "dim", "scale": "number"},
    "linear": {"dim": "dim", "bias": "bool", "transpose": "bool"},
    "layernorm": {"dim": "dim", "eps": "number"},
    "rmsnorm": {"dim": "dim", "eps": "number", "cast_float": "bool", "unit_offset": "bool"},
    "attention": {
        "backend": "str",
        "causal": "bool",
        "mask": "str",
        "scale": "number",
        "rope_theta": "number",
        "sliding_window": "dim",
        "causal_mask_buffer": "str",
    },
    "causal_mask": {"key": "str", "window": "dim", "padding_mask": "str"},
    "reshape_heads": {"heads": "dim", "head_dim": "dim"},
    "split_last": {"parts": "int", "sizes": "list_int"},
    "apply_rope_pair": {"position_ids": "str", "theta": "number"},
    "repeat_kv": {"heads": "dim", "kv_heads": "dim"},
    "arange_positions": {"attention_mask": "str"},
    "kv_cache_update": {},
    "coalesce": {},
    "topk": {"k": "dim", "dim": "int"},
    "softmax": {"dim": "int", "dtype": "str"},
    "zeros_like": {},
    "moe_select_tokens": {"expert": "dim"},
    "moe_scatter_add": {},
    "index": {},
    "init_list": {},
    "append": {},
    "add": {},
    "mul": {},
    "merge_heads": {},
    "activation": {"kind": "str"},
    "kv_seq_len": {},
}


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
            return {"op": "coalesce", "in": args, "out": out}

    default_op: dict[str, Any] = {"op": callee, "out": out}
    if args:
        default_op["in"] = args[0] if len(args) == 1 else args
    for key, value in kwargs.items():
        default_op[key] = value
    return default_op


def _canonical_op_name(callee: str) -> str:
    if "@" in callee:
        op_name = callee.split("@", 1)[0]
        if op_name == "embed":
            return "embedding"
        return op_name
    if "::" in callee:
        ns, name = callee.split("::", 1)
        if ns == "act":
            return "activation"
        if ns == "cache" and name == "update":
            return "kv_cache_update"
        if ns == "cache" and name == "coalesce":
            return "coalesce"
        if ns == "cache" and name == "seq_len":
            return "kv_seq_len"
    return callee


def _normalize_dim_token(value: Any) -> Any:
    if isinstance(value, str):
        token = value.strip()
        if re.fullmatch(r"-?[0-9]+", token):
            return int(token)
        return token
    return value


def _dims_compatible(left: Any, right: Any) -> bool:
    return _normalize_dim_token(left) == _normalize_dim_token(right)


def _is_kind(value: Any, kind: str) -> bool:
    if kind == "bool":
        return isinstance(value, bool)
    if kind == "int":
        return isinstance(value, int) and not isinstance(value, bool)
    if kind == "number":
        if isinstance(value, bool):
            return False
        return isinstance(value, (int, float, str))
    if kind == "str":
        return isinstance(value, str)
    if kind == "dim":
        if isinstance(value, bool):
            return False
        return isinstance(value, (int, str))
    if kind == "list_int":
        return isinstance(value, list) and all(
            isinstance(v, int) and not isinstance(v, bool) for v in value
        )
    return True


def _validate_op_signature(op_name: str, args: list[str], kwargs: dict[str, Any]) -> None:
    arity = _OP_ARITY.get(op_name)
    if arity is not None:
        min_args, max_args = arity
        if len(args) < min_args or len(args) > max_args:
            raise ValueError(
                f"{op_name} expects {min_args}"
                + (f"..{max_args}" if min_args != max_args else "")
                + f" positional args, got {len(args)}"
            )
    allowed = _OP_ALLOWED_KWARGS.get(op_name)
    if allowed is not None:
        unknown = sorted(set(kwargs) - allowed)
        if unknown:
            raise ValueError(f"{op_name} unsupported kwargs: {', '.join(unknown)}")
    kinds = _OP_KWARG_KINDS.get(op_name, {})
    for key, value in kwargs.items():
        expected = kinds.get(key)
        if expected is None:
            continue
        if not _is_kind(value, expected):
            raise ValueError(
                f"{op_name} kwarg {key!r} expects {expected}, got {type(value).__name__}"
            )


def _merge_when(outer: str | None, inner: Any) -> str | None:
    if inner is None:
        return outer
    if isinstance(inner, bool):
        inner_text = "true" if inner else "false"
    else:
        inner_text = str(inner).strip()
    if not inner_text or inner_text == "true":
        return outer
    if inner_text == "false":
        return "false"
    if outer is None or not outer.strip() or outer.strip() == "true":
        return inner_text
    if outer.strip() == "false":
        return "false"
    return f"({outer}) and ({inner_text})"


@dataclass
class _LowerCtx:
    counter: int = 0
    block_signatures: dict[str, tuple[list[str], list[str]]] | None = None
    block_path_params: dict[str, str | None] | None = None
    block_param_last_dims: dict[str, dict[str, Any]] | None = None
    block_output_last_dims: dict[str, dict[str, Any]] | None = None
    tensor_last_dim: dict[str, Any] = field(default_factory=dict)
    tensor_heads: dict[str, Any] = field(default_factory=dict)
    scope_stack: list[str] = field(default_factory=list)
    path_param_names: set[str] = field(default_factory=set)

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


def _op_name_from_callee(callee: str) -> str:
    if "@" in callee:
        return callee.split("@", 1)[0]
    if "::" in callee:
        return callee.split("::", 1)[1]
    return callee


def _maybe_int_list(value: Any) -> list[int] | None:
    if isinstance(value, list):
        try:
            return [int(v) for v in value]
        except Exception:
            return None
    if isinstance(value, str) and value.strip().startswith("[") and value.strip().endswith("]"):
        try:
            parsed = ast.literal_eval(value)
        except Exception:
            return None
        if isinstance(parsed, list):
            try:
                return [int(v) for v in parsed]
            except Exception:
                return None
    return None


def _infer_split_sizes_from_last_dim(last_dim: Any, parts: int) -> list[Any] | None:
    if parts <= 0:
        return None
    if isinstance(last_dim, int):
        if last_dim % parts != 0:
            return None
        return [last_dim // parts for _ in range(parts)]
    if not isinstance(last_dim, str):
        return None
    token = last_dim.strip().replace(" ", "")
    if not token:
        return None
    if re.fullmatch(r"-?[0-9]+", token):
        value = int(token)
        if value % parts != 0:
            return None
        return [value // parts for _ in range(parts)]
    m = re.fullmatch(r"([0-9]+)\*(.+)", token)
    if m is None:
        m = re.fullmatch(r"(.+)\*([0-9]+)", token)
        if m is None:
            return None
        term = m.group(1)
        factor = int(m.group(2))
    else:
        factor = int(m.group(1))
        term = m.group(2)
    if factor % parts != 0:
        return None
    each = factor // parts
    piece: Any = term if each == 1 else f"{each}*{term}"
    return [piece for _ in range(parts)]


def _record_last_dim_for_call(
    *, callee: str, args: list[str], kwargs: dict[str, Any], out: str | list[str], ctx: _LowerCtx
) -> None:
    resolved_block = _resolve_block_call(callee, ctx)
    if resolved_block is not None and ctx.block_signatures is not None:
        block_name, _ = resolved_block
        input_names, output_names = ctx.block_signatures[block_name]
        provided: dict[str, str] = {}
        for idx, value in enumerate(args):
            if idx < len(input_names):
                provided[input_names[idx]] = value.strip()
        for key, value in kwargs.items():
            if key in input_names:
                provided[key] = str(value).strip()

        symbol_bindings: dict[str, Any] = {}
        for param_name, raw in provided.items():
            if _is_name_token(raw):
                if raw in ctx.tensor_last_dim:
                    symbol_bindings[param_name] = ctx.tensor_last_dim[raw]
                else:
                    symbol_bindings[param_name] = raw
                continue
            parsed = _parse_scalar(raw)
            symbol_bindings[param_name] = parsed

        param_last_dims = (
            ctx.block_param_last_dims.get(block_name, {})
            if isinstance(ctx.block_param_last_dims, dict)
            else {}
        )
        for param_name, sym in param_last_dims.items():
            if param_name in symbol_bindings and isinstance(sym, str):
                symbol_bindings[sym] = symbol_bindings[param_name]

        output_last_dims = (
            ctx.block_output_last_dims.get(block_name, {})
            if isinstance(ctx.block_output_last_dims, dict)
            else {}
        )
        out_targets = [out] if isinstance(out, str) else list(out)
        for output_name, target in zip(output_names, out_targets, strict=False):
            dim_token = output_last_dims.get(output_name)
            if isinstance(dim_token, str):
                resolved_dim = symbol_bindings.get(dim_token, dim_token)
                ctx.tensor_last_dim[target] = resolved_dim
            elif dim_token is not None:
                ctx.tensor_last_dim[target] = dim_token

    op_name = _canonical_op_name(callee)
    first_in = args[0].strip() if args else None
    second_in = args[1].strip() if len(args) > 1 else None
    first_dim = (
        ctx.tensor_last_dim.get(first_in)
        if isinstance(first_in, str) and _is_name_token(first_in)
        else None
    )
    second_dim = (
        ctx.tensor_last_dim.get(second_in)
        if isinstance(second_in, str) and _is_name_token(second_in)
        else None
    )

    if isinstance(out, list):
        if op_name == "split_last":
            sizes = _maybe_int_list(kwargs.get("sizes"))
            if sizes is not None and len(sizes) == len(out):
                for name, dim in zip(out, sizes, strict=True):
                    ctx.tensor_last_dim[name] = dim
            return
        if op_name == "reshape_heads_triplet":
            heads = kwargs.get("heads")
            head_dim = kwargs.get("head_dim")
            if head_dim is not None:
                for name in out:
                    ctx.tensor_last_dim[name] = head_dim
            if heads is not None:
                for name in out:
                    ctx.tensor_heads[name] = heads
            return
        return

    last_dim: Any | None = None
    if op_name in {"add", "mul"}:
        if (
            first_dim is not None
            and second_dim is not None
            and not _dims_compatible(first_dim, second_dim)
        ):
            raise ValueError(
                f"{op_name} requires matching last-dim; got {first_dim!r} and {second_dim!r}"
            )
        unified = first_dim if first_dim is not None else second_dim
        if (
            unified is not None
            and isinstance(first_in, str)
            and _is_name_token(first_in)
            and first_dim is None
        ):
            ctx.tensor_last_dim[first_in] = unified
        if (
            unified is not None
            and isinstance(second_in, str)
            and _is_name_token(second_in)
            and second_dim is None
        ):
            ctx.tensor_last_dim[second_in] = unified
        last_dim = unified
    elif op_name in {"layernorm", "rmsnorm", "activation", "merge_heads"}:
        last_dim = first_dim
    if op_name in {"layernorm", "rmsnorm"}:
        norm_dim = kwargs.get("dim")
        if norm_dim is not None:
            if first_dim is not None and not _dims_compatible(norm_dim, first_dim):
                raise ValueError(
                    f"{op_name} dim={norm_dim!r} mismatches input last-dim {first_dim!r}"
                )
            if first_dim is None and isinstance(first_in, str) and _is_name_token(first_in):
                ctx.tensor_last_dim[first_in] = norm_dim
    elif op_name == "embedding":
        last_dim = kwargs.get("embedding_dim")
    elif op_name == "linear":
        last_dim = kwargs.get("dim", first_dim)

    if last_dim is not None:
        ctx.tensor_last_dim[out] = last_dim
    if op_name == "reshape_heads":
        heads = kwargs.get("heads")
        if heads is not None:
            ctx.tensor_heads[out] = heads
    if op_name == "repeat_kv":
        heads = kwargs.get("heads")
        if heads is not None:
            ctx.tensor_heads[out] = heads


def _lower_simple_call(
    expr: str, out: str | list[str], ctx: _LowerCtx, *, when: str | None = None
) -> list[dict[str, Any]]:
    callee, args, kwargs = _parse_call(expr)
    pre_graph: list[dict[str, Any]] = []
    effective_when = _merge_when(when, kwargs.pop("when", None))

    resolved_args: list[str] = []
    for arg in args:
        token = arg.strip()
        inner = token
        if token.startswith("(") and token.endswith(")"):
            inner = token[1:-1].strip()
        if inner != token and (not _is_name_token(inner)):
            tmp = ctx.fresh("arg")
            pre_graph.extend(_lower_expr(inner, tmp, ctx, when=when))
            resolved_args.append(tmp)
        else:
            resolved_args.append(token)
    args = resolved_args

    is_absolute_path = "@@" in callee
    if is_absolute_path:
        callee = callee.replace("@@", "@", 1)
    if _op_name_from_callee(callee) == "reshape_heads_triplet":
        if not isinstance(out, list) or len(out) != 3 or len(args) != 3:
            raise ValueError("reshape_heads_triplet requires 3 inputs and 3 outputs")
        if "heads" not in kwargs and "head_dim" not in kwargs:
            raise ValueError("reshape_heads_triplet requires heads or head_dim")
        lowered_nodes: list[dict[str, Any]] = []
        for src, dst in zip(args, out, strict=True):
            head_kwargs: dict[str, Any] = {}
            if "heads" in kwargs:
                head_kwargs["heads"] = kwargs["heads"]
            if "head_dim" in kwargs:
                head_kwargs["head_dim"] = kwargs["head_dim"]
            lowered_nodes.extend(
                _lower_simple_call(
                    _render_call("reshape_heads", [src], head_kwargs),
                    dst,
                    ctx,
                    when=effective_when,
                )
            )
        return [*pre_graph, *lowered_nodes]
    if "@" in callee and ctx.scope_stack and not is_absolute_path:
        op_name_with_at, param_path = callee.split("@", 1)
        scope_prefix = ".".join(part for part in ctx.scope_stack if part)
        if scope_prefix:
            scoped_path = f"{scope_prefix}.{param_path}" if param_path.strip() else scope_prefix
            callee = f"{op_name_with_at}@{scoped_path}"
    op_name = _canonical_op_name(callee)
    if op_name == "embedding":
        if "embedding_dim" in kwargs:
            raise ValueError("embedding does not support embedding_dim; use dim")
        allowed_embedding_kwargs = {"dim", "scale"}
        invalid_embedding_kwargs = sorted(set(kwargs) - allowed_embedding_kwargs)
        if invalid_embedding_kwargs:
            bad = ", ".join(invalid_embedding_kwargs)
            raise ValueError(f"embedding unsupported kwargs: {bad}; allowed: dim, scale")
        if "dim" in kwargs:
            kwargs["embedding_dim"] = kwargs.pop("dim")
    if op_name == "linear" and "weight_layout" in kwargs:
        raise ValueError("linear does not support weight_layout; use transpose=true/false")
    if op_name == "linear" and "tie_weight" in kwargs:
        raise ValueError("linear does not support tie_weight; use linear@<path>")
    if op_name == "linear" and "out_features" in kwargs:
        raise ValueError("linear does not support out_features; use dim")
    if op_name == "linear" and "out_dim" in kwargs:
        raise ValueError("linear does not support out_dim; use dim")
    if op_name == "linear" and "transpose" in kwargs:
        raw_transpose = kwargs["transpose"]
        if isinstance(raw_transpose, bool):
            pass
        elif isinstance(raw_transpose, str) and raw_transpose.lower() in {"true", "false"}:
            kwargs["transpose"] = raw_transpose.lower() == "true"
        else:
            raise ValueError("linear transpose must be true/false")
    if op_name in {"layernorm", "rmsnorm"} and "dim" not in kwargs and args:
        first_arg = args[0].strip()
        if _is_name_token(first_arg):
            inferred = ctx.tensor_last_dim.get(first_arg)
            if inferred is not None:
                kwargs["dim"] = inferred
    if op_name == "linear" and "dim" not in kwargs and isinstance(out, str):
        inferred = ctx.tensor_last_dim.get(out)
        if inferred is not None:
            kwargs["dim"] = inferred
    if op_name == "embedding" and "embedding_dim" not in kwargs and isinstance(out, str):
        inferred = ctx.tensor_last_dim.get(out)
        if inferred is not None:
            kwargs["embedding_dim"] = inferred
    validation_kwargs = dict(kwargs)
    if op_name == "embedding" and "embedding_dim" in validation_kwargs:
        validation_kwargs["dim"] = validation_kwargs.pop("embedding_dim")
    _validate_op_signature(op_name, args, validation_kwargs)
    if op_name == "repeat_kv" and args:
        src_name = args[0].strip()
        if _is_name_token(src_name):
            if "kv_heads" not in kwargs:
                inferred_kv_heads = ctx.tensor_heads.get(src_name)
                if inferred_kv_heads is not None:
                    kwargs["kv_heads"] = inferred_kv_heads
            if "heads" not in kwargs and isinstance(out, str):
                inferred_heads = ctx.tensor_heads.get(out)
                if inferred_heads is not None:
                    kwargs["heads"] = inferred_heads
    if (
        op_name == "split_last"
        and isinstance(out, list)
        and "sizes" not in kwargs
        and "parts" not in kwargs
        and args
    ):
        first_arg = args[0].strip()
        if _is_name_token(first_arg):
            inferred = ctx.tensor_last_dim.get(first_arg)
            split_sizes = _infer_split_sizes_from_last_dim(inferred, len(out))
            if split_sizes is not None:
                kwargs["sizes"] = split_sizes
    resolved_block = _resolve_block_call(callee, ctx)
    if resolved_block is not None and ctx.block_signatures:
        block_name, path_bindings = resolved_block
        input_names, output_names = ctx.block_signatures[block_name]
        provided: dict[str, str] = {}
        for idx, value in enumerate(args):
            if idx >= len(input_names):
                raise ValueError(f"too many positional args for block call {callee!r}")
            provided[input_names[idx]] = value
        for key, value in kwargs.items():
            if key not in input_names:
                raise ValueError(f"unknown block input {key!r} for call {callee!r}")
            provided[key] = str(value)
        for key, concrete_path in path_bindings.items():
            if key not in input_names:
                raise ValueError(f"unknown block path parameter {key!r} for call {callee!r}")
            provided[key] = repr(concrete_path)
        in_map = {name: provided[name] for name in input_names if name in provided}

        out_values = [out] if isinstance(out, str) else list(out)
        if len(out_values) != len(output_names):
            raise ValueError(
                f"block call {callee!r} expects {len(output_names)} outputs, got {len(out_values)}"
            )
        out_map = {name: out_values[idx] for idx, name in enumerate(output_names)}

        node_name = f"n_{ctx.fresh('use')}"
        node_spec: dict[str, Any] = {"use": block_name, "in": in_map, "out": out_map}
        nodes = _with_when([{node_name: node_spec}], effective_when)
        _record_last_dim_for_call(callee=block_name, args=args, kwargs=kwargs, out=out, ctx=ctx)
        return [*pre_graph, *nodes]

    node_spec = _to_synapse_op(callee, args, kwargs, out)
    if "@" in callee:
        op_name, param_path = callee.split("@", 1)
        if param_path in ctx.path_param_names:
            node_name = f"n_{ctx.fresh('op')}"
            templated_node = _to_synapse_op(op_name, args, kwargs, out)
            templated_node["param_base"] = param_path
            nodes = _with_when([{node_name: templated_node}], effective_when)
            _record_last_dim_for_call(callee=op_name, args=args, kwargs=kwargs, out=out, ctx=ctx)
            return [*pre_graph, *nodes]
        segments = [part.strip() for part in param_path.split(".") if part.strip()]
        if not segments:
            raise ValueError(f"invalid @ path in Axon call: {expr!r}")
        item: dict[str, Any] = {segments[-1]: node_spec}
        for segment in reversed(segments[:-1]):
            item = {segment: {"graph": [item]}}
        nodes = _with_when([item], effective_when)
        _record_last_dim_for_call(callee=callee, args=args, kwargs=kwargs, out=out, ctx=ctx)
        return [*pre_graph, *nodes]
    node_name = f"n_{ctx.fresh('op')}"
    nodes = _with_when([{node_name: node_spec}], effective_when)
    _record_last_dim_for_call(callee=callee, args=args, kwargs=kwargs, out=out, ctx=ctx)
    return [*pre_graph, *nodes]


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
        node = {"op": "_ir_alias", "in": token, "out": out}
        if token in ctx.tensor_last_dim:
            ctx.tensor_last_dim[out] = ctx.tensor_last_dim[token]
    else:
        node = {"op": "_ir_const", "value": scalar, "out": out}
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


def _word_boundary(text: str, index: int) -> bool:
    if index < 0 or index >= len(text):
        return True
    return not (text[index].isalnum() or text[index] == "_")


def _find_top_level_keyword(text: str, keyword: str, *, start: int = 0) -> int:
    depth = 0
    i = start
    size = len(keyword)
    while i <= len(text) - size:
        ch = text[i]
        if ch in "([":
            depth += 1
            i += 1
            continue
        if ch in ")]":
            depth -= 1
            i += 1
            continue
        if depth == 0 and text.startswith(keyword, i):
            left_ok = _word_boundary(text, i - 1)
            right_ok = _word_boundary(text, i + size)
            if left_ok and right_ok:
                return i
        i += 1
    return -1


def _split_if_then_else(expr: str) -> tuple[str, str, str] | None:
    text = expr.strip()
    if not text.startswith("if") or not _word_boundary(text, 2):
        return None

    then_pos = _find_top_level_keyword(text, "then", start=2)
    if then_pos < 0:
        return None

    cond = text[2:then_pos].strip()
    if not cond:
        return None

    body_start = then_pos + len("then")
    depth = 0
    nested_if = 0
    i = body_start
    else_pos = -1
    while i < len(text):
        ch = text[i]
        if ch in "([":
            depth += 1
            i += 1
            continue
        if ch in ")]":
            depth -= 1
            i += 1
            continue
        if depth == 0:
            if (
                text.startswith("if", i)
                and _word_boundary(text, i - 1)
                and _word_boundary(text, i + 2)
            ):
                nested_if += 1
                i += 2
                continue
            if (
                text.startswith("else", i)
                and _word_boundary(text, i - 1)
                and _word_boundary(text, i + 4)
            ):
                if nested_if == 0:
                    else_pos = i
                    break
                nested_if -= 1
                i += 4
                continue
        i += 1

    if else_pos < 0:
        return None

    true_expr = text[body_start:else_pos].strip()
    false_expr = text[else_pos + len("else") :].strip()
    if not true_expr or not false_expr:
        return None
    return cond, true_expr, false_expr


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


def _resolve_block_call(callee: str, ctx: _LowerCtx) -> tuple[str, dict[str, str]] | None:
    if not ctx.block_signatures:
        return None
    if callee in ctx.block_signatures:
        return callee, {}
    if "@" not in callee:
        return None
    base, concrete_path = callee.split("@", 1)
    if base not in ctx.block_signatures:
        return None
    path_param = (
        ctx.block_path_params.get(base) if isinstance(ctx.block_path_params, dict) else None
    )
    if not isinstance(path_param, str):
        return None
    return base, {path_param: concrete_path}


def _known_output_arity(callee: str, ctx: _LowerCtx) -> int | None:
    resolved = _resolve_block_call(callee, ctx)
    if resolved is not None and ctx.block_signatures:
        block_name, _ = resolved
        _, output_names = ctx.block_signatures[block_name]
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

    conditional = _split_if_then_else(expr)
    if conditional is None:
        conditional = _split_ternary(expr)
    if conditional is not None:
        cond, true_expr, false_expr = conditional
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
            stage_args = list(args)
            if stage_args:
                if isinstance(pipe_ref, str):
                    if stage_args[0].strip() == pipe_ref:
                        stage_args = stage_args[1:]
                else:
                    n = len(piped_args)
                    if len(stage_args) >= n and all(
                        stage_args[i].strip() == piped_args[i] for i in range(n)
                    ):
                        stage_args = stage_args[n:]
            call_expr = _render_call(callee, [*piped_args, *stage_args], kwargs)
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


def _module_return_last_dims(module: AxonModule, returns: tuple[str, ...]) -> dict[str, Any]:
    if not returns or module.return_shape is None or len(module.return_shape) == 0:
        return {}
    if len(returns) != 1:
        return {}
    return {returns[0]: module.return_shape[-1]}


def _module_param_last_dims(module: AxonModule) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for param in module.params:
        if param.shape is None or len(param.shape) == 0:
            continue
        out[param.name] = param.shape[-1]
    return out


def _module_return_heads(module: AxonModule, returns: tuple[str, ...]) -> dict[str, Any]:
    if not returns or module.return_shape is None or len(module.return_shape) < 2:
        return {}
    if len(returns) != 1:
        return {}
    return {returns[0]: module.return_shape[1]}


def lower_axon_module_to_synapse_block(module: AxonModule) -> dict[str, Any]:
    inputs = {param.name: {"optional": param.optional} for param in module.params}
    if module.path_param is not None:
        inputs[module.path_param] = {"optional": False}
    graph: list[dict[str, Any]] = []
    outputs: dict[str, str] = {}
    returns = _module_return_names(module)
    initial_dims = {
        param.name: param.shape[-1]
        for param in module.params
        if param.shape is not None and len(param.shape) > 0
    }
    initial_heads = {
        param.name: param.shape[1]
        for param in module.params
        if param.shape is not None and len(param.shape) >= 2
    }
    initial_dims.update(_module_return_last_dims(module, returns))
    initial_heads.update(_module_return_heads(module, returns))
    ctx = _LowerCtx(
        block_signatures={},
        block_path_params={module.name: module.path_param},
        block_param_last_dims={module.name: _module_param_last_dims(module)},
        block_output_last_dims={module.name: _module_return_last_dims(module, returns)},
        tensor_last_dim=initial_dims,
        tensor_heads=initial_heads,
        path_param_names={p for p in [module.path_param] if isinstance(p, str)},
    )

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
    if module.symbols:
        model["symbols"] = dict(module.symbols)
    return {
        "synapse": 1,
        "model": model,
    }


def _lower_statements(
    *,
    statements: tuple[AxonStatement, ...],
    graph: list[dict[str, Any]],
    outputs: dict[str, str],
    returns: tuple[str, ...],
    ctx: _LowerCtx,
) -> None:
    for stmt in statements:
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
            if ctx.scope_stack:
                scope_prefix = ".".join(part for part in ctx.scope_stack if part)
                if scope_prefix:
                    repeat_name = f"{scope_prefix}.{repeat_name}"
            repeat_item: dict[str, Any] = {
                repeat_name.split(".")[-1]: {
                    "op": "repeat",
                    "var": stmt.var,
                    "range": stmt.range_expr,
                    "body": body_graph,
                }
            }
            if stmt.start_expr != "0":
                repeat_item[repeat_name.split(".")[-1]]["start"] = stmt.start_expr
            segments = [part for part in repeat_name.split(".") if part]
            for segment in reversed(segments[:-1]):
                repeat_item = {segment: {"graph": [repeat_item]}}
            graph.append(repeat_item)
            continue

        if isinstance(stmt, AxonScope):
            ctx.scope_stack.append(stmt.prefix)
            try:
                _lower_statements(
                    statements=stmt.body,
                    graph=graph,
                    outputs=outputs,
                    returns=returns,
                    ctx=ctx,
                )
            finally:
                ctx.scope_stack.pop()
            continue

        if isinstance(stmt, AxonScopeBind):
            ctx.scope_stack.append(stmt.prefix)
            scoped_outputs: dict[str, str] = {}
            try:
                _lower_statements(
                    statements=stmt.body,
                    graph=graph,
                    outputs=scoped_outputs,
                    returns=(),
                    ctx=ctx,
                )
            finally:
                ctx.scope_stack.pop()
            for idx, target in enumerate(stmt.targets):
                output_name = f"out_{idx}"
                if output_name not in scoped_outputs:
                    raise ValueError(
                        f"scope bind for {stmt.prefix!r} must return value {idx} via `return`"
                    )
                source_name = scoped_outputs[output_name]
                if target == source_name:
                    continue
                graph.extend(_lower_expr(source_name, target, ctx))
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
    block_path_params: dict[str, str | None] = {}
    block_param_last_dims: dict[str, dict[str, Any]] = {}
    block_output_last_dims: dict[str, dict[str, Any]] = {}
    for module in modules:
        input_names = [param.name for param in module.params]
        if module.path_param is not None:
            input_names.append(module.path_param)
        output_names = list(_module_return_names(module))
        signatures[module.name] = (
            input_names,
            output_names,
        )
        block_path_params[module.name] = module.path_param
        block_param_last_dims[module.name] = _module_param_last_dims(module)
        block_output_last_dims[module.name] = _module_return_last_dims(module, tuple(output_names))

    main = by_name[main_name]
    main_returns = _module_return_names(main)
    main_inputs = {param.name: {"optional": param.optional} for param in main.params}
    main_graph: list[dict[str, Any]] = []
    main_outputs: dict[str, str] = {}
    main_initial_dims = {
        param.name: param.shape[-1]
        for param in main.params
        if param.shape is not None and len(param.shape) > 0
    }
    main_initial_heads = {
        param.name: param.shape[1]
        for param in main.params
        if param.shape is not None and len(param.shape) >= 2
    }
    main_initial_dims.update(_module_return_last_dims(main, main_returns))
    main_initial_heads.update(_module_return_heads(main, main_returns))
    _lower_statements(
        statements=main.statements,
        graph=main_graph,
        outputs=main_outputs,
        returns=main_returns,
        ctx=_LowerCtx(
            block_signatures=signatures,
            block_path_params=block_path_params,
            block_param_last_dims=block_param_last_dims,
            block_output_last_dims=block_output_last_dims,
            tensor_last_dim=main_initial_dims,
            tensor_heads=main_initial_heads,
            path_param_names={p for p in [main.path_param] if isinstance(p, str)},
        ),
    )
    if not main_outputs:
        for name in main_returns:
            main_outputs[name] = name
    model: dict[str, Any] = {"inputs": main_inputs, "graph": main_graph, "outputs": main_outputs}
    if main.symbols:
        model["symbols"] = dict(main.symbols)
    spec: dict[str, Any] = {"synapse": 1, "model": model}

    blocks: dict[str, Any] = {}
    for module in modules:
        if module.name == main_name:
            continue
        block_inputs = {param.name: {"optional": param.optional} for param in module.params}
        if module.path_param is not None:
            block_inputs[module.path_param] = {"optional": False}
        block_returns = _module_return_names(module)
        block_graph: list[dict[str, Any]] = []
        block_outputs: dict[str, str] = {}
        block_initial_dims = {
            param.name: param.shape[-1]
            for param in module.params
            if param.shape is not None and len(param.shape) > 0
        }
        block_initial_heads = {
            param.name: param.shape[1]
            for param in module.params
            if param.shape is not None and len(param.shape) >= 2
        }
        block_initial_dims.update(_module_return_last_dims(module, block_returns))
        block_initial_heads.update(_module_return_heads(module, block_returns))
        _lower_statements(
            statements=module.statements,
            graph=block_graph,
            outputs=block_outputs,
            returns=block_returns,
            ctx=_LowerCtx(
                block_signatures=signatures,
                block_path_params=block_path_params,
                block_param_last_dims=block_param_last_dims,
                block_output_last_dims=block_output_last_dims,
                tensor_last_dim=block_initial_dims,
                tensor_heads=block_initial_heads,
                path_param_names={p for p in [module.path_param] if isinstance(p, str)},
            ),
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
