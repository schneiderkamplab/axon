from __future__ import annotations

import ast
import copy
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
_ZERO_ARG_CALLS = {"init_list", "init", "Cache.init", "List.init", "_list_init"}
_INVALID_POSITIONAL_TOKENS = {"+", "-", "*", "/", "%", "==", "!=", "<", ">", "<=", ">="}

_OP_ARITY: dict[str, tuple[int, int]] = {
    "embedding": (1, 1),
    "linear": (1, 1),
    "layernorm": (1, 1),
    "rmsnorm": (1, 1),
    "attention": (3, 3),
    "causal_mask": (2, 2),
    "reshape_heads": (1, 1),
    "split": (1, 1),
    "apply_rope_pair": (2, 2),
    "repeat_kv": (1, 2),
    "position_ids": (2, 2),
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
        "causal",
        "mask",
        "scale",
    },
    "causal_mask": {"window", "padding_mask"},
    "reshape_heads": {"heads", "head_dim"},
    "split": {"dim", "parts", "sizes"},
    "apply_rope_pair": {"position_ids", "theta"},
    "repeat_kv": {"heads", "kv_heads", "repeats", "times", "dim"},
    "position_ids": {"past_length"},
    "kv_cache_update": {"when"},
    "coalesce": set(),
    "topk": {"k", "dim", "largest", "sorted"},
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
        "causal": "bool",
        "mask": "str",
        "scale": "number",
    },
    "causal_mask": {"window": "dim", "padding_mask": "str"},
    "reshape_heads": {"heads": "dim", "head_dim": "dim"},
    "split": {"dim": "int", "parts": "int", "sizes": "list_dim"},
    "apply_rope_pair": {"position_ids": "str", "theta": "number"},
    "repeat_kv": {
        "heads": "dim",
        "kv_heads": "dim",
        "repeats": "dim",
        "times": "dim",
        "dim": "int",
    },
    "position_ids": {"past_length": "dim"},
    "kv_cache_update": {},
    "coalesce": {},
    "topk": {"k": "dim", "dim": "int", "largest": "bool", "sorted": "bool"},
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

_OP_REQUIRED_KWARGS: dict[str, set[str]] = {
    "topk": {"k"},
    "moe_select_tokens": {"expert"},
}

_SOFTMAX_SUPPORTED_DTYPES: set[str] = {"float32", "float16", "bfloat16"}
_ACTIVATION_KINDS: set[str] = {
    "gelu",
    "gelu_new",
    "gelu_pytorch_tanh",
    "relu",
    "silu",
    "swiglu",
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


def _strip_wrapping_parens(text: str) -> str:
    current = text.strip()
    while current.startswith("(") and current.endswith(")"):
        depth = 0
        valid = True
        for idx, ch in enumerate(current):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth < 0:
                    valid = False
                    break
                if depth == 0 and idx != len(current) - 1:
                    valid = False
                    break
        if not valid or depth != 0:
            break
        current = current[1:-1].strip()
    return current


def _render_call(callee: str, args: list[str], kwargs: dict[str, Any]) -> str:
    items = [*args, *[f"{k}={v}" for k, v in kwargs.items()]]
    return f"{callee}({', '.join(items)})"


def _to_synapse_op(
    callee: str,
    args: list[str],
    kwargs: dict[str, Any],
    out: str | list[str],
) -> dict[str, Any]:
    if callee in _ACTIVATION_KINDS:
        if len(args) != 1:
            raise ValueError(f"{callee} activation expects exactly one positional argument")
        activation_op: dict[str, Any] = {
            "_op": "activation",
            "_args": args[0],
            "_bind": out,
            "kind": callee,
        }
        for key, value in kwargs.items():
            activation_op[key] = value
        return activation_op
    if callee.startswith("_act_"):
        kind = callee[len("_act_") :]
        if not kind:
            raise ValueError("invalid primitive activation call: missing kind in _act_*")
        if len(args) != 1:
            raise ValueError("_act_* primitive requires exactly one positional argument")
        primitive_act: dict[str, Any] = {
            "_op": "activation",
            "_args": args[0],
            "_bind": out,
            "kind": kind,
        }
        for key, value in kwargs.items():
            primitive_act[key] = value
        return primitive_act
    if callee.startswith("_cache_"):
        cache_suffix = callee[len("_cache_") :]
        if cache_suffix == "update":
            return _to_synapse_op("cache::update", args, kwargs, out)
        if cache_suffix == "coalesce":
            return _to_synapse_op("cache::coalesce", args, kwargs, out)
        if cache_suffix == "seq_len":
            return _to_synapse_op("cache::seq_len", args, kwargs, out)
        raise ValueError(f"unsupported cache primitive alias: {callee!r}")
    if callee == "_repeat":
        return _to_synapse_op("repeat_kv", args, kwargs, out)
    if callee == "_list_init":
        return _to_synapse_op("init_list", args, kwargs, out)
    if callee == "_list_index":
        return _to_synapse_op("index", args, kwargs, out)
    if callee == "_list_append":
        return _to_synapse_op("append", args, kwargs, out)
    if callee == "_moe_select":
        return _to_synapse_op("moe_select_tokens", args, kwargs, out)
    if callee.startswith("_") and len(callee) > 1 and callee[1].isalpha():
        primitive = callee[1:]
        return _to_synapse_op(primitive, args, kwargs, out)

    if callee == "split":
        split_op: dict[str, Any] = {"_op": "split", "_bind": out}
        if args:
            split_op["_args"] = args[0] if len(args) == 1 else args
        for key, value in kwargs.items():
            split_op[key] = value
        return split_op

    if "@" in callee:
        op_name, _ = callee.split("@", 1)
        at_op: dict[str, Any] = {"_op": op_name, "_bind": out}
        if op_name == "embed":
            at_op["_op"] = "embedding"
        if args:
            at_op["_args"] = args[0] if len(args) == 1 else args
        for key, value in kwargs.items():
            at_op[key] = value
        return at_op

    if "::" in callee:
        ns, name = callee.split("::", 1)
        if ns == "act":
            return {"_op": "activation", "_args": args[0], "_bind": out, "kind": name}
        if ns == "cache" and name == "update":
            return {"_op": "kv_cache_update", "_args": args, "_bind": out}
        if ns == "cache" and name == "seq_len":
            return {"_op": "kv_seq_len", "_args": args[0], "_bind": out}
        if ns == "cache" and name == "coalesce":
            return {"_op": "coalesce", "_args": args, "_bind": out}

    default_op: dict[str, Any] = {"_op": callee, "_bind": out}
    if args:
        default_op["_args"] = args[0] if len(args) == 1 else args
    for key, value in kwargs.items():
        default_op[key] = value
    return default_op


def _canonical_op_name(callee: str) -> str:
    if callee in _ACTIVATION_KINDS:
        return "activation"
    if callee.startswith("_act_"):
        kind = callee[len("_act_") :]
        if not kind:
            raise ValueError("invalid primitive activation call: missing kind in _act_*")
        return "activation"
    if callee.startswith("_cache_"):
        cache_suffix = callee[len("_cache_") :]
        if cache_suffix == "update":
            return "kv_cache_update"
        if cache_suffix == "coalesce":
            return "coalesce"
        if cache_suffix == "seq_len":
            return "kv_seq_len"
        raise ValueError(f"unsupported cache primitive alias: {callee!r}")
    if callee == "_repeat":
        return "repeat_kv"
    if callee == "_list_init":
        return "init_list"
    if callee == "_list_index":
        return "index"
    if callee == "_list_append":
        return "append"
    if callee == "_moe_select":
        return "moe_select_tokens"
    if callee.startswith("_") and len(callee) > 1 and callee[1].isalpha():
        return _canonical_op_name(callee[1:])
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


def _is_symbolic_dim_token(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    token = value.strip()
    return re.fullmatch(r"-?[0-9]+", token) is None


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
    if kind == "list_dim":
        return isinstance(value, list) and all(
            (isinstance(v, int) and not isinstance(v, bool)) or isinstance(v, str) for v in value
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
    required = _OP_REQUIRED_KWARGS.get(op_name)
    if required:
        missing = sorted(required - set(kwargs))
        if missing:
            raise ValueError(f"{op_name} missing required kwargs: {', '.join(missing)}")
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
    block_path_params: dict[str, tuple[str, ...]] | None = None
    block_param_last_dims: dict[str, dict[str, Any]] | None = None
    block_output_last_dims: dict[str, dict[str, Any]] | None = None
    block_param_shapes: dict[str, dict[str, tuple[str, ...]]] | None = None
    block_output_shapes: dict[str, dict[str, tuple[str, ...]]] | None = None
    tensor_last_dim: dict[str, Any] = field(default_factory=dict)
    tensor_heads: dict[str, Any] = field(default_factory=dict)
    tensor_shape: dict[str, tuple[Any, ...]] = field(default_factory=dict)
    scope_stack: list[str] = field(default_factory=list)
    path_param_names: set[str] = field(default_factory=set)
    imported_namespaces: set[str] = field(default_factory=set)
    imported_member_namespaces: dict[str, set[str]] = field(default_factory=dict)
    prelude_aliases: dict[str, tuple[str, int]] = field(default_factory=dict)
    primitive_aliases: dict[str, tuple[str, int]] = field(default_factory=dict)
    current_module: str | None = None

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
    _validate_namespaced_block_call(callee, ctx)
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
        output_shapes = (
            ctx.block_output_shapes.get(block_name, {})
            if isinstance(ctx.block_output_shapes, dict)
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
            shape_tokens = output_shapes.get(output_name)
            if shape_tokens is not None:
                resolved_shape = tuple(symbol_bindings.get(tok, tok) for tok in shape_tokens)
                ctx.tensor_shape[target] = resolved_shape

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
        if op_name == "split":
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
        last_dim = kwargs.get("dim")
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


def _normalize_embedding_kwargs(kwargs: dict[str, Any]) -> None:
    if "embedding_dim" in kwargs:
        raise ValueError("embedding does not support embedding_dim; use dim")
    allowed_embedding_kwargs = {"dim", "scale"}
    invalid_embedding_kwargs = sorted(set(kwargs) - allowed_embedding_kwargs)
    if invalid_embedding_kwargs:
        bad = ", ".join(invalid_embedding_kwargs)
        raise ValueError(f"embedding unsupported kwargs: {bad}; allowed: dim, scale")
    # Keep Synapse/YAML key aligned with Axon surface naming (`dim`).


def _normalize_linear_kwargs(kwargs: dict[str, Any]) -> None:
    if "weight_layout" in kwargs:
        raise ValueError("linear does not support weight_layout; use transpose=true/false")
    if "tie_weight" in kwargs:
        raise ValueError("linear does not support tie_weight; use linear@<path>")
    if "out_features" in kwargs:
        raise ValueError("linear does not support out_features; use dim")
    if "out_dim" in kwargs:
        raise ValueError("linear does not support out_dim; use dim")
    if "transpose" not in kwargs:
        return
    raw_transpose = kwargs["transpose"]
    if isinstance(raw_transpose, bool):
        return
    if isinstance(raw_transpose, str) and raw_transpose.lower() in {"true", "false"}:
        kwargs["transpose"] = raw_transpose.lower() == "true"
        return
    raise ValueError("linear transpose must be true/false")


def _infer_norm_dim_from_input(
    op_name: str, args: list[str], kwargs: dict[str, Any], ctx: _LowerCtx
) -> None:
    if op_name not in {"layernorm", "rmsnorm"} or "dim" in kwargs or not args:
        return
    first_arg = args[0].strip()
    if not _is_name_token(first_arg):
        return
    inferred = ctx.tensor_last_dim.get(first_arg)
    if inferred is not None:
        kwargs["dim"] = inferred


def _infer_linear_dim_from_output(
    op_name: str, out: str | list[str], kwargs: dict[str, Any], ctx: _LowerCtx
) -> None:
    if op_name != "linear" or "dim" in kwargs or not isinstance(out, str):
        return
    inferred = ctx.tensor_last_dim.get(out)
    if inferred is not None:
        kwargs["dim"] = inferred


def _infer_embedding_dim_from_output(
    op_name: str, out: str | list[str], kwargs: dict[str, Any], ctx: _LowerCtx
) -> None:
    if op_name != "embedding" or "dim" in kwargs or not isinstance(out, str):
        return
    inferred = ctx.tensor_last_dim.get(out)
    if inferred is not None:
        kwargs["dim"] = inferred


def _normalize_split_kwargs(
    op_name: str,
    args: list[str],
    out: str | list[str],
    kwargs: dict[str, Any],
    ctx: _LowerCtx,
) -> None:
    if op_name != "split":
        return
    dim = kwargs.get("dim", -1)
    if dim != -1:
        raise ValueError("split currently supports only dim=-1 (last axis)")
    kwargs.pop("dim", None)
    if not isinstance(out, list):
        raise ValueError(f"{op_name} requires tuple/list binding outputs")
    has_parts = "parts" in kwargs
    has_sizes = "sizes" in kwargs
    if has_parts and has_sizes:
        raise ValueError(f"{op_name} accepts either parts or sizes, not both")
    if has_sizes and isinstance(kwargs["sizes"], str):
        parsed_sizes = _maybe_int_list(kwargs["sizes"])
        if parsed_sizes is not None:
            kwargs["sizes"] = parsed_sizes
    if not has_parts and not has_sizes and args:
        first_arg = args[0].strip()
        if _is_name_token(first_arg):
            inferred = ctx.tensor_last_dim.get(first_arg)
            split_sizes = _infer_split_sizes_from_last_dim(inferred, len(out))
            if split_sizes is not None:
                kwargs["sizes"] = split_sizes
                has_sizes = True
    if has_parts:
        parts_raw = kwargs["parts"]
        if not isinstance(parts_raw, int) or isinstance(parts_raw, bool) or parts_raw <= 0:
            raise ValueError(f"{op_name} parts must be a positive integer")
        if len(out) != parts_raw:
            raise ValueError(
                f"{op_name} parts={parts_raw} requires {parts_raw} outputs, got {len(out)}"
            )
    if has_sizes:
        sizes_raw = kwargs["sizes"]
        if not isinstance(sizes_raw, list) or len(sizes_raw) == 0:
            raise ValueError(f"{op_name} sizes must be a non-empty list")
        if not all(
            (isinstance(v, int) and not isinstance(v, bool)) or isinstance(v, str)
            for v in sizes_raw
        ):
            raise ValueError(f"{op_name} sizes must contain only ints or symbolic dims")
        if len(out) != len(sizes_raw):
            raise ValueError(
                f"{op_name} sizes length {len(sizes_raw)} requires {len(sizes_raw)} outputs, got {len(out)}"
            )


def _infer_repeat_kv_kwargs(
    op_name: str,
    args: list[str],
    out: str | list[str],
    kwargs: dict[str, Any],
    ctx: _LowerCtx,
) -> None:
    if op_name != "repeat_kv" or not args:
        return
    if "times" in kwargs and "repeats" not in kwargs:
        kwargs["repeats"] = kwargs.pop("times")
    if len(args) >= 2 and "repeats" not in kwargs:
        kwargs["repeats"] = args[1]
        del args[1:]
    dim = kwargs.pop("dim", 1)
    if dim != 1:
        raise ValueError("repeat currently supports only dim=1 (head axis)")
    if "repeats" in kwargs:
        return
    src_name = args[0].strip()
    if not _is_name_token(src_name):
        return
    if "kv_heads" not in kwargs:
        inferred_kv_heads = ctx.tensor_heads.get(src_name)
        if inferred_kv_heads is not None:
            kwargs["kv_heads"] = inferred_kv_heads
    if "heads" not in kwargs and isinstance(out, str):
        inferred_heads = ctx.tensor_heads.get(out)
        if inferred_heads is not None:
            kwargs["heads"] = inferred_heads


def _normalize_coalesce_call(op_name: str, args: list[str], out: str | list[str]) -> None:
    if op_name != "coalesce":
        return
    if not isinstance(out, list) or len(out) == 0:
        raise ValueError("coalesce requires tuple/list binding outputs")
    if len(args) % len(out) != 0:
        raise ValueError("coalesce input count must be divisible by output count")
    for arg in args:
        if not _is_name_token(arg.strip()):
            raise ValueError("coalesce inputs must be variable names")


def _normalize_call_kwargs(
    *,
    op_name: str,
    args: list[str],
    out: str | list[str],
    kwargs: dict[str, Any],
    ctx: _LowerCtx,
) -> dict[str, Any]:
    normalized = dict(kwargs)
    passes = [
        lambda: _normalize_embedding_kwargs(normalized) if op_name == "embedding" else None,
        lambda: _normalize_linear_kwargs(normalized) if op_name == "linear" else None,
        lambda: _infer_norm_dim_from_input(op_name, args, normalized, ctx),
        lambda: _infer_linear_dim_from_output(op_name, out, normalized, ctx),
        lambda: _infer_embedding_dim_from_output(op_name, out, normalized, ctx),
        lambda: _normalize_split_kwargs(op_name, args, out, normalized, ctx),
        lambda: _normalize_coalesce_call(op_name, args, out),
        lambda: _infer_repeat_kv_kwargs(op_name, args, out, normalized, ctx),
    ]
    for normalize in passes:
        normalize()
    return normalized


def _validate_normalized_kwargs(op_name: str, kwargs: dict[str, Any], args: list[str]) -> None:
    _validate_op_signature(op_name, args, kwargs)


def _lower_simple_call(
    expr: str, out: str | list[str], ctx: _LowerCtx, *, when: str | None = None
) -> list[dict[str, Any]]:
    callee, args, kwargs = _parse_call(expr)
    callee = _rewrite_prelude_alias_callee(callee, kwargs, ctx)
    callee = _rewrite_primitive_alias_callee(callee, kwargs, ctx)
    pre_graph: list[dict[str, Any]] = []
    effective_when = _merge_when(when, kwargs.pop("when", None))

    resolved_args: list[str] = []
    raw_op_name = _op_name_from_callee(callee)
    for idx, arg in enumerate(args):
        token = arg.strip()
        inner = token
        if token.startswith("(") and token.endswith(")"):
            inner = token[1:-1].strip()
        if inner != token and (not _is_name_token(inner)):
            if raw_op_name in {"repeat", "_repeat", "repeat_kv"} and idx >= 1:
                resolved_args.append(inner)
                continue
            tmp = ctx.fresh("arg")
            pre_graph.extend(_lower_expr(inner, tmp, ctx, when=when))
            resolved_args.append(tmp)
        else:
            resolved_args.append(token)
    args = resolved_args

    resolved_kwargs: dict[str, Any] = {}
    for key, value in kwargs.items():
        if isinstance(value, str):
            stripped_value = _strip_wrapping_parens(value)
            if _looks_like_call(stripped_value):
                tmp = ctx.fresh("kwarg")
                pre_graph.extend(_lower_expr(stripped_value, tmp, ctx, when=when))
                resolved_kwargs[key] = tmp
                continue
        resolved_kwargs[key] = value
    kwargs = resolved_kwargs

    # Canonicalize positional expert for the primitive MoE selector.
    if callee == "_moe_select" and "expert" not in kwargs and len(args) >= 4:
        kwargs["expert"] = args[3]
        args = args[:3]

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
        scope_prefix = ".".join(part for part in ctx.scope_stack if part)
        if scope_prefix:
            parts = callee.split("@")
            base = parts[0]
            if ctx.block_signatures and base in ctx.block_signatures and len(parts) > 1:
                scoped_paths = [
                    f"{scope_prefix}.{param_path}" if param_path.strip() else scope_prefix
                    for param_path in parts[1:]
                ]
                callee = "@".join([base, *scoped_paths])
            else:
                op_name_with_at, param_path = callee.split("@", 1)
                scoped_path = f"{scope_prefix}.{param_path}" if param_path.strip() else scope_prefix
                callee = f"{op_name_with_at}@{scoped_path}"
    op_name = _canonical_op_name(callee)
    kwargs = _normalize_call_kwargs(op_name=op_name, args=args, out=out, kwargs=kwargs, ctx=ctx)
    if op_name == "topk":
        if not isinstance(out, list) or len(out) != 2:
            raise ValueError("topk requires exactly two outputs: values, indices")
    if op_name == "moe_select_tokens":
        if not isinstance(out, list) or len(out) != 4:
            raise ValueError(
                "moe_select_tokens requires exactly four outputs: "
                "selected_hidden, token_idx, topk_pos, selected_scores"
            )
    if op_name == "moe_scatter_add":
        if not isinstance(out, str):
            raise ValueError("moe_scatter_add requires a single scalar output binding")
    if op_name == "softmax":
        if not isinstance(out, str):
            raise ValueError("softmax requires a single scalar output binding")
        dtype_name = kwargs.get("dtype")
        if dtype_name is not None:
            if not isinstance(dtype_name, str):
                raise ValueError("softmax dtype must be a string when provided")
            if dtype_name not in _SOFTMAX_SUPPORTED_DTYPES:
                supported = ", ".join(sorted(_SOFTMAX_SUPPORTED_DTYPES))
                raise ValueError(
                    f"Unsupported softmax dtype: {dtype_name} (supported: {supported})"
                )
    if op_name == "zeros_like":
        if not isinstance(out, str):
            raise ValueError("zeros_like requires a single scalar output binding")
    if op_name in {"add", "mul"}:
        if not isinstance(out, str):
            raise ValueError(f"{op_name} requires a single scalar output binding")
    _validate_normalized_kwargs(op_name, kwargs, args)
    resolved_block = _resolve_block_call(callee, ctx)
    if resolved_block is None and "." in callee and "@" not in callee and "::" not in callee:
        namespace = callee.split(".", 1)[0]
        raise ValueError(
            f"unknown namespaced module call {callee!r}; add `import {namespace}` and parse from file"
        )
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
        if isinstance(ctx.block_param_shapes, dict):
            param_shapes = ctx.block_param_shapes.get(block_name, {})
        else:
            param_shapes = {}
        symbol_bindings: dict[str, Any] = {}
        for param_name, raw in provided.items():
            token = str(raw).strip()
            if _is_name_token(token) and token in ctx.tensor_last_dim:
                symbol_bindings[param_name] = ctx.tensor_last_dim[token]
            elif not _is_name_token(token):
                symbol_bindings[param_name] = _parse_scalar(token)
        for param_name, param_shape in param_shapes.items():
            if param_name not in provided:
                continue
            token = str(provided[param_name]).strip()
            if not _is_name_token(token):
                continue
            arg_shape = ctx.tensor_shape.get(token)
            if arg_shape is None:
                continue
            if len(arg_shape) != len(param_shape):
                raise ValueError(
                    f"shape mismatch in call {callee!r} for param {param_name!r}: "
                    f"expected rank {len(param_shape)} from signature {param_shape}, got rank {len(arg_shape)} from {arg_shape}"
                )
            for sym, actual in zip(param_shape, arg_shape, strict=True):
                if _is_symbolic_dim_token(sym) and sym not in symbol_bindings:
                    symbol_bindings[sym] = actual
            expected_shape = tuple(symbol_bindings.get(sym, sym) for sym in param_shape)
            if len(expected_shape) != len(arg_shape) or any(
                not _dims_compatible(exp, got)
                for exp, got in zip(expected_shape, arg_shape, strict=True)
            ):
                raise ValueError(
                    f"shape mismatch in call {callee!r} for param {param_name!r}: "
                    f"expected {expected_shape} from signature, got {arg_shape} from argument {token!r}"
                )
        out_values = [out] if isinstance(out, str) else list(out)
        if len(out_values) != len(output_names):
            raise ValueError(
                f"block call {callee!r} expects {len(output_names)} outputs, got {len(out_values)}"
            )
        positional_args: list[str] = []
        extra_kwargs: dict[str, str] = {}
        for input_name in input_names:
            if input_name not in provided:
                continue
            value = provided[input_name]
            if input_name in kwargs or input_name in path_bindings:
                extra_kwargs[input_name] = value
            elif len(positional_args) < len(args):
                positional_args.append(value)
            else:
                extra_kwargs[input_name] = value
        node_name = f"n_{ctx.fresh('call')}"
        node_spec: dict[str, Any] = {"_op": "call", "_target": block_name}
        if positional_args:
            node_spec["_args"] = (
                positional_args[0] if len(positional_args) == 1 else positional_args
            )
        node_spec["_bind"] = out_values[0] if len(out_values) == 1 else out_values
        for key, value in extra_kwargs.items():
            node_spec[key] = value
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
        concrete_node = _to_synapse_op(op_name, args, kwargs, out)
        try:
            bound_params = _path_bound_param_names(concrete_node)
        except ValueError:
            bound_params = []
        if bound_params:
            if not param_path.strip():
                raise ValueError(f"invalid @ path in Axon call: {expr!r}")
            node_name = f"n_{ctx.fresh('op')}"
            params = {param_name: f"{param_path}.{param_name}" for param_name in bound_params}
            concrete_node["_params"] = params
            nodes = _with_when([{node_name: concrete_node}], effective_when)
            _record_last_dim_for_call(callee=callee, args=args, kwargs=kwargs, out=out, ctx=ctx)
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
        node = {"_op": "_ir_alias", "_args": token, "_bind": out}
        if token in ctx.tensor_last_dim:
            ctx.tensor_last_dim[out] = ctx.tensor_last_dim[token]
    else:
        node = {"_op": "_ir_const", "value": scalar, "_bind": out}
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
    if "@" not in callee and "." not in callee and "::" not in callee:
        imported_namespaces = ctx.imported_member_namespaces.get(callee, set())
        if imported_namespaces:
            if len(imported_namespaces) > 1:
                choices = ", ".join(sorted(imported_namespaces))
                raise ValueError(
                    f"ambiguous imported member {callee!r}; found in namespaces: {choices}"
                )
            namespace = next(iter(imported_namespaces))
            namespaced_callee = f"{namespace}.{callee}"
            if namespaced_callee in ctx.block_signatures:
                return namespaced_callee, {}
            raise ValueError(
                f"imported member {callee!r} from {namespace!r} not found as module "
                f"{namespaced_callee!r}"
            )
    if "@" not in callee:
        return None
    parts = callee.split("@")
    base = parts[0]
    concrete_paths = parts[1:]
    if base not in ctx.block_signatures:
        return None
    path_params = (
        ctx.block_path_params.get(base, ()) if isinstance(ctx.block_path_params, dict) else ()
    )
    if not path_params:
        return None
    if len(concrete_paths) != len(path_params):
        raise ValueError(
            f"block call {callee!r} expects {len(path_params)} @path arguments, got {len(concrete_paths)}"
        )
    return base, {
        path_param: concrete
        for path_param, concrete in zip(path_params, concrete_paths, strict=True)
    }


def _validate_namespaced_block_call(callee: str, ctx: _LowerCtx) -> None:
    if "." not in callee or "@" in callee or "::" in callee:
        return
    if not ctx.block_signatures or callee not in ctx.block_signatures:
        return
    namespace = callee.split(".", 1)[0].strip()
    if not namespace:
        return
    if namespace in ctx.imported_namespaces:
        return
    if isinstance(ctx.current_module, str) and ctx.current_module.startswith(f"{namespace}."):
        return
    raise ValueError(f"namespaced call {callee!r} requires `import {namespace}` in the Axon source")


def _rewrite_prelude_alias_callee(callee: str, kwargs: dict[str, Any], ctx: _LowerCtx) -> str:
    if not ctx.prelude_aliases:
        return callee
    if "::" in callee:
        return callee
    parts = callee.split("@")
    base = parts[0]
    path_parts = parts[1:]

    member_name: str | None = None
    if "." not in base:
        member_name = base
    elif base.startswith("Prelude."):
        member_name = base.split(".", 1)[1]
    if not member_name:
        return callee

    alias = ctx.prelude_aliases.get(member_name)
    if alias is None:
        return callee
    imported_for_member = ctx.imported_member_namespaces.get(member_name, set())
    if imported_for_member and imported_for_member != {"Prelude"}:
        return callee
    target_base, expected_path_count = alias
    if expected_path_count != len(path_parts):
        if expected_path_count == 0 and not path_parts:
            return target_base
        if expected_path_count > 0 and not path_parts:
            return target_base
        raise ValueError(
            f"Prelude alias {member_name!r} expects {expected_path_count} @path arguments, got {len(path_parts)}"
        )
    if path_parts:
        return "@".join([target_base, *path_parts])
    return target_base


def _rewrite_primitive_alias_callee(callee: str, kwargs: dict[str, Any], ctx: _LowerCtx) -> str:
    if not ctx.primitive_aliases:
        return callee
    if "::" in callee:
        return callee
    parts = callee.split("@")
    base = parts[0]
    path_parts = parts[1:]

    full_name: str | None = None
    if "." in base:
        full_name = base
    else:
        imported_for_member = ctx.imported_member_namespaces.get(base, set())
        if len(imported_for_member) == 1:
            namespace = next(iter(imported_for_member))
            full_name = f"{namespace}.{base}"
    if not full_name:
        return callee

    alias = ctx.primitive_aliases.get(full_name)
    if alias is None:
        return callee
    target_base, expected_path_count = alias
    if expected_path_count != len(path_parts):
        if expected_path_count == 0 and not path_parts:
            return target_base
        if expected_path_count > 0 and not path_parts:
            return target_base
        raise ValueError(
            f"primitive alias {full_name!r} expects {expected_path_count} @path arguments, got {len(path_parts)}"
        )
    if path_parts:
        return "@".join([target_base, *path_parts])
    return target_base


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


def _infer_split_arity(kwargs: dict[str, Any]) -> int | None:
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
    if callee == "split":
        arity = _infer_split_arity(kwargs)
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


def _module_param_shapes(module: AxonModule) -> dict[str, tuple[str, ...]]:
    out: dict[str, tuple[str, ...]] = {}
    for param in module.params:
        if param.shape is None:
            continue
        out[param.name] = tuple(param.shape)
    return out


def _module_return_shapes(
    module: AxonModule, returns: tuple[str, ...]
) -> dict[str, tuple[str, ...]]:
    if not returns or module.return_shape is None:
        return {}
    if len(returns) != 1:
        return {}
    return {returns[0]: tuple(module.return_shape)}


def _module_return_heads(module: AxonModule, returns: tuple[str, ...]) -> dict[str, Any]:
    if not returns or module.return_shape is None or len(module.return_shape) < 2:
        return {}
    if len(returns) != 1:
        return {}
    return {returns[0]: module.return_shape[1]}


def _module_inputs(module: AxonModule) -> dict[str, dict[str, bool]]:
    inputs = {param.name: {"optional": param.optional} for param in module.params}
    for path_param in module.path_params:
        inputs[path_param] = {"optional": False}
    if not module.path_params and module.path_param is not None:
        inputs[module.path_param] = {"optional": False}
    return inputs


def _module_initial_dims(module: AxonModule, returns: tuple[str, ...]) -> dict[str, Any]:
    initial_dims = {
        param.name: param.shape[-1]
        for param in module.params
        if param.shape is not None and len(param.shape) > 0
    }
    initial_dims.update(_module_return_last_dims(module, returns))
    return initial_dims


def _module_initial_shapes(
    module: AxonModule, returns: tuple[str, ...]
) -> dict[str, tuple[Any, ...]]:
    initial_shapes = {
        param.name: tuple(param.shape) for param in module.params if param.shape is not None
    }
    initial_shapes.update(_module_return_shapes(module, returns))
    return initial_shapes


def _module_initial_heads(module: AxonModule, returns: tuple[str, ...]) -> dict[str, Any]:
    initial_heads = {
        param.name: param.shape[1]
        for param in module.params
        if param.shape is not None and len(param.shape) >= 2
    }
    initial_heads.update(_module_return_heads(module, returns))
    return initial_heads


def _module_path_param_names(module: AxonModule) -> set[str]:
    names = {p for p in module.path_params if isinstance(p, str)}
    if not names and isinstance(module.path_param, str):
        names.add(module.path_param)
    return names


def _ensure_outputs_from_returns(outputs: dict[str, str], returns: tuple[str, ...]) -> None:
    if outputs:
        return
    for name in returns:
        outputs[name] = name


def _extract_primitive_aliases(modules: tuple[AxonModule, ...]) -> dict[str, tuple[str, int]]:
    direct_aliases: dict[str, tuple[str, int]] = {}
    for module in modules:
        if not isinstance(module.name, str) or "." not in module.name:
            continue
        if len(module.statements) != 1:
            continue
        stmt = module.statements[0]
        if not isinstance(stmt, AxonReturn) or len(stmt.values) != 1:
            continue
        expr = stmt.values[0].strip()
        try:
            callee, _, _ = _parse_call(expr)
        except ValueError:
            continue
        target_base = callee.split("@", 1)[0]
        direct_aliases[module.name] = (target_base, len(module.path_params))

    aliases: dict[str, tuple[str, int]] = {}
    for name, (target_base, expected_path_count) in direct_aliases.items():
        seen: set[str] = set()
        resolved = target_base
        while not resolved.startswith("_"):
            if resolved in seen:
                break
            seen.add(resolved)
            next_alias = direct_aliases.get(resolved)
            if next_alias is None:
                break
            next_base, next_path_count = next_alias
            if next_path_count != expected_path_count:
                break
            resolved = next_base
        if resolved.startswith("_"):
            aliases[name] = (resolved, expected_path_count)
    return aliases


def _extract_prelude_aliases(modules: tuple[AxonModule, ...]) -> dict[str, tuple[str, int]]:
    aliases: dict[str, tuple[str, int]] = {}
    primitive_aliases = _extract_primitive_aliases(modules)
    for full_name, alias in primitive_aliases.items():
        if not full_name.startswith("Prelude."):
            continue
        member_name = full_name.split(".", 1)[1]
        if not member_name:
            continue
        aliases[member_name] = alias
    return aliases


def _new_lower_ctx(
    *,
    module: AxonModule,
    returns: tuple[str, ...],
    signatures: dict[str, tuple[list[str], list[str]]] | None,
    block_path_params: dict[str, tuple[str, ...]] | None,
    block_param_last_dims: dict[str, dict[str, Any]] | None,
    block_output_last_dims: dict[str, dict[str, Any]] | None,
    block_param_shapes: dict[str, dict[str, tuple[str, ...]]] | None = None,
    block_output_shapes: dict[str, dict[str, tuple[str, ...]]] | None = None,
    implicit_prelude_members: set[str] | None = None,
    prelude_aliases: dict[str, tuple[str, int]] | None = None,
    primitive_aliases: dict[str, tuple[str, int]] | None = None,
) -> _LowerCtx:
    imported_member_namespaces: dict[str, set[str]] = {}
    if module.imported_members:
        for namespace, members in module.imported_members.items():
            for member in members:
                bucket = imported_member_namespaces.setdefault(member, set())
                bucket.add(namespace)
    if implicit_prelude_members:
        for member in implicit_prelude_members:
            if member in imported_member_namespaces:
                continue
            bucket = imported_member_namespaces.setdefault(member, set())
            bucket.add("Prelude")
    return _LowerCtx(
        block_signatures=signatures,
        block_path_params=block_path_params,
        block_param_last_dims=block_param_last_dims,
        block_output_last_dims=block_output_last_dims,
        block_param_shapes=block_param_shapes,
        block_output_shapes=block_output_shapes,
        tensor_last_dim=_module_initial_dims(module, returns),
        tensor_heads=_module_initial_heads(module, returns),
        tensor_shape=_module_initial_shapes(module, returns),
        path_param_names=_module_path_param_names(module),
        imported_namespaces=set(module.imports) | {"Prelude"},
        imported_member_namespaces=imported_member_namespaces,
        prelude_aliases=dict(prelude_aliases or {}),
        primitive_aliases=dict(primitive_aliases or {}),
        current_module=module.name,
    )


def lower_axon_module_to_synapse_block(module: AxonModule) -> dict[str, Any]:
    inputs = _module_inputs(module)
    graph: list[dict[str, Any]] = []
    outputs: dict[str, str] = {}
    returns = _module_return_names(module)
    ctx = _new_lower_ctx(
        module=module,
        returns=returns,
        signatures={},
        block_path_params={
            module.name: module.path_params
            if module.path_params
            else tuple([module.path_param] if module.path_param is not None else [])
        },
        block_param_last_dims={module.name: _module_param_last_dims(module)},
        block_output_last_dims={module.name: _module_return_last_dims(module, returns)},
        block_param_shapes={module.name: _module_param_shapes(module)},
        block_output_shapes={module.name: _module_return_shapes(module, returns)},
        implicit_prelude_members=set(),
        prelude_aliases={},
    )

    _lower_statements(
        statements=module.statements,
        graph=graph,
        outputs=outputs,
        returns=returns,
        ctx=ctx,
    )

    _ensure_outputs_from_returns(outputs, returns)

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
            node_name = f"n_{ctx.fresh('for')}"
            base_loop_scope = stmt.name if isinstance(stmt.name, str) and stmt.name else node_name
            loop_scope = base_loop_scope
            if ctx.scope_stack:
                scope_prefix = ".".join(part for part in ctx.scope_stack if part)
                if scope_prefix:
                    loop_scope = f"{scope_prefix}.{base_loop_scope}"
            repeat_item: dict[str, Any] = {
                node_name: {
                    "_op": "for",
                    "_scope": loop_scope,
                    "_var": stmt.var,
                    "_to": stmt.to_expr,
                    "_body": body_graph,
                }
            }
            if stmt.from_expr != "0":
                repeat_item[node_name]["_from"] = stmt.from_expr
            if stmt.step_expr != "1":
                repeat_item[node_name]["_step"] = stmt.step_expr
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


def _as_concrete_path(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    token = value.strip()
    if not token:
        return None
    try:
        parsed = ast.literal_eval(token)
    except Exception:
        parsed = token
    if isinstance(parsed, str) and parsed.strip():
        return parsed.strip()
    return None


def _sanitize_path_suffix(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_")
    return sanitized or "path"


def _path_bound_param_names(node_spec: dict[str, Any]) -> list[str]:
    op = node_spec.get("_op")
    if op == "embedding":
        return ["weight"]
    if op == "linear":
        names = ["weight"]
        if bool(node_spec.get("bias", False)):
            names.append("bias")
        return names
    if op == "rmsnorm":
        return ["weight"]
    if op == "layernorm":
        return ["weight", "bias"]
    raise ValueError(f"unsupported param_base resolution for op {op!r}")


def _resolve_paths_at_lowering_time(
    model: dict[str, Any], block_path_params: dict[str, tuple[str, ...]]
) -> None:
    blocks_raw = model.get("blocks")
    if not isinstance(blocks_raw, dict) or not blocks_raw:
        return

    base_blocks: dict[str, dict[str, Any]] = {
        name: copy.deepcopy(spec) for name, spec in blocks_raw.items() if isinstance(spec, dict)
    }
    all_blocks: dict[str, dict[str, Any]] = dict(base_blocks)
    path_params_by_block: dict[str, tuple[str, ...]] = {
        name: tuple(p for p in block_path_params.get(name, ()) if isinstance(p, str) and p)
        for name in base_blocks
    }
    specialization_cache: dict[tuple[str, tuple[tuple[str, str], ...]], str] = {}

    def _to_list(value: Any) -> list[Any]:
        if isinstance(value, list):
            return list(value)
        if value is None:
            return []
        return [value]

    def _substitute_names(value: Any, mapping: dict[str, Any]) -> Any:
        if isinstance(value, str):
            return mapping.get(value, value)
        if isinstance(value, list):
            return [_substitute_names(item, mapping) for item in value]
        if isinstance(value, dict):
            return {k: _substitute_names(v, mapping) for k, v in value.items()}
        return value

    def _map_call_inputs_to_values(
        *, call_spec: dict[str, Any], block_spec: dict[str, Any]
    ) -> dict[str, Any] | None:
        inputs = block_spec.get("inputs")
        if not isinstance(inputs, dict):
            return None
        ordered_inputs = list(inputs.keys())
        positional = _to_list(call_spec.get("_args"))
        kwargs = {
            key: value
            for key, value in call_spec.items()
            if isinstance(key, str) and not key.startswith("_")
        }
        env: dict[str, Any] = {}
        for idx, name in enumerate(ordered_inputs):
            if idx < len(positional):
                env[name] = positional[idx]
                continue
            if name in kwargs:
                env[name] = kwargs[name]
                continue
            return None
        return env

    def _bind_map_for_inline(*, template_bind: Any, call_bind: Any) -> dict[str, Any] | None:
        template = _to_list(template_bind)
        target = _to_list(call_bind)
        if len(template) != len(target):
            return None
        return {
            str(src): dst
            for src, dst in zip(template, target, strict=False)
            if isinstance(src, str)
        }

    def _inline_simple_call_if_possible(
        *, call_spec: dict[str, Any], target: str, concrete_paths: dict[str, str]
    ) -> dict[str, Any] | None:
        block_spec = base_blocks.get(target)
        if not isinstance(block_spec, dict):
            return None
        graph = block_spec.get("graph")
        if not isinstance(graph, list) or len(graph) != 1:
            return None
        node_item = graph[0]
        if not isinstance(node_item, dict) or len(node_item) != 1:
            return None
        _, template_spec = next(iter(node_item.items()))
        if not isinstance(template_spec, dict):
            return None
        template_op = template_spec.get("_op")
        if not isinstance(template_op, str) or template_op in {"call", "for"}:
            return None

        input_values = _map_call_inputs_to_values(call_spec=call_spec, block_spec=block_spec)
        if input_values is None:
            return None
        input_values.update(concrete_paths)

        bind_map = _bind_map_for_inline(
            template_bind=template_spec.get("_bind"),
            call_bind=call_spec.get("_bind"),
        )
        if bind_map is None:
            return None
        input_values.update(bind_map)

        inlined = copy.deepcopy(template_spec)
        inlined = _substitute_names(inlined, input_values)

        param_base = inlined.get("param_base")
        if isinstance(param_base, str):
            explicit_params = inlined.get("_params")
            params = dict(explicit_params) if isinstance(explicit_params, dict) else {}
            for param_name in _path_bound_param_names(inlined):
                params.setdefault(param_name, f"{param_base}.{param_name}")
            inlined["_params"] = params
            inlined.pop("param_base", None)

        return inlined

    def ensure_specialized_block(block_name: str, path_bindings: dict[str, str]) -> str:
        cache_key = (block_name, tuple(sorted(path_bindings.items())))
        cached = specialization_cache.get(cache_key)
        if cached is not None:
            return cached
        base_spec = base_blocks.get(block_name)
        if base_spec is None:
            return block_name
        specialized = copy.deepcopy(base_spec)
        inputs = specialized.get("inputs")
        if isinstance(inputs, dict):
            for path_name in path_bindings:
                inputs.pop(path_name, None)
        parts = [block_name]
        for key in sorted(path_bindings):
            parts.append(f"{key}_{_sanitize_path_suffix(path_bindings[key])}")
        base_name = "__".join(parts)
        candidate = base_name
        idx = 2
        while candidate in all_blocks:
            candidate = f"{base_name}_{idx}"
            idx += 1
        rewrite_graph(
            specialized.get("graph"),
            inherited_path_bindings=path_bindings,
        )
        all_blocks[candidate] = specialized
        path_params_by_block[candidate] = ()
        specialization_cache[cache_key] = candidate
        return candidate

    def rewrite_graph(graph: Any, *, inherited_path_bindings: dict[str, str]) -> None:
        if not isinstance(graph, list):
            return
        for item in graph:
            if not isinstance(item, dict) or len(item) != 1:
                continue
            _, node_spec = next(iter(item.items()))
            if not isinstance(node_spec, dict):
                continue

            param_base = node_spec.get("param_base")
            if isinstance(param_base, str) and param_base in inherited_path_bindings:
                base_path = inherited_path_bindings[param_base]
                explicit_params = node_spec.get("_params")
                params: dict[str, str]
                if isinstance(explicit_params, dict):
                    params = dict(explicit_params)
                else:
                    params = {}
                for param_name in _path_bound_param_names(node_spec):
                    params.setdefault(param_name, f"{base_path}.{param_name}")
                node_spec["_params"] = params
                node_spec.pop("param_base", None)

            if node_spec.get("_op") == "call":
                target = node_spec.get("_target")
                if isinstance(target, str):
                    required_path_params = path_params_by_block.get(target, ())
                    if required_path_params:
                        concrete: dict[str, str] = {}
                        for path_name in required_path_params:
                            concrete_value = _as_concrete_path(node_spec.get(path_name))
                            if concrete_value is None:
                                concrete = {}
                                break
                            concrete[path_name] = concrete_value
                        if concrete:
                            inlined = _inline_simple_call_if_possible(
                                call_spec=node_spec,
                                target=target,
                                concrete_paths=concrete,
                            )
                            if inlined is not None:
                                node_spec.clear()
                                node_spec.update(inlined)
                                target = None
                                required_path_params = ()
                                continue
                            specialized_name = ensure_specialized_block(target, concrete)
                            node_spec["_target"] = specialized_name
                            for path_name in required_path_params:
                                node_spec.pop(path_name, None)

            nested = node_spec.get("graph")
            if isinstance(nested, list):
                rewrite_graph(nested, inherited_path_bindings=inherited_path_bindings)
            body = node_spec.get("_body")
            if isinstance(body, list):
                rewrite_graph(body, inherited_path_bindings=inherited_path_bindings)

    rewrite_graph(model.get("graph"), inherited_path_bindings={})
    for block_spec in list(all_blocks.values()):
        rewrite_graph(block_spec.get("graph"), inherited_path_bindings={})

    reachable: set[str] = set()

    def collect_called_blocks(graph: Any) -> list[str]:
        called: list[str] = []
        if not isinstance(graph, list):
            return called
        for item in graph:
            if not isinstance(item, dict) or len(item) != 1:
                continue
            _, node_spec = next(iter(item.items()))
            if not isinstance(node_spec, dict):
                continue
            if node_spec.get("_op") == "call" and isinstance(node_spec.get("_target"), str):
                called.append(node_spec["_target"])
            nested = node_spec.get("graph")
            if isinstance(nested, list):
                called.extend(collect_called_blocks(nested))
            body = node_spec.get("_body")
            if isinstance(body, list):
                called.extend(collect_called_blocks(body))
        return called

    worklist = collect_called_blocks(model.get("graph"))
    while worklist:
        target = worklist.pop()
        if target in reachable:
            continue
        if target not in all_blocks:
            continue
        block_spec = all_blocks[target]
        reachable.add(target)
        worklist.extend(collect_called_blocks(block_spec.get("graph")))

    for block_name in sorted(reachable):
        block_spec = all_blocks[block_name]
        for item in block_spec.get("graph", []):
            if not isinstance(item, dict) or len(item) != 1:
                continue
            _, node_spec = next(iter(item.items()))
            if isinstance(node_spec, dict) and "param_base" in node_spec:
                raise ValueError(f"unresolved param_base in reachable block {block_name!r}")

    if reachable:
        model["blocks"] = {name: all_blocks[name] for name in all_blocks if name in reachable}
    else:
        model.pop("blocks", None)


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
    block_path_params: dict[str, tuple[str, ...]] = {}
    block_param_last_dims: dict[str, dict[str, Any]] = {}
    block_output_last_dims: dict[str, dict[str, Any]] = {}
    block_param_shapes: dict[str, dict[str, tuple[str, ...]]] = {}
    block_output_shapes: dict[str, dict[str, tuple[str, ...]]] = {}
    for module in modules:
        input_names = [param.name for param in module.params]
        path_params = module.path_params
        if not path_params and module.path_param is not None:
            path_params = (module.path_param,)
        input_names.extend(path_params)
        output_names = list(_module_return_names(module))
        signatures[module.name] = (input_names, output_names)
        block_path_params[module.name] = path_params
        block_param_last_dims[module.name] = _module_param_last_dims(module)
        block_output_last_dims[module.name] = _module_return_last_dims(module, tuple(output_names))
        block_param_shapes[module.name] = _module_param_shapes(module)
        block_output_shapes[module.name] = _module_return_shapes(module, tuple(output_names))
    primitive_aliases = _extract_primitive_aliases(modules)
    prelude_aliases = _extract_prelude_aliases(modules)
    implicit_prelude_members = {
        name.split(".", 1)[1]
        for name in signatures
        if isinstance(name, str) and name.startswith("Prelude.") and "." in name
    }

    main = by_name[main_name]
    main_returns = _module_return_names(main)
    main_inputs = _module_inputs(main)
    main_graph: list[dict[str, Any]] = []
    main_outputs: dict[str, str] = {}
    _lower_statements(
        statements=main.statements,
        graph=main_graph,
        outputs=main_outputs,
        returns=main_returns,
        ctx=_new_lower_ctx(
            module=main,
            returns=main_returns,
            signatures=signatures,
            block_path_params=block_path_params,
            block_param_last_dims=block_param_last_dims,
            block_output_last_dims=block_output_last_dims,
            block_param_shapes=block_param_shapes,
            block_output_shapes=block_output_shapes,
            implicit_prelude_members=implicit_prelude_members,
            prelude_aliases=prelude_aliases,
            primitive_aliases=primitive_aliases,
        ),
    )
    _ensure_outputs_from_returns(main_outputs, main_returns)
    model: dict[str, Any] = {"inputs": main_inputs, "graph": main_graph, "outputs": main_outputs}
    if main.symbols:
        model["symbols"] = dict(main.symbols)
    spec: dict[str, Any] = {"synapse": 1, "model": model}

    blocks: dict[str, Any] = {}
    for module in modules:
        if module.name == main_name:
            continue
        if module.name in primitive_aliases:
            continue
        block_inputs = _module_inputs(module)
        block_returns = _module_return_names(module)
        block_graph: list[dict[str, Any]] = []
        block_outputs: dict[str, str] = {}
        _lower_statements(
            statements=module.statements,
            graph=block_graph,
            outputs=block_outputs,
            returns=block_returns,
            ctx=_new_lower_ctx(
                module=module,
                returns=block_returns,
                signatures=signatures,
                block_path_params=block_path_params,
                block_param_last_dims=block_param_last_dims,
                block_output_last_dims=block_output_last_dims,
                block_param_shapes=block_param_shapes,
                block_output_shapes=block_output_shapes,
                implicit_prelude_members=implicit_prelude_members,
                prelude_aliases=prelude_aliases,
                primitive_aliases=primitive_aliases,
            ),
        )
        _ensure_outputs_from_returns(block_outputs, block_returns)
        blocks[module.name] = {
            "inputs": block_inputs,
            "graph": block_graph,
            "outputs": block_outputs,
        }
    if blocks:
        spec["model"]["blocks"] = blocks

    _resolve_paths_at_lowering_time(spec["model"], block_path_params)

    return spec


__all__ = [
    "lower_axon_module_to_synapse_block",
    "lower_axon_module_to_synapse_spec",
    "lower_axon_program_to_synapse_spec",
]
