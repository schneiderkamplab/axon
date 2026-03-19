from __future__ import annotations

import re

from .types import (
    AxonBind,
    AxonModule,
    AxonParam,
    AxonRepeat,
    AxonReturn,
    AxonScope,
    AxonStatement,
)

_HEADER_RE = re.compile(r"^module\s+([A-Za-z_][A-Za-z0-9_]*)\s*\((.*?)\)\s*->\s*\((.*?)\)\s*do\s*$")
_SIG_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*::\s*(.+)\s*$")
_DEF_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*(.*?)\s*=\s*do\s*$")
_FOR_AT_RANGE_RE = re.compile(
    r"^for(?:@([A-Za-z_][A-Za-z0-9_.]*))?\s+([A-Za-z_][A-Za-z0-9_]*)\s*<-\s*([\[\(])\s*(.+?)\s*\.\.\s*(.+?)\s*([\]\)\[])\s+do\s*$"
)
_SCOPE_RE = re.compile(r"^scope(?:@|\s+)([A-Za-z_][A-Za-z0-9_.]*)\s+do\s*$")
_TOP_CONST_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+?)\s*$")
_TYPE_SHAPE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*\[(.+)\]$")


def _strip_haskell_comment(line: str) -> str:
    in_single = False
    in_double = False
    for idx, ch in enumerate(line):
        prev = line[idx - 1] if idx > 0 else ""
        if ch == "'" and not in_double and prev != "\\":
            in_single = not in_single
            continue
        if ch == '"' and not in_single and prev != "\\":
            in_double = not in_double
            continue
        if (
            ch == "-"
            and not in_single
            and not in_double
            and idx + 1 < len(line)
            and line[idx + 1] == "-"
        ):
            return line[:idx]
    return line


def _normalized_source_lines(source: str) -> list[str]:
    out: list[str] = []
    for raw in source.splitlines():
        line = _strip_haskell_comment(raw).rstrip()
        if line.strip():
            out.append(line)
    return out


def _split_top_level_csv(text: str) -> list[str]:
    parts: list[str] = []
    depth = 0
    start = 0
    for idx, ch in enumerate(text):
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append(text[start:idx].strip())
            start = idx + 1
    tail = text[start:].strip()
    if tail:
        parts.append(tail)
    return parts


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


def _parse_params(raw: str) -> tuple[AxonParam, ...]:
    if not raw.strip():
        return ()
    out: list[AxonParam] = []
    for token in _split_top_level_csv(raw):
        if token.endswith("?"):
            out.append(AxonParam(name=token[:-1].strip(), optional=True))
        else:
            out.append(AxonParam(name=token.strip(), optional=False))
    return tuple(out)


def _shape_dims_from_type(type_expr: str) -> tuple[str, ...] | None:
    match = _TYPE_SHAPE_RE.match(type_expr.strip())
    if match is None:
        return None
    dims = tuple(
        part.strip() for part in _split_top_level_csv(match.group(1).strip()) if part.strip()
    )
    if not dims:
        return None
    return dims


def _parse_const_scalar(token: str) -> object:
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
    if re.fullmatch(r"-?[0-9]+(\.[0-9]+)?([eE][+-]?[0-9]+)?", value):
        return float(value)
    return value


def _extract_top_level_constants(lines: list[str]) -> tuple[list[str], dict[str, object]]:
    body: list[str] = []
    constants: dict[str, object] = {}
    for line in lines:
        if len(line) != len(line.lstrip(" ")):
            body.append(line)
            continue
        stripped = line.strip()
        match = _TOP_CONST_RE.match(stripped)
        if (
            match is not None
            and _SIG_RE.match(stripped) is None
            and _DEF_RE.match(stripped) is None
            and stripped != "do"
        ):
            constants[match.group(1)] = _parse_const_scalar(match.group(2))
            continue
        body.append(line)
    return body, constants


def _inject_symbols_meta(module: AxonModule, symbols: dict[str, object]) -> AxonModule:
    if not symbols:
        return module
    merged: dict[str, object] = dict(symbols)
    if module.symbols:
        merged.update({str(k): v for k, v in module.symbols.items()})
    return AxonModule(
        name=module.name,
        params=module.params,
        returns=module.returns,
        statements=module.statements,
        symbols=merged,
        return_type_expr=module.return_type_expr,
        return_shape=module.return_shape,
    )


def _parse_haskell_header(
    lines: list[str],
) -> (
    tuple[
        str,
        tuple[AxonParam, ...],
        tuple[str, ...],
        int,
        dict[str, object],
        str | None,
        tuple[str, ...] | None,
    ]
    | None
):
    if len(lines) < 2:
        return None
    sig_match = _SIG_RE.match(lines[0])
    def_match = _DEF_RE.match(lines[1])
    if sig_match is None or def_match is None:
        return None

    name_sig = sig_match.group(1)
    name_def = def_match.group(1)
    if name_sig != name_def:
        raise ValueError(f"signature/definition name mismatch: {name_sig!r} != {name_def!r}")

    sig_expr = sig_match.group(2).strip()
    parts = _split_top_level(sig_expr, "->")
    if len(parts) < 1:
        raise ValueError("invalid Axon type signature")
    arg_types = parts[:-1]
    raw_return_type = parts[-1].strip()
    opt_flags = [arg.strip().startswith("?") for arg in arg_types]

    arg_names = [p for p in def_match.group(2).strip().split() if p]
    if len(arg_names) != len(opt_flags):
        raise ValueError(
            f"signature arg count ({len(opt_flags)}) does not match definition args ({len(arg_names)})"
        )
    annotation_symbols: dict[str, object] = {}
    params_out: list[AxonParam] = []
    for idx, arg_name in enumerate(arg_names):
        raw_type = arg_types[idx].strip()
        clean_type = raw_type[1:].strip() if raw_type.startswith("?") else raw_type
        shape = _shape_dims_from_type(clean_type)
        if shape is not None:
            for dim in shape:
                annotation_symbols.setdefault(dim, None)
        params_out.append(
            AxonParam(
                name=arg_name.strip(),
                optional=opt_flags[idx],
                type_expr=clean_type,
                shape=shape,
            )
        )
    ret_shape = _shape_dims_from_type(raw_return_type)
    if ret_shape is not None:
        for dim in ret_shape:
            annotation_symbols.setdefault(dim, None)
    params = tuple(params_out)
    # Haskell-style signatures carry output types, not names. Return names will be inferred from `return`.
    return name_sig, params, (), 2, annotation_symbols, raw_return_type, ret_shape


def parse_axon_module(source: str) -> AxonModule:
    lines, top_constants = _extract_top_level_constants(_normalized_source_lines(source))
    if not lines:
        raise ValueError("empty Axon source")

    parsed = _parse_haskell_header(lines)
    if parsed is None:
        raise ValueError("expected haskell-style pair: '<name> :: ...' + '<name> ... = do'")
    (
        module_name,
        params,
        returns,
        body_start,
        annotation_symbols,
        return_type_expr,
        return_shape,
    ) = parsed

    entries = _line_entries(lines[body_start:])
    if not entries:
        return AxonModule(
            name=module_name,
            params=params,
            returns=returns,
            statements=(),
            symbols=top_constants if top_constants else None,
            return_type_expr=return_type_expr,
            return_shape=return_shape,
        )
    base_indent = min(indent for indent, _ in entries)
    statements, index = _parse_statements(entries, 0, base_indent)
    if index != len(entries):
        raise ValueError("unexpected trailing lines in module body")

    module = AxonModule(
        name=module_name,
        params=params,
        returns=returns,
        statements=tuple(statements),
        symbols=None,
        return_type_expr=return_type_expr,
        return_shape=return_shape,
    )
    module = _inject_symbols_meta(module, annotation_symbols)
    return _inject_symbols_meta(module, top_constants)


def _parse_simple_line(line: str) -> AxonBind | AxonReturn:
    if line.startswith("return "):
        values = tuple(_split_top_level_csv(line[len("return ") :].strip()))
        return AxonReturn(values=values)
    if "<-" in line:
        left, right = line.split("<-", 1)
        targets = tuple(part.strip() for part in _split_top_level_csv(left.strip()))
        return AxonBind(targets=targets, expr=right.strip())
    raise ValueError(f"unsupported Axon statement: {line!r}")


def _line_entries(lines: list[str]) -> list[tuple[int, str]]:
    entries: list[tuple[int, str]] = []
    for raw in lines:
        if not raw.strip():
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        entries.append((indent, raw.strip()))
    return entries


def _parse_statements(
    lines: list[tuple[int, str]],
    start: int,
    current_indent: int,
) -> tuple[list[AxonStatement], int]:
    out: list[AxonStatement] = []
    i = start
    while i < len(lines):
        indent, line = lines[i]
        if indent < current_indent:
            return out, i
        if indent > current_indent:
            raise ValueError(f"unexpected indentation at line: {line!r}")

        for_at_match = _FOR_AT_RANGE_RE.match(line)
        scope_match = _SCOPE_RE.match(line)
        if for_at_match is not None:
            repeat_name = for_at_match.group(1).strip() if for_at_match.group(1) else None
            var = for_at_match.group(2).strip()
            start_delim = for_at_match.group(3)
            start_raw = for_at_match.group(4).strip()
            end_raw = for_at_match.group(5).strip()
            end_delim = for_at_match.group(6)

            start_expr = start_raw if start_delim == "[" else f"({start_raw}) + 1"
            end_exclusive = f"({end_raw}) + 1" if end_delim == "]" else end_raw
            range_expr = (
                end_exclusive if start_expr == "0" else f"({end_exclusive}) - ({start_expr})"
            )
            if i + 1 >= len(lines):
                raise ValueError("for@ requires indented body")
            next_indent, _ = lines[i + 1]
            if next_indent <= indent:
                raise ValueError("for@ requires indented body")
            body, new_i = _parse_statements(lines, i + 1, next_indent)
            out.append(
                AxonRepeat(
                    name=repeat_name,
                    var=var,
                    range_expr=range_expr,
                    start_expr=start_expr,
                    body=tuple(body),
                )
            )
            i = new_i
            continue
        if scope_match is not None:
            prefix = scope_match.group(1).strip()
            if i + 1 >= len(lines):
                raise ValueError("scope requires indented body")
            next_indent, _ = lines[i + 1]
            if next_indent <= indent:
                raise ValueError("scope requires indented body")
            body, new_i = _parse_statements(lines, i + 1, next_indent)
            out.append(AxonScope(prefix=prefix, body=tuple(body)))
            i = new_i
            continue

        while i + 1 < len(lines):
            nxt_indent, nxt = lines[i + 1]
            current = line.rstrip()
            nxt_line = nxt.strip()
            current_continues = current.endswith("|>") or current.endswith(">>=")
            next_is_continuation = nxt_line.startswith("|>") or nxt_line.startswith(">>=")
            if nxt_indent > indent and (current_continues or next_is_continuation):
                line = f"{current} {nxt_line}"
                i += 1
                continue
            break

        out.append(_parse_simple_line(line))
        i += 1

    return out, i


def parse_axon_program(source: str) -> tuple[AxonModule, ...]:
    raw_lines, top_constants = _extract_top_level_constants(_normalized_source_lines(source))
    module_starts: list[int] = []
    for idx, line in enumerate(raw_lines):
        if len(line) != len(line.lstrip(" ")):
            continue
        stripped = line.strip()
        if _SIG_RE.match(stripped) is not None:
            module_starts.append(idx)
    if not module_starts:
        return (parse_axon_module(source),)

    modules: list[AxonModule] = []
    for i, start in enumerate(module_starts):
        end = module_starts[i + 1] if i + 1 < len(module_starts) else len(raw_lines)
        chunk = "\n".join(raw_lines[start:end]).strip()
        if not chunk:
            continue
        modules.append(parse_axon_module(chunk))
    if modules:
        modules[-1] = _inject_symbols_meta(modules[-1], top_constants)
    return tuple(modules)


__all__ = ["parse_axon_module", "parse_axon_program"]
