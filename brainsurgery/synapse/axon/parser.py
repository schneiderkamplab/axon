from __future__ import annotations

import json
import re

from .types import AxonBind, AxonMeta, AxonModule, AxonParam, AxonRawNode, AxonRepeat, AxonReturn

_HEADER_RE = re.compile(r"^module\s+([A-Za-z_][A-Za-z0-9_]*)\s*\((.*?)\)\s*->\s*\((.*?)\)\s*do\s*$")
_SIG_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*::\s*(.+)\s*$")
_DEF_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*(.*?)\s*=\s*do\s*$")
_REPEAT_RE = re.compile(
    r"^repeat(?:\s+([A-Za-z_][A-Za-z0-9_.]*)\s*:)?\s+([A-Za-z_][A-Za-z0-9_]*)\s+in\s+(.+?)(?:\s+do)?\s*$"
)


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


def _parse_haskell_header(
    lines: list[str],
) -> tuple[str, tuple[AxonParam, ...], tuple[str, ...], int] | None:
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
    opt_flags = [arg.strip().startswith("?") for arg in arg_types]

    arg_names = [p for p in def_match.group(2).strip().split() if p]
    if len(arg_names) != len(opt_flags):
        raise ValueError(
            f"signature arg count ({len(opt_flags)}) does not match definition args ({len(arg_names)})"
        )
    params = tuple(
        AxonParam(name=arg_name.strip(), optional=opt_flags[idx])
        for idx, arg_name in enumerate(arg_names)
    )
    # Haskell-style signatures carry output types, not names. Return names will be inferred from `return`.
    return name_sig, params, (), 2


def parse_axon_module(source: str) -> AxonModule:
    lines = _normalized_source_lines(source)
    if not lines:
        raise ValueError("empty Axon source")

    header_match = _HEADER_RE.match(lines[0])
    if header_match is not None:
        module_name = header_match.group(1)
        params = _parse_params(header_match.group(2))
        returns = tuple(part.strip() for part in _split_top_level_csv(header_match.group(3)))
        body_start = 1
    else:
        parsed = _parse_haskell_header(lines)
        if parsed is None:
            raise ValueError(
                "expected module header or haskell-style pair: "
                "'module <name>(...) -> (...) do' OR '<name> :: ...' + '<name> ... = do'"
            )
        module_name, params, returns, body_start = parsed

    entries = _line_entries(lines[body_start:])
    if not entries:
        return AxonModule(name=module_name, params=params, returns=returns, statements=())
    base_indent = min(indent for indent, _ in entries)
    statements, index = _parse_statements(entries, 0, base_indent)
    if index != len(entries):
        raise ValueError("unexpected trailing lines in module body")

    return AxonModule(
        name=module_name, params=params, returns=returns, statements=tuple(statements)
    )


def _parse_simple_line(line: str) -> AxonBind | AxonReturn | AxonRawNode | AxonMeta:
    if line.startswith("node ") and " = " in line:
        left, right = line.split(" = ", 1)
        _, node_name = left.split(" ", 1)
        node_spec = json.loads(right)
        if not isinstance(node_spec, dict):
            raise ValueError(f"node statement expects JSON object: {line!r}")
        return AxonRawNode(name=node_name.strip(), node_spec=node_spec)
    if line.startswith("meta ") and " = " in line:
        left, right = line.split(" = ", 1)
        _, key = left.split(" ", 1)
        return AxonMeta(key=key.strip(), value=json.loads(right))
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
) -> tuple[list[AxonBind | AxonReturn | AxonRawNode | AxonMeta | AxonRepeat], int]:
    out: list[AxonBind | AxonReturn | AxonRawNode | AxonMeta | AxonRepeat] = []
    i = start
    while i < len(lines):
        indent, line = lines[i]
        if indent < current_indent:
            return out, i
        if indent > current_indent:
            raise ValueError(f"unexpected indentation at line: {line!r}")

        repeat_match = _REPEAT_RE.match(line)
        if repeat_match is not None:
            repeat_name = repeat_match.group(1).strip() if repeat_match.group(1) else None
            var = repeat_match.group(2).strip()
            range_expr = repeat_match.group(3).strip()
            if i + 1 >= len(lines):
                raise ValueError("repeat requires indented body")
            next_indent, _ = lines[i + 1]
            if next_indent <= indent:
                raise ValueError("repeat requires indented body")
            body, new_i = _parse_statements(lines, i + 1, next_indent)
            out.append(
                AxonRepeat(name=repeat_name, var=var, range_expr=range_expr, body=tuple(body))
            )
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
    raw_lines = _normalized_source_lines(source)
    module_starts: list[int] = []
    for idx, line in enumerate(raw_lines):
        if len(line) != len(line.lstrip(" ")):
            continue
        stripped = line.strip()
        if _HEADER_RE.match(stripped) is not None or _SIG_RE.match(stripped) is not None:
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
    return tuple(modules)


__all__ = ["parse_axon_module", "parse_axon_program"]
