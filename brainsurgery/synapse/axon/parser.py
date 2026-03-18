from __future__ import annotations

import json
import re

from .types import AxonBind, AxonMeta, AxonModule, AxonParam, AxonRawNode, AxonRepeat, AxonReturn

_HEADER_RE = re.compile(r"^module\s+([A-Za-z_][A-Za-z0-9_]*)\s*\((.*?)\)\s*->\s*\((.*?)\)\s*do\s*$")
_REPEAT_RE = re.compile(
    r"^repeat(?:\s+([A-Za-z_][A-Za-z0-9_.]*)\s*:)?\s+([A-Za-z_][A-Za-z0-9_]*)\s+in\s+(.+?)(?:\s+do)?\s*$"
)


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


def parse_axon_module(source: str) -> AxonModule:
    lines = [line.rstrip() for line in source.splitlines() if line.strip()]
    if not lines:
        raise ValueError("empty Axon source")

    header_match = _HEADER_RE.match(lines[0])
    if header_match is None:
        raise ValueError("expected module header: module <name>(...) -> (...) do")

    module_name = header_match.group(1)
    params = _parse_params(header_match.group(2))
    returns = tuple(part.strip() for part in _split_top_level_csv(header_match.group(3)))
    entries = _line_entries(lines[1:])
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
            if nxt_indent > indent and (nxt.startswith("|>") or nxt.startswith(">>=")):
                line = line.rstrip() + " " + nxt
                i += 1
                continue
            break

        out.append(_parse_simple_line(line))
        i += 1

    return out, i


def parse_axon_program(source: str) -> tuple[AxonModule, ...]:
    raw_lines = [line.rstrip("\n") for line in source.splitlines()]
    module_starts: list[int] = []
    for idx, line in enumerate(raw_lines):
        if _HEADER_RE.match(line.strip()) is not None:
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
