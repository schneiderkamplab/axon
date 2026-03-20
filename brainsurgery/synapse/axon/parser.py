from __future__ import annotations

import re
from pathlib import Path

from .types import (
    AxonBind,
    AxonModule,
    AxonParam,
    AxonRepeat,
    AxonReturn,
    AxonScopeBind,
    AxonStatement,
)

_HEADER_RE = re.compile(r"^module\s+([A-Za-z_][A-Za-z0-9_]*)\s*\((.*?)\)\s*->\s*\((.*?)\)\s*do\s*$")
_MOD_NAME_RE = r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*"
_SIG_RE = re.compile(rf"^({_MOD_NAME_RE}(?:@[A-Za-z_][A-Za-z0-9_]*)*)\s*::\s*(.+)\s*$")
_DEF_DO_RE = re.compile(rf"^({_MOD_NAME_RE}(?:@[A-Za-z_][A-Za-z0-9_]*)*)\s*(.*?)\s*=\s*do\s*$")
_DEF_EXPR_RE = re.compile(rf"^({_MOD_NAME_RE}(?:@[A-Za-z_][A-Za-z0-9_]*)*)\s*(.*?)\s*=\s*(.+?)\s*$")
_FOR_AT_RANGE_RE = re.compile(
    r"^for(?:@([A-Za-z_][A-Za-z0-9_.]*))?\s+([A-Za-z_][A-Za-z0-9_]*)\s*<-\s*([\[\(])\s*(.+?)\s*\.\.\s*(.+?)\s*([\]\)\[])\s+do\s*$"
)
_SCOPE_RE = re.compile(r"^scope(?:@|\s+)([A-Za-z_][A-Za-z0-9_.]*)\s+do\s*$")
_BIND_SCOPE_RE = re.compile(r"^(.+?)<-\s*scope(?:@|\s+)([A-Za-z_][A-Za-z0-9_.]*)\s+do\s*$")
_TOP_CONST_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+?)\s*$")
_TYPE_SHAPE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*\[(.+)\]$")
_PATH_SIG_ARG_RE = re.compile(r"^@([A-Za-z_][A-Za-z0-9_]*)\s*:\s*([A-Za-z_][A-Za-z0-9_]*)$")
_PATH_SIG_SHORT_RE = re.compile(r"^@([A-Za-z_][A-Za-z0-9_]*)$")
_IMPORT_RE = re.compile(rf"^import\s+({_MOD_NAME_RE})(?:\s+(.*))?$")
_IMPORT_MEMBER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SIMPLE_CALLEE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_:.@]*$")


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
    prev_was_sig = False
    for line in lines:
        if len(line) != len(line.lstrip(" ")):
            body.append(line)
            prev_was_sig = False
            continue
        stripped = line.strip()
        if _SIG_RE.match(stripped) is not None:
            body.append(line)
            prev_was_sig = True
            continue
        match = _TOP_CONST_RE.match(stripped)
        if match is not None and not prev_was_sig:
            constants[match.group(1)] = _parse_const_scalar(match.group(2))
            prev_was_sig = False
            continue
        body.append(line)
        prev_was_sig = False
    return body, constants


def _inject_symbols_meta(module: AxonModule, symbols: dict[str, object]) -> AxonModule:
    if not symbols:
        return module
    merged: dict[str, object] = dict(symbols)
    if module.symbols:
        merged.update({str(k): v for k, v in module.symbols.items()})
    return AxonModule(
        name=module.name,
        path_param=module.path_param,
        path_params=module.path_params,
        params=module.params,
        returns=module.returns,
        statements=module.statements,
        imports=module.imports,
        imported_members=module.imported_members,
        symbols=merged,
        return_type_expr=module.return_type_expr,
        return_shape=module.return_shape,
    )


def _split_module_path_params(name: str) -> tuple[str, tuple[str, ...]]:
    if "@" not in name:
        return name, ()
    parts = name.split("@")
    base = parts[0]
    path_params = tuple(parts[1:])
    if not re.fullmatch(_MOD_NAME_RE, base):
        raise ValueError(f"invalid module name: {name!r}")
    for path_param in path_params:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", path_param):
            raise ValueError(f"invalid module path parameter: {name!r}")
    if len(set(path_params)) != len(path_params):
        raise ValueError(f"duplicate module path parameter in {name!r}")
    return base, path_params


def _parse_import_members(raw: str) -> tuple[str, ...]:
    text = raw.strip()
    if not text:
        return ()
    if text.startswith("("):
        if not text.endswith(")"):
            raise ValueError(f"invalid import member list: {raw!r}")
        inner = text[1:-1].strip()
        if not inner:
            return ()
        tokens = _split_top_level_csv(inner)
    else:
        normalized = text.replace(",", " ")
        tokens = [part.strip() for part in normalized.split() if part.strip()]
    if not tokens:
        return ()
    for token in tokens:
        if _IMPORT_MEMBER_RE.fullmatch(token) is None:
            raise ValueError(f"invalid imported member name: {token!r}")
    deduped = tuple(dict.fromkeys(tokens))
    return deduped


def _extract_top_level_imports(
    lines: list[str],
) -> tuple[list[str], tuple[str, ...], dict[str, tuple[str, ...]]]:
    body: list[str] = []
    imports: list[str] = []
    imported_members: dict[str, tuple[str, ...]] = {}
    for line in lines:
        if len(line) != len(line.lstrip(" ")):
            body.append(line)
            continue
        match = _IMPORT_RE.match(line.strip())
        if match is not None:
            namespace = match.group(1)
            imports.append(namespace)
            raw_members = match.group(2) or ""
            members = _parse_import_members(raw_members)
            if members:
                prev = imported_members.get(namespace, ())
                imported_members[namespace] = tuple(dict.fromkeys([*prev, *members]))
            continue
        body.append(line)
    deduped = tuple(dict.fromkeys(imports))
    return body, deduped, imported_members


def _parse_haskell_header(
    lines: list[str],
) -> (
    tuple[
        str,
        str | None,
        tuple[str, ...],
        tuple[AxonParam, ...],
        tuple[str, ...],
        int,
        str | None,
        dict[str, object],
        str | None,
        tuple[str, ...] | None,
    ]
    | None
):
    if len(lines) < 2:
        return None
    sig_match = _SIG_RE.match(lines[0])
    if sig_match is None:
        return None
    def_do_match = _DEF_DO_RE.match(lines[1])
    def_expr_match = None if def_do_match is not None else _DEF_EXPR_RE.match(lines[1])
    def_match = def_do_match if def_do_match is not None else def_expr_match
    if def_match is None:
        return None

    name_sig_raw = sig_match.group(1)
    name_def_raw = def_match.group(1)
    name_sig, path_params_sig = _split_module_path_params(name_sig_raw)
    name_def, path_params_def = _split_module_path_params(name_def_raw)
    if name_sig != name_def:
        raise ValueError(
            f"signature/definition name mismatch: {name_sig_raw!r} != {name_def_raw!r}"
        )
    if path_params_sig and path_params_def and path_params_sig != path_params_def:
        raise ValueError(
            f"signature/definition path parameter mismatch: {name_sig_raw!r} != {name_def_raw!r}"
        )
    path_params = path_params_sig if path_params_sig else path_params_def
    path_param = path_params[0] if path_params else None

    sig_expr = sig_match.group(2).strip()
    parts = _split_top_level(sig_expr, "->")
    if len(parts) < 1:
        raise ValueError("invalid Axon type signature")
    arg_types = parts[:-1]
    consumed_path_types = 0
    while consumed_path_types < len(arg_types):
        current = arg_types[consumed_path_types].strip()
        path_sig_match = _PATH_SIG_ARG_RE.match(current)
        path_sig_short = _PATH_SIG_SHORT_RE.match(current)
        if path_sig_match is None and path_sig_short is None:
            break
        if path_sig_match is not None:
            path_sig_name = path_sig_match.group(1)
            path_sig_type = path_sig_match.group(2)
        else:
            path_sig_type = path_sig_short.group(1) if path_sig_short is not None else ""
            if consumed_path_types >= len(path_params):
                raise ValueError(
                    "path signature annotation count exceeds module path parameters in definition"
                )
            path_sig_name = path_params[consumed_path_types]
        if path_sig_type != "Path":
            raise ValueError(
                f"path signature type must be Path, got {path_sig_type!r}. Use '@Path'."
            )
        if not path_params:
            raise ValueError(
                "path signature annotation requires a module path parameter in the definition"
            )
        expected_name = (
            path_params[consumed_path_types] if consumed_path_types < len(path_params) else None
        )
        if path_sig_name != expected_name:
            raise ValueError(
                "path signature parameter does not match module path parameter:"
                f" {path_sig_name!r} != {expected_name!r}"
            )
        consumed_path_types += 1
    if consumed_path_types != len(path_params):
        raise ValueError("path signature annotation count must match module path parameter count")
    arg_types = arg_types[consumed_path_types:]
    raw_return_type = parts[-1].strip()
    opt_flags = [arg.strip().startswith("?") for arg in arg_types]

    arg_names = [p for p in def_match.group(2).strip().split() if p]
    inline_expr = def_expr_match.group(3).strip() if def_expr_match is not None else None
    if len(arg_names) != len(opt_flags):
        allow_pointfree_eta = (
            len(arg_names) == 0
            and len(opt_flags) > 0
            and inline_expr is not None
            and _SIMPLE_CALLEE_RE.fullmatch(inline_expr) is not None
        )
        if not allow_pointfree_eta:
            raise ValueError(
                f"signature arg count ({len(opt_flags)}) does not match definition args ({len(arg_names)})"
            )
        arg_names = [f"arg_{idx}" for idx in range(len(opt_flags))]
        inline_expr = f"{inline_expr} {' '.join(arg_names)}"
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
    return (
        name_sig,
        path_param,
        path_params,
        params,
        (),
        2,
        inline_expr,
        annotation_symbols,
        raw_return_type,
        ret_shape,
    )


def parse_axon_module(source: str) -> AxonModule:
    lines, top_constants = _extract_top_level_constants(_normalized_source_lines(source))
    lines, imports, imported_members = _extract_top_level_imports(lines)
    if not lines:
        raise ValueError("empty Axon source")

    parsed = _parse_haskell_header(lines)
    if parsed is None:
        raise ValueError("expected haskell-style pair: '<name> :: ...' + '<name> ... = do|<expr>'")
    (
        module_name,
        module_path_param,
        module_path_params,
        params,
        returns,
        body_start,
        inline_expr,
        annotation_symbols,
        return_type_expr,
        return_shape,
    ) = parsed

    if inline_expr is not None:
        module = AxonModule(
            name=module_name,
            path_param=module_path_param,
            path_params=module_path_params,
            params=params,
            returns=returns,
            statements=(AxonReturn(values=(inline_expr,)),),
            imports=imports,
            imported_members=imported_members or None,
            symbols=None,
            return_type_expr=return_type_expr,
            return_shape=return_shape,
        )
        module = _inject_symbols_meta(module, annotation_symbols)
        return _inject_symbols_meta(module, top_constants)

    entries = _line_entries(lines[body_start:])
    if not entries:
        return AxonModule(
            name=module_name,
            path_param=module_path_param,
            path_params=module_path_params,
            params=params,
            returns=returns,
            statements=(),
            imports=imports,
            imported_members=imported_members or None,
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
        path_param=module_path_param,
        path_params=module_path_params,
        params=params,
        returns=returns,
        statements=tuple(statements),
        imports=imports,
        imported_members=imported_members or None,
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
        bind_scope_match = _BIND_SCOPE_RE.match(line)
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
        if bind_scope_match is not None:
            raw_targets = bind_scope_match.group(1).strip()
            targets = tuple(part.strip() for part in _split_top_level_csv(raw_targets))
            if not targets:
                raise ValueError("scope bind requires one or more targets")
            prefix = bind_scope_match.group(2).strip()
            if i + 1 >= len(lines):
                raise ValueError("scope bind requires indented body")
            next_indent, _ = lines[i + 1]
            if next_indent <= indent:
                raise ValueError("scope bind requires indented body")
            body, new_i = _parse_statements(lines, i + 1, next_indent)
            out.append(AxonScopeBind(targets=targets, prefix=prefix, body=tuple(body)))
            i = new_i
            continue
        if scope_match is not None:
            del scope_match
            raise ValueError(
                "scope statement form is not supported; use '<target> <- scope@name do ... return ...'"
            )

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
    raw_lines, top_imports, top_imported_members = _extract_top_level_imports(raw_lines)
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
        module = parse_axon_module(chunk)
        merged_imports = tuple(dict.fromkeys([*top_imports, *module.imports]))
        merged_imported_members: dict[str, tuple[str, ...]] = dict(top_imported_members)
        if module.imported_members:
            for namespace, members in module.imported_members.items():
                prev = merged_imported_members.get(namespace, ())
                merged_imported_members[namespace] = tuple(dict.fromkeys([*prev, *members]))
        modules.append(
            AxonModule(
                name=module.name,
                path_param=module.path_param,
                path_params=module.path_params,
                params=module.params,
                returns=module.returns,
                statements=module.statements,
                imports=merged_imports,
                imported_members=merged_imported_members or None,
                symbols=module.symbols,
                return_type_expr=module.return_type_expr,
                return_shape=module.return_shape,
            )
        )
    if modules:
        modules[-1] = _inject_symbols_meta(modules[-1], top_constants)
    return tuple(modules)


def parse_axon_program_from_path(path: Path) -> tuple[AxonModule, ...]:
    root = path.resolve()
    if not root.exists():
        raise FileNotFoundError(f"Axon file not found: {root}")
    if not root.is_file():
        raise ValueError(f"Axon import root must be a file: {root}")

    seen_paths: set[Path] = set()
    visiting: list[Path] = []
    ordered_modules: list[AxonModule] = []

    builtins_dir = (Path(__file__).resolve().parents[1] / "builtins").resolve()
    prelude_file = (builtins_dir / "Prelude.axon").resolve()

    def _apply_namespace(
        modules: tuple[AxonModule, ...], namespace: str | None
    ) -> tuple[AxonModule, ...]:
        if not namespace:
            return modules
        namespaced: list[AxonModule] = []
        for module in modules:
            if "." in module.name:
                namespaced.append(module)
                continue
            namespaced.append(
                AxonModule(
                    name=f"{namespace}.{module.name}",
                    path_param=module.path_param,
                    path_params=module.path_params,
                    params=module.params,
                    returns=module.returns,
                    statements=module.statements,
                    imports=module.imports,
                    imported_members=module.imported_members,
                    symbols=module.symbols,
                    return_type_expr=module.return_type_expr,
                    return_shape=module.return_shape,
                )
            )
        return tuple(namespaced)

    def _resolve_import_path(base_file: Path, import_name: str) -> Path:
        rel = Path(*import_name.split(".")).with_suffix(".axon")
        local_candidate = (base_file.parent / rel).resolve()
        if local_candidate.exists():
            return local_candidate
        builtin_candidate = (builtins_dir / rel).resolve()
        if builtin_candidate.exists():
            return builtin_candidate
        raise FileNotFoundError(
            f"Axon import {import_name!r} not found from {base_file}: "
            f"tried {local_candidate} and {builtin_candidate}"
        )

    def _load_file(file_path: Path, *, namespace: str | None = None) -> None:
        resolved = file_path.resolve()
        if resolved in seen_paths:
            return
        if resolved in visiting:
            cycle = " -> ".join(str(p) for p in [*visiting, resolved])
            raise ValueError(f"Cyclic Axon imports detected: {cycle}")
        visiting.append(resolved)
        source = resolved.read_text(encoding="utf-8")
        modules = _apply_namespace(parse_axon_program(source), namespace)
        import_names: set[str] = set()
        for module in modules:
            import_names.update(module.imports)
        for import_name in sorted(import_names):
            _load_file(_resolve_import_path(resolved, import_name), namespace=import_name)
        ordered_modules.extend(modules)
        seen_paths.add(resolved)
        visiting.pop()

    if prelude_file.exists() and prelude_file != root:
        _load_file(prelude_file, namespace="Prelude")
    _load_file(root)
    return tuple(ordered_modules)


__all__ = ["parse_axon_module", "parse_axon_program", "parse_axon_program_from_path"]
