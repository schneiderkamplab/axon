from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Iterator, Mapping
from typing import Any


def parameter_pack_bindings(
    state_keys: Iterable[str], spec: Mapping[str, Any], *, target_key: str | None = None
) -> Iterator[tuple[str, list[str]]]:
    """Resolve a parameter-pack recipe, including multi-segment path captures."""

    def pattern_regex(pattern: str) -> re.Pattern[str]:
        pieces, used, cursor = [], set(), 0
        for match in re.finditer(r"\{([A-Za-z_][A-Za-z0-9_]*)\}", pattern):
            pieces.append(re.escape(pattern[cursor : match.start()]))
            name = match.group(1)
            pieces.append(f"(?P={name})" if name in used else f"(?P<{name}>.+?)")
            used.add(name)
            cursor = match.end()
        pieces.append(re.escape(pattern[cursor:]))
        return re.compile("^" + "".join(pieces) + "$")

    def substitute(pattern: str, values: Mapping[str, str]) -> str:
        for name, value in values.items():
            pattern = pattern.replace("{" + name + "}", value)
        return pattern

    output = str(spec["output"])
    inputs = tuple(str(item) for item in spec["inputs"])
    candidates = []
    regex = pattern_regex(output)
    if target_key is not None:
        match = regex.fullmatch(str(target_key))
        if match is not None:
            candidates.append((str(target_key), match.groupdict()))
    elif "{" not in output:
        candidates.append((output, {}))
    else:
        keys = tuple(str(key) for key in state_keys)
        for key in keys:
            match = regex.fullmatch(key)
            if match is not None:
                candidates.append((key, match.groupdict()))
        if inputs:
            input_regex = pattern_regex(inputs[0])
            for key in keys:
                match = input_regex.fullmatch(key)
                if match is not None:
                    values = match.groupdict()
                    candidates.append((substitute(output, values), values))
    seen = set()
    for key, values in candidates:
        if key in seen:
            continue
        seen.add(key)
        yield key, [substitute(pattern, values) for pattern in inputs]


def normalize_primitive_op(name: str) -> str:
    if name.startswith("_activations_"):
        return name[1:]
    if name.startswith("_"):
        return name[1:]
    return name


def is_null(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value.strip().lower() == "null")


def path_parts(value: Any) -> tuple[bool, str]:
    if not isinstance(value, str):
        raise ValueError(f"expected Path value, got {type(value).__name__}")
    token = value.strip()
    if token.startswith("@@"):
        return True, token.lstrip("@")
    if token.startswith("@"):
        return False, token.lstrip("@")
    return True, token


def compose_path(base: Any, leaf: Any) -> str:
    leaf_text = "" if leaf is None else str(leaf).strip()
    if leaf_text.startswith("@@"):
        return leaf_text.lstrip("@")
    base_key = "" if base is None else str(base).strip().lstrip("@")
    leaf_key = leaf_text.lstrip("@")
    if not base_key:
        return leaf_key
    if not leaf_key:
        return base_key
    return f"{base_key}.{leaf_key}"


def render_path(prefix: str, parts: list[Any]) -> str:
    clean: list[str] = []
    for part in parts:
        if part is None:
            continue
        text = str(part).strip()
        if not text or text == "None":
            continue
        clean.append(text.strip("@"))
    return str(prefix) + ".".join(clean)


def required_state_value(state: Mapping[str, Any], path: Any) -> Any:
    key = str(path).lstrip("@")
    try:
        return state[key]
    except KeyError as exc:
        raise KeyError(f"missing parameter {key!r}") from exc


def optional_state_value(state: Mapping[str, Any], path: Any) -> Any | None:
    return state.get(str(path).lstrip("@"))


def lookup_config(config: Any, key: str) -> tuple[bool, Any]:
    if not isinstance(config, dict):
        return False, None
    current: Any = config
    for part in str(key).split("."):
        if not isinstance(current, dict) or part not in current:
            return False, None
        current = current[part]
    return True, current


def config_value(config: Any, path: Any, default: Any = None) -> Any:
    key = str(path).lstrip("@").strip(".")
    found, value = lookup_config(config, key)
    if not found or value is None:
        return default
    return value


def has_config_value(config: Any, path: Any) -> bool:
    key = str(path).lstrip("@").strip(".")
    found, value = lookup_config(config, key)
    return found and value is not None


def require_value(value: Any) -> Any:
    if value is None:
        raise ValueError("require expected non-null value")
    return value


def read_config_value(
    *,
    primitive: str,
    args: list[Any],
    kwargs: Mapping[str, Any],
    config: Any,
) -> Any:
    if len(args) < 1:
        raise ValueError(f"{primitive} expects one Path argument")
    _absolute, key = path_parts(args[0])
    found, value = lookup_config(config, key)
    if primitive == "config_has":
        return found
    if primitive == "config_has_value":
        return found and value is not None
    if not found or value is None:
        value = args[1] if len(args) > 1 else kwargs.get("default")
    if primitive in {"config_int", "config_dim"}:
        return int(value)
    if primitive == "config_float":
        return float(value)
    if primitive == "config_bool":
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes"}
        return bool(value)
    if primitive == "config_str":
        return str(value)
    if primitive == "config_list":
        return list(value)
    return value


def execute_common_primitive(
    *,
    primitive: str,
    args: list[Any],
    kwargs: Mapping[str, Any],
    config: Any,
    state_keys: Callable[[], Any],
    require_param: Callable[[str], Any],
) -> tuple[bool, Any]:
    if primitive == "params_param":
        if len(args) != 1:
            raise ValueError("params_param expects one Path argument")
        _absolute, key = path_parts(args[0])
        return True, require_param(key)

    if primitive == "params_has_root":
        if len(args) != 1:
            raise ValueError("params_has_root expects one root argument")
        root = str(args[0])
        prefix = f"{root}." if root else ""
        return True, root == "" or any(key == root or key.startswith(prefix) for key in state_keys())

    if primitive.startswith("config_"):
        return True, read_config_value(
            primitive=primitive,
            args=args,
            kwargs=kwargs,
            config=config,
        )

    if primitive == "require":
        if args[0] is None:
            raise ValueError("require expected non-null value")
        return True, args[0]

    if primitive == "list_init":
        return True, []
    if primitive == "list_append":
        values = [] if args[0] is None else list(args[0])
        return True, [*values, args[1]]
    if primitive == "list_index":
        return True, None if args[0] is None else args[0][int(args[1])]
    if primitive == "shape":
        return True, list(args[0].shape)

    return False, None


__all__ = [
    "config_value",
    "compose_path",
    "execute_common_primitive",
    "has_config_value",
    "is_null",
    "lookup_config",
    "normalize_primitive_op",
    "optional_state_value",
    "parameter_pack_bindings",
    "path_parts",
    "read_config_value",
    "render_path",
    "required_state_value",
    "require_value",
]
