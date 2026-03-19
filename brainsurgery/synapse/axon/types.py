from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AxonParam:
    name: str
    optional: bool = False
    type_expr: str | None = None
    shape: tuple[str, ...] | None = None


@dataclass(frozen=True)
class AxonBind:
    targets: tuple[str, ...]
    expr: str


@dataclass(frozen=True)
class AxonReturn:
    values: tuple[str, ...]


@dataclass(frozen=True)
class AxonRepeat:
    name: str | None
    var: str
    range_expr: str
    start_expr: str
    body: tuple["AxonStatement", ...]


@dataclass(frozen=True)
class AxonScope:
    prefix: str
    body: tuple["AxonStatement", ...]


@dataclass(frozen=True)
class AxonScopeBind:
    targets: tuple[str, ...]
    prefix: str
    body: tuple["AxonStatement", ...]


AxonStatement = AxonBind | AxonReturn | AxonRepeat | AxonScope | AxonScopeBind


@dataclass(frozen=True)
class AxonModule:
    name: str
    params: tuple[AxonParam, ...]
    returns: tuple[str, ...]
    statements: tuple[AxonStatement, ...]
    symbols: dict[str, object] | None = None
    return_type_expr: str | None = None
    return_shape: tuple[str, ...] | None = None


__all__ = [
    "AxonParam",
    "AxonBind",
    "AxonReturn",
    "AxonRepeat",
    "AxonScope",
    "AxonScopeBind",
    "AxonStatement",
    "AxonModule",
]
