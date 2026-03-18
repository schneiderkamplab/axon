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
class AxonRawNode:
    name: str
    node_spec: dict[str, object]


@dataclass(frozen=True)
class AxonMeta:
    key: str
    value: object


@dataclass(frozen=True)
class AxonRepeat:
    name: str | None
    var: str
    range_expr: str
    start_expr: str
    body: tuple["AxonBind | AxonReturn | AxonRawNode | AxonMeta | AxonRepeat", ...]


@dataclass(frozen=True)
class AxonModule:
    name: str
    params: tuple[AxonParam, ...]
    returns: tuple[str, ...]
    statements: tuple[AxonBind | AxonReturn | AxonRawNode | AxonMeta | AxonRepeat, ...]
    return_type_expr: str | None = None
    return_shape: tuple[str, ...] | None = None


__all__ = [
    "AxonParam",
    "AxonBind",
    "AxonReturn",
    "AxonRawNode",
    "AxonMeta",
    "AxonRepeat",
    "AxonModule",
]
