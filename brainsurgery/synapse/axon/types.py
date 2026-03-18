from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AxonParam:
    name: str
    optional: bool = False


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
    body: tuple["AxonBind | AxonReturn | AxonRawNode | AxonMeta | AxonRepeat", ...]


@dataclass(frozen=True)
class AxonModule:
    name: str
    params: tuple[AxonParam, ...]
    returns: tuple[str, ...]
    statements: tuple[AxonBind | AxonReturn | AxonRawNode | AxonMeta | AxonRepeat, ...]


__all__ = [
    "AxonParam",
    "AxonBind",
    "AxonReturn",
    "AxonRawNode",
    "AxonMeta",
    "AxonRepeat",
    "AxonModule",
]
