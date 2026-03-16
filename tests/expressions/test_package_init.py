import pkgutil
from importlib import import_module, reload
from types import SimpleNamespace

import pytest


def test_expressions_package_skips_private_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        pkgutil,
        "iter_modules",
        lambda _path: [SimpleNamespace(name="_private"), SimpleNamespace(name="visible")],
    )
    import importlib as _importlib

    monkeypatch.setattr(_importlib, "import_module", lambda name: calls.append(name))
    package = import_module("brainsurgery.expressions")
    reload(package)
    assert calls == ["brainsurgery.expressions.visible"]


def test_expressions_package_imports_modules_in_sorted_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        pkgutil,
        "iter_modules",
        lambda _path: [
            SimpleNamespace(name="zeta"),
            SimpleNamespace(name="alpha"),
            SimpleNamespace(name="_private"),
        ],
    )
    import importlib as _importlib

    monkeypatch.setattr(_importlib, "import_module", lambda name: calls.append(name))
    package = import_module("brainsurgery.expressions")
    reload(package)
    assert calls == ["brainsurgery.expressions.alpha", "brainsurgery.expressions.zeta"]


def test_expressions_package_duplicate_discovery_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        pkgutil,
        "iter_modules",
        lambda _path: [SimpleNamespace(name="dup"), SimpleNamespace(name="dup")],
    )
    package = import_module("brainsurgery.expressions")
    with pytest.raises(RuntimeError, match="Duplicate expression module names discovered"):
        reload(package)


def test_expressions_package_import_failure_reports_module(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pkgutil, "iter_modules", lambda _path: [SimpleNamespace(name="bad")])
    import importlib as _importlib

    def _raise(name: str):
        raise ValueError(name)

    monkeypatch.setattr(_importlib, "import_module", _raise)
    package = import_module("brainsurgery.expressions")
    with pytest.raises(
        RuntimeError,
        match=r"Failed to import discovered expression module: brainsurgery\.expressions\.bad",
    ):
        reload(package)
