from importlib import import_module, reload
from types import SimpleNamespace

import pkgutil
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
