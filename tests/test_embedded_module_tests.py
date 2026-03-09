from __future__ import annotations

from collections.abc import Callable, Iterable
from importlib import import_module
from pkgutil import iter_modules

import pytest

import brainsurgery.expressions as expressions_pkg
import brainsurgery.transforms as transforms_pkg


def _discover_embedded_tests(package) -> Iterable[tuple[str, Callable[[], None]]]:
    """
    Discover module-embedded tests.

    Modules may expose a `__unit_tests__` list of no-arg callables.
    """
    for module_info in iter_modules(package.__path__):  # type: ignore[attr-defined]
        module_name = module_info.name
        if module_name.startswith("_"):
            continue
        module = import_module(f"{package.__name__}.{module_name}")
        unit_tests = getattr(module, "__unit_tests__", [])
        if not isinstance(unit_tests, list):
            continue
        for idx, test_fn in enumerate(unit_tests):
            if callable(test_fn):
                yield f"{package.__name__}.{module_name}[{idx}]", test_fn


_DISCOVERED = list(_discover_embedded_tests(expressions_pkg)) + list(
    _discover_embedded_tests(transforms_pkg)
)


def _modules_missing_embedded_tests(package, *, exempt: set[str]) -> list[str]:
    missing: list[str] = []
    for module_info in iter_modules(package.__path__):  # type: ignore[attr-defined]
        module_name = module_info.name
        if module_name.startswith("_") or module_name in exempt:
            continue
        module = import_module(f"{package.__name__}.{module_name}")
        unit_tests = getattr(module, "__unit_tests__", None)
        if not isinstance(unit_tests, list) or not unit_tests:
            missing.append(f"{package.__name__}.{module_name}")
    return sorted(missing)


@pytest.mark.parametrize("test_id,test_fn", _DISCOVERED, ids=[item[0] for item in _DISCOVERED])
def test_embedded_unit_hook(test_id: str, test_fn: Callable[[], None]) -> None:
    del test_id
    test_fn()


def test_all_expression_modules_define_embedded_tests() -> None:
    missing = _modules_missing_embedded_tests(expressions_pkg, exempt=set())
    assert not missing, f"Expression modules missing __unit_tests__: {missing}"


def test_all_transform_modules_define_embedded_tests() -> None:
    missing = _modules_missing_embedded_tests(
        transforms_pkg,
        exempt={"binary", "unary"},
    )
    assert not missing, f"Transform modules missing __unit_tests__: {missing}"
