import ast
from pathlib import Path

PACKAGE_ROOT = Path("brainsurgery")
ENGINE_DIR = PACKAGE_ROOT / "engine"
ENGINE_INIT = ENGINE_DIR / "__init__.py"


def _iter_python_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)


def _engine_exports() -> set[str]:
    tree = ast.parse(ENGINE_INIT.read_text(encoding="utf-8"), filename=str(ENGINE_INIT))
    exports: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name != "*":
                    exports.add(alias.asname or alias.name)
    return exports


def test_engine_modules_import_concrete_engine_modules_only() -> None:
    offenders: list[str] = []
    for path in _iter_python_files(ENGINE_DIR):
        if path.name == "__init__.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.level == 0 and node.module == "brainsurgery.engine":
                offenders.append(f"{path}:{node.lineno}")
            if node.level == 1 and node.module is None:
                offenders.append(f"{path}:{node.lineno}")

    assert not offenders, (
        "Engine modules must import from concrete engine modules, not package-level re-exports. "
        f"Offenders: {', '.join(offenders)}"
    )


def test_engine_modules_do_not_import_other_brainsurgery_subpackages_except_core_and_io() -> None:
    offenders: list[str] = []

    allowed_absolute_prefixes = (
        "brainsurgery.engine",
        "brainsurgery.core",
        "brainsurgery.io",
    )
    allowed_parent_relative = ("core", "io")

    for path in _iter_python_files(ENGINE_DIR):
        if path.name == "__init__.py":
            continue

        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""

                if (
                    node.level == 0
                    and module.startswith("brainsurgery.")
                    and not module.startswith(allowed_absolute_prefixes)
                ):
                    offenders.append(f"{path}:{node.lineno}")

                if node.level >= 3:
                    offenders.append(f"{path}:{node.lineno}")
                if node.level == 2:
                    if module not in allowed_parent_relative and not module.startswith(
                        tuple(f"{name}." for name in allowed_parent_relative)
                    ):
                        offenders.append(f"{path}:{node.lineno}")

            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("brainsurgery.") and not alias.name.startswith(
                        allowed_absolute_prefixes
                    ):
                        offenders.append(f"{path}:{node.lineno}")

    assert not offenders, (
        "Engine modules may not import from brainsurgery subpackages other than engine, core, and io. "
        f"Offenders: {', '.join(offenders)}"
    )


def test_non_engine_modules_do_not_import_engine_submodules_directly() -> None:
    offenders: list[str] = []
    for path in _iter_python_files(PACKAGE_ROOT):
        if ENGINE_DIR in path.parents:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if node.level == 0 and module.startswith("brainsurgery.engine."):
                    offenders.append(f"{path}:{node.lineno}")
                if node.level == 2 and module.startswith("engine."):
                    offenders.append(f"{path}:{node.lineno}")
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("brainsurgery.engine."):
                        offenders.append(f"{path}:{node.lineno}")

    assert not offenders, (
        "Non-engine package modules must import engine symbols through brainsurgery.engine "
        "(engine.__init__), not engine submodules directly. "
        f"Offenders: {', '.join(offenders)}"
    )


def test_engine_reexports_are_used_by_non_engine_package_modules() -> None:
    exports = _engine_exports()
    used: set[str] = set()

    for path in _iter_python_files(PACKAGE_ROOT):
        if ENGINE_DIR in path.parents:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.level == 0 and node.module == "brainsurgery.engine":
                for alias in node.names:
                    if alias.name != "*":
                        used.add(alias.name)
            if node.level == 2 and node.module == "engine":
                for alias in node.names:
                    if alias.name != "*":
                        used.add(alias.name)

    unused = sorted(exports - used)
    assert not unused, (
        "brainsurgery.engine re-exports should be kept minimal and used by non-engine package modules. "
        f"Unused exports: {', '.join(unused)}"
    )
