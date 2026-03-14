import ast
from pathlib import Path

PACKAGE_ROOT = Path("brainsurgery")
CORE_DIR = PACKAGE_ROOT / "core"
CORE_INIT = CORE_DIR / "__init__.py"

def _iter_python_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)

def _core_exports() -> set[str]:
    tree = ast.parse(CORE_INIT.read_text(encoding="utf-8"), filename=str(CORE_INIT))
    exports: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name != "*":
                    exports.add(alias.asname or alias.name)
    return exports

def test_core_modules_import_concrete_core_modules_only() -> None:
    offenders: list[str] = []
    for path in _iter_python_files(CORE_DIR):
        if path.name == "__init__.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.level == 0 and node.module == "brainsurgery.core":
                offenders.append(f"{path}:{node.lineno}")
            if node.level == 1 and node.module is None:
                offenders.append(f"{path}:{node.lineno}")

    assert not offenders, (
        "Core modules must import from concrete core modules, not package-level re-exports. "
        f"Offenders: {', '.join(offenders)}"
    )

def test_core_modules_do_not_import_other_brainsurgery_subpackages() -> None:
    offenders: list[str] = []

    for path in _iter_python_files(CORE_DIR):
        if path.name == "__init__.py":
            continue

        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""

                # Absolute imports from brainsurgery must stay inside brainsurgery.core.
                if node.level == 0 and module.startswith("brainsurgery.") and not module.startswith(
                    "brainsurgery.core"
                ):
                    offenders.append(f"{path}:{node.lineno}")

                # Relative imports with level>=2 from brainsurgery.core.* escape core.
                if node.level >= 2:
                    offenders.append(f"{path}:{node.lineno}")

            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("brainsurgery.") and not alias.name.startswith("brainsurgery.core"):
                        offenders.append(f"{path}:{node.lineno}")

    assert not offenders, (
        "Core modules may not import from non-core brainsurgery subpackages. "
        f"Offenders: {', '.join(offenders)}"
    )

def test_non_core_modules_do_not_import_core_submodules_directly() -> None:
    offenders: list[str] = []
    for path in _iter_python_files(PACKAGE_ROOT):
        if CORE_DIR in path.parents:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if node.level == 0 and module.startswith("brainsurgery.core."):
                    offenders.append(f"{path}:{node.lineno}")
                if node.level == 2 and module.startswith("core."):
                    offenders.append(f"{path}:{node.lineno}")
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("brainsurgery.core."):
                        offenders.append(f"{path}:{node.lineno}")

    assert not offenders, (
        "Non-core package modules must import core symbols through brainsurgery.core "
        "(core.__init__), not core submodules directly. "
        f"Offenders: {', '.join(offenders)}"
    )

def test_core_reexports_are_used_by_non_core_package_modules() -> None:
    exports = _core_exports()
    used: set[str] = set()

    for path in _iter_python_files(PACKAGE_ROOT):
        if CORE_DIR in path.parents:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.level == 0 and node.module == "brainsurgery.core":
                for alias in node.names:
                    if alias.name != "*":
                        used.add(alias.name)
            if node.level == 2 and node.module == "core":
                for alias in node.names:
                    if alias.name != "*":
                        used.add(alias.name)

    unused = sorted(exports - used)
    assert not unused, (
        "brainsurgery.core re-exports should be kept minimal and used by non-core package modules. "
        f"Unused exports: {', '.join(unused)}"
    )
