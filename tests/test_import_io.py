import ast
from pathlib import Path

PACKAGE_ROOT = Path("brainsurgery")
IO_DIR = PACKAGE_ROOT / "io"
IO_INIT = IO_DIR / "__init__.py"


def _iter_python_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)


def _io_exports() -> set[str]:
    tree = ast.parse(IO_INIT.read_text(encoding="utf-8"), filename=str(IO_INIT))
    exports: set[str] = set()
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        if node.targets[0].id != "__all__":
            continue
        if not isinstance(node.value, (ast.List, ast.Tuple)):
            continue
        for element in node.value.elts:
            if isinstance(element, ast.Constant) and isinstance(element.value, str):
                exports.add(element.value)
    return exports


def test_io_modules_import_concrete_io_modules_only() -> None:
    offenders: list[str] = []
    for path in _iter_python_files(IO_DIR):
        if path.name == "__init__.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.level == 0 and node.module == "brainsurgery.io":
                offenders.append(f"{path}:{node.lineno}")
            if node.level == 1 and node.module is None:
                offenders.append(f"{path}:{node.lineno}")

    assert not offenders, (
        "IO modules must import from concrete io modules, not package-level re-exports. "
        f"Offenders: {', '.join(offenders)}"
    )


def test_io_modules_do_not_import_other_brainsurgery_subpackages() -> None:
    offenders: list[str] = []

    for path in _iter_python_files(IO_DIR):
        if path.name == "__init__.py":
            continue

        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""

                if (
                    node.level == 0
                    and module.startswith("brainsurgery.")
                    and not module.startswith("brainsurgery.io")
                ):
                    offenders.append(f"{path}:{node.lineno}")

                if node.level >= 2:
                    offenders.append(f"{path}:{node.lineno}")

            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("brainsurgery.") and not alias.name.startswith(
                        "brainsurgery.io"
                    ):
                        offenders.append(f"{path}:{node.lineno}")

    assert not offenders, (
        "IO modules may not import from non-io brainsurgery subpackages. "
        f"Offenders: {', '.join(offenders)}"
    )


def test_non_io_modules_do_not_import_io_submodules_directly() -> None:
    offenders: list[str] = []
    for path in _iter_python_files(PACKAGE_ROOT):
        if IO_DIR in path.parents:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if node.level == 0 and module.startswith("brainsurgery.io."):
                    offenders.append(f"{path}:{node.lineno}")
                if node.level == 2 and module.startswith("io."):
                    offenders.append(f"{path}:{node.lineno}")
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("brainsurgery.io."):
                        offenders.append(f"{path}:{node.lineno}")

    assert not offenders, (
        "Non-io package modules must import io symbols through brainsurgery.io (io.__init__), "
        "not io submodules directly. "
        f"Offenders: {', '.join(offenders)}"
    )


def test_io_reexports_are_used_by_non_io_package_modules() -> None:
    exports = _io_exports()
    used: set[str] = set()

    for path in _iter_python_files(PACKAGE_ROOT):
        if IO_DIR in path.parents:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.level == 0 and node.module == "brainsurgery.io":
                for alias in node.names:
                    if alias.name != "*":
                        used.add(alias.name)
            if node.level == 2 and node.module == "io":
                for alias in node.names:
                    if alias.name != "*":
                        used.add(alias.name)

    unused = sorted(exports - used)
    assert not unused, (
        "brainsurgery.io re-exports should be kept minimal and used by non-io package modules. "
        f"Unused exports: {', '.join(unused)}"
    )
