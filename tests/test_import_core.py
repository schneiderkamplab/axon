import ast
from pathlib import Path

PACKAGE_ROOT = Path("brainsurgery")
CORE_DIR = PACKAGE_ROOT / "core"
CORE_INIT = CORE_DIR / "__init__.py"
SPECS_DIR = CORE_DIR / "specs"
COMPILE_DIR = CORE_DIR / "compile"
RUNTIME_DIR = CORE_DIR / "runtime"

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

def _assert_subpackage_import_policy(
    *,
    subpackage_dir: Path,
    subpackage_name: str,
    allowed_core_packages: set[str],
) -> None:
    offenders: list[str] = []

    for path in _iter_python_files(subpackage_dir):
        if path.name == "__init__.py":
            continue

        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if node.level == 0 and module.startswith("brainsurgery.") and not module.startswith(
                    "brainsurgery.core"
                ):
                    offenders.append(f"{path}:{node.lineno}")
                    continue

                if node.level == 1 and node.module is None:
                    offenders.append(f"{path}:{node.lineno}")
                    continue

                if node.level == 2:
                    # Cross-subpackage imports must target package __init__, never submodules.
                    if module not in allowed_core_packages:
                        offenders.append(f"{path}:{node.lineno}")
                    continue

                if node.level >= 3:
                    offenders.append(f"{path}:{node.lineno}")
                    continue

                if node.level == 0 and module.startswith("brainsurgery.core."):
                    tail = module[len("brainsurgery.core.") :]
                    if tail.startswith(f"{subpackage_name}."):
                        # Same-subpackage imports should be direct relative imports.
                        offenders.append(f"{path}:{node.lineno}")
                        continue
                    if "." in tail:
                        # Cross-subpackage imports must go through package __init__.
                        offenders.append(f"{path}:{node.lineno}")
                        continue
                    if tail and tail not in allowed_core_packages:
                        offenders.append(f"{path}:{node.lineno}")
                        continue

            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("brainsurgery.") and not alias.name.startswith("brainsurgery.core"):
                        offenders.append(f"{path}:{node.lineno}")
                        continue
                    if alias.name.startswith(f"brainsurgery.core.{subpackage_name}."):
                        offenders.append(f"{path}:{node.lineno}")
                        continue
                    if alias.name.startswith("brainsurgery.core.") and alias.name.count(".") >= 2:
                        offenders.append(f"{path}:{node.lineno}")

    assert not offenders, (
        f"core.{subpackage_name}: invalid imports detected. "
        f"Offenders: {', '.join(offenders)}"
    )

def test_core_specs_import_policy() -> None:
    _assert_subpackage_import_policy(
        subpackage_dir=SPECS_DIR,
        subpackage_name="specs",
        allowed_core_packages=set(),
    )

def test_core_compile_import_policy() -> None:
    _assert_subpackage_import_policy(
        subpackage_dir=COMPILE_DIR,
        subpackage_name="compile",
        allowed_core_packages={"specs"},
    )

def test_core_runtime_import_policy() -> None:
    _assert_subpackage_import_policy(
        subpackage_dir=RUNTIME_DIR,
        subpackage_name="runtime",
        allowed_core_packages={"specs", "compile"},
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
