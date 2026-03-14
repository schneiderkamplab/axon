import ast
from pathlib import Path

def test_core_modules_import_core_symbols_from_concrete_modules() -> None:
    core_dir = Path("brainsurgery/core")
    offenders: list[str] = []

    for path in sorted(core_dir.glob("*.py")):
        if path.name == "__init__.py":
            continue

        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                # Intra-core imports must target concrete modules, not the package
                # re-export surface (`brainsurgery.core` or `from . import ...`).
                if node.level == 0 and node.module == "brainsurgery.core":
                    offenders.append(f"{path}:{node.lineno}")
                if node.level == 1 and node.module is None:
                    offenders.append(f"{path}:{node.lineno}")

    assert not offenders, (
        "Core modules must import from concrete core modules, not package-level "
        f"re-exports. Offenders: {', '.join(offenders)}"
    )
