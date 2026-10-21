"""Compile the Timestral Axon definition and export a standalone MLX package."""

import ast
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "synapse/models/timestral/generic-timestral.axon"


def generate_core(backend="mlx", source=SOURCE, optimize=False):
    from synapse.axon import (
        GraphOptimizeConfig,
        elaborate_closed_axon_file,
        flatten_closed_axon_file,
        lower_axon_program_to_graph_ir,
        normalize_closed_axon_file,
        optimize_graph_program,
        resolve_axon_program_from_path,
        typecheck2_flat_axon_file,
    )

    program = resolve_axon_program_from_path(source).ast
    program = normalize_closed_axon_file(program)
    program = elaborate_closed_axon_file(program)
    program = flatten_closed_axon_file(program)
    program = typecheck2_flat_axon_file(program)
    graph = lower_axon_program_to_graph_ir(program)
    if backend == "mlx":
        from synapse.axon.codegen2_mlx import emit_model_code_from_graph_ir
    elif backend == "torch":
        from synapse.axon.codegen2_torch import emit_model_code_from_graph_ir
    else:
        raise ValueError(f"unsupported backend {backend!r}")
    if optimize:
        # Backend intrinsics require an explicit opt-in. Both selected MLX
        # fusions validate primitive provenance, independent of model names.
        intrinsics = f"codegen2-{backend}:__{backend}_sdpa"
        if backend == "mlx":
            intrinsics += ",__mlx_rmsnorm_scaled"
        graph = optimize_graph_program(
            graph, config=GraphOptimizeConfig(backend_intrinsics=intrinsics)
        )
    code = emit_model_code_from_graph_ir(graph, class_name="TimestralCore")
    ast.parse(code)
    return code


def export_package(output, *, optimize=False):
    """Write a new importable package; never overwrite an existing export."""
    output = Path(output).resolve()
    package = output / "timestral_axon"
    if package.exists():
        raise FileExistsError(f"export exists: {package}; choose a fresh output directory")
    code = generate_core(optimize=optimize)
    old_import = "from synapse.axon.codegen2_common import ("
    if code.count(old_import) != 1:
        raise ValueError("generated runtime helper import changed; inspect exporter")
    code = code.replace(old_import, "from ._axon_common import (")
    package.mkdir(parents=True)
    for path in Path(__file__).parent.iterdir():
        if path.suffix == ".py" and path.name != "build.py":
            shutil.copy2(path, package / path.name)
        elif path.name in {"LICENSE", "NOTICE", "README.md"}:
            shutil.copy2(path, package / path.name)
    # Reuse the backend-neutral standard-library helpers without importing the compiler.
    shutil.copy2(ROOT / "synapse/axon/codegen2_common/core.py", package / "_axon_common.py")
    shutil.copy2(ROOT / "LICENSE", package / "AXON_LICENSE")
    (package / "_generated.py").write_text(code)
    shutil.copy2(SOURCE, package / "model.axon")
    readme = package / "README.md"
    if readme.exists():
        readme.write_text(
            readme.read_text().replace(
                "(../../synapse/models/timestral/generic-timestral.axon)", "(model.axon)"
            )
        )
    (package / "build-info.json").write_text(
        json.dumps(
            {
                "backend": "mlx",
                "optimize": optimize,
                "axon_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                "generated_sha256": hashlib.sha256(code.encode()).hexdigest(),
            },
            indent=2,
        )
        + "\n"
    )
    (output / "requirements.txt").write_text("mlx\nnumpy\nsafetensors\nhuggingface_hub\n")
    return package
