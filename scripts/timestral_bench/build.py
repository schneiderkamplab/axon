"""Compile the Timestral Axon definition into standalone MLX core modules.

Only this module imports the compiler (which imports PyTorch). Worker processes load
the written modules, so Metal timing runs without PyTorch in the process.
"""

import ast
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "synapse/models/timestral/generic-timestral.axon"
COMMON = ROOT / "synapse/axon/codegen2_common/core.py"
COMMON_IMPORT = "from synapse.axon.codegen2_common import ("
LOCAL_IMPORT = "from _axon_common import ("
# Backend intrinsics require an explicit opt-in. Both MLX fusions validate primitive
# provenance, independent of model names.
VARIANTS = {
    "unfused": None,
    "fused": "codegen2-mlx:__mlx_sdpa,__mlx_rmsnorm_scaled",
}


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def generate_core(variant="fused", *, source=SOURCE):
    """Return generated MLX source for one optimization variant of the Axon model."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; expected one of {sorted(VARIANTS)}")
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
    from synapse.axon.codegen2_mlx import emit_model_code_from_graph_ir

    program = resolve_axon_program_from_path(Path(source)).ast
    for transform in (
        normalize_closed_axon_file,
        elaborate_closed_axon_file,
        flatten_closed_axon_file,
        typecheck2_flat_axon_file,
    ):
        program = transform(program)
    graph = lower_axon_program_to_graph_ir(program)
    intrinsics = VARIANTS[variant]
    if intrinsics is not None:
        graph = optimize_graph_program(
            graph, config=GraphOptimizeConfig(backend_intrinsics=intrinsics)
        )
    code = emit_model_code_from_graph_ir(graph, class_name="TimestralCore")
    ast.parse(code)
    return code


def rewrite_common_import(code):
    """Point the generated helper import at a local copy of the backend-neutral helpers."""
    if code.count(COMMON_IMPORT) != 1:
        raise ValueError("generated runtime helper import changed; inspect the exporter")
    return code.replace(COMMON_IMPORT, LOCAL_IMPORT)


def _git_state():
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout
        return {"head": head, "dirty": bool(status.strip())}
    except (OSError, subprocess.CalledProcessError) as error:
        return {"head": None, "dirty": None, "error": repr(error)}


def write_cores(run_dir, variants=tuple(VARIANTS)):
    """Write ``<run_dir>/cores/`` with one generated module per variant; never overwrite."""
    cores = Path(run_dir) / "cores"
    if cores.exists():
        raise FileExistsError(f"cores exist: {cores}; choose a fresh run directory")
    cores.mkdir(parents=True)
    shutil.copy2(COMMON, cores / "_axon_common.py")
    shutil.copy2(SOURCE, cores / "model.axon")
    info = {
        "axon_source": str(SOURCE.relative_to(ROOT)),
        "axon_sha256": _sha256(SOURCE.read_bytes()),
        "common_source": str(COMMON.relative_to(ROOT)),
        "common_sha256": _sha256(COMMON.read_bytes()),
        "git": _git_state(),
        "variants": {},
    }
    for variant in variants:
        code = rewrite_common_import(generate_core(variant))
        path = cores / f"axon_core_{variant}.py"
        path.write_text(code)
        info["variants"][variant] = {
            "file": path.name,
            "backend_intrinsics": VARIANTS[variant],
            "sha256": _sha256(code.encode()),
            "lines": len(code.splitlines()),
        }
    (cores / "build-info.json").write_text(json.dumps(info, indent=2) + "\n")
    return cores
