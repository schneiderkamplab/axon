"""Load a materialized ``.axon`` model into a generated, backend-native model.

This reuses the exact compiler pipeline the test/benchmark harnesses use
(resolve → normalize → elaborate → flatten → typecheck → lower → optimize)
and the generated model classes' own weight-loading classmethods, so the
serving path cannot drift from the parity-tested path.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

from synapse.axon import (
    GraphOptimizeConfig,
    elaborate_closed_axon_file,
    flatten_closed_axon_file,
    lower_axon_program_to_graph_ir,
    normalize_closed_axon_file,
    optimize_graph_program,
    optimize_safe_flat_typed_axon_file,
    resolve_axon_program_from_path,
    typecheck2_flat_axon_file,
)

__all__ = [
    "MLX_INTRINSICS_SPEC",
    "default_checkpoint_dir",
    "load_mlx_model",
    "load_torch_model",
    "resolve_safetensors_paths",
]

# Default opt-in MLX fast-path intrinsics (matches the harness default).
MLX_INTRINSICS_SPEC = "codegen2-mlx:__mlx_sdpa,__mlx_rope"

_VALID_EMIT_PROFILES = {"inference", "serving"}


def default_checkpoint_dir(axon_path: Path) -> Path:
    """Default checkpoint location: ``<repo>/models/<model-family>/``.

    Materialized model files live at ``synapse/models/<family>/generic-*.axon``
    and the downloaded HF checkpoints at ``models/<family>/``.
    """
    family = axon_path.resolve().parent.name
    repo_root = _find_repo_root(axon_path.resolve())
    return repo_root / "models" / family


def _find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    return Path.cwd()


def resolve_safetensors_paths(checkpoint_dir: Path) -> list[Path]:
    """Shard resolution: every ``*.safetensors`` file in the checkpoint dir."""
    files = sorted(checkpoint_dir.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(
            f"No *.safetensors files found in checkpoint dir: {checkpoint_dir}"
        )
    return files


def load_model_config(checkpoint_dir: Path) -> dict[str, Any]:
    """Minimal HF ``config.json`` reader (the harness augments more; this is enough
    for the generated models' config lookups: hidden sizes, eos id, rope, ...)."""
    config_path = checkpoint_dir / "config.json"
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    return loaded if isinstance(loaded, dict) else {}


def _compile_graph_program(
    axon_path: Path,
    *,
    checkpoint_dir: Path | str | None,
    optimize_ast: bool,
    optimize_graph: bool,
    backend_intrinsics: str | None,
) -> tuple[Any, dict[str, Any]]:
    """Run the harness pipeline through the graph IR; returns (graph, config).

    Shared by every backend loader so the serving path cannot drift from the
    parity-tested compile path.
    """
    resolved_axon = resolve_axon_program_from_path(axon_path).ast
    normalized_axon = normalize_closed_axon_file(resolved_axon)
    elaborated_axon = elaborate_closed_axon_file(normalized_axon)
    flat_axon = flatten_closed_axon_file(elaborated_axon)
    typed_axon = typecheck2_flat_axon_file(flat_axon)
    if optimize_ast:
        typed_axon = optimize_safe_flat_typed_axon_file(typed_axon)
    graph_program = lower_axon_program_to_graph_ir(typed_axon)

    checkpoint = (
        Path(checkpoint_dir).resolve() if checkpoint_dir else default_checkpoint_dir(axon_path)
    )
    model_config = load_model_config(checkpoint)

    # Preserve the embedding scale the graph optimizer drops (e.g. Gemma's
    # sqrt(hidden_size)) — same extraction as the harness.
    if "embedding_scale" not in model_config:
        for module in graph_program.modules:
            for node in module.nodes:
                if node.op.name == "NN.embedding" and len(node.inputs) >= 4:
                    scale_input = node.inputs[3]
                    is_null = hasattr(scale_input, "value") and scale_input.value is None
                    if not is_null:
                        hidden = model_config.get("hidden_size")
                        if isinstance(hidden, (int, float)) and hidden > 0:
                            model_config["embedding_scale"] = hidden**0.5
                    break

    if optimize_graph:
        graph_program = optimize_graph_program(
            graph_program,
            config=GraphOptimizeConfig(backend_intrinsics=backend_intrinsics),
        )
    return graph_program, model_config


def load_mlx_model(
    axon_path: Path | str,
    *,
    checkpoint_dir: Path | str | None = None,
    dtype: str = "float16",
    optimize_graph: bool = True,
    optimize_ast: bool = True,
    emit_profile: str = "serving",
    backend_intrinsics: str | None = MLX_INTRINSICS_SPEC,
) -> Any:
    """Compile a materialized ``.axon`` file and instantiate the MLX model.

    ``emit_profile`` selects the generated artifact shape:
    - ``"inference"``: forward + fast-path ``compile()``, no standalone
      ``generate()`` loop (the ``Session`` engine owns the decode loop).
    - ``"serving"``: full artifact including ``generate()`` (HF-compatible use).
    """
    if emit_profile not in _VALID_EMIT_PROFILES:
        raise ValueError(
            f"emit_profile must be one of {sorted(_VALID_EMIT_PROFILES)}, got {emit_profile!r}"
        )
    path = Path(axon_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Axon model file not found: {path}")

    graph_program, model_config = _compile_graph_program(
        path,
        checkpoint_dir=checkpoint_dir,
        optimize_ast=optimize_ast,
        optimize_graph=optimize_graph,
        backend_intrinsics=backend_intrinsics,
    )

    from synapse.axon.codegen2_mlx import emit_model_code_from_graph_ir as emit_mlx

    class_name = _class_name_for(path, suffix="Mlx")
    code = emit_mlx(
        graph_program,
        class_name=class_name,
        model_config=model_config,
        emit_profile=emit_profile,
    )
    model_cls = _exec_generated_class(code, class_name)

    shards = resolve_safetensors_paths(Path(checkpoint_dir).resolve() if checkpoint_dir else default_checkpoint_dir(path))
    return model_cls.from_safetensors(shards, model_config=model_config, dtype=dtype).eval()


def load_torch_model(
    axon_path: Path | str,
    *,
    checkpoint_dir: Path | str | None = None,
    dtype: str = "float32",
    optimize_graph: bool = True,
    optimize_ast: bool = True,
    emit_profile: str = "inference",
    device: str = "cpu",
) -> Any:
    """Compile a materialized ``.axon`` file and instantiate the torch model.

    Mirrors ``load_mlx_model`` with the torch emitter and torch weight
    loading; ``device`` places the model (CPU is the agnosticism proof path).
    """
    if emit_profile not in _VALID_EMIT_PROFILES:
        raise ValueError(
            f"emit_profile must be one of {sorted(_VALID_EMIT_PROFILES)}, got {emit_profile!r}"
        )
    path = Path(axon_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Axon model file not found: {path}")

    graph_program, model_config = _compile_graph_program(
        path,
        checkpoint_dir=checkpoint_dir,
        optimize_ast=optimize_ast,
        optimize_graph=optimize_graph,
        # No intrinsics on the default torch path: torch has no opted-in
        # backend intrinsics yet; the plain graph lowering is the contract.
        backend_intrinsics=None,
    )

    from synapse.axon.codegen2_torch import emit_model_code_from_graph_ir as emit_torch

    class_name = _class_name_for(path, suffix="Torch")
    code = emit_torch(
        graph_program,
        class_name=class_name,
        model_config=model_config,
        # codegen2-torch does not take emit profiles yet (the serving-profile
        # artifact shape is 2b work); its `profile` arg is the optional
        # timing-profiler flag and must stay False.
        profile=False,
    )
    model_cls = _exec_generated_class(code, class_name)

    checkpoint = Path(checkpoint_dir).resolve() if checkpoint_dir else default_checkpoint_dir(path)
    shards = resolve_safetensors_paths(checkpoint)
    state_dict = _load_torch_state_dict(shards, device="cpu", dtype=dtype)
    return model_cls.from_state_dict(state_dict).to(device).eval()


def _load_torch_state_dict(paths: list[Path], *, device: str, dtype: str) -> dict[str, Any]:
    """Minimal safetensors -> torch state dict (the harness loader's core loop).

    Exotic model-specific branches (fp8 dequant, mxfp4 aliases, ...) stay in
    the harness until the server actually serves such models; at that point
    the helper moves into a shared module rather than being re-duplicated.
    """
    import torch
    from safetensors import safe_open

    torch_dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
    out: dict[str, Any] = {}
    for path in paths:
        with safe_open(str(path), framework="pt") as handle:
            for key in handle.keys():
                if key in out:
                    raise ValueError(f"Duplicate tensor key while reading safetensors shards: {key}")
                tensor = handle.get_tensor(key)
                if tensor.is_floating_point():
                    tensor = tensor.to(device=device, dtype=torch_dtype)
                else:
                    tensor = tensor.to(device=device)
                out[key] = tensor
    return out


def _class_name_for(axon_path: Path, suffix: str = "Mlx") -> str:
    stem = axon_path.stem.replace("-", "_").replace(".", "_")
    return f"{''.join(part.capitalize() for part in stem.split('_'))}{suffix}"


def _exec_generated_class(code: str, class_name: str) -> type[Any]:
    module_name = f"_axon_serve_generated_{int(time.time() * 1e9)}"
    compiled = compile(code, f"<axon_serve:{class_name}>", "exec")
    module = importlib.util.module_from_spec(
        importlib.util.spec_from_loader(module_name, loader=None)  # type: ignore[arg-type]
    )
    sys.modules[module_name] = module
    exec(compiled, module.__dict__)  # noqa: S102 - generated, trusted repo code
    model_cls = getattr(module, class_name, None)
    if model_cls is None:
        raise RuntimeError(f"Generated class {class_name!r} not found in emitted module")
    return model_cls
