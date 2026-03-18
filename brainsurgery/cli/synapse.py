from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import typer
from omegaconf import OmegaConf
from typer.models import OptionInfo

app = typer.Typer(help="Synapse tooling.")


def _synapse_module() -> Any:
    return importlib.import_module("brainsurgery.synapse")


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    loaded = OmegaConf.load(path)
    data = OmegaConf.to_container(loaded, resolve=True)
    if not isinstance(data, dict):
        raise typer.BadParameter(f"Expected YAML mapping at {path}, got {type(data).__name__}")
    return {str(key): value for key, value in data.items()}


def _emit_model_code(spec: dict[str, Any], class_name: str) -> str:
    module = _synapse_module()
    emit_fn = getattr(module, "emit_model_code_from_synapse_spec")
    return emit_fn(spec, class_name=class_name)


def _parse_axon_to_synapse_spec(source: str, *, main_module: str | None = None) -> dict[str, Any]:
    module = _synapse_module()
    parse_fn = getattr(module, "parse_axon_program")
    lower_fn = getattr(module, "lower_axon_program_to_synapse_spec")
    parsed = parse_fn(source)
    return lower_fn(parsed, main_module=main_module)


def _render_synapse_to_axon_text(spec: dict[str, Any], *, module_name: str) -> str:
    module = _synapse_module()
    render_fn = getattr(module, "synapse_spec_to_axon_module_text")
    return render_fn(spec, module_name=module_name)


def _ensure_overwrite_allowed(path: Path, *, force: bool) -> None:
    if path.exists() and not force:
        raise typer.BadParameter(
            f"Refusing to overwrite existing file: {path}. Use --force to overwrite."
        )


@app.command("emit")
def emit_generic(
    spec_path: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Path to a Synapse YAML spec.",
    ),
    output_path: Path = typer.Argument(
        ...,
        help="Destination Python file for generated model code.",
    ),
    class_name: str = typer.Option(
        "GeneratedSynapseModel",
        "--class-name",
        help="Name of the generated model class.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Overwrite output file if it already exists.",
    ),
) -> None:
    """Generate standalone PyTorch model code from any Synapse YAML spec."""
    _ensure_overwrite_allowed(output_path, force=force)
    if output_path.suffix != ".py":
        raise typer.BadParameter("Output path must end with .py")

    spec = _load_yaml_mapping(spec_path)
    try:
        source = _emit_model_code(spec, class_name)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(source, encoding="utf-8")
    typer.echo(f"Wrote generated model code to {output_path}")


@app.command("axon-to-synapse")
def axon_to_synapse(
    axon_path: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Path to an Axon source file.",
    ),
    output_path: Path = typer.Argument(
        ...,
        help="Destination YAML file for lowered Synapse spec.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Overwrite output file if it already exists.",
    ),
    main_module: str | None = typer.Option(
        None,
        "--main-module",
        help="Main model module name when Axon file contains multiple modules (defaults to last).",
    ),
) -> None:
    """Lower an Axon module into a Synapse YAML spec."""
    _ensure_overwrite_allowed(output_path, force=force)
    if output_path.suffix not in {".yaml", ".yml"}:
        raise typer.BadParameter("Output path must end with .yaml or .yml")

    source = axon_path.read_text(encoding="utf-8")
    if isinstance(main_module, OptionInfo):
        main_module = None
    try:
        spec = _parse_axon_to_synapse_spec(source, main_module=main_module)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(OmegaConf.to_yaml(spec, resolve=True), encoding="utf-8")
    typer.echo(f"Wrote Synapse YAML to {output_path}")


@app.command("synapse-to-axon")
def synapse_to_axon(
    spec_path: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Path to a Synapse YAML spec.",
    ),
    output_path: Path = typer.Argument(
        ...,
        help="Destination Axon file.",
    ),
    module_name: str = typer.Option(
        "main",
        "--module-name",
        help="Module name to use in emitted Axon source.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Overwrite output file if it already exists.",
    ),
) -> None:
    """Render an Axon source file from a Synapse YAML spec."""
    _ensure_overwrite_allowed(output_path, force=force)
    if output_path.suffix != ".axon":
        raise typer.BadParameter("Output path must end with .axon")
    if not module_name.isidentifier():
        raise typer.BadParameter(f"Invalid module name: {module_name!r}")

    spec = _load_yaml_mapping(spec_path)
    try:
        text = _render_synapse_to_axon_text(spec, module_name=module_name)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    typer.echo(f"Wrote Axon source to {output_path}")


@app.command("axon-test")
def axon_test(
    axon_path: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Path to an Axon source file.",
    ),
    weights: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=True,
        readable=True,
        help="Path to a .safetensors file or a model directory containing safetensors.",
    ),
    device: str = typer.Option(
        "cpu",
        "--device",
        help="Torch device (cpu/auto/cuda/mps or explicit like cuda:0).",
    ),
    text: list[str] = typer.Option(
        ["The future of AI is"],
        "--text",
        help="Prompt text to complete. Repeat --text for batched prompts.",
    ),
    max_len: int = typer.Option(
        32,
        "--max-len",
        help="Total sequence length target for generation.",
    ),
    hf_model_dir: Path | None = typer.Option(
        None,
        "--hf-model-dir",
        help="HF model directory for AutoModel (defaults to weights directory).",
    ),
    tokenizer: str | None = typer.Option(
        None,
        "--tokenizer",
        help="Tokenizer source override (local path or HF repo id).",
    ),
    class_name: str = typer.Option(
        "AxonGeneratedModel",
        "--class-name",
        help="Generated PyTorch class name.",
    ),
    main_module: str | None = typer.Option(
        None,
        "--main-module",
        help="Main Axon module name (defaults to last module in file).",
    ),
    dtype: str = typer.Option(
        "float32",
        "--dtype",
        help="Floating-point dtype for loaded safetensors parameters (float32/bfloat16/float16).",
    ),
    strip_model_prefix: bool = typer.Option(
        False,
        "--strip-model-prefix",
        help="Strip leading 'model.' from safetensors keys.",
    ),
) -> None:
    """Run HF vs Axon-derived model benchmark for an Axon spec + weights."""
    module = _synapse_module()
    run_fn = getattr(module, "run_axon_test")
    try:
        run_fn(
            axon_file=axon_path,
            weights=weights,
            device=device,
            text=text,
            max_len=max_len,
            hf_model_dir=hf_model_dir,
            tokenizer=tokenizer,
            class_name=class_name,
            main_module=main_module,
            dtype=dtype,
            strip_model_prefix=strip_model_prefix,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


__all__ = ["app", "emit_generic", "axon_to_synapse", "synapse_to_axon", "axon_test"]
