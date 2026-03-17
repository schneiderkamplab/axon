from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import typer
from omegaconf import OmegaConf

app = typer.Typer(help="Synapse tooling.")


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    loaded = OmegaConf.load(path)
    data = OmegaConf.to_container(loaded, resolve=True)
    if not isinstance(data, dict):
        raise typer.BadParameter(f"Expected YAML mapping at {path}, got {type(data).__name__}")
    return {str(key): value for key, value in data.items()}


def _emit_model_code(spec: dict[str, Any], class_name: str) -> str:
    module = importlib.import_module("brainsurgery.synapse")
    emit_fn = getattr(module, "emit_model_code_from_synapse_spec")
    return emit_fn(spec, class_name=class_name)


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
    if output_path.exists() and not force:
        raise typer.BadParameter(
            f"Refusing to overwrite existing file: {output_path}. Use --force to overwrite."
        )
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


__all__ = ["app", "emit_generic"]
