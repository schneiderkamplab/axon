from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from brainsurgery.engine.plan import (
    PlanLoaderError,
    compile_plan,
    parse_inputs,
    parse_input_entry,
    parse_output,
    parse_output_mapping,
    parse_raw_transforms,
    validate_model_aliases,
)

class _Spec:
    def collect_models(self) -> set[str]:
        return {"missing"}

def test_parse_input_entry_and_output_cover_validation_branches() -> None:
    assert parse_input_entry("base::/tmp/model.safetensors") == ("base", Path("/tmp/model.safetensors"))
    assert parse_input_entry("/tmp/model.safetensors") == (None, Path("/tmp/model.safetensors"))
    assert parse_output("/tmp/out.safetensors").path == Path("/tmp/out.safetensors")
    assert parse_output({}) is None

    output = parse_output_mapping({"path": "/tmp/out", "format": "safetensors", "shard": "1GB"})
    assert output.path == Path("/tmp/out")
    assert output.format == "safetensors"
    assert output.shard == "1GB"
    dcp_output = parse_output_mapping({"path": "/tmp/dcp_out", "format": "dcp"})
    assert dcp_output.format == "dcp"

    with pytest.raises(PlanLoaderError, match="output.path is required"):
        parse_output_mapping({"format": "torch"})

def test_compile_plan_from_loaded_yaml(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.yaml"
    plan_path.write_text("transforms:\n  - help: {}\n", encoding="utf-8")

    loaded = compile_plan(yaml.safe_load(plan_path.read_text(encoding="utf-8")))
    assert len(loaded.transforms) == 1

def test_validate_model_aliases_rejects_unknown_models() -> None:
    with pytest.raises(PlanLoaderError, match="unknown model alias"):
        validate_model_aliases(_Spec(), {"base"}, index=0)

def test_compile_plan_rejects_non_mapping_root() -> None:
    with pytest.raises(PlanLoaderError, match="plan must be a YAML mapping"):
        compile_plan([])

def test_parse_inputs_validation_errors() -> None:
    with pytest.raises(PlanLoaderError, match="inputs must be a list"):
        parse_inputs("x")

    with pytest.raises(PlanLoaderError, match="must not be empty when multiple inputs"):
        parse_inputs(["::/tmp/a", "::/tmp/b"])

    with pytest.raises(PlanLoaderError, match="duplicate input alias"):
        parse_inputs(["a::/tmp/a", "a::/tmp/b"])

    with pytest.raises(PlanLoaderError, match="must be a non-empty string"):
        parse_input_entry("")

    with pytest.raises(PlanLoaderError, match="input path must not be empty"):
        parse_input_entry("a::")

def test_parse_output_validation_errors() -> None:
    with pytest.raises(PlanLoaderError, match="output must be either empty"):
        parse_output(1)

    assert parse_output("") is None

    with pytest.raises(PlanLoaderError, match="unknown keys"):
        parse_output_mapping({"path": "/tmp/out", "extra": 1})

    with pytest.raises(PlanLoaderError, match="output.path must be a non-empty string"):
        parse_output_mapping({"path": ""})

    with pytest.raises(PlanLoaderError, match="output.format must be a non-empty string"):
        parse_output_mapping({"path": "/tmp/out", "format": ""})

    with pytest.raises(PlanLoaderError, match="must be one of"):
        parse_output_mapping({"path": "/tmp/out", "format": "bad"})

    with pytest.raises(PlanLoaderError, match="output.shard must be a non-empty string"):
        parse_output_mapping({"path": "/tmp/out", "shard": ""})

class _NoCollectSpec:
    pass

def test_validate_model_aliases_rejects_non_collecting_spec() -> None:
    with pytest.raises(PlanLoaderError, match="does not expose collect_models"):
        validate_model_aliases(_NoCollectSpec(), {"base"}, index=1)

def test_parse_raw_transforms_rejects_non_list() -> None:
    with pytest.raises(PlanLoaderError, match="transforms must be a list"):
        parse_raw_transforms({})

def test_parse_transform_entry_validation_errors() -> None:
    with pytest.raises(PlanLoaderError, match="single-key mapping"):
        compile_plan({"transforms": ["bad"]})

    with pytest.raises(PlanLoaderError, match="operation name must be a non-empty string"):
        compile_plan({"transforms": [{1: {}}]})  # type: ignore[dict-item]

    with pytest.raises(PlanLoaderError, match="unknown transform"):
        compile_plan({"transforms": [{"unknown": {}}]})

    with pytest.raises(PlanLoaderError, match="missing required keys"):
        compile_plan({"inputs": ["model::/tmp/x.safetensors"], "transforms": [{"copy": {}}]})
