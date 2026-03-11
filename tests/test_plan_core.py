from __future__ import annotations

from pathlib import Path

import pytest

from brainsurgery.engine.plan import (
    PlanLoaderError,
    load_plan,
    parse_input_entry,
    parse_output,
    parse_output_mapping,
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

    with pytest.raises(PlanLoaderError, match="output.path is required"):
        parse_output_mapping({"format": "torch"})


def test_load_plan_reads_yaml_and_wraps_parse_errors(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.yaml"
    plan_path.write_text("transforms:\n  - help: {}\n", encoding="utf-8")

    loaded = load_plan(plan_path)
    assert len(loaded.transforms) == 1

    bad_path = tmp_path / "bad.yaml"
    bad_path.write_text("transforms: [", encoding="utf-8")
    with pytest.raises(PlanLoaderError, match="failed to parse yaml"):
        load_plan(bad_path)


def test_validate_model_aliases_rejects_unknown_models() -> None:
    with pytest.raises(PlanLoaderError, match="unknown model alias"):
        validate_model_aliases(_Spec(), {"base"}, index=0)
