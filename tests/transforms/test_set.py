import pytest

import brainsurgery.transforms.set as set_module
from brainsurgery.engine import (
    RuntimeFlagLifecycleScope,
    get_runtime_flags,
    reset_runtime_flags_for_scope,
    use_output_emitter,
)
from brainsurgery.transforms.set import SetSpec, SetTransform, SetTransformError


def test_set_compile_accepts_bool_and_string_forms() -> None:
    transform = SetTransform()

    spec = transform.compile({"dry-run": True, "preview": "T", "verbose": "F"}, default_model=None)
    assert spec == SetSpec(dry_run=True, preview=True, verbose=False)

    spec = transform.compile({"dry-run": "T", "verbose": "false"}, default_model=None)
    assert spec == SetSpec(dry_run=True, preview=None, verbose=False)


def test_set_compile_rejects_invalid_payloads() -> None:
    transform = SetTransform()

    with pytest.raises(SetTransformError, match="payload must be a mapping"):
        transform.compile("dry-run: true", default_model=None)

    spec = transform.compile({}, default_model=None)
    assert spec == SetSpec(dry_run=None, preview=None, verbose=None)

    with pytest.raises(SetTransformError, match="unknown keys"):
        transform.compile({"unknown": True}, default_model=None)

    with pytest.raises(SetTransformError, match="must be a boolean"):
        transform.compile({"dry-run": "yes"}, default_model=None)


def test_set_apply_updates_runtime_flags_and_reports_count() -> None:
    reset_runtime_flags_for_scope(RuntimeFlagLifecycleScope.CLI_RUN)


def test_set_apply_without_flags_prints_current_values_and_no_changes() -> None:
    reset_runtime_flags_for_scope(RuntimeFlagLifecycleScope.CLI_RUN)
    transform = SetTransform()
    lines: list[str] = []
    with use_output_emitter(lines.append):
        result = transform.apply(SetSpec(), provider=object())
    assert result.name == "set"
    assert result.count == 0
    assert lines == ["set flags: dry-run=False, preview=False, verbose=False"]
    transform = SetTransform()

    spec = SetSpec(dry_run=True, preview=None, verbose=None)
    result = transform.apply(spec, provider=object())

    assert result.name == "set"
    assert result.count == 1
    assert get_runtime_flags().dry_run is True
    assert get_runtime_flags().preview is False
    assert get_runtime_flags().verbose is False

    spec = SetSpec(dry_run=None, preview=True, verbose=True)
    result = transform.apply(spec, provider=object())
    assert result.count == 2
    assert get_runtime_flags().dry_run is True
    assert get_runtime_flags().preview is True
    assert get_runtime_flags().verbose is True

    reset_runtime_flags_for_scope(RuntimeFlagLifecycleScope.CLI_RUN)


def test_set_transform_infer_output_model_and_completion() -> None:
    transform = SetTransform()
    with pytest.raises(SetTransformError, match="does not infer an output model"):
        transform._infer_output_model(SetSpec())

    assert transform.contributes_output_model(SetSpec()) is False
    assert transform.completion_key_candidates("set: { ", "d") == ["dry-run: "]
    assert transform.completion_value_candidates("dry-run", "T", []) == ["T", "True"]
    assert transform.completion_value_candidates("unknown", "", []) is None


def test_set_parse_bool_direct_helper_error() -> None:
    with pytest.raises(SetTransformError, match="set.verbose must be a boolean"):
        set_module._parse_bool(1, field_name="verbose")
