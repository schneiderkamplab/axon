from __future__ import annotations

from pathlib import Path

import pytest

from brainsurgery.interactive import (
    _collect_completion_candidates,
    _collect_payload_candidates,
    _infer_active_transform,
    _is_top_level_completion_position,
    parse_transform_block,
)


def test_parse_transform_block_accepts_canonical_help_mapping() -> None:
    parsed = parse_transform_block("help: { assert: all }")
    assert parsed == [{"help": {"assert": "all"}}]


def test_parse_transform_block_rejects_help_shorthand() -> None:
    with pytest.raises(ValueError):
        parse_transform_block("help: assert: all")


def test_collect_completion_candidates_includes_commands_keys_and_refs() -> None:
    candidates = _collect_completion_candidates(None)

    assert "copy:" in candidates
    assert "- copy:" not in candidates
    assert "from:" not in candidates
    assert "base::" not in candidates
    assert "ln_f.weight" not in candidates


def test_collect_completion_candidates_without_provider() -> None:
    candidates = _collect_completion_candidates(None)
    assert "help" in candidates
    assert "exit" in candidates
    assert "help:" not in candidates
    assert "exit:" not in candidates


def test_is_top_level_completion_position() -> None:
    assert _is_top_level_completion_position("", 0) is True
    assert _is_top_level_completion_position("co", 0) is True
    assert _is_top_level_completion_position("- ", 2) is True
    assert _is_top_level_completion_position("copy: {", 7) is False
    assert _is_top_level_completion_position("  from: x", 8) is False


class _DummyProvider:
    def __init__(self) -> None:
        self.model_paths = {"base": Path("/tmp/base.safetensors")}
        self.state_dicts = {
            "base": {"ln_f.weight": object()},
            "scratch": {"new.weight": object()},
        }

    def list_model_aliases(self) -> set[str]:
        return {"base", "scratch"}


def test_infer_active_transform_from_current_or_previous_lines() -> None:
    assert _infer_active_transform([], "copy: {") == "copy"
    assert _infer_active_transform(["copy: {"], "from: ") == "copy"


def test_collect_payload_candidates_include_keys_aliases_tensors_and_yaml_tokens() -> None:
    candidates = _collect_payload_candidates(
        active_transform="copy",
        state_dict_provider=_DummyProvider(),
    )
    assert "from: " in candidates
    assert "to: " in candidates
    assert "base::" in candidates
    assert "ln_f.weight" in candidates
    assert "base::ln_f.weight" in candidates
    assert "{ " in candidates
