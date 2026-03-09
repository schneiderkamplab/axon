from __future__ import annotations

from pathlib import Path

import pytest

from brainsurgery.interactive import _collect_completion_candidates, parse_transform_block


def test_parse_transform_block_accepts_canonical_help_mapping() -> None:
    parsed = parse_transform_block("help: { assert: all }")
    assert parsed == [{"help": {"assert": "all"}}]


def test_parse_transform_block_rejects_help_shorthand() -> None:
    with pytest.raises(ValueError):
        parse_transform_block("help: assert: all")


class _DummyProvider:
    def __init__(self) -> None:
        self.model_paths = {"base": Path("/tmp/base.safetensors")}
        self.state_dicts = {
            "base": {"ln_f.weight": object()},
            "scratch": {"new.weight": object()},
        }

    def list_model_aliases(self) -> set[str]:
        return {"base", "scratch"}


def test_collect_completion_candidates_includes_commands_keys_and_refs() -> None:
    candidates = _collect_completion_candidates(_DummyProvider())

    assert "copy" in candidates
    assert "copy:" in candidates
    assert "from:" in candidates
    assert "to:" in candidates
    assert "base::" in candidates
    assert "ln_f.weight" in candidates
    assert "base::ln_f.weight" in candidates


def test_collect_completion_candidates_without_provider() -> None:
    candidates = _collect_completion_candidates(None)
    assert "help" in candidates
    assert "exit" in candidates
