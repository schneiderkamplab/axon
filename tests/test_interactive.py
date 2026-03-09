from __future__ import annotations

import pytest

from brainsurgery.interactive import parse_transform_block


def test_parse_transform_block_accepts_canonical_help_mapping() -> None:
    parsed = parse_transform_block("help: { assert: all }")
    assert parsed == [{"help": {"assert": "all"}}]


def test_parse_transform_block_rejects_help_shorthand() -> None:
    with pytest.raises(ValueError):
        parse_transform_block("help: assert: all")
