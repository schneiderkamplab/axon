from __future__ import annotations

import pytest

from brainsurgery.core import MatchError, StructuredMatch, StructuredPathMatcher


def test_structured_path_matcher_matches_and_rewrites_captures() -> None:
    matcher = StructuredPathMatcher()

    match = matcher.match(["encoder", "$layer", "*tail"], "encoder.3.attn.proj")
    assert match == StructuredMatch(bindings={"layer": "3", "tail": ["attn", "proj"]})
    assert matcher.rewrite(["decoder", "${layer}", "*tail"], match) == "decoder.3.attn.proj"


def test_structured_path_matcher_supports_regex_captures() -> None:
    matcher = StructuredPathMatcher()

    match = matcher.match(["layer", "~idx::([0-9]+)"], "layer.12")
    assert match == StructuredMatch(bindings={"idx": "12"})
    assert matcher.rewrite(["copy_${idx}"], match) == "copy_12"


def test_structured_path_matcher_rejects_invalid_output_patterns() -> None:
    matcher = StructuredPathMatcher()
    match = StructuredMatch(bindings={"name": "proj"})

    with pytest.raises(MatchError, match="regex token not allowed"):
        matcher.rewrite(["~x::.*"], match)

    with pytest.raises(MatchError, match="unknown interpolation variable"):
        matcher.rewrite(["${missing}"], match)


def test_structured_path_matcher_rejects_binding_count_mismatches() -> None:
    matcher = StructuredPathMatcher()

    with pytest.raises(MatchError, match="binds 2 variables but regex has 1 capturing groups"):
        matcher.match(["~left,right::(x)"], "x")
