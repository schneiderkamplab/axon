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
    assert matcher.rewrite(["$name"], match) == "proj"


def test_structured_path_matcher_rejects_binding_count_mismatches() -> None:
    matcher = StructuredPathMatcher()

    with pytest.raises(MatchError, match="binds 2 variables but regex has 1 capturing groups"):
        matcher.match(["~left,right::(x)"], "x")


def test_structured_path_matcher_match_and_rewrite_returns_none_when_unmatched() -> None:
    matcher = StructuredPathMatcher()
    assert matcher.match_and_rewrite(
        from_pattern=["encoder", "$layer"],
        to_pattern=["decoder", "${layer}"],
        name="decoder.1",
    ) is None


def test_structured_path_matcher_rejects_invalid_capture_and_regex_forms() -> None:
    matcher = StructuredPathMatcher()

    with pytest.raises(MatchError, match="invalid capture name"):
        matcher.match(["$1bad"], "x")

    with pytest.raises(MatchError, match="invalid regex token"):
        matcher.match(["~::(x)"], "x")

    with pytest.raises(MatchError, match="missing regex body"):
        matcher.match(["~x::"], "x")

    with pytest.raises(MatchError, match="invalid structured regex"):
        matcher.match(["~x::("], "x")

    with pytest.raises(MatchError, match="named groups are not allowed"):
        matcher.match(["~x::(?P<v>x)"], "x")


def test_structured_path_matcher_rejects_regex_capture_none_values() -> None:
    matcher = StructuredPathMatcher()

    with pytest.raises(MatchError, match="captured None"):
        matcher.match(["~x::(a)?"], "")

    with pytest.raises(MatchError, match="captured None"):
        matcher.match(["~x,y::(a)?(b)?"], "")


def test_structured_path_matcher_rejects_invalid_variadic_output_bindings() -> None:
    matcher = StructuredPathMatcher()

    with pytest.raises(MatchError, match="unknown variadic variable"):
        matcher.rewrite(["*tail"], StructuredMatch(bindings={"x": "1"}))

    with pytest.raises(MatchError, match="is not variadic"):
        matcher.rewrite(["*tail"], StructuredMatch(bindings={"tail": "x"}))

    with pytest.raises(MatchError, match="contains non-string segments"):
        matcher.rewrite(["*tail"], StructuredMatch(bindings={"tail": ["x", 1]}))  # type: ignore[list-item]
