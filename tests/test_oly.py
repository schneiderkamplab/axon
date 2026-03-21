from __future__ import annotations

import pytest
import yaml  # type: ignore[import-untyped]

from brainsurgery.cli.oly import _parse_oly_line, _Parser
from brainsurgery.cli.parse import _parse_transform_block

OLY_EXAMPLES = [
    "exit",
    "exit:",
    "exit: {}",
    "help: assert: all",
    "help: { assert: all }",
    "copy: from: ln_f.weight, to: ln_f_copy.weight",
    "copy: from: [*prefix, $i, attn, *suffix], to: [*prefix, $i, attention, *suffix]",
    "copy: from: [*prefix, $i, attn, *suffix], to: [*prefix, ${i}, attention, *suffix]",
    "copy: from: [layer, ~idx::([0-9]+), attn, *suffix], to: [layer, $idx, attention, *suffix]",
    "scale: from: model::h.0.attn.bias, to: model::h.0.attn.bias_scaled, by: 2.5",
    "assert: equal: { left: model::h.0.attn.bias, right: model::h.0.attn.bias, eps: 1e-6 }",
    "prefixes: mode: rename, from: work, to: merged",
]


@pytest.mark.parametrize("oly_text", OLY_EXAMPLES)
def test_oly_parse_roundtrips_through_yaml_parser(oly_text: str) -> None:
    ast = [_parse_oly_line(oly_text)]
    yaml_text = yaml.safe_dump(ast[0], sort_keys=False)
    assert _parse_transform_block(yaml_text) == ast


def test_parse_oly_rejects_duplicate_keys() -> None:
    with pytest.raises(ValueError, match="duplicate key"):
        _parse_oly_line("copy: from: a, from: b")


def test_parse_oly_rejects_unclosed_list() -> None:
    with pytest.raises(ValueError, match="expected ',' or ']' in list|unexpected end of input"):
        _parse_oly_line("copy: from: [a, b, to: c")


def test_parse_oly_exercises_string_and_map_error_paths() -> None:
    assert _parse_oly_line("copy: from: 'a b', to: \"x\\n\"") == {
        "copy": {"from": "a b", "to": "x\n"}
    }
    assert _parse_oly_line("copy: from: {}, to: []") == {"copy": {"from": {}, "to": []}}

    with pytest.raises(ValueError, match="unclosed single-quoted string"):
        _parse_oly_line("copy: from: 'abc")
    with pytest.raises(ValueError, match="unclosed double-quoted string"):
        _parse_oly_line('copy: from: "abc')
    with pytest.raises(ValueError, match="dangling escape in string"):
        _parse_oly_line('copy: from: "abc\\')
    with pytest.raises(ValueError, match="trailing comma in list"):
        _parse_oly_line("copy: from: [a,], to: b")
    with pytest.raises(ValueError, match="trailing comma in map"):
        _parse_oly_line("copy: from: { a: b, }, to: c")
    with pytest.raises(ValueError, match="expected key/value pair after comma"):
        _parse_oly_line("copy: from: a,")
    with pytest.raises(ValueError, match="expected ',' or '}' in map"):
        _parse_oly_line("copy: from: { a: b x }, to: c")
    with pytest.raises(ValueError, match="expected ',' between key/value pairs"):
        _parse_oly_line("copy: from: a to: b")
    with pytest.raises(ValueError, match="empty input"):
        _parse_oly_line("   ")
    with pytest.raises(ValueError, match="expected ',' between key/value pairs"):
        _parse_oly_line("copy: from: a, to: b !")
    assert _parse_oly_line("copy") == {"copy": {}}
    with pytest.raises(ValueError, match="expected value"):
        _parse_oly_line("copy: from: , to: b")
    with pytest.raises(ValueError, match="unexpected token after top-level map"):
        _parse_oly_line("copy: { from: a } junk")


def test_oly_scalar_coercion() -> None:
    assert _parse_oly_line("assert: a: true, b: false, c: null, d: ~, e: 1, f: 1.5") == {
        "assert": {"a": True, "b": False, "c": None, "d": None, "e": 1, "f": 1.5}
    }


def test_parser_private_advance_and_expect_errors() -> None:
    parser = _Parser("x")
    assert parser._advance() == "x"
    with pytest.raises(ValueError, match="unexpected end of input"):
        parser._advance()
    with pytest.raises(ValueError, match="expected ':'"):
        _Parser("x")._expect(":")
