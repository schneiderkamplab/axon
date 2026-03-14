from __future__ import annotations

import pytest

from brainsurgery.cli.oly import _Parser, emit_oly_line, emit_yaml_transform, parse_oly_line
from brainsurgery.cli.parse import parse_transform_block


YAML_EXAMPLES = [
    "exit",
    "help: { assert: all }",
    "copy: { from: ln_f.weight, to: ln_f_copy.weight }",
    'copy: { from: ["*prefix", "$i", "attn", "*suffix"], to: ["*prefix", "${i}", "attention", "*suffix"] }',
    'copy: { from: ["layer", "~idx::([0-9]+)", "attn", "*suffix"], to: ["layer", "$idx", "attention", "*suffix"] }',
    "scale: { from: model::h.0.attn.bias, to: model::h.0.attn.bias_scaled, by: 2.5 }",
    "assert: { equal: { left: model::h.0.attn.bias, right: model::h.0.attn.bias, eps: 1e-6 } }",
    "prefixes: { mode: rename, from: work, to: merged }",
]


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


@pytest.mark.parametrize("yaml_text", YAML_EXAMPLES)
def test_yaml_to_oly_to_yaml_roundtrip_ast_equality(yaml_text: str) -> None:
    ast = parse_transform_block(yaml_text)
    assert len(ast) == 1

    oly = emit_oly_line(ast[0])
    ast_from_oly = [parse_oly_line(oly)]
    assert ast_from_oly == ast

    yaml_from_oly = emit_yaml_transform(ast_from_oly[0])
    ast_from_yaml = parse_transform_block(yaml_from_oly)
    assert ast_from_yaml == ast


@pytest.mark.parametrize("oly_text", OLY_EXAMPLES)
def test_oly_to_yaml_to_oly_roundtrip_ast_equality(oly_text: str) -> None:
    ast = [parse_oly_line(oly_text)]

    yaml_text = emit_yaml_transform(ast[0])
    ast_from_yaml = parse_transform_block(yaml_text)
    assert ast_from_yaml == ast

    oly_from_yaml = emit_oly_line(ast_from_yaml[0])
    ast_from_oly = [parse_oly_line(oly_from_yaml)]
    assert ast_from_oly == ast


def test_parse_oly_rejects_duplicate_keys() -> None:
    with pytest.raises(ValueError, match="duplicate key"):
        parse_oly_line("copy: from: a, from: b")


def test_parse_oly_rejects_unclosed_list() -> None:
    with pytest.raises(ValueError, match="expected ',' or ']' in list|unexpected end of input"):
        parse_oly_line("copy: from: [a, b, to: c")


def test_parse_oly_exercises_string_and_map_error_paths() -> None:
    assert parse_oly_line("copy: from: 'a b', to: \"x\\n\"") == {
        "copy": {"from": "a b", "to": "x\n"}
    }
    assert parse_oly_line("copy: from: {}, to: []") == {"copy": {"from": {}, "to": []}}

    with pytest.raises(ValueError, match="unclosed single-quoted string"):
        parse_oly_line("copy: from: 'abc")
    with pytest.raises(ValueError, match="unclosed double-quoted string"):
        parse_oly_line('copy: from: "abc')
    with pytest.raises(ValueError, match="dangling escape in string"):
        parse_oly_line('copy: from: "abc\\')
    with pytest.raises(ValueError, match="trailing comma in list"):
        parse_oly_line("copy: from: [a,], to: b")
    with pytest.raises(ValueError, match="trailing comma in map"):
        parse_oly_line("copy: from: { a: b, }, to: c")
    with pytest.raises(ValueError, match="expected key/value pair after comma"):
        parse_oly_line("copy: from: a,")
    with pytest.raises(ValueError, match="expected ',' or '}' in map"):
        parse_oly_line("copy: from: { a: b x }, to: c")
    with pytest.raises(ValueError, match="expected ',' between key/value pairs"):
        parse_oly_line("copy: from: a to: b")
    with pytest.raises(ValueError, match="empty input"):
        parse_oly_line("   ")
    with pytest.raises(ValueError, match="expected ',' between key/value pairs"):
        parse_oly_line("copy: from: a, to: b !")
    assert parse_oly_line("copy") == {"copy": {}}
    with pytest.raises(ValueError, match="expected value"):
        parse_oly_line("copy: from: , to: b")
    with pytest.raises(ValueError, match="unexpected token after top-level map"):
        parse_oly_line("copy: { from: a } junk")


def test_oly_scalar_coercion_and_emit_branches() -> None:
    assert parse_oly_line("assert: a: true, b: false, c: null, d: ~, e: 1, f: 1.5") == {
        "assert": {"a": True, "b": False, "c": None, "d": None, "e": 1, "f": 1.5}
    }
    assert emit_oly_line({"assert": {"a": None, "b": True, "c": False}}) == "assert: { a: null, b: true, c: false }"
    with pytest.raises(TypeError, match="unsupported OLY value type"):
        emit_oly_line({"copy": {"from": object()}})
    with pytest.raises(ValueError, match="exactly one transform"):
        emit_oly_line({"copy": {}, "move": {}})
    with pytest.raises(ValueError, match="valid identifier"):
        emit_oly_line({"1copy": {}})
    with pytest.raises(ValueError, match="payload must be a mapping"):
        emit_oly_line({"copy": []})  # type: ignore[arg-type]
    assert emit_oly_line({"copy": None}) == "copy: {}"


def test_parser_private_advance_and_expect_errors() -> None:
    parser = _Parser("x")
    assert parser._advance() == "x"
    with pytest.raises(ValueError, match="unexpected end of input"):
        parser._advance()
    with pytest.raises(ValueError, match="expected ':'"):
        _Parser("x")._expect(":")


def test_oly_additional_emit_and_parse_paths() -> None:
    assert parse_oly_line("copy: from: a, to: b") == {"copy": {"from": "a", "to": "b"}}
    assert emit_oly_line({"copy": {"nested": {"k": "v"}}}) == 'copy: { nested: { k: v } }'
    with pytest.raises(ValueError, match="payload must be a mapping"):
        emit_oly_line({"copy": "bad"})  # type: ignore[arg-type]
