from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import pytest

from brainsurgery.interactive import (
    _collect_completion_candidates,
    _collect_payload_candidates,
    _infer_active_transform,
    _is_transform_payload_start,
    _match_payload_candidates,
    _payload_context,
    _render_completion_preview,
    _is_top_level_completion_position,
    parse_transform_block,
    prompt_interactive_transform,
)


def test_parse_transform_block_accepts_canonical_help_mapping() -> None:
    parsed = parse_transform_block("help: { assert: all }")
    assert parsed == [{"help": {"assert": "all"}}]


def test_parse_transform_block_rejects_help_shorthand() -> None:
    with pytest.raises(ValueError):
        parse_transform_block("help: assert: all")


@contextmanager
def _no_completion(*args: object, **kwargs: object):
    del args, kwargs
    yield


def test_prompt_interactive_transform_ctrl_c_at_fresh_prompt_restarts(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = iter([KeyboardInterrupt(), EOFError()])

    def fake_input(prompt: str) -> str:
        response = next(responses)
        if isinstance(response, BaseException):
            raise response
        return response

    monkeypatch.setattr("brainsurgery.interactive._interactive_completion", _no_completion)
    monkeypatch.setattr("builtins.input", fake_input)

    assert prompt_interactive_transform() is None


def test_prompt_interactive_transform_ctrl_c_discards_partial_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = iter(
        [
            "copy: {",
            KeyboardInterrupt(),
            "exit",
            "",
        ]
    )
    history_entries: list[str] = []

    def fake_input(prompt: str) -> str:
        response = next(responses)
        if isinstance(response, BaseException):
            raise response
        return response

    monkeypatch.setattr("brainsurgery.interactive._interactive_completion", _no_completion)
    monkeypatch.setattr("brainsurgery.interactive.add_history_entry", history_entries.append)
    monkeypatch.setattr("builtins.input", fake_input)

    assert prompt_interactive_transform() == [{"exit": {}}]
    assert history_entries == ["exit"]


def test_collect_completion_candidates_includes_commands_keys_and_refs() -> None:
    candidates = _collect_completion_candidates(None)

    assert "copy: " in candidates
    assert "- copy:" not in candidates
    assert "from:" not in candidates
    assert "base::" not in candidates
    assert "ln_f.weight" not in candidates


def test_collect_completion_candidates_without_provider() -> None:
    candidates = _collect_completion_candidates(None)
    assert "help: " in candidates
    assert "exit" in candidates
    assert "prefixes" in candidates
    assert "help" not in candidates
    assert "exit:" not in candidates
    assert "prefixes: " not in candidates


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
    assert ": " not in candidates


def test_payload_context_key_and_value_detection() -> None:
    assert _payload_context("copy: { ") == "key"
    assert _payload_context("copy: { from: ") == "value"
    assert _payload_context("copy: { from: x, ") == "key"


def test_match_payload_candidates_filters_by_prefix_and_context() -> None:
    candidates = [
        "from: ",
        "to: ",
        "base",
        "base::",
        "ln_f.weight",
        "base::ln_f.weight",
        "{ ",
    ]
    key_matches = _match_payload_candidates(
        text="f",
        line_buffer="copy: { f",
        begidx=len("copy: { f"),
        payload_candidates=candidates,
    )
    assert key_matches == ["from: "]

    value_matches = _match_payload_candidates(
        text="ba",
        line_buffer="copy: { from: ba",
        begidx=len("copy: { from: ba"),
        payload_candidates=candidates,
        active_transform="copy",
    )
    assert "base::" in value_matches
    assert "base" not in value_matches
    assert "from: " not in value_matches

    ref_matches = _match_payload_candidates(
        text="base::",
        line_buffer="copy: { from: base::",
        begidx=len("copy: { from: base::"),
        payload_candidates=candidates,
        active_transform="copy",
    )
    assert "base::" in ref_matches
    assert "base::ln_f.weight" in ref_matches


def test_transform_payload_start_only_suggests_open_brace() -> None:
    assert _is_transform_payload_start(
        line_buffer="copy: ",
        begidx=len("copy: "),
        active_transform="copy",
    )
    matches = _match_payload_candidates(
        text="",
        line_buffer="copy: ",
        begidx=len("copy: "),
        payload_candidates=["base::x", "from: ", "{ "],
    )
    assert matches == ["{ "]


def test_copy_mapping_start_suggests_keys_not_yaml_colon() -> None:
    matches = _match_payload_candidates(
        text="",
        line_buffer="copy: { ",
        begidx=len("copy: { "),
        payload_candidates=["from: ", "to: ", "{ ", "}"],
    )
    assert "from: " in matches
    assert "to: " in matches
    assert "{ " not in matches


def test_key_context_filters_already_used_keys() -> None:
    matches = _match_payload_candidates(
        text="",
        line_buffer="copy: { from: ln_f.weight, ",
        begidx=len("copy: { from: ln_f.weight, "),
        payload_candidates=["from: ", "to: "],
        active_transform="copy",
    )
    assert matches == ["to: "]


def test_value_context_for_reference_key_shows_aliases_and_tensors() -> None:
    matches = _match_payload_candidates(
        text="",
        line_buffer="copy: { from: ",
        begidx=len("copy: { from: "),
        payload_candidates=["base", "base::", "base::ln_f.weight", "from: ", "{ ", "}"],
        active_transform="copy",
    )
    assert "base::" in matches
    assert "base" not in matches
    assert "base::ln_f.weight" in matches
    assert "from: " not in matches


def test_value_context_short_prefix_keeps_reference_candidates() -> None:
    matches = _match_payload_candidates(
        text="b",
        line_buffer="copy: { from: b",
        begidx=len("copy: { from: b"),
        payload_candidates=["base", "base::", "base::ln_f.weight", "from: "],
        active_transform="copy",
    )
    assert "base::" in matches
    assert "base::ln_f.weight" in matches
    assert "b, to: " not in matches


def test_reference_value_completion_prefers_aliases_when_multiple_aliases_exist() -> None:
    matches = _match_payload_candidates(
        text="",
        line_buffer="copy: { from: ",
        begidx=len("copy: { from: "),
        payload_candidates=[
            "base",
            "base::",
            "scratch",
            "scratch::",
            "ln_f.weight",
            "base::ln_f.weight",
            "scratch::ln_f.weight",
            "from: ",
        ],
        active_transform="copy",
        model_aliases=["base", "scratch"],
    )
    assert matches == ["base::", "scratch::"]


def test_reference_value_prefix_with_multiple_aliases_filters_to_aliases() -> None:
    matches = _match_payload_candidates(
        text="s",
        line_buffer="copy: { from: s",
        begidx=len("copy: { from: s"),
        payload_candidates=[
            "base",
            "base::",
            "scratch",
            "scratch::",
            "scratch::ln_f.weight",
            "shared.weight",
            "from: ",
        ],
        active_transform="copy",
        model_aliases=["base", "scratch"],
    )
    assert matches == ["scratch::"]


def test_reference_completion_adds_copy_to_continuation_snippet() -> None:
    matches = _match_payload_candidates(
        text="base::ln_f.weight",
        line_buffer="copy: { from: base::ln_f.weight",
        begidx=len("copy: { from: base::ln_f.weight"),
        payload_candidates=["base::", "base::ln_f.weight"],
        active_transform="copy",
    )
    assert matches == ["base::ln_f.weight", "base::ln_f.weight, to: "]


def test_reference_completion_does_not_add_continuation_for_bare_alias() -> None:
    matches = _match_payload_candidates(
        text="base::",
        line_buffer="copy: { from: base::",
        begidx=len("copy: { from: base::"),
        payload_candidates=["base::", "base::ln_f.weight"],
        active_transform="copy",
        model_aliases=["base"],
    )
    assert matches == ["base::", "base::ln_f.weight"]


def test_reference_completion_adds_assign_to_continuation_snippet() -> None:
    matches = _match_payload_candidates(
        text="base::ln_f.weight",
        line_buffer="assign: { from: base::ln_f.weight",
        begidx=len("assign: { from: base::ln_f.weight"),
        payload_candidates=["base::", "base::ln_f.weight"],
        active_transform="assign",
    )
    assert matches == ["base::ln_f.weight", "base::ln_f.weight, to: "]


def test_reference_completion_adds_ternary_next_reference_key() -> None:
    matches = _match_payload_candidates(
        text="base::a.weight",
        line_buffer="add: { from_a: base::a.weight",
        begidx=len("add: { from_a: base::a.weight"),
        payload_candidates=["base::", "base::a.weight"],
        active_transform="add",
    )
    assert matches == ["base::a.weight", "base::a.weight, from_b: "]


def test_copy_after_both_keys_only_suggests_close_brace() -> None:
    matches = _match_payload_candidates(
        text="",
        line_buffer="copy: { from: model::h.0.attn.bias, to: model::h.1.attn.bias ",
        begidx=len("copy: { from: model::h.0.attn.bias, to: model::h.1.attn.bias "),
        payload_candidates=["from: ", "to: ", "model::h.0.attn.bias", "}"],
        active_transform="copy",
    )
    assert matches == ["}"]


def test_copy_after_completed_to_with_trailing_space_only_suggests_close_brace() -> None:
    matches = _match_payload_candidates(
        text="",
        line_buffer="copy: { from: model::h.0.attn.bias, to: model::h.1.attn.bias ",
        begidx=len("copy: { from: model::h.0.attn.bias, to: model::h.1.attn.bias "),
        payload_candidates=["from: ", "to: ", "model::h.0.attn.bias", "model::h.1.attn.bias", "}"],
        active_transform="copy",
    )
    assert matches == ["}"]


def test_copy_after_first_key_suggests_remaining_key_and_close() -> None:
    matches = _match_payload_candidates(
        text="",
        line_buffer="copy: { from: model::h.0.attn.bias ",
        begidx=len("copy: { from: model::h.0.attn.bias "),
        payload_candidates=["from: ", "to: ", "model::h.0.attn.bias", "}"],
        active_transform="copy",
    )
    assert ", to: " in matches
    assert "}" in matches


def test_copy_after_comma_without_space_tab_can_insert_spaced_next_key() -> None:
    matches = _match_payload_candidates(
        text="model::h.0.attn.bias,",
        line_buffer="copy: { from: model::h.0.attn.bias,",
        begidx=len("copy: { from: "),
        endidx=len("copy: { from: model::h.0.attn.bias,"),
        payload_candidates=["from: ", "to: ", "model::h.0.attn.bias", "}"],
        active_transform="copy",
    )
    assert matches == ["model::h.0.attn.bias, to: "]


def test_help_value_completion_suggests_commands() -> None:
    matches = _match_payload_candidates(
        text="",
        line_buffer="help: ",
        begidx=len("help: "),
        payload_candidates=["{ ", "}"],
        active_transform="help",
    )
    assert "copy" in matches
    assert "exit" in matches


def test_assert_value_completion_suggests_assert_expressions() -> None:
    matches = _match_payload_candidates(
        text="",
        line_buffer="assert: ",
        begidx=len("assert: "),
        payload_candidates=["{ ", "}"],
        active_transform="assert",
    )
    assert "equal" in matches
    assert "exists" in matches


def test_assert_mapping_key_completion_suggests_assert_expressions() -> None:
    matches = _match_payload_candidates(
        text="",
        line_buffer="assert: { ",
        begidx=len("assert: { "),
        payload_candidates=["{ ", "}"],
        active_transform="assert",
    )
    assert "equal: " in matches
    assert "exists: " in matches


def test_help_mapping_key_completion_suggests_command_keys() -> None:
    matches = _match_payload_candidates(
        text="",
        line_buffer="help: { ",
        begidx=len("help: { "),
        payload_candidates=["{ ", "}"],
        active_transform="help",
    )
    assert "assert: " in matches
    assert "copy: " not in matches


def test_help_mapping_start_with_open_brace_only_suggests_assert_key() -> None:
    matches = _match_payload_candidates(
        text="{",
        line_buffer="help: {",
        begidx=len("help: "),
        endidx=len("help: {"),
        payload_candidates=["{ ", "}"],
        active_transform="help",
    )
    assert matches == ["{ assert: "]


def test_help_assert_value_completion_suggests_assert_expressions() -> None:
    matches = _match_payload_candidates(
        text="",
        line_buffer="help: { assert: ",
        begidx=len("help: { assert: "),
        payload_candidates=["{ ", "}"],
        active_transform="help",
    )
    assert "equal" in matches
    assert "exists" in matches


def test_help_assert_committed_value_only_suggests_close_brace() -> None:
    matches = _match_payload_candidates(
        text="",
        line_buffer="help: { assert: not ",
        begidx=len("help: { assert: not "),
        payload_candidates=["assert: ", "{ ", "}"],
        active_transform="help",
    )
    assert matches == ["}"]


def test_prefixes_key_suggestions_depend_on_mode() -> None:
    matches_remove = _match_payload_candidates(
        text="",
        line_buffer="prefixes: { mode: remove, ",
        begidx=len("prefixes: { mode: remove, "),
        payload_candidates=["mode: ", "alias: ", "from: ", "to: ", "}"],
        active_transform="prefixes",
    )
    assert "alias: " in matches_remove
    assert "from: " not in matches_remove
    assert "to: " not in matches_remove

    matches_rename = _match_payload_candidates(
        text="",
        line_buffer="prefixes: { mode: rename, ",
        begidx=len("prefixes: { mode: rename, "),
        payload_candidates=["mode: ", "alias: ", "from: ", "to: ", "}"],
        active_transform="prefixes",
    )
    assert "from: " in matches_rename
    assert "to: " in matches_rename
    assert "alias: " not in matches_rename


def test_prefixes_alias_value_completion_uses_aliases() -> None:
    matches = _match_payload_candidates(
        text="s",
        line_buffer="prefixes: { mode: remove, alias: s",
        begidx=len("prefixes: { mode: remove, alias: s"),
        payload_candidates=["mode: ", "alias: ", "}"],
        active_transform="prefixes",
        model_aliases=["scratch", "source", "base"],
    )
    assert "scratch" in matches
    assert "source" in matches
    assert "base" not in matches


def test_prefixes_rename_from_value_completion_uses_aliases_not_tensor_refs() -> None:
    matches = _match_payload_candidates(
        text="s",
        line_buffer="prefixes: { mode: rename, from: s",
        begidx=len("prefixes: { mode: rename, from: s"),
        payload_candidates=["scratch", "scratch::", "source", "source::", "mode: ", "from: ", "to: "],
        active_transform="prefixes",
        model_aliases=["scratch", "source", "base"],
    )
    assert matches == ["scratch", "source"]


def test_render_completion_preview_compacts_and_limits() -> None:
    preview = _render_completion_preview(["a", "b", "c"], limit=2)
    assert preview == "a  b (+1 more)"
