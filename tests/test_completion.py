from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import brainsurgery
import pytest

from brainsurgery.cli.interactive import (
    _collect_completion_candidates,
    _collect_payload_candidates,
    _infer_active_transform,
    _is_top_level_completion_position,
    _list_model_aliases,
    _match_payload_candidates,
)
from brainsurgery.core import get_transform, list_transforms


@dataclass
class _MiniProvider:
    model_paths: dict[str, Path]
    state_dicts: dict[str, dict[str, object]]

    def list_model_aliases(self) -> set[str]:
        return set(self.state_dicts)


SINGLE_ALIAS_PROVIDER = _MiniProvider(
    model_paths={"model": Path("/tmp/model.safetensors")},
    state_dicts={
        "model": {
            "ln_f.weight": object(),
            "ln_f.bias": object(),
            "h.0.attn.c_attn.weight": object(),
            "h.0.attn.c_proj.weight": object(),
            "mlp.c_fc.weight": object(),
        }
    },
)

MULTI_ALIAS_PROVIDER = _MiniProvider(
    model_paths={
        "base": Path("/tmp/base.safetensors"),
        "scratch": Path("/tmp/scratch.safetensors"),
    },
    state_dicts={
        "base": {
            "ln_f.weight": object(),
            "ln_f.bias": object(),
            "h.0.attn.c_attn.weight": object(),
        },
        "scratch": {
            "ln_f.weight": object(),
            "new.weight": object(),
            "h.1.attn.c_proj.weight": object(),
        },
    },
)

_SPECIAL_TRANSFORMS = {"assert", "diff", "exit", "help", "prefixes"}
_PREFERRED_KEYS = {
    "add": "from_a",
    "add_": "from",
    "assign": "from",
    "cast": "from",
    "cast_": "target",
    "clamp": "from",
    "clamp_": "target",
    "concat": "from",
    "copy": "from",
    "delete": "target",
    "dump": "target",
    "fill": "from",
    "fill_": "target",
    "load": "path",
    "matmul": "from_a",
    "move": "from",
    "multiply": "from_a",
    "permute": "from",
    "phlora": "target",
    "phlora_": "target",
    "reshape": "from",
    "reshape_": "target",
    "save": "target",
    "scale": "from",
    "scale_": "target",
    "split": "from",
    "subtract": "from_a",
    "subtract_": "from",
}

_REFERENCE_TRANSFORMS = [
    name
    for name in list_transforms()
    if _PREFERRED_KEYS.get(name) in set(get_transform(name).completion_reference_keys())
]


def _top_level_candidate(transform_name: str) -> str:
    candidates = _collect_completion_candidates(None)
    with_payload = f"{transform_name}: "
    if with_payload in candidates:
        return with_payload
    return transform_name


def _shortest_unique_prefix(target: str, candidates: list[str]) -> str:
    for index in range(1, len(target) + 1):
        prefix = target[:index]
        matches = [candidate for candidate in candidates if candidate.startswith(prefix)]
        if matches == [target]:
            return prefix
    return target


def _shortest_unique_key_prefix(target_key_candidate: str, candidates: list[str]) -> str:
    bare_key = target_key_candidate[:-2]
    for index in range(1, len(bare_key) + 1):
        prefix = bare_key[:index]
        matches = [candidate for candidate in candidates if candidate.startswith(prefix)]
        if matches == [target_key_candidate]:
            return prefix
    return bare_key


def _current_token_bounds(buffer: str) -> tuple[int, int]:
    last_delim = max(buffer.rfind(" "), buffer.rfind("\t"), buffer.rfind("\n"))
    begidx = last_delim + 1
    endidx = len(buffer)
    return begidx, endidx


def _completion_matches(buffer: str, provider: object | None) -> list[str]:
    begidx, endidx = _current_token_bounds(buffer)
    text = buffer[begidx:endidx]

    if _is_top_level_completion_position(buffer, begidx):
        candidates = _collect_completion_candidates(provider)
        return [candidate for candidate in candidates if candidate.startswith(text)]

    active_transform = _infer_active_transform([], buffer)
    payload_candidates = _collect_payload_candidates(
        active_transform=active_transform,
        state_dict_provider=provider,
    )
    return _match_payload_candidates(
        text=text,
        line_buffer=buffer,
        begidx=begidx,
        endidx=endidx,
        payload_candidates=payload_candidates,
        active_transform=active_transform,
        model_aliases=sorted(_list_model_aliases(provider)),
    )


def _complete_unique(buffer: str, provider: object | None) -> str:
    matches = _completion_matches(buffer, provider)
    assert matches, f"expected completion matches for {buffer!r}"
    assert len(matches) == 1, f"expected unique completion for {buffer!r}, got {matches!r}"
    begidx, endidx = _current_token_bounds(buffer)
    return f"{buffer[:begidx]}{matches[0]}{buffer[endidx:]}"


def _transform_reference_keys(transform_name: str) -> list[str]:
    transform = get_transform(transform_name)
    return transform.completion_reference_keys()


def _next_reference_key(transform_name: str, current_key: str) -> str | None:
    ordered = _transform_reference_keys(transform_name)
    if current_key not in ordered:
        return None
    current_index = ordered.index(current_key)
    if current_index + 1 >= len(ordered):
        return None
    return ordered[current_index + 1]


def _first_base_tensor_match(matches: list[str]) -> str:
    for match in matches:
        if match.startswith("base::") and match != "base::":
            return match
    raise AssertionError(f"expected a base-qualified tensor match, got {matches!r}")


@pytest.mark.parametrize("transform_name", list_transforms(), ids=list_transforms())
def test_top_level_unique_completion_for_each_transform(transform_name: str) -> None:
    candidate = _top_level_candidate(transform_name)
    prefix = _shortest_unique_prefix(candidate, _collect_completion_candidates(None))
    assert _complete_unique(prefix, None) == candidate


@pytest.mark.parametrize(
    "transform_name",
    [name for name in list_transforms() if name not in _SPECIAL_TRANSFORMS],
    ids=[name for name in list_transforms() if name not in _SPECIAL_TRANSFORMS],
)
def test_single_alias_typical_completion_sequence_for_each_non_special_transform(
    transform_name: str,
) -> None:
    candidate = _top_level_candidate(transform_name)
    assert _complete_unique(
        _shortest_unique_prefix(candidate, _collect_completion_candidates(None)),
        SINGLE_ALIAS_PROVIDER,
    ) == candidate

    assert _complete_unique(candidate, SINGLE_ALIAS_PROVIDER) == f"{candidate}{{ "

    preferred_key = _PREFERRED_KEYS[transform_name]
    mapping_start_matches = _completion_matches(f"{candidate}{{ ", SINGLE_ALIAS_PROVIDER)
    key_candidate = f"{preferred_key}: "
    assert key_candidate in mapping_start_matches

    key_prefix = _shortest_unique_key_prefix(key_candidate, mapping_start_matches)
    key_prefix_matches = [match for match in mapping_start_matches if match.startswith(key_prefix)]
    if key_prefix_matches == [key_candidate]:
        assert _complete_unique(
            f"{candidate}{{ {key_prefix}",
            SINGLE_ALIAS_PROVIDER,
        ) == f"{candidate}{{ {key_candidate}"
    else:
        assert key_candidate in key_prefix_matches

    if preferred_key not in set(_transform_reference_keys(transform_name)):
        return

    reference_matches = _completion_matches(
        f"{candidate}{{ {key_candidate}",
        SINGLE_ALIAS_PROVIDER,
    )
    assert "model::" in reference_matches
    assert any(
        match.startswith("model::") and match != "model::"
        for match in reference_matches
    )
    assert any(
        "::" not in match and not match.endswith(": ") and match not in {"{ ", "}", "[ ", "]", ", "}
        for match in reference_matches
    )

    next_reference_key = _next_reference_key(transform_name, preferred_key)
    if next_reference_key is None:
        return

    continuation_matches = _completion_matches(
        f"{candidate}{{ {key_candidate}model::ln_f.weight",
        SINGLE_ALIAS_PROVIDER,
    )
    assert f"model::ln_f.weight, {next_reference_key}: " in continuation_matches


@pytest.mark.parametrize(
    "transform_name",
    _REFERENCE_TRANSFORMS,
    ids=_REFERENCE_TRANSFORMS,
)
def test_multi_alias_reference_completion_sequence_for_reference_transforms(
    transform_name: str,
) -> None:
    candidate = _top_level_candidate(transform_name)
    preferred_key = _PREFERRED_KEYS[transform_name]
    key_candidate = f"{preferred_key}: "

    alias_selection_matches = _completion_matches(
        f"{candidate}{{ {key_candidate}",
        MULTI_ALIAS_PROVIDER,
    )
    assert alias_selection_matches == ["base::", "scratch::"]

    base_matches = _completion_matches(
        f"{candidate}{{ {key_candidate}base::",
        MULTI_ALIAS_PROVIDER,
    )
    assert "base::" in base_matches
    assert any(match.startswith("base::") and match != "base::" for match in base_matches)
    assert all(not match.startswith("scratch::") for match in base_matches)

    next_reference_key = _next_reference_key(transform_name, preferred_key)
    if next_reference_key is None:
        return

    full_ref = _first_base_tensor_match(base_matches)
    continuation_matches = _completion_matches(
        f"{candidate}{{ {key_candidate}{full_ref}",
        MULTI_ALIAS_PROVIDER,
    )
    assert f"{full_ref}, {next_reference_key}: " in continuation_matches


def test_help_completion_sequence() -> None:
    candidate = _top_level_candidate("help")
    assert _complete_unique(
        _shortest_unique_prefix(candidate, _collect_completion_candidates(None)),
        SINGLE_ALIAS_PROVIDER,
    ) == candidate

    payload_start_matches = _completion_matches(candidate, SINGLE_ALIAS_PROVIDER)
    assert "{ " in payload_start_matches
    assert "copy" in payload_start_matches
    assert "exit" in payload_start_matches

    mapping_start_matches = _completion_matches("help: { ", SINGLE_ALIAS_PROVIDER)
    assert mapping_start_matches == ["assert: "]

    assert _complete_unique("help: { assert: eq", SINGLE_ALIAS_PROVIDER) == "help: { assert: equal"


def test_assert_completion_sequence() -> None:
    candidate = _top_level_candidate("assert")
    assert _complete_unique(
        _shortest_unique_prefix(candidate, _collect_completion_candidates(None)),
        SINGLE_ALIAS_PROVIDER,
    ) == candidate

    payload_start_matches = _completion_matches(candidate, SINGLE_ALIAS_PROVIDER)
    assert "{ " in payload_start_matches
    assert "equal" in payload_start_matches
    assert "exists" in payload_start_matches

    mapping_start_matches = _completion_matches("assert: { ", SINGLE_ALIAS_PROVIDER)
    assert "equal: " in mapping_start_matches
    assert "exists: " in mapping_start_matches

    assert "equal: " in mapping_start_matches
    assert "exists: " in mapping_start_matches


def test_prefixes_completion_sequence() -> None:
    candidate = _top_level_candidate("prefixes")
    assert _complete_unique(
        _shortest_unique_prefix(candidate, _collect_completion_candidates(None)),
        MULTI_ALIAS_PROVIDER,
    ) == candidate

    assert _complete_unique("prefixes: { m", MULTI_ALIAS_PROVIDER) == "prefixes: { mode: "

    mode_matches = _completion_matches("prefixes: { mode: r", MULTI_ALIAS_PROVIDER)
    assert mode_matches == ["remove", "rename"]

    rename_value = _complete_unique("prefixes: { mode: ren", MULTI_ALIAS_PROVIDER)
    assert rename_value == "prefixes: { mode: rename"

    rename_key_matches = _completion_matches(
        "prefixes: { mode: rename, ",
        MULTI_ALIAS_PROVIDER,
    )
    assert rename_key_matches == ["from: ", "to: "]

    alias_matches = _completion_matches(
        "prefixes: { mode: rename, from: ",
        MULTI_ALIAS_PROVIDER,
    )
    assert alias_matches == ["base", "scratch"]


def test_diff_completion_sequence() -> None:
    candidate = _top_level_candidate("diff")
    assert _complete_unique(
        _shortest_unique_prefix(candidate, _collect_completion_candidates(None)),
        MULTI_ALIAS_PROVIDER,
    ) == candidate

    assert _complete_unique(candidate, MULTI_ALIAS_PROVIDER) == f"{candidate}{{ "

    mapping_start_matches = _completion_matches(f"{candidate}{{ ", MULTI_ALIAS_PROVIDER)
    assert "mode: " in mapping_start_matches
    assert "left: " in mapping_start_matches

    mode_matches = _completion_matches(f"{candidate}{{ mode: a", MULTI_ALIAS_PROVIDER)
    assert mode_matches == ["aliases"]

    ref_matches = _completion_matches(f"{candidate}{{ left: ", MULTI_ALIAS_PROVIDER)
    assert ref_matches == ["base::", "scratch::"]

    alias_matches = _completion_matches(
        f"{candidate}{{ mode: aliases, left_alias: ",
        MULTI_ALIAS_PROVIDER,
    )
    assert alias_matches == ["base", "scratch"]


def test_exit_completion_sequence() -> None:
    candidate = _top_level_candidate("exit")
    assert _complete_unique(
        _shortest_unique_prefix(candidate, _collect_completion_candidates(None)),
        None,
    ) == candidate
