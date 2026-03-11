from importlib import import_module
from types import SimpleNamespace

import pytest

_module = import_module("brainsurgery.transforms.help")
globals().update({name: getattr(_module, name) for name in dir(_module) if not name.startswith("_")})


def test_help_compile_rejects_multi_key_mapping() -> None:
    try:
        HelpTransform().compile({"a": "x", "b": "y"}, default_model=None)
    except HelpTransformError as exc:
        assert "exactly one key" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected help mapping payload validation error")


def test_help_compile_string_payload() -> None:
    spec = HelpTransform().compile("copy", default_model=None)
    assert spec.command == "copy"
    assert spec.subcommand is None


def test_help_compile_mapping_with_subcommand() -> None:
    spec = HelpTransform().compile({"assert": "equal"}, default_model=None)
    assert spec.command == "assert"
    assert spec.subcommand == "equal"


def test_help_apply_rejects_subcommand_for_non_assert_command() -> None:
    transform = HelpTransform()
    with pytest.raises(HelpTransformError, match="does not support subcommand"):
        transform.apply(HelpSpec(command="copy", subcommand="from"), provider=object())


def test_help_print_command_help_emits_unavailable_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    lines: list[str] = []
    monkeypatch.setattr(_module, "get_transform", lambda _: object())
    monkeypatch.setattr(_module, "emit_line", lines.append)

    HelpTransform()._print_command_help("noop")

    assert "Command: noop" in lines
    assert "Key metadata: unavailable" in lines


def test_help_print_assert_expr_help_emits_required_optional_and_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    lines: list[str] = []
    monkeypatch.setattr(
        _module,
        "get_assert_expr_help",
        lambda _: SimpleNamespace(
            name="equal",
            payload_kind="mapping",
            description="desc",
            required_keys={"left"},
            allowed_keys={"left", "right"},
        ),
    )
    monkeypatch.setattr(_module, "emit_line", lines.append)

    HelpTransform()._print_assert_expr_help("equal")

    assert "Assert expression: equal" in lines
    assert "Payload: mapping" in lines
    assert "Required keys:" in lines
    assert "  - left" in lines
    assert "Optional keys:" in lines
    assert "  - right" in lines
