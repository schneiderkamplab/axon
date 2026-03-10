from importlib import import_module

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
