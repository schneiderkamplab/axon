import brainsurgery.transforms.prefixes as prefixes_module
from brainsurgery.transforms.prefixes import PrefixesSpec, PrefixesTransform, PrefixesTransformError


def test_prefixes_compile_accepts_none_and_empty_mapping() -> None:
    transform = PrefixesTransform()
    assert isinstance(transform.compile(None, default_model=None), PrefixesSpec)
    assert isinstance(transform.compile({}, default_model=None), PrefixesSpec)


def test_prefixes_compile_add_mode() -> None:
    spec = PrefixesTransform().compile(
        {"mode": "add", "alias": "scratch"},
        default_model=None,
    )
    assert spec.mode == "add"
    assert spec.alias == "scratch"


def test_prefixes_compile_rejects_invalid_mode() -> None:
    try:
        PrefixesTransform().compile({"mode": "explode"}, default_model=None)
    except PrefixesTransformError as exc:
        assert "must be one of" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected prefixes mode error")


def test_prefixes_list_aliases_from_provider_state() -> None:
    class _Provider:
        model_paths = {"base": object()}
        state_dicts = {"scratch": object()}

    assert prefixes_module._list_aliases(_Provider()) == {"base", "scratch"}
