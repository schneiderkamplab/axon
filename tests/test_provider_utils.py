from brainsurgery.provider_utils import list_model_aliases, resolve_single_model_alias
from brainsurgery.transform_types import TransformError


def test_list_model_aliases_from_duck_typed_provider() -> None:
    class _Provider:
        model_paths = {"base": object()}
        state_dicts = {"scratch": object()}

    assert list_model_aliases(_Provider()) == {"base", "scratch"}


def test_resolve_single_model_alias_rejects_multiple() -> None:
    class _Error(TransformError):
        pass

    class _Provider:
        state_dicts = {"a": object(), "b": object()}

    try:
        resolve_single_model_alias(_Provider(), error_type=_Error, op_name="save")
    except _Error as exc:
        assert "save.alias is required" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected single-alias validation error")
