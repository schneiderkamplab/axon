from brainsurgery.utils.provider_utils import (
    find_alias_mapping,
    get_or_create_alias_state_dict,
    list_loaded_tensor_names,
    list_model_aliases,
    new_empty_state_dict,
    resolve_single_model_alias,
)
from brainsurgery.providers import InMemoryStateDict
from brainsurgery.core import TransformError


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


def test_get_or_create_alias_state_dict_requires_extension_support() -> None:
    class _Provider:
        def __init__(self) -> None:
            self.state_dicts = {"base": InMemoryStateDict()}

        def get_state_dict(self, model: str) -> InMemoryStateDict:
            return self.state_dicts[model]

    class _Error(TransformError):
        pass

    try:
        get_or_create_alias_state_dict(_Provider(), "new", error_type=_Error, op_name="prefixes")
    except _Error as exc:
        assert "supports creating new aliases" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected create-alias validation error")


def test_list_loaded_tensor_names_find_alias_mapping_and_new_empty_state_dict() -> None:
    state_dict = InMemoryStateDict()
    state_dict["weight"] = __import__("torch").ones(1)

    class _Provider:
        model_paths = {"disk": object()}
        state_dicts = {"loaded": state_dict}

    assert list_loaded_tensor_names(_Provider()) == {"loaded": {"weight"}}
    attr_name, mapping, value = find_alias_mapping(_Provider(), "loaded", error_type=TransformError)
    assert attr_name == "state_dicts"
    assert mapping["loaded"] is value
    assert isinstance(new_empty_state_dict([("state_dicts", {"loaded": state_dict})]), InMemoryStateDict)
