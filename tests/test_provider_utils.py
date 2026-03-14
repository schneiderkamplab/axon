import pytest

from brainsurgery.engine.state_dicts import _InMemoryStateDict
from brainsurgery.engine.providers import InMemoryStateDictProvider
from brainsurgery.engine.provider_utils import (
    list_loaded_tensor_names,
    find_alias_mapping,
    get_or_create_alias_state_dict,
    iter_alias_mappings,
    list_model_aliases,
    new_empty_state_dict,
    resolve_single_model_alias,
)

from brainsurgery.core import TransformError

def test_list_model_aliases_from_duck_typed_provider() -> None:
    class _Provider:
        model_paths = {"base": object()}
        state_dicts = {"scratch": object()}

    assert list_model_aliases(_Provider()) == {"base", "scratch"}

def test_list_model_aliases_handles_none_and_iterable_return() -> None:
    class _Provider:
        def list_model_aliases(self):
            return ["a", "b"]

    assert list_model_aliases(None) == set()
    assert list_model_aliases(_Provider()) == {"a", "b"}

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
            self.state_dicts = {"base": _InMemoryStateDict()}

        def get_state_dict(self, model: str) -> _InMemoryStateDict:
            return self.state_dicts[model]

    class _Error(TransformError):
        pass

    try:
        get_or_create_alias_state_dict(_Provider(), "new", error_type=_Error, op_name="prefixes")
    except _Error as exc:
        assert "supports creating new aliases" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected create-alias validation error")

def test_get_or_create_alias_state_dict_returns_existing_alias() -> None:
    class _Provider:
        def __init__(self) -> None:
            self.state_dicts = {"base": _InMemoryStateDict()}

        def get_state_dict(self, model: str) -> _InMemoryStateDict:
            return self.state_dicts[model]

    provider = _Provider()
    assert get_or_create_alias_state_dict(
        provider,
        "base",
        error_type=TransformError,
        op_name="prefixes",
    ) is provider.state_dicts["base"]

def test_get_or_create_alias_state_dict_uses_base_provider_creation() -> None:
    provider = InMemoryStateDictProvider({}, max_io_workers=1)
    created = get_or_create_alias_state_dict(
        provider,
        "scratch",
        error_type=TransformError,
        op_name="prefixes",
    )
    assert created is provider.get_state_dict("scratch")

def test_list_loaded_tensor_names_find_alias_mapping_and_new_empty_state_dict() -> None:
    state_dict = _InMemoryStateDict()
    state_dict["weight"] = __import__("torch").ones(1)

    class _Provider:
        model_paths = {"disk": object()}
        state_dicts = {"loaded": state_dict}

    assert list_loaded_tensor_names(_Provider()) == {"loaded": {"weight"}}
    attr_name, mapping, value = find_alias_mapping(_Provider(), "loaded", error_type=TransformError)
    assert attr_name == "state_dicts"
    assert mapping["loaded"] is value
    assert isinstance(new_empty_state_dict([("state_dicts", {"loaded": state_dict})]), _InMemoryStateDict)

def test_provider_utils_misc_fallbacks_and_errors() -> None:
    class _KeysRaises:
        def keys(self):
            raise RuntimeError("broken keys")

    class _ProviderNoStateDicts:
        model_paths = {"disk": object()}
        _state_dicts = {"shadow": object()}

    assert [name for name, _ in iter_alias_mappings(_ProviderNoStateDicts())] == ["model_paths", "_state_dicts"]

    class _ProviderWithBadKeys:
        state_dicts = {"bad": _KeysRaises()}

    assert list_loaded_tensor_names(_ProviderWithBadKeys()) == {}

    with pytest.raises(TransformError, match="unknown model prefix"):
        find_alias_mapping(_ProviderNoStateDicts(), "missing", error_type=TransformError)

    class _NoDefaultCtor(dict):
        def __init__(self, arg):
            super().__init__(arg)

    assert new_empty_state_dict([("state_dicts", {"x": _NoDefaultCtor({"a": 1})})]) == {}

def test_resolve_single_model_alias_success() -> None:
    class _Provider:
        state_dicts = {"only": object()}

    assert resolve_single_model_alias(_Provider(), error_type=TransformError, op_name="save") == "only"
