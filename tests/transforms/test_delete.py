from importlib import import_module

from brainsurgery.core import TensorRef
from brainsurgery.engine import (
    RuntimeFlagLifecycleScope,
    reset_runtime_flags_for_scope,
    set_runtime_flag,
)

_module = import_module("brainsurgery.transforms.delete")
DeleteTransform = _module.DeleteTransform
DeleteTransformError = _module.DeleteTransformError
UnarySpec = _module.UnarySpec


def test_delete_compile_rejects_sliced_target() -> None:
    try:
        DeleteTransform().compile({"target": "x::[:]"}, default_model="model")
    except DeleteTransformError as exc:
        assert "target must not be sliced" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected sliced target error")


def test_delete_apply_removes_target() -> None:
    class _Provider:
        def __init__(self) -> None:
            self._state_dict = {"x": object()}

        def get_state_dict(self, model: str):
            assert model == "model"
            return self._state_dict

    provider = _Provider()
    spec = UnarySpec(target_ref=TensorRef(model="model", expr="x"))
    DeleteTransform().apply_to_target(spec, "x", provider)
    assert "x" not in provider._state_dict


def test_delete_apply_raises_for_missing_target() -> None:
    class _Provider:
        def __init__(self) -> None:
            self._state_dict = {}

        def get_state_dict(self, model: str):
            assert model == "model"
            return self._state_dict

    spec = UnarySpec(target_ref=TensorRef(model="model", expr="x"))
    try:
        DeleteTransform().apply_to_target(spec, "x", _Provider())
    except DeleteTransformError as exc:
        assert "disappeared" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected missing target error")


def test_delete_apply_emits_verbose_activity_line(capsys) -> None:
    class _Provider:
        def __init__(self) -> None:
            self._state_dict = {"x": object()}

        def get_state_dict(self, model: str):
            assert model == "model"
            return self._state_dict

    provider = _Provider()
    spec = UnarySpec(target_ref=TensorRef(model="model", expr="x"))

    reset_runtime_flags_for_scope(RuntimeFlagLifecycleScope.CLI_RUN)
    set_runtime_flag("verbose", True)
    DeleteTransform().apply_to_target(spec, "x", provider)
    assert "delete: x" in capsys.readouterr().out
    reset_runtime_flags_for_scope(RuntimeFlagLifecycleScope.CLI_RUN)
