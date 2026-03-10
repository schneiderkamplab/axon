from importlib import import_module

_module = import_module("brainsurgery.transforms.delete")
globals().update({name: getattr(_module, name) for name in dir(_module) if not name.startswith("_")})


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
