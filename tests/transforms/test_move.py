from importlib import import_module

_module = import_module("brainsurgery.transforms.move")
globals().update({name: getattr(_module, name) for name in dir(_module) if not name.startswith("_")})


def test_move_compile_rejects_sliced_source() -> None:
    try:
        MoveTransform().compile({"from": "a::[:]", "to": "b"}, default_model="model")
    except MoveTransformError as exc:
        assert "source must not be sliced" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected move sliced source error")


def test_move_compile_rejects_sliced_destination() -> None:
    try:
        MoveTransform().compile({"from": "a", "to": "b::[:]"}, default_model="model")
    except MoveTransformError as exc:
        assert "destination must not be sliced" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected move sliced destination error")


def test_move_apply_moves_slot() -> None:
    class _StateDict(dict):
        def slot(self, key):
            return self[key]

        def bind_slot(self, key, slot):
            self[key] = slot

    class _Provider:
        def __init__(self) -> None:
            self._state_dict = _StateDict({"src": object()})

        def get_state_dict(self, model: str):
            assert model == "model"
            return self._state_dict

    provider = _Provider()
    item = ResolvedMapping(
        src_model="model",
        src_name="src",
        src_slice=None,
        dst_model="model",
        dst_name="dst",
        dst_slice=None,
    )
    MoveTransform().apply_mapping(item, provider)
    assert "src" not in provider._state_dict
    assert "dst" in provider._state_dict
