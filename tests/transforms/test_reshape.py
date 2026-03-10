from importlib import import_module

_module = import_module("brainsurgery.transforms.reshape")
globals().update({name: getattr(_module, name) for name in dir(_module) if not name.startswith("_")})


def test_reshape_compile_rejects_multiple_infer_dims() -> None:
    try:
        ReshapeTransform().compile({"from": "x", "to": "y", "shape": [-1, -1]}, default_model="m")
    except ReshapeTransformError as exc:
        assert "at most one '-1'" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected reshape.shape validation error")


def test_reshape_apply_success() -> None:
    class _Provider:
        def __init__(self) -> None:
            self._state_dict = {"x": torch.arange(6)}

        def get_state_dict(self, model: str):
            assert model == "m"
            return self._state_dict

    provider = _Provider()
    spec = ReshapeTransform().compile({"from": "x", "to": "y", "shape": [2, 3]}, default_model="m")
    ReshapeTransform().apply(spec, provider)
    assert provider._state_dict["y"].shape == (2, 3)
