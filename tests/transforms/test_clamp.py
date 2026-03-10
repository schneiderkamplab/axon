from importlib import import_module

_module = import_module("brainsurgery.transforms.clamp")
globals().update({name: getattr(_module, name) for name in dir(_module) if not name.startswith("_")})


def test_clamp_compile_requires_bound() -> None:
    try:
        ClampTransform().compile({"from": "x", "to": "y"}, default_model="m")
    except ClampTransformError as exc:
        assert "at least one of: min, max" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected clamp min/max validation error")


def test_clamp_apply_success() -> None:
    class _Provider:
        def __init__(self) -> None:
            self._state_dict = {"x": torch.tensor([-2.0, 0.0, 3.0])}

        def get_state_dict(self, model: str):
            assert model == "m"
            return self._state_dict

    provider = _Provider()
    spec = ClampTransform().compile(
        {"from": "x", "to": "y", "min": -1.0, "max": 1.0},
        default_model="m",
    )
    ClampTransform().apply(spec, provider)
    assert provider._state_dict["y"].tolist() == [-1.0, 0.0, 1.0]
