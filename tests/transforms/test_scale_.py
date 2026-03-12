from importlib import import_module

import torch

from brainsurgery.engine import InMemoryStateDict, reset_runtime_flags, set_runtime_flag

_module = import_module("brainsurgery.transforms.scale")
globals().update({name: getattr(_module, name) for name in dir(_module) if not name.startswith("_")})


def test_scale_in_place_compile_rejects_non_numeric_factor() -> None:
    try:
        ScaleInPlaceTransform().compile({"target": "x", "by": "nan?!"}, default_model="model")
    except TransformError as exc:
        assert "scale_.by must be numeric" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected scale_ numeric validation error")


def test_scale_in_place_compile_accepts_numeric_string_factor() -> None:
    spec = ScaleInPlaceTransform().compile({"target": "x", "by": "2.5"}, default_model="model")
    assert spec.factor == 2.5


def test_scale_in_place_apply_with_slice() -> None:
    class _Provider:
        def __init__(self) -> None:
            self._state_dict = InMemoryStateDict()
            self._state_dict["x"] = torch.tensor([1.0, 2.0, 3.0, 4.0])

        def get_state_dict(self, model: str):
            assert model == "model"
            return self._state_dict

    provider = _Provider()
    spec = ScaleInPlaceSpec(
        target_ref=TensorRef(model="model", expr="x", slice_spec="[1:3]"),
        factor=10.0,
    )
    ScaleInPlaceTransform().apply_to_target(spec, "x", provider)
    assert provider._state_dict["x"].tolist() == [1.0, 20.0, 30.0, 4.0]


def test_scale_in_place_emits_verbose_activity_line(capsys) -> None:
    class _Provider:
        def __init__(self) -> None:
            self._state_dict = InMemoryStateDict()
            self._state_dict["x"] = torch.tensor([1.0, 2.0])

        def get_state_dict(self, model: str):
            assert model == "model"
            return self._state_dict

    provider = _Provider()
    spec = ScaleInPlaceSpec(
        target_ref=TensorRef(model="model", expr="x", slice_spec=None),
        factor=2.0,
    )

    reset_runtime_flags()
    set_runtime_flag("verbose", True)
    ScaleInPlaceTransform().apply_to_target(spec, "x", provider)
    assert "scale_: x" in capsys.readouterr().out
    reset_runtime_flags()
