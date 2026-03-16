from importlib import import_module

import torch

from brainsurgery.engine.state_dicts import _InMemoryStateDict

_module = import_module("brainsurgery.transforms.fill")
FillConfig = _module.FillConfig
FillInPlaceSpec = _module.FillInPlaceSpec
FillInPlaceTransform = _module.FillInPlaceTransform
TensorRef = _module.TensorRef


def test_fill_in_place_tensor_mode() -> None:
    class _Provider:
        def __init__(self) -> None:
            self._state_dict = _InMemoryStateDict()
            self._state_dict["x"] = torch.zeros((2,), dtype=torch.float32)

        def get_state_dict(self, model: str):
            assert model == "m"
            return self._state_dict

    provider = _Provider()
    spec = FillInPlaceSpec(
        target_ref=TensorRef(model="m", expr="x"),
        config=FillConfig(
            mode="tensor",
            constant_value=None,
            values_payload=[5.0, 6.0],
            distribution="uniform",
            low=0.0,
            high=1.0,
            mean=0.0,
            std=1.0,
            seed=None,
        ),
    )
    FillInPlaceTransform().apply_to_target(spec, "x", provider)
    assert provider._state_dict["x"].tolist() == [5.0, 6.0]
