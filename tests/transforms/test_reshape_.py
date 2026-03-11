from importlib import import_module

import torch

_module = import_module("brainsurgery.transforms.reshape")
globals().update({name: getattr(_module, name) for name in dir(_module) if not name.startswith("_")})


def test_reshape_in_place_apply_success() -> None:
    class _Provider:
        def __init__(self) -> None:
            self._state_dict = {"x": torch.arange(6)}

        def get_state_dict(self, model: str):
            assert model == "m"
            return self._state_dict

    provider = _Provider()
    spec = ReshapeInPlaceSpec(target_ref=TensorRef(model="m", expr="x"), shape=(2, 3))
    ReshapeInPlaceTransform().apply_to_target(spec, "x", provider)
    assert provider._state_dict["x"].shape == (2, 3)
