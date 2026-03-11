from importlib import import_module

import torch

from brainsurgery.engine import InMemoryStateDict

_module = import_module("brainsurgery.transforms.clamp_")
globals().update({name: getattr(_module, name) for name in dir(_module) if not name.startswith("_")})


def test_clamp_in_place_apply_with_slice() -> None:
    class _Provider:
        def __init__(self) -> None:
            self._state_dict = InMemoryStateDict()
            self._state_dict["x"] = torch.tensor([-3.0, 0.0, 4.0])

        def get_state_dict(self, model: str):
            assert model == "m"
            return self._state_dict

    provider = _Provider()
    spec = ClampInPlaceSpec(
        target_ref=TensorRef(model="m", expr="x", slice_spec="[0:2]"),
        min_value=-1.0,
        max_value=1.0,
    )
    ClampInPlaceTransform().apply_to_target(spec, "x", provider)
    assert provider._state_dict["x"].tolist() == [-1.0, 0.0, 4.0]
