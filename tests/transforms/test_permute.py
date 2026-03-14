from importlib import import_module

import torch

_module = import_module("brainsurgery.transforms.permute")
globals().update({name: getattr(_module, name) for name in dir(_module) if not name.startswith("_")})

def test_permute_compile_rejects_non_list_order() -> None:
    try:
        PermuteTransform().compile({"from": "x", "to": "y", "order": "01"}, default_model="m")
    except TransformError as exc:
        assert "non-empty list of integers" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected order validation error")

def test_permute_apply_success() -> None:
    class _Provider:
        def __init__(self) -> None:
            self._state_dict = {"x": torch.arange(6).reshape(2, 3)}

        def get_state_dict(self, model: str):
            assert model == "m"
            return self._state_dict

    provider = _Provider()
    spec = PermuteTransform().compile({"from": "x", "to": "y", "order": [1, 0]}, default_model="m")
    PermuteTransform().apply(spec, provider)
    assert provider._state_dict["y"].shape == (3, 2)
