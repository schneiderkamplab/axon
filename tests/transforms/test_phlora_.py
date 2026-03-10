import torch

from brainsurgery.transforms.phlora_ import PhloraInPlaceTransform, PhloraInPlaceTransformError


def test_phlora_in_place_compile_rejects_non_integral_rank() -> None:
    try:
        PhloraInPlaceTransform().compile({"target": "x", "rank": 3.5}, default_model="model")
    except PhloraInPlaceTransformError as exc:
        assert "positive integer" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected rank validation error")


def test_phlora_in_place_rewrites_target_with_ranked_matrix() -> None:
    class _Provider:
        def __init__(self) -> None:
            self._state_dict = {
                "w": torch.tensor([[3.0, 0.0], [0.0, 2.0]], dtype=torch.float32),
            }

        def get_state_dict(self, model: str):
            assert model == "model"
            return self._state_dict

    provider = _Provider()
    transform = PhloraInPlaceTransform()
    spec = transform.compile({"target": "w", "rank": 1}, default_model="model")
    transform.apply(spec, provider)

    assert "w" in provider._state_dict
    assert provider._state_dict["w"].shape == (2, 2)
    assert torch.allclose(provider._state_dict["w"], torch.tensor([[3.0, 0.0], [0.0, 0.0]]))


def test_phlora_in_place_rejects_non_2d_target() -> None:
    class _Provider:
        def __init__(self) -> None:
            self._state_dict = {"x": torch.tensor([1.0, 2.0, 3.0])}

        def get_state_dict(self, model: str):
            assert model == "model"
            return self._state_dict

    provider = _Provider()
    transform = PhloraInPlaceTransform()
    spec = transform.compile({"target": "x", "rank": 1}, default_model="model")
    try:
        transform.apply(spec, provider)
    except PhloraInPlaceTransformError as exc:
        assert "must be 2D" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected 2D validation error")
