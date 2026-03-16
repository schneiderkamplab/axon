import torch

from brainsurgery.transforms.phlora import PhloraTransform, PhloraTransformError


def test_phlora_compile_rejects_non_integral_rank() -> None:
    try:
        PhloraTransform().compile(
            {"target": "w", "target_a": "w.a", "target_b": "w.b", "rank": 3.5},
            default_model="model",
        )
    except PhloraTransformError as exc:
        assert "positive integer" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected rank validation error")


def test_phlora_split_mode_writes_a_b_and_deletes_original_by_default() -> None:
    class _Provider:
        def __init__(self) -> None:
            self._state_dict = {
                "proj.weight": torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32),
            }

        def get_state_dict(self, model: str):
            assert model == "model"
            return self._state_dict

    provider = _Provider()
    transform = PhloraTransform()
    spec = transform.compile(
        {
            "target": "(.*)\\.weight",
            "rank": 1,
            "target_a": "\\1.a",
            "target_b": "\\1.b",
        },
        default_model="model",
    )
    transform.apply(spec, provider)

    sd = provider._state_dict
    assert "proj.weight" not in sd
    assert "proj.a" in sd
    assert "proj.b" in sd


def test_phlora_split_mode_can_keep_original_when_configured() -> None:
    class _Provider:
        def __init__(self) -> None:
            self._state_dict = {
                "proj.weight": torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32),
            }

        def get_state_dict(self, model: str):
            assert model == "model"
            return self._state_dict

    provider = _Provider()
    transform = PhloraTransform()
    spec = transform.compile(
        {
            "target": "(.*)\\.weight",
            "rank": 1,
            "target_a": "\\1.a",
            "target_b": "\\1.b",
            "delete_original": False,
        },
        default_model="model",
    )
    transform.apply(spec, provider)

    sd = provider._state_dict
    assert "proj.weight" in sd
    assert "proj.a" in sd
    assert "proj.b" in sd
