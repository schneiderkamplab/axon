from importlib import import_module

_module = import_module("brainsurgery.transforms.split")
SplitTransform = _module.SplitTransform
SplitTransformError = _module.SplitTransformError
torch = _module.torch


def test_split_compile_requires_matching_sizes_and_outputs() -> None:
    try:
        SplitTransform().compile(
            {"from": "x", "to": ["a", "b"], "sizes": [1]},
            default_model="m",
        )
    except SplitTransformError as exc:
        assert "length must match" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected split.sizes length validation error")


def test_split_apply_success() -> None:
    class _Provider:
        def __init__(self) -> None:
            self._state_dict = {"x": torch.tensor([1.0, 2.0, 3.0, 4.0])}

        def get_state_dict(self, model: str):
            assert model == "m"
            return self._state_dict

    provider = _Provider()
    spec = SplitTransform().compile(
        {"from": "x", "to": ["x0", "x1"], "sizes": [2, 2], "dim": 0},
        default_model="m",
    )
    SplitTransform().apply(spec, provider)
    assert provider._state_dict["x0"].tolist() == [1.0, 2.0]
    assert provider._state_dict["x1"].tolist() == [3.0, 4.0]
