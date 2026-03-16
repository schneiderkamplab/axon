from importlib import import_module

_module = import_module("brainsurgery.transforms.matmul")
globals().update(
    {name: getattr(_module, name) for name in dir(_module) if not name.startswith("_")}
)


def test_matmul_compile_rejects_sliced_destination() -> None:
    try:
        MatmulTransform().compile(
            {"from_a": "a", "from_b": "b", "to": "c::[:]"},
            default_model="m",
        )
    except TransformError as exc:
        assert "destination must not be sliced" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected sliced destination rejection")


def test_matmul_apply_success() -> None:
    class _Provider:
        def __init__(self) -> None:
            self._state_dict = {
                "a": torch.tensor([[1.0, 2.0]], dtype=torch.float32),
                "b": torch.tensor([[3.0], [4.0]], dtype=torch.float32),
            }

        def get_state_dict(self, model: str):
            assert model == "m"
            return self._state_dict

    provider = _Provider()
    spec = MatmulTransform().compile({"from_a": "a", "from_b": "b", "to": "c"}, default_model="m")
    MatmulTransform().apply(spec, provider)
    assert provider._state_dict["c"].tolist() == [[11.0]]
