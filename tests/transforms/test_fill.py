from importlib import import_module

_module = import_module("brainsurgery.transforms.fill")
globals().update(
    {name: getattr(_module, name) for name in dir(_module) if not name.startswith("_")}
)


def test_fill_compile_requires_mode_specific_payload() -> None:
    try:
        FillTransform().compile(
            {"from": "x", "to": "y", "mode": "constant"},
            default_model="m",
        )
    except TransformError as exc:
        assert "fill.value is required" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected fill.value validation error")


def test_fill_tensor_mode_broadcasts_values() -> None:
    template = torch.zeros((2, 2), dtype=torch.float32)
    config = parse_fill_config(
        {"mode": "tensor", "values": [1.0, 2.0]},
        op_name="fill",
        error_type=TransformError,
    )
    out = build_filled_tensor_like(template, config, TransformError)
    assert out.tolist() == [[1.0, 2.0], [1.0, 2.0]]


def test_fill_rand_mode_is_seeded() -> None:
    template = torch.zeros((3,), dtype=torch.float32)
    config = parse_fill_config(
        {"mode": "rand", "seed": 7},
        op_name="fill",
        error_type=TransformError,
    )
    a = build_filled_tensor_like(template, config, TransformError)
    b = build_filled_tensor_like(template, config, TransformError)
    assert torch.equal(a, b)
