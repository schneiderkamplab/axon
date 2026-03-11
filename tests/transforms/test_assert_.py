from importlib import import_module

import pytest
import torch

from brainsurgery.engine import InMemoryStateDict

_module = import_module("brainsurgery.transforms.assert_")
globals().update({name: getattr(_module, name) for name in dir(_module) if not name.startswith("_")})


def test_assert_compile_rejects_unknown_op() -> None:
    try:
        AssertTransform().compile({"unknown": {}}, default_model="model")
    except TransformError as exc:
        assert "unknown assert op" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected unknown assert op error")


def test_assert_compile_builds_spec() -> None:
    spec = AssertTransform().compile({"exists": "x"}, default_model="model")
    assert isinstance(spec, AssertSpec)
    assert spec.collect_models() == {"model"}


def test_assert_infer_output_model_requires_single_model() -> None:
    spec = AssertTransform().compile(
        {"equal": {"left": "a::x", "right": "b::x"}},
        default_model=None,
    )
    try:
        AssertTransform().infer_output_model(spec)
    except TransformError as exc:
        assert "exactly one model" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected single-model inference error")


def test_assert_apply_supports_model_wide_reads_check() -> None:
    state_dict = InMemoryStateDict()
    state_dict["a"] = torch.ones(1)
    state_dict["b"] = torch.ones(1)
    _ = state_dict["a"]
    _ = state_dict["b"]

    class _Provider:
        def get_state_dict(self, model: str):
            assert model == "model"
            return state_dict

    spec = AssertTransform().compile(
        {"reads": {"of": ".*", "ge": 1}},
        default_model="model",
    )
    result = AssertTransform().apply(spec, _Provider())
    assert result.count == 1


def test_assert_apply_fails_when_tensor_has_not_been_read() -> None:
    state_dict = InMemoryStateDict()
    state_dict["a"] = torch.ones(1)

    class _Provider:
        def get_state_dict(self, model: str):
            assert model == "model"
            return state_dict

    spec = AssertTransform().compile(
        {"reads": {"of": ".*", "ge": 1}},
        default_model="model",
    )
    with pytest.raises(TransformError, match="reads=0"):
        AssertTransform().apply(spec, _Provider())
