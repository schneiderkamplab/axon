from importlib import import_module

import pytest
import torch

from brainsurgery.core import TensorRef, TransformError

_module = import_module("brainsurgery.expressions.iszero")
globals().update({name: getattr(_module, name) for name in dir(_module) if not name.startswith("_")})

def test_iszero_compile_rejects_boolean_eps() -> None:
    with pytest.raises(TransformError, match="non-negative number"):
        compile_iszero_expr({"of": "x", "eps": False}, default_model="model")

def test_iszero_evaluate_complex_tolerance_branch() -> None:
    class _Provider:
        def __init__(self) -> None:
            self._state_dict = {"x": torch.tensor([0 + 0j], dtype=torch.complex64)}

        def get_state_dict(self, model: str):
            assert model == "model"
            return self._state_dict

    expr = IsZeroExpr(ref=TensorRef(model="model", expr="x"), eps=1e-6)
    expr.evaluate(_Provider())
