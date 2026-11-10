"""Compiler defects found while writing the PP-OCR model files.

Each test is the smallest program that showed the defect.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from synapse.axon.parse import parse_axon_program
from tests.mlx_core_support import generate_core, load_core

KEYWORD_PREFIXED_NAMES = [
    "downsample",
    "done",
    "format",
    "forward",
    "stepper",
    "typed",
    "scoped",
    "returns",
    "carry_x",
    "yielded",
    "trueish",
    "falsey",
    "nullable",
    "importer",
    "exported",
]


@pytest.mark.parametrize("name", KEYWORD_PREFIXED_NAMES)
def test_name_may_start_with_a_keyword(name: str) -> None:
    # The lexer used to cut `downsample` into the keyword `do` and `wnsample`.
    source = (
        f"{name} :: Int -> Int\n"
        f"{name} x = x + 1\n"
        "\n"
        "caller :: Int -> Int\n"
        "caller x = do\n"
        f"  y <- {name} x\n"
        f"  return {name} y\n"
    )
    program = parse_axon_program(source)
    assert [module.name for module in program.modules] == [name, "caller"]


# `up_like` reads the sizes of both arguments. Its `H` is the height of `x`; at both call
# sites that height is an expression over the caller's `H`, so the call substitution has
# no binding for it. Inlining used to read the callee's `H` as the caller's `H` and fold
# the size ratio to 0.
INLINE_DIM_CAPTURE = """\
{-# MAIN "m" #-}
import Tensor
import Vision

up_like :: Tensor[B,H,W,C] -> Tensor[B,Ho,Wo,D] -> Tensor[B,Ho,Wo,C]
up_like x ref = Vision.upsample_nearest2d x ((Tensor.size ref 1) / (Tensor.size x 1)) ((Tensor.size ref 2) / (Tensor.size x 2))

m :: Tensor[B,H,W,C] -> Tensor[B,H,W,C]
m pixel_values = do
  y <- Vision.max_pool2d pixel_values 2 2 stride_h=2 stride_w=2
  z <- Vision.max_pool2d y 2 2 stride_h=2 stride_w=2
  u <- up_like z y
  return up_like (y + u) pixel_values
"""


def test_inlining_keeps_callee_dim_apart_from_caller_dim_of_the_same_name(tmp_path: Path) -> None:
    source = tmp_path / "inline_dim_capture.axon"
    source.write_text(INLINE_DIM_CAPTURE)
    x = torch.randn(1, 16, 24, 3)
    outputs = []
    for optimize in (False, True):
        core = load_core(generate_core(source, backend="torch", optimize=optimize), "AxonCore")
        outputs.append(core.from_state_dict({}, model_config={}).forward(pixel_values=x))
    plain, optimized = outputs
    assert tuple(plain.shape) == (1, 16, 24, 3)
    assert tuple(optimized.shape) == tuple(plain.shape)
    torch.testing.assert_close(optimized, plain)
