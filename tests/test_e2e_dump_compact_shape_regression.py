from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

import brainsurgery.transforms.dump as dump_module

from brainsurgery.engine.execution import execute_transform_pairs
from brainsurgery.engine.plan import compile_plan

from brainsurgery.engine.state_dicts import _InMemoryStateDict
@dataclass(frozen=True)
class _Case:
    state_dict: _InMemoryStateDict
    transforms: list[dict[str, object]]
    expected_dump: str

class _Provider:
    def __init__(self, state_dict: _InMemoryStateDict) -> None:
        self._state_dict = state_dict

    def get_state_dict(self, model: str) -> _InMemoryStateDict:
        assert model == "model"
        return self._state_dict

def _make_state_dict(values: dict[str, torch.Tensor]) -> _InMemoryStateDict:
    sd = _InMemoryStateDict()
    for key, value in values.items():
        sd[key] = value
    return sd

def _dump_line(name: str, shape: tuple[int, ...]) -> str:
    return f"└── {name}  shape={list(shape)}"

def _shape_for_slot(slot: int) -> tuple[int, int]:
    grid = [
        (1, 1),
        (1, 2),
        (2, 1),
        (2, 2),
        (2, 3),
        (3, 2),
        (3, 3),
        (1, 4),
        (4, 1),
        (4, 2),
    ]
    return grid[slot]

def _build_case(case_id: int) -> _Case:
    family = case_id % 10
    slot = case_id // 10
    rows, cols = _shape_for_slot(slot)
    out = f"out_{case_id:03d}"
    base = torch.arange(1, rows * cols + 1, dtype=torch.float32).reshape(rows, cols)

    if family == 0:
        state_dict = _make_state_dict({"x": base})
        transforms = [
            {"copy": {"from": "x", "to": out}},
            {"assert": {"equal": {"left": "x", "right": out}}},
            {"dump": {"target": out, "format": "compact", "verbosity": "shape"}},
        ]
        expected_shape = (rows, cols)
    elif family == 1:
        state_dict = _make_state_dict({"x": base})
        factor = (slot + 2) / 3.0
        transforms = [
            {"scale": {"from": "x", "to": out, "by": factor}},
            {"assert": {"shape": {"of": out, "is": [rows, cols]}}},
            {"assert": {"dtype": {"of": out, "is": "float32"}}},
            {"dump": {"target": out, "format": "compact", "verbosity": "shape"}},
        ]
        expected_shape = (rows, cols)
    elif family == 2:
        state_dict = _make_state_dict({"x": base})
        mid = f"{out}_fp16"
        transforms = [
            {"cast": {"from": "x", "to": mid, "dtype": "float16"}},
            {"assert": {"dtype": {"of": mid, "is": "float16"}}},
            {"cast": {"from": mid, "to": out, "dtype": "float32"}},
            {"assert": {"dtype": {"of": out, "is": "float32"}}},
            {"dump": {"target": out, "format": "compact", "verbosity": "shape"}},
        ]
        expected_shape = (rows, cols)
    elif family == 3:
        state_dict = _make_state_dict({"x": base})
        transforms = [
            {"fill": {"from": "x", "to": out, "mode": "constant", "value": 0}},
            {"assert": {"iszero": out}},
            {"dump": {"target": out, "format": "compact", "verbosity": "shape"}},
        ]
        expected_shape = (rows, cols)
    elif family == 4:
        split_cols = max(cols, 2)
        split_base = torch.arange(1, rows * split_cols + 1, dtype=torch.float32).reshape(rows, split_cols)
        left = max(1, split_cols // 2)
        right = split_cols - left
        if right == 0:
            left -= 1
            right = 1
        a = f"{out}_a"
        b = f"{out}_b"
        state_dict = _make_state_dict({"x": split_base})
        transforms = [
            {"split": {"from": "x", "to": [a, b], "sizes": [left, right], "dim": 1}},
            {"concat": {"from": [a, b], "to": out, "dim": 1}},
            {"assert": {"equal": {"left": "x", "right": out}}},
            {"dump": {"target": out, "format": "compact", "verbosity": "shape"}},
        ]
        expected_shape = (rows, split_cols)
    elif family == 5:
        state_dict = _make_state_dict({"x": base})
        permuted = f"{out}_p"
        flat_size = rows * cols
        transforms = [
            {"permute": {"from": "x", "to": permuted, "order": [1, 0]}},
            {"reshape": {"from": permuted, "to": out, "shape": [flat_size]}},
            {"assert": {"shape": {"of": out, "is": [flat_size]}}},
            {"dump": {"target": out, "format": "compact", "verbosity": "shape"}},
        ]
        expected_shape = (flat_size,)
    elif family == 6:
        k = (slot % 4) + 1
        a = torch.arange(1, rows * cols + 1, dtype=torch.float32).reshape(rows, cols)
        b = torch.arange(1, cols * k + 1, dtype=torch.float32).reshape(cols, k)
        state_dict = _make_state_dict({"a": a, "b": b})
        transforms = [
            {"matmul": {"from_a": "a", "from_b": "b", "to": out}},
            {"assert": {"shape": {"of": out, "is": [rows, k]}}},
            {"dump": {"target": out, "format": "compact", "verbosity": "shape"}},
        ]
        expected_shape = (rows, k)
    elif family == 7:
        y = torch.full((rows, cols), fill_value=float(slot + 1), dtype=torch.float32)
        state_dict = _make_state_dict({"x": base, "y": y})
        transforms = [
            {"copy": {"from": "x", "to": out}},
            {"add_": {"from": "y", "to": out}},
            {"subtract_": {"from": "y", "to": out}},
            {"assert": {"equal": {"left": "x", "right": out}}},
            {"dump": {"target": out, "format": "compact", "verbosity": "shape"}},
        ]
        expected_shape = (rows, cols)
    elif family == 8:
        state_dict = _make_state_dict({"x": base})
        transforms = [
            {"move": {"from": "x", "to": out}},
            {"assert": {"exists": out}},
            {"dump": {"target": out, "format": "compact", "verbosity": "shape"}},
        ]
        expected_shape = (rows, cols)
    else:
        out_b = f"{out}_b"
        state_dict = _make_state_dict({"x": base})
        transforms = [
            {
                "phlora": {
                    "target": "x",
                    "target_a": out,
                    "target_b": out_b,
                    "rank": 1,
                    "delete_original": False,
                    "require_missing_dest": True,
                }
            },
            {"assert": {"exists": out}},
            {"dump": {"target": out, "format": "compact", "verbosity": "shape"}},
        ]
        expected_shape = (1, cols)

    return _Case(
        state_dict=state_dict,
        transforms=transforms,
        expected_dump=_dump_line(out, expected_shape),
    )

@pytest.mark.parametrize("case_id", range(100), ids=lambda value: f"e2e_{value:03d}")
def test_e2e_multi_transform_dump_compact_shape_regression(
    case_id: int,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    case = _build_case(case_id)
    raw = {"inputs": ["/tmp/model.safetensors"], "transforms": case.transforms}
    plan = compile_plan(raw)
    monkeypatch.setattr(dump_module, "tqdm", lambda iterable, **_: iterable)

    should_continue, executed = execute_transform_pairs(
        zip(raw["transforms"], plan.transforms, strict=False),
        _Provider(case.state_dict),
        interactive=False,
    )

    assert should_continue is True
    assert len(executed) == len(case.transforms)
    assert capsys.readouterr().out.strip() == case.expected_dump
