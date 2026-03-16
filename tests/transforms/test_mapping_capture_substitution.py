from __future__ import annotations

import torch

from brainsurgery.core import get_transform
from brainsurgery.engine.state_dicts import _InMemoryStateDict


class _Provider:
    def __init__(self, state_dict: _InMemoryStateDict) -> None:
        self._state_dict = state_dict

    def get_state_dict(self, model: str) -> _InMemoryStateDict:
        assert model == "model"
        return self._state_dict


def test_binary_mapping_transforms_support_regex_capture_substitution() -> None:
    copy_sd = _InMemoryStateDict()
    copy_sd["block.0.weight"] = torch.tensor([1.0, 2.0], dtype=torch.float32)
    copy_sd["block.1.weight"] = torch.tensor([3.0, 4.0], dtype=torch.float32)
    copy_provider = _Provider(copy_sd)
    copy_transform = get_transform("copy")
    copy_spec = copy_transform.compile(
        {"from": r"block\.(\d+)\.weight", "to": r"out.\1.weight"},
        default_model="model",
    )
    copy_result = copy_transform.apply(copy_spec, copy_provider)
    assert copy_result.count == 2
    assert torch.equal(copy_sd["out.0.weight"], copy_sd["block.0.weight"])
    assert torch.equal(copy_sd["out.1.weight"], copy_sd["block.1.weight"])

    move_sd = _InMemoryStateDict()
    move_sd["block.0.weight"] = torch.tensor([1.0, 2.0], dtype=torch.float32)
    move_sd["block.1.weight"] = torch.tensor([3.0, 4.0], dtype=torch.float32)
    move_provider = _Provider(move_sd)
    move_transform = get_transform("move")
    move_spec = move_transform.compile(
        {"from": r"block\.(\d+)\.weight", "to": r"out.\1.weight"},
        default_model="model",
    )
    move_result = move_transform.apply(move_spec, move_provider)
    assert move_result.count == 2
    assert "block.0.weight" not in move_sd and "block.1.weight" not in move_sd
    assert "out.0.weight" in move_sd and "out.1.weight" in move_sd

    assign_sd = _InMemoryStateDict()
    assign_sd["block.0.weight"] = torch.tensor([1.0, 2.0], dtype=torch.float32)
    assign_sd["block.1.weight"] = torch.tensor([3.0, 4.0], dtype=torch.float32)
    assign_sd["out.0.weight"] = torch.zeros(2, dtype=torch.float32)
    assign_sd["out.1.weight"] = torch.zeros(2, dtype=torch.float32)
    assign_provider = _Provider(assign_sd)
    assign_transform = get_transform("assign")
    assign_spec = assign_transform.compile(
        {"from": r"block\.(\d+)\.weight", "to": r"out.\1.weight"},
        default_model="model",
    )
    assign_result = assign_transform.apply(assign_spec, assign_provider)
    assert assign_result.count == 2
    assert torch.equal(assign_sd["out.0.weight"], assign_sd["block.0.weight"])
    assert torch.equal(assign_sd["out.1.weight"], assign_sd["block.1.weight"])

    add_in_place_sd = _InMemoryStateDict()
    add_in_place_sd["delta.0.weight"] = torch.tensor([1.0, 2.0], dtype=torch.float32)
    add_in_place_sd["delta.1.weight"] = torch.tensor([3.0, 4.0], dtype=torch.float32)
    add_in_place_sd["out.0.weight"] = torch.tensor([10.0, 20.0], dtype=torch.float32)
    add_in_place_sd["out.1.weight"] = torch.tensor([30.0, 40.0], dtype=torch.float32)
    add_in_place_provider = _Provider(add_in_place_sd)
    add_in_place_transform = get_transform("add_")
    add_in_place_spec = add_in_place_transform.compile(
        {"from": r"delta\.(\d+)\.weight", "to": r"out.\1.weight"},
        default_model="model",
    )
    add_in_place_result = add_in_place_transform.apply(add_in_place_spec, add_in_place_provider)
    assert add_in_place_result.count == 2
    assert torch.equal(
        add_in_place_sd["out.0.weight"],
        torch.tensor([11.0, 22.0], dtype=torch.float32),
    )
    assert torch.equal(
        add_in_place_sd["out.1.weight"],
        torch.tensor([33.0, 44.0], dtype=torch.float32),
    )

    subtract_in_place_sd = _InMemoryStateDict()
    subtract_in_place_sd["delta.0.weight"] = torch.tensor([1.0, 2.0], dtype=torch.float32)
    subtract_in_place_sd["delta.1.weight"] = torch.tensor([3.0, 4.0], dtype=torch.float32)
    subtract_in_place_sd["out.0.weight"] = torch.tensor([10.0, 20.0], dtype=torch.float32)
    subtract_in_place_sd["out.1.weight"] = torch.tensor([30.0, 40.0], dtype=torch.float32)
    subtract_in_place_provider = _Provider(subtract_in_place_sd)
    subtract_in_place_transform = get_transform("subtract_")
    subtract_in_place_spec = subtract_in_place_transform.compile(
        {"from": r"delta\.(\d+)\.weight", "to": r"out.\1.weight"},
        default_model="model",
    )
    subtract_in_place_result = subtract_in_place_transform.apply(
        subtract_in_place_spec, subtract_in_place_provider
    )
    assert subtract_in_place_result.count == 2
    assert torch.equal(
        subtract_in_place_sd["out.0.weight"],
        torch.tensor([9.0, 18.0], dtype=torch.float32),
    )
    assert torch.equal(
        subtract_in_place_sd["out.1.weight"],
        torch.tensor([27.0, 36.0], dtype=torch.float32),
    )


def test_binary_mapping_transforms_support_structured_capture_substitution() -> None:
    copy_sd = _InMemoryStateDict()
    copy_sd["block.0.weight"] = torch.tensor([1.0, 2.0], dtype=torch.float32)
    copy_sd["block.1.weight"] = torch.tensor([3.0, 4.0], dtype=torch.float32)
    copy_provider = _Provider(copy_sd)
    copy_transform = get_transform("copy")
    copy_spec = copy_transform.compile(
        {"from": ["block", "$i", "weight"], "to": ["out", "${i}", "weight"]},
        default_model="model",
    )
    copy_result = copy_transform.apply(copy_spec, copy_provider)
    assert copy_result.count == 2
    assert torch.equal(copy_sd["out.0.weight"], copy_sd["block.0.weight"])
    assert torch.equal(copy_sd["out.1.weight"], copy_sd["block.1.weight"])

    assign_sd = _InMemoryStateDict()
    assign_sd["block.0.weight"] = torch.tensor([1.0, 2.0], dtype=torch.float32)
    assign_sd["block.1.weight"] = torch.tensor([3.0, 4.0], dtype=torch.float32)
    assign_sd["out.0.weight"] = torch.zeros(2, dtype=torch.float32)
    assign_sd["out.1.weight"] = torch.zeros(2, dtype=torch.float32)
    assign_provider = _Provider(assign_sd)
    assign_transform = get_transform("assign")
    assign_spec = assign_transform.compile(
        {"from": ["block", "$i", "weight"], "to": ["out", "${i}", "weight"]},
        default_model="model",
    )
    assign_result = assign_transform.apply(assign_spec, assign_provider)
    assert assign_result.count == 2
    assert torch.equal(assign_sd["out.0.weight"], assign_sd["block.0.weight"])
    assert torch.equal(assign_sd["out.1.weight"], assign_sd["block.1.weight"])


def test_ternary_mapping_transforms_support_capture_substitution() -> None:
    add_sd = _InMemoryStateDict()
    add_sd["a.0.weight"] = torch.tensor([1.0, 2.0], dtype=torch.float32)
    add_sd["a.1.weight"] = torch.tensor([3.0, 4.0], dtype=torch.float32)
    add_sd["b.0.weight"] = torch.tensor([10.0, 20.0], dtype=torch.float32)
    add_sd["b.1.weight"] = torch.tensor([30.0, 40.0], dtype=torch.float32)
    add_sd["out.0.weight"] = torch.zeros(2, dtype=torch.float32)
    add_sd["out.1.weight"] = torch.zeros(2, dtype=torch.float32)
    add_provider = _Provider(add_sd)
    add_transform = get_transform("add")
    add_spec = add_transform.compile(
        {
            "from_a": r"a\.(\d+)\.weight",
            "from_b": r"b.\1.weight",
            "to": r"out.\1.weight",
        },
        default_model="model",
    )
    add_result = add_transform.apply(add_spec, add_provider)
    assert add_result.count == 2
    assert torch.equal(add_sd["out.0.weight"], torch.tensor([11.0, 22.0], dtype=torch.float32))
    assert torch.equal(add_sd["out.1.weight"], torch.tensor([33.0, 44.0], dtype=torch.float32))

    multiply_sd = _InMemoryStateDict()
    multiply_sd["a.0.weight"] = torch.tensor([1.0, 2.0], dtype=torch.float32)
    multiply_sd["a.1.weight"] = torch.tensor([3.0, 4.0], dtype=torch.float32)
    multiply_sd["b.0.weight"] = torch.tensor([10.0, 20.0], dtype=torch.float32)
    multiply_sd["b.1.weight"] = torch.tensor([30.0, 40.0], dtype=torch.float32)
    multiply_sd["out.0.weight"] = torch.zeros(2, dtype=torch.float32)
    multiply_sd["out.1.weight"] = torch.zeros(2, dtype=torch.float32)
    multiply_provider = _Provider(multiply_sd)
    multiply_transform = get_transform("multiply")
    multiply_spec = multiply_transform.compile(
        {
            "from_a": ["a", "$i", "weight"],
            "from_b": ["b", "${i}", "weight"],
            "to": ["out", "${i}", "weight"],
        },
        default_model="model",
    )
    multiply_result = multiply_transform.apply(multiply_spec, multiply_provider)
    assert multiply_result.count == 2
    assert torch.equal(
        multiply_sd["out.0.weight"],
        torch.tensor([10.0, 40.0], dtype=torch.float32),
    )
    assert torch.equal(
        multiply_sd["out.1.weight"],
        torch.tensor([90.0, 160.0], dtype=torch.float32),
    )

    subtract_sd = _InMemoryStateDict()
    subtract_sd["a.0.weight"] = torch.tensor([1.0, 2.0], dtype=torch.float32)
    subtract_sd["a.1.weight"] = torch.tensor([3.0, 4.0], dtype=torch.float32)
    subtract_sd["b.0.weight"] = torch.tensor([10.0, 20.0], dtype=torch.float32)
    subtract_sd["b.1.weight"] = torch.tensor([30.0, 40.0], dtype=torch.float32)
    subtract_sd["out.0.weight"] = torch.zeros(2, dtype=torch.float32)
    subtract_sd["out.1.weight"] = torch.zeros(2, dtype=torch.float32)
    subtract_provider = _Provider(subtract_sd)
    subtract_transform = get_transform("subtract")
    subtract_spec = subtract_transform.compile(
        {
            "from_a": r"a\.(\d+)\.weight",
            "from_b": r"b.\1.weight",
            "to": r"out.\1.weight",
        },
        default_model="model",
    )
    subtract_result = subtract_transform.apply(subtract_spec, subtract_provider)
    assert subtract_result.count == 2
    assert torch.equal(
        subtract_sd["out.0.weight"],
        torch.tensor([-9.0, -18.0], dtype=torch.float32),
    )
    assert torch.equal(
        subtract_sd["out.1.weight"],
        torch.tensor([-27.0, -36.0], dtype=torch.float32),
    )

    matmul_sd = _InMemoryStateDict()
    matmul_sd["a.0.weight"] = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    matmul_sd["a.1.weight"] = torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.float32)
    matmul_sd["b.0.weight"] = torch.tensor([[2.0, 0.0], [1.0, 2.0]], dtype=torch.float32)
    matmul_sd["b.1.weight"] = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    matmul_provider = _Provider(matmul_sd)
    matmul_transform = get_transform("matmul")
    matmul_spec = matmul_transform.compile(
        {
            "from_a": ["a", "$i", "weight"],
            "from_b": ["b", "${i}", "weight"],
            "to": ["out", "${i}", "weight"],
        },
        default_model="model",
    )
    matmul_result = matmul_transform.apply(matmul_spec, matmul_provider)
    assert matmul_result.count == 2
    assert torch.equal(
        matmul_sd["out.0.weight"],
        torch.tensor([[4.0, 4.0], [10.0, 8.0]], dtype=torch.float32),
    )
    assert torch.equal(
        matmul_sd["out.1.weight"],
        torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.float32),
    )
