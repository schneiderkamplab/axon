from __future__ import annotations

from pathlib import Path

import pytest

from brainsurgery.engine.plan import PlanLoaderError, compile_plan

def test_compile_plan_allows_missing_inputs_for_non_tensor_transforms_and_load() -> None:
    plan = compile_plan(
        {
            "transforms": [
                {"help": {}},
                {"prefixes": {}},
                {"load": {"path": "/tmp/model.safetensors", "alias": "loaded"}},
                {"exit": {}},
            ]
        }
    )
    assert plan.inputs == {}
    assert len(plan.transforms) == 4

def test_compile_plan_allows_empty_inputs_list() -> None:
    plan = compile_plan({"inputs": [], "transforms": [{"help": {}}]})
    assert plan.inputs == {}
    assert len(plan.transforms) == 1

def test_compile_plan_allows_missing_transforms() -> None:
    plan = compile_plan({"inputs": ["model::/tmp/model.safetensors"]})
    assert plan.inputs == {"model": Path("/tmp/model.safetensors")}
    assert plan.transforms == []

def test_compile_plan_rejects_non_basic_transform_without_inputs() -> None:
    with pytest.raises(PlanLoaderError, match="missing model alias in reference"):
        compile_plan(
            {
                "transforms": [
                    {"copy": {"from": "a", "to": "b"}},
                ]
            }
        )

@pytest.mark.parametrize(
    ("inputs", "output"),
    [
        (["model::/tmp/model.safetensors"], None),  # 1 input, 0 outputs
        (["left::/tmp/left.safetensors", "right::/tmp/right.safetensors"], None),  # 2 inputs, 0 outputs
        (["model::/tmp/model.safetensors"], "/tmp/out.safetensors"),  # 1 input, 1 output
        (  # 2 inputs, 1 output
            ["left::/tmp/left.safetensors", "right::/tmp/right.safetensors"],
            "/tmp/out.safetensors",
        ),
    ],
)
def test_compile_plan_input_output_count_combinations(
    inputs: list[str],
    output: str | None,
) -> None:
    raw_plan: dict[str, object] = {
        "inputs": inputs,
        "transforms": [{"copy": {"from": f"{inputs[0].split('::', 1)[0]}::a", "to": f"{inputs[0].split('::', 1)[0]}::b"}}],
    }
    if output is not None:
        raw_plan["output"] = output

    plan = compile_plan(raw_plan)
    assert len(plan.inputs) == len(inputs)
    if output is None:
        assert plan.output is None
    else:
        assert plan.output is not None
        assert plan.output.path == Path(output)
