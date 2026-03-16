from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file as save_safetensors_file

import brainsurgery.transforms.dump as dump_module
from brainsurgery.core import TransformError
from brainsurgery.engine import (
    RuntimeFlagLifecycleScope,
    create_state_dict_provider,
    reset_runtime_flags_for_scope,
)
from brainsurgery.engine.execution import _execute_transform_pairs
from brainsurgery.engine.plan import compile_plan
from brainsurgery.engine.state_dicts import _InMemoryStateDict


class _Provider:
    def __init__(self, state_dicts: dict[str, _InMemoryStateDict]) -> None:
        self._state_dicts = state_dicts

    def get_state_dict(self, model: str) -> _InMemoryStateDict:
        return self._state_dicts[model]


def _make_state_dict(values: dict[str, torch.Tensor]) -> _InMemoryStateDict:
    sd = _InMemoryStateDict()
    for key, tensor in values.items():
        sd[key] = tensor
    return sd


def test_cross_compile_execute_copy_then_assert_equal() -> None:
    raw = {
        "inputs": ["/tmp/model.safetensors"],
        "transforms": [
            {"copy": {"from": "src", "to": "dst"}},
            {"assert": {"equal": {"left": "src", "right": "dst"}}},
        ],
    }
    plan = compile_plan(raw)
    provider = _Provider(
        {
            "model": _make_state_dict(
                {
                    "src": torch.tensor([1.0, 2.0], dtype=torch.float32),
                }
            )
        }
    )
    should_continue, executed = _execute_transform_pairs(
        zip(raw["transforms"], plan.transforms, strict=False),
        provider,
        interactive=False,
    )

    assert should_continue is True
    assert len(executed) == 2
    model_sd = provider.get_state_dict("model")
    assert torch.equal(model_sd["src"], model_sd["dst"])


def test_cross_execute_interactive_stops_current_block_on_failure() -> None:
    raw = {
        "inputs": ["/tmp/model.safetensors"],
        "transforms": [
            {"copy": {"from": "src", "to": "tmp"}},
            {"assign": {"from": "src", "to": "dst"}},
            {"copy": {"from": "src", "to": "after_fail"}},
        ],
    }
    plan = compile_plan(raw)
    provider = _Provider(
        {
            "model": _make_state_dict(
                {
                    "src": torch.tensor([1.0, 2.0], dtype=torch.float32),
                    "dst": torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32),
                }
            )
        }
    )

    should_continue, executed = _execute_transform_pairs(
        zip(raw["transforms"], plan.transforms, strict=False),
        provider,
        interactive=True,
    )

    assert should_continue is True
    assert executed == [raw["transforms"][0]]
    model_sd = provider.get_state_dict("model")
    assert "tmp" in model_sd
    assert "after_fail" not in model_sd


def test_cross_execute_non_interactive_raises_on_failure() -> None:
    raw = {
        "inputs": ["/tmp/model.safetensors"],
        "transforms": [
            {"assign": {"from": "src", "to": "dst"}},
        ],
    }
    plan = compile_plan(raw)
    provider = _Provider(
        {
            "model": _make_state_dict(
                {
                    "src": torch.tensor([1.0, 2.0], dtype=torch.float32),
                    "dst": torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32),
                }
            )
        }
    )

    with pytest.raises(TransformError):
        _execute_transform_pairs(
            zip(raw["transforms"], plan.transforms, strict=False),
            provider,
            interactive=False,
        )


def test_cross_structured_mapping_with_copy_transform() -> None:
    raw = {
        "inputs": ["/tmp/model.safetensors"],
        "transforms": [
            {
                "copy": {
                    "from": ["block", "$i", "weight"],
                    "to": ["backup", "${i}", "weight"],
                }
            }
        ],
    }
    plan = compile_plan(raw)
    provider = _Provider(
        {
            "model": _make_state_dict(
                {
                    "block.0.weight": torch.tensor([1.0], dtype=torch.float32),
                    "block.1.weight": torch.tensor([2.0], dtype=torch.float32),
                }
            )
        }
    )

    should_continue, executed = _execute_transform_pairs(
        zip(raw["transforms"], plan.transforms, strict=False),
        provider,
        interactive=False,
    )

    assert should_continue is True
    assert len(executed) == 1
    model_sd = provider.get_state_dict("model")
    assert torch.equal(model_sd["backup.0.weight"], model_sd["block.0.weight"])
    assert torch.equal(model_sd["backup.1.weight"], model_sd["block.1.weight"])


def test_cross_assert_transform_covers_all_expression_ops() -> None:
    raw = {
        "inputs": ["/tmp/model.safetensors"],
        "transforms": [
            {"assert": {"exists": "x"}},
            {"assert": {"count": {"of": "vec_.*", "is": 2}}},
            {"assert": {"dimensions": {"of": "x", "is": 2}}},
            {"assert": {"dtype": {"of": "x", "is": "float32"}}},
            {"assert": {"shape": {"of": "x", "is": [2, 2]}}},
            {"assert": {"iszero": "x"}},
            {"assert": {"equal": {"left": "y", "right": "z"}}},
            {"assert": {"all": [{"exists": "x"}, {"dtype": {"of": "x", "is": "float32"}}]}},
            {"assert": {"any": [{"exists": "missing.*"}, {"exists": "x"}]}},
            {"assert": {"not": {"exists": "missing.*"}}},
        ],
    }
    plan = compile_plan(raw)
    provider = _Provider(
        {
            "model": _make_state_dict(
                {
                    "x": torch.zeros((2, 2), dtype=torch.float32),
                    "y": torch.tensor([1.0, 2.0], dtype=torch.float32),
                    "z": torch.tensor([1.0, 2.0], dtype=torch.float32),
                    "vec_0": torch.tensor([1.0]),
                    "vec_1": torch.tensor([2.0]),
                }
            )
        }
    )

    should_continue, executed = _execute_transform_pairs(
        zip(raw["transforms"], plan.transforms, strict=False),
        provider,
        interactive=False,
    )

    assert should_continue is True
    assert len(executed) == len(raw["transforms"])


def test_cross_assert_equal_with_eps() -> None:
    raw = {
        "inputs": ["/tmp/model.safetensors"],
        "transforms": [
            {"assert": {"equal": {"left": "y", "right": "z", "eps": 1e-3}}},
        ],
    }
    plan = compile_plan(raw)
    provider = _Provider(
        {
            "model": _make_state_dict(
                {
                    "y": torch.tensor([1.0, 2.0], dtype=torch.float32),
                    "z": torch.tensor([1.0, 2.0005], dtype=torch.float32),
                }
            )
        }
    )

    should_continue, executed = _execute_transform_pairs(
        zip(raw["transforms"], plan.transforms, strict=False),
        provider,
        interactive=False,
    )

    assert should_continue is True
    assert len(executed) == 1


def test_cross_help_dump_exit_in_single_flow(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    raw = {
        "inputs": ["/tmp/model.safetensors"],
        "transforms": [
            {"help": "assert"},
            {"dump": {"target": "x", "format": "json", "verbosity": "shape"}},
            {"exit": {}},
            {"copy": {"from": "x", "to": "x_copy_should_not_exist"}},
        ],
    }
    plan = compile_plan(raw)
    provider = _Provider(
        {
            "model": _make_state_dict(
                {
                    "x": torch.zeros((2, 2), dtype=torch.float32),
                }
            )
        }
    )
    monkeypatch.setattr(dump_module, "tqdm", lambda iterable, **_: iterable)

    should_continue, executed = _execute_transform_pairs(
        zip(raw["transforms"], plan.transforms, strict=False),
        provider,
        interactive=False,
    )

    output = capsys.readouterr().out
    assert "Command: assert" in output
    assert '{"x":{"shape":[2,2]}}' in output
    assert should_continue is False
    assert executed == raw["transforms"][:3]
    assert "x_copy_should_not_exist" not in provider.get_state_dict("model")


def test_cross_prefixes_lists_available_aliases(capsys: pytest.CaptureFixture[str]) -> None:
    raw = {
        "inputs": ["/tmp/model.safetensors"],
        "transforms": [
            {"prefixes": {}},
        ],
    }
    plan = compile_plan(raw)
    provider = _Provider(
        {
            "model": _make_state_dict({"x": torch.zeros((1,), dtype=torch.float32)}),
            "scratch": _make_state_dict({"y": torch.ones((1,), dtype=torch.float32)}),
        }
    )

    should_continue, executed = _execute_transform_pairs(
        zip(raw["transforms"], plan.transforms, strict=False),
        provider,
        interactive=False,
    )

    output = capsys.readouterr().out
    assert "Available model prefixes:" in output
    assert "  model::" in output
    assert "  scratch::" in output
    assert should_continue is True
    assert executed == raw["transforms"]


def test_cross_set_verbose_then_copy_emits_activity(capsys: pytest.CaptureFixture[str]) -> None:
    raw = {
        "inputs": ["/tmp/model.safetensors"],
        "transforms": [
            {"set": {"verbose": True}},
            {"copy": {"from": "h.0.attn.bias", "to": "i.0.attn.bias"}},
        ],
    }
    plan = compile_plan(raw)
    provider = _Provider(
        {
            "model": _make_state_dict(
                {
                    "h.0.attn.bias": torch.ones((1,), dtype=torch.float32),
                }
            )
        }
    )
    reset_runtime_flags_for_scope(RuntimeFlagLifecycleScope.CLI_RUN)
    should_continue, executed = _execute_transform_pairs(
        zip(raw["transforms"], plan.transforms, strict=False),
        provider,
        interactive=False,
    )
    output = capsys.readouterr().out
    reset_runtime_flags_for_scope(RuntimeFlagLifecycleScope.CLI_RUN)

    assert should_continue is True
    assert executed == raw["transforms"]
    assert "copy: h.0.attn.bias -> i.0.attn.bias" in output


def test_cross_prefixes_add_creates_empty_alias() -> None:
    raw = {
        "inputs": ["/tmp/model.safetensors"],
        "transforms": [
            {"prefixes": {"mode": "add", "alias": "scratch"}},
        ],
    }
    plan = compile_plan(raw)
    provider = _Provider(
        {
            "model": _make_state_dict({"x": torch.zeros((1,), dtype=torch.float32)}),
        }
    )

    should_continue, executed = _execute_transform_pairs(
        zip(raw["transforms"], plan.transforms, strict=False),
        provider,
        interactive=False,
    )

    assert should_continue is True
    assert executed == raw["transforms"]
    assert set(provider._state_dicts.keys()) == {"model", "scratch"}
    assert len(provider._state_dicts["scratch"]) == 0
    assert provider._state_dicts["scratch"] is not provider._state_dicts["model"]


def test_cross_prefixes_rename_remove_mutate_aliases() -> None:
    raw = {
        "inputs": ["/tmp/model.safetensors"],
        "transforms": [
            {"prefixes": {"mode": "rename", "from": "scratch", "to": "edited"}},
            {"prefixes": {"mode": "remove", "alias": "edited"}},
        ],
    }
    plan = compile_plan(raw)
    provider = _Provider(
        {
            "model": _make_state_dict({"x": torch.zeros((1,), dtype=torch.float32)}),
            "scratch": _make_state_dict({}),
        }
    )

    should_continue, executed = _execute_transform_pairs(
        zip(raw["transforms"], plan.transforms, strict=False),
        provider,
        interactive=False,
    )

    assert should_continue is True
    assert executed == raw["transforms"]
    assert set(provider._state_dicts.keys()) == {"model"}


def _write_checkpoint(path: Path) -> None:
    save_safetensors_file(
        {
            "src": torch.tensor([1.0, 2.0], dtype=torch.float32),
            "dst": torch.tensor([0.0, 0.0], dtype=torch.float32),
            "x": torch.tensor([3.0, 4.0], dtype=torch.float32),
        },
        str(path),
    )


@pytest.mark.parametrize("provider_name", ["inmemory", "arena"])
def test_cross_provider_backends_load_execute_and_save(provider_name: str, tmp_path: Path) -> None:
    in_path = tmp_path / "in.safetensors"
    out_path = tmp_path / f"out-{provider_name}.safetensors"
    _write_checkpoint(in_path)

    raw = {
        "inputs": [str(in_path)],
        "output": str(out_path),
        "transforms": [
            {"assign": {"from": "src", "to": "dst"}},
            {"scale_": {"target": "dst", "by": 2}},
            {"cast_": {"target": "dst", "to": "float16"}},
            {"copy": {"from": "x", "to": "x_copy"}},
            {"move": {"from": "x_copy", "to": "x_moved"}},
            {"delete": {"target": "x_moved"}},
            {"assert": {"dtype": {"of": "dst", "is": "float16"}}},
        ],
    }
    plan = compile_plan(raw)
    provider = create_state_dict_provider(
        provider=provider_name,
        model_paths=plan.inputs,
        max_io_workers=2,
        arena_root=tmp_path / "arena",
        arena_segment_size="1MB",
    )
    try:
        should_continue, executed = _execute_transform_pairs(
            zip(raw["transforms"], plan.transforms, strict=False),
            provider,
            interactive=False,
        )
        written = provider.save_output(
            plan,
            default_shard_size="1GB",
            max_io_workers=2,
        )
    finally:
        provider.close()

    assert should_continue is True
    assert len(executed) == len(raw["transforms"])
    assert written == out_path
    assert out_path.exists()


def test_cross_scale_creates_new_tensor() -> None:
    raw = {
        "inputs": ["/tmp/model.safetensors"],
        "transforms": [
            {"scale": {"from": "src", "to": "scaled", "by": 2}},
            {"assert": {"equal": {"left": "scaled", "right": "expected"}}},
        ],
    }
    plan = compile_plan(raw)
    provider = _Provider(
        {
            "model": _make_state_dict(
                {
                    "src": torch.tensor([1.0, 2.0], dtype=torch.float32),
                    "expected": torch.tensor([2.0, 4.0], dtype=torch.float32),
                }
            )
        }
    )

    should_continue, executed = _execute_transform_pairs(
        zip(raw["transforms"], plan.transforms, strict=False),
        provider,
        interactive=False,
    )

    assert should_continue is True
    assert len(executed) == len(raw["transforms"])
    model_sd = provider.get_state_dict("model")
    assert torch.equal(model_sd["scaled"], torch.tensor([2.0, 4.0], dtype=torch.float32))
    assert torch.equal(model_sd["src"], torch.tensor([1.0, 2.0], dtype=torch.float32))


def test_cross_cast_creates_new_tensor() -> None:
    raw = {
        "inputs": ["/tmp/model.safetensors"],
        "transforms": [
            {"cast": {"from": "src", "to": "src_fp16", "dtype": "float16"}},
            {"assert": {"dtype": {"of": "src_fp16", "is": "float16"}}},
            {"assert": {"dtype": {"of": "src", "is": "float32"}}},
        ],
    }
    plan = compile_plan(raw)
    provider = _Provider(
        {
            "model": _make_state_dict(
                {
                    "src": torch.tensor([1.0, 2.0], dtype=torch.float32),
                }
            )
        }
    )

    should_continue, executed = _execute_transform_pairs(
        zip(raw["transforms"], plan.transforms, strict=False),
        provider,
        interactive=False,
    )

    assert should_continue is True
    assert len(executed) == len(raw["transforms"])
    model_sd = provider.get_state_dict("model")
    assert model_sd["src"].dtype == torch.float32
    assert model_sd["src_fp16"].dtype == torch.float16


def test_cross_cast__casts_in_place() -> None:
    raw = {
        "inputs": ["/tmp/model.safetensors"],
        "transforms": [
            {"cast_": {"target": "src", "to": "float16"}},
            {"assert": {"dtype": {"of": "src", "is": "float16"}}},
        ],
    }
    plan = compile_plan(raw)
    provider = _Provider(
        {
            "model": _make_state_dict(
                {
                    "src": torch.tensor([1.0, 2.0], dtype=torch.float32),
                }
            )
        }
    )

    should_continue, executed = _execute_transform_pairs(
        zip(raw["transforms"], plan.transforms, strict=False),
        provider,
        interactive=False,
    )

    assert should_continue is True
    assert len(executed) == len(raw["transforms"])
    model_sd = provider.get_state_dict("model")
    assert model_sd["src"].dtype == torch.float16


def test_cross_add_subtract_multiply_pipeline() -> None:
    raw = {
        "inputs": ["/tmp/model.safetensors"],
        "transforms": [
            {"add": {"from_a": "x", "from_b": "y", "to": "sum"}},
            {"subtract": {"from_a": "sum", "from_b": "y", "to": "back_x"}},
            {"multiply": {"from_a": "x", "from_b": "y", "to": "prod"}},
            {"assert": {"equal": {"left": "back_x", "right": "x"}}},
            {"assert": {"equal": {"left": "prod", "right": "expected_prod"}}},
        ],
    }
    plan = compile_plan(raw)
    provider = _Provider(
        {
            "model": _make_state_dict(
                {
                    "x": torch.tensor([2.0, 3.0], dtype=torch.float32),
                    "y": torch.tensor([4.0, 5.0], dtype=torch.float32),
                    "sum": torch.zeros((2,), dtype=torch.float32),
                    "back_x": torch.zeros((2,), dtype=torch.float32),
                    "prod": torch.zeros((2,), dtype=torch.float32),
                    "expected_prod": torch.tensor([8.0, 15.0], dtype=torch.float32),
                }
            )
        }
    )

    should_continue, executed = _execute_transform_pairs(
        zip(raw["transforms"], plan.transforms, strict=False),
        provider,
        interactive=False,
    )

    assert should_continue is True
    assert len(executed) == len(raw["transforms"])
    model_sd = provider.get_state_dict("model")
    assert torch.equal(model_sd["sum"], torch.tensor([6.0, 8.0], dtype=torch.float32))
    assert torch.equal(model_sd["back_x"], model_sd["x"])
    assert torch.equal(model_sd["prod"], model_sd["expected_prod"])


def test_cross_structured_mapping_with_add_transform() -> None:
    raw = {
        "inputs": ["/tmp/model.safetensors"],
        "transforms": [
            {
                "add": {
                    "from_a": ["block", "$i", "weight"],
                    "from_b": ["delta", "${i}", "weight"],
                    "to": ["out", "${i}", "weight"],
                }
            }
        ],
    }
    plan = compile_plan(raw)
    provider = _Provider(
        {
            "model": _make_state_dict(
                {
                    "block.0.weight": torch.tensor([1.0, 2.0], dtype=torch.float32),
                    "block.1.weight": torch.tensor([3.0, 4.0], dtype=torch.float32),
                    "delta.0.weight": torch.tensor([10.0, 20.0], dtype=torch.float32),
                    "delta.1.weight": torch.tensor([30.0, 40.0], dtype=torch.float32),
                    "out.0.weight": torch.zeros((2,), dtype=torch.float32),
                    "out.1.weight": torch.zeros((2,), dtype=torch.float32),
                }
            )
        }
    )

    should_continue, executed = _execute_transform_pairs(
        zip(raw["transforms"], plan.transforms, strict=False),
        provider,
        interactive=False,
    )

    assert should_continue is True
    assert len(executed) == 1
    model_sd = provider.get_state_dict("model")
    assert torch.equal(
        model_sd["out.0.weight"],
        torch.tensor([11.0, 22.0], dtype=torch.float32),
    )
    assert torch.equal(
        model_sd["out.1.weight"],
        torch.tensor([33.0, 44.0], dtype=torch.float32),
    )


def test_cross_add_requires_existing_destination() -> None:
    raw = {
        "inputs": ["/tmp/model.safetensors"],
        "transforms": [
            {"add": {"from_a": "x", "from_b": "y", "to": "missing_dst"}},
        ],
    }
    plan = compile_plan(raw)
    provider = _Provider(
        {
            "model": _make_state_dict(
                {
                    "x": torch.tensor([1.0, 2.0], dtype=torch.float32),
                    "y": torch.tensor([3.0, 4.0], dtype=torch.float32),
                }
            )
        }
    )

    with pytest.raises(TransformError, match="destination missing"):
        _execute_transform_pairs(
            zip(raw["transforms"], plan.transforms, strict=False),
            provider,
            interactive=False,
        )


def test_cross_add__subtract__in_place_pipeline() -> None:
    raw = {
        "inputs": ["/tmp/model.safetensors"],
        "transforms": [
            {"add_": {"from": "delta", "to": "base"}},
            {"subtract_": {"from": "delta", "to": "base"}},
            {"assert": {"equal": {"left": "base", "right": "expected"}}},
        ],
    }
    plan = compile_plan(raw)
    provider = _Provider(
        {
            "model": _make_state_dict(
                {
                    "base": torch.tensor([2.0, 3.0], dtype=torch.float32),
                    "delta": torch.tensor([4.0, 5.0], dtype=torch.float32),
                    "expected": torch.tensor([2.0, 3.0], dtype=torch.float32),
                }
            )
        }
    )

    should_continue, executed = _execute_transform_pairs(
        zip(raw["transforms"], plan.transforms, strict=False),
        provider,
        interactive=False,
    )

    assert should_continue is True
    assert len(executed) == len(raw["transforms"])
    model_sd = provider.get_state_dict("model")
    assert torch.equal(model_sd["base"], torch.tensor([2.0, 3.0], dtype=torch.float32))


def test_cross_dry_run_executes_flow_without_persisting_changes_and_prefixes_verbose(
    capsys: pytest.CaptureFixture[str],
) -> None:
    raw = {
        "inputs": ["/tmp/model.safetensors"],
        "transforms": [
            {"set": {"dry-run": True, "verbose": True}},
            {"copy": {"from": "src", "to": "dst"}},
            {"assert": {"equal": {"left": "src", "right": "dst"}}},
        ],
    }
    plan = compile_plan(raw)
    provider = _Provider(
        {
            "model": _make_state_dict(
                {
                    "src": torch.tensor([1.0, 2.0], dtype=torch.float32),
                }
            )
        }
    )
    baseline_counts = provider.get_state_dict("model").access_counts("src")

    reset_runtime_flags_for_scope(RuntimeFlagLifecycleScope.CLI_RUN)
    should_continue, executed = _execute_transform_pairs(
        zip(raw["transforms"], plan.transforms, strict=False),
        provider,
        interactive=False,
    )
    output = capsys.readouterr().out

    assert should_continue is True
    assert len(executed) == len(raw["transforms"])
    assert "dry-run copy: src -> dst" in output
    assert "dry-run assert: ok" in output

    reset_runtime_flags_for_scope(RuntimeFlagLifecycleScope.CLI_RUN)
    model_sd = provider.get_state_dict("model")
    assert "dst" not in model_sd
    assert model_sd.access_counts("src") == baseline_counts
