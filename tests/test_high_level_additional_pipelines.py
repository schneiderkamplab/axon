from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file as save_safetensors_file

from brainsurgery.engine import create_state_dict_provider
from brainsurgery.engine.execution import execute_transform_pairs
from brainsurgery.engine.plan import compile_plan

def _run_pipeline(raw_plan: dict[str, object], provider) -> tuple[bool, list[dict[str, object]]]:
    plan = compile_plan(raw_plan)
    return execute_transform_pairs(
        zip(raw_plan["transforms"], plan.transforms, strict=False),  # type: ignore[index]
        provider,
        interactive=False,
    )

def _assert_no_diff_output(output: str) -> None:
    assert "No differences found." in output
    assert "Missing on left:\n  (none)\n" in output
    assert "Missing on right:\n  (none)\n" in output
    assert "Differing:\n  (none)\n" in output

def _toy_tensors() -> dict[str, torch.Tensor]:
    return {
        "wte.weight": torch.arange(24, dtype=torch.float32).reshape(6, 4),
        "ln_f.weight": torch.arange(4, dtype=torch.float32),
        "h.0.attn.c_attn.weight": torch.arange(48, dtype=torch.float32).reshape(4, 12),
        "h.0.attn.c_proj.weight": torch.arange(16, dtype=torch.float32).reshape(4, 4),
        "h.0.mlp.c_fc.bias": torch.arange(8, dtype=torch.float32),
        "h.0.mlp.c_proj.bias": torch.arange(4, dtype=torch.float32),
        "mat.left": torch.arange(6, dtype=torch.float32).reshape(2, 3),
        "mat.right": torch.arange(6, dtype=torch.float32).reshape(3, 2),
        "mat.target": torch.arange(4, dtype=torch.float32).reshape(2, 2),
        "ones.2x2": torch.ones((2, 2), dtype=torch.float32),
        "x.square": torch.arange(16, dtype=torch.float32).reshape(4, 4),
    }

def _write_toy_checkpoint(path: Path) -> None:
    save_safetensors_file(_toy_tensors(), str(path))

def test_sharded_hf_style_migration_roundtrip_has_no_diff(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    original_path = tmp_path / "original.safetensors"
    sharded_dir = tmp_path / "migrated_sharded"

    save_safetensors_file(
        {
            "wte.weight": torch.arange(24, dtype=torch.float32).reshape(6, 4),
            "ln_f.weight": torch.arange(4, dtype=torch.float32),
            "h.0.attn.c_attn.weight": torch.arange(48, dtype=torch.float32).reshape(4, 12),
            "h.0.attn.c_proj.weight": torch.arange(16, dtype=torch.float32).reshape(4, 4),
            "h.0.mlp.c_fc.bias": torch.arange(8, dtype=torch.float32),
            "h.0.mlp.c_proj.bias": torch.arange(4, dtype=torch.float32),
        },
        str(original_path),
    )

    forward_rules = [
        (r"base::wte\.weight", r"hf::model.embed_tokens.weight"),
        (r"base::ln_f\.weight", r"hf::model.norm.weight"),
        (r"base::h\.(\d+)\.attn\.c_attn\.(weight|bias)", r"hf::model.layers.\1.self_attn.qkv_proj.\2"),
        (r"base::h\.(\d+)\.attn\.c_proj\.(weight|bias)", r"hf::model.layers.\1.self_attn.o_proj.\2"),
        (r"base::h\.(\d+)\.mlp\.c_fc\.(weight|bias)", r"hf::model.layers.\1.mlp.up_proj.\2"),
        (r"base::h\.(\d+)\.mlp\.c_proj\.(weight|bias)", r"hf::model.layers.\1.mlp.down_proj.\2"),
    ]
    reverse_rules = [
        (r"hf::model\.embed_tokens\.weight", r"hf::wte.weight"),
        (r"hf::model\.norm\.weight", r"hf::ln_f.weight"),
        (r"hf::model\.layers\.(\d+)\.self_attn\.qkv_proj\.(weight|bias)", r"hf::h.\1.attn.c_attn.\2"),
        (r"hf::model\.layers\.(\d+)\.self_attn\.o_proj\.(weight|bias)", r"hf::h.\1.attn.c_proj.\2"),
        (r"hf::model\.layers\.(\d+)\.mlp\.up_proj\.(weight|bias)", r"hf::h.\1.mlp.c_fc.\2"),
        (r"hf::model\.layers\.(\d+)\.mlp\.down_proj\.(weight|bias)", r"hf::h.\1.mlp.c_proj.\2"),
    ]

    pipeline1 = {
        "inputs": [f"base::{original_path}"],
        "transforms": [{"prefixes": {"mode": "add", "alias": "hf"}}]
        + [{"copy": {"from": src, "to": dst}} for src, dst in forward_rules]
        + [{"save": {"path": str(sharded_dir), "alias": "hf", "format": "safetensors", "shard": "1KB"}}],
    }
    provider = create_state_dict_provider(
        provider="arena",
        model_paths=compile_plan(pipeline1).inputs,
        max_io_workers=2,
        arena_root=tmp_path / "arena_migrate_1",
        arena_segment_size="256MB",
    )
    try:
        should_continue, executed = _run_pipeline(pipeline1, provider)
        assert should_continue is True
        assert len(executed) == len(pipeline1["transforms"])
    finally:
        provider.close()

    assert (sharded_dir / "model.safetensors.index.json").exists()

    pipeline2 = {
        "inputs": [],
        "transforms": [{"load": {"path": str(sharded_dir), "alias": "hf"}}]
        + [{"move": {"from": src, "to": dst}} for src, dst in reverse_rules]
        + [
            {"load": {"path": str(original_path), "alias": "base"}},
            {"diff": {"mode": "aliases", "left_alias": "base", "right_alias": "hf"}},
        ],
    }
    provider = create_state_dict_provider(
        provider="inmemory",
        model_paths={},
        max_io_workers=2,
        arena_root=tmp_path / "arena_migrate_2",
        arena_segment_size="256MB",
    )
    try:
        should_continue, executed = _run_pipeline(pipeline2, provider)
        assert should_continue is True
        assert len(executed) == len(pipeline2["transforms"])
    finally:
        provider.close()

    output = capsys.readouterr().out
    assert "No differences found." in output
    assert "Missing on left:\n  (none)\n" in output
    assert "Missing on right:\n  (none)\n" in output
    assert "Differing:\n  (none)\n" in output

def test_reversible_arithmetic_pipeline_roundtrip_has_no_diff(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    original_path = tmp_path / "arith_original.safetensors"
    edited_path = tmp_path / "arith_edited.safetensors"
    targets = [
        "wte.weight",
        "ln_f.weight",
        "h.0.attn.c_attn.weight",
        "h.0.attn.c_proj.weight",
        "h.0.mlp.c_fc.bias",
    ]
    delta = 2.0

    save_safetensors_file(
        {
            "wte.weight": torch.arange(24, dtype=torch.float32).reshape(6, 4),
            "ln_f.weight": torch.arange(4, dtype=torch.float32),
            "h.0.attn.c_attn.weight": torch.arange(48, dtype=torch.float32).reshape(4, 12),
            "h.0.attn.c_proj.weight": torch.arange(16, dtype=torch.float32).reshape(4, 4),
            "h.0.mlp.c_fc.bias": torch.arange(8, dtype=torch.float32),
        },
        str(original_path),
    )

    forward_transforms: list[dict[str, object]] = [
        {"prefixes": {"mode": "add", "alias": "work"}},
        {"copy": {"from": r"base::(.+)", "to": r"work::\1"}},
    ]
    for name in targets:
        delta_name = f"delta.{name}"
        forward_transforms.extend(
            [
                {"fill": {"from": f"work::{name}", "to": f"work::{delta_name}", "mode": "constant", "value": delta}},
                {"add_": {"from": f"work::{delta_name}", "to": f"work::{name}"}},
                {"scale_": {"target": f"work::{name}", "by": 0.5}},
                {"delete": {"target": f"work::{delta_name}"}},
            ]
        )
    forward_transforms.append({"save": {"path": str(edited_path), "alias": "work", "format": "safetensors"}})

    pipeline1 = {"inputs": [f"base::{original_path}"], "transforms": forward_transforms}
    provider = create_state_dict_provider(
        provider="inmemory",
        model_paths=compile_plan(pipeline1).inputs,
        max_io_workers=2,
        arena_root=tmp_path / "arena_arith_1",
        arena_segment_size="256MB",
    )
    try:
        should_continue, executed = _run_pipeline(pipeline1, provider)
        assert should_continue is True
        assert len(executed) == len(pipeline1["transforms"])
    finally:
        provider.close()

    assert edited_path.exists()

    reverse_transforms: list[dict[str, object]] = [{"load": {"path": str(edited_path), "alias": "work"}}]
    for name in targets:
        delta_name = f"delta.{name}"
        reverse_transforms.extend(
            [
                {"scale_": {"target": f"work::{name}", "by": 2.0}},
                {"fill": {"from": f"work::{name}", "to": f"work::{delta_name}", "mode": "constant", "value": delta}},
                {"subtract_": {"from": f"work::{delta_name}", "to": f"work::{name}"}},
                {"delete": {"target": f"work::{delta_name}"}},
            ]
        )
    reverse_transforms.extend(
        [
            {"load": {"path": str(original_path), "alias": "base"}},
            {"diff": {"mode": "aliases", "left_alias": "base", "right_alias": "work"}},
        ]
    )

    pipeline2 = {"inputs": [], "transforms": reverse_transforms}
    provider = create_state_dict_provider(
        provider="inmemory",
        model_paths={},
        max_io_workers=2,
        arena_root=tmp_path / "arena_arith_2",
        arena_segment_size="256MB",
    )
    try:
        should_continue, executed = _run_pipeline(pipeline2, provider)
        assert should_continue is True
        assert len(executed) == len(pipeline2["transforms"])
    finally:
        provider.close()

    output = capsys.readouterr().out
    _assert_no_diff_output(output)

def test_prefix_rename_remove_roundtrip_has_no_diff(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    original_path = tmp_path / "prefix_original.safetensors"
    edited_path = tmp_path / "prefix_edited.safetensors"
    _write_toy_checkpoint(original_path)

    pipeline1 = {
        "inputs": [f"base::{original_path}"],
        "transforms": [
            {"prefixes": {"mode": "add", "alias": "temp"}},
            {"copy": {"from": r"base::(.+)", "to": r"temp::\1"}},
            {"prefixes": {"mode": "rename", "from": "temp", "to": "work"}},
            {"prefixes": {"mode": "add", "alias": "trash"}},
            {"copy": {"from": "work::wte.weight", "to": "trash::tmp.weight"}},
            {"prefixes": {"mode": "remove", "alias": "trash"}},
            {"save": {"path": str(edited_path), "alias": "work", "format": "safetensors"}},
        ],
    }
    provider = create_state_dict_provider(
        provider="inmemory",
        model_paths=compile_plan(pipeline1).inputs,
        max_io_workers=2,
        arena_root=tmp_path / "arena_prefix_1",
        arena_segment_size="256MB",
    )
    try:
        should_continue, executed = _run_pipeline(pipeline1, provider)
        assert should_continue is True
        assert len(executed) == len(pipeline1["transforms"])
    finally:
        provider.close()

    pipeline2 = {
        "inputs": [],
        "transforms": [
            {"load": {"path": str(edited_path), "alias": "work"}},
            {"load": {"path": str(original_path), "alias": "base"}},
            {"diff": {"mode": "aliases", "left_alias": "base", "right_alias": "work"}},
        ],
    }
    provider = create_state_dict_provider(
        provider="arena",
        model_paths={},
        max_io_workers=2,
        arena_root=tmp_path / "arena_prefix_2",
        arena_segment_size="256MB",
    )
    try:
        should_continue, executed = _run_pipeline(pipeline2, provider)
        assert should_continue is True
        assert len(executed) == len(pipeline2["transforms"])
    finally:
        provider.close()

    _assert_no_diff_output(capsys.readouterr().out)

def test_split_concat_permute_reshape_roundtrip_has_no_diff(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    original_path = tmp_path / "shape_original.safetensors"
    edited_path = tmp_path / "shape_edited.safetensors"
    _write_toy_checkpoint(original_path)

    pipeline1 = {
        "inputs": [f"base::{original_path}"],
        "transforms": [
            {"prefixes": {"mode": "add", "alias": "work"}},
            {"copy": {"from": r"base::(.+)", "to": r"work::\1"}},
            {"split": {"from": "work::x.square", "to": ["work::x.left", "work::x.right"], "sizes": [2, 2], "dim": 1}},
            {"concat": {"from": ["work::x.left", "work::x.right"], "to": "work::x.rebuilt", "dim": 1}},
            {"delete": {"target": "work::x.square"}},
            {"move": {"from": "work::x.rebuilt", "to": "work::x.square"}},
            {"permute": {"from": "work::x.square", "to": "work::x.transposed", "order": [1, 0]}},
            {"permute": {"from": "work::x.transposed", "to": "work::x.back", "order": [1, 0]}},
            {"delete": {"target": "work::x.square"}},
            {"move": {"from": "work::x.back", "to": "work::x.square"}},
            {"reshape": {"from": "work::ln_f.weight", "to": "work::ln_f.tmp", "shape": [2, 2]}},
            {"reshape_": {"target": "work::ln_f.tmp", "shape": [4]}},
            {"delete": {"target": "work::ln_f.weight"}},
            {"move": {"from": "work::ln_f.tmp", "to": "work::ln_f.weight"}},
            {"delete": {"target": "work::x.left"}},
            {"delete": {"target": "work::x.right"}},
            {"delete": {"target": "work::x.transposed"}},
            {"save": {"path": str(edited_path), "alias": "work", "format": "safetensors"}},
        ],
    }
    provider = create_state_dict_provider(
        provider="inmemory",
        model_paths=compile_plan(pipeline1).inputs,
        max_io_workers=2,
        arena_root=tmp_path / "arena_shape_1",
        arena_segment_size="256MB",
    )
    try:
        should_continue, executed = _run_pipeline(pipeline1, provider)
        assert should_continue is True
        assert len(executed) == len(pipeline1["transforms"])
    finally:
        provider.close()

    pipeline2 = {
        "inputs": [],
        "transforms": [
            {"load": {"path": str(edited_path), "alias": "work"}},
            {"load": {"path": str(original_path), "alias": "base"}},
            {"diff": {"mode": "aliases", "left_alias": "base", "right_alias": "work"}},
        ],
    }
    provider = create_state_dict_provider(
        provider="inmemory",
        model_paths={},
        max_io_workers=2,
        arena_root=tmp_path / "arena_shape_2",
        arena_segment_size="256MB",
    )
    try:
        should_continue, executed = _run_pipeline(pipeline2, provider)
        assert should_continue is True
        assert len(executed) == len(pipeline2["transforms"])
    finally:
        provider.close()

    _assert_no_diff_output(capsys.readouterr().out)

def test_cast_and_dtype_assert_roundtrip_has_no_diff(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    original_path = tmp_path / "cast_original.safetensors"
    edited_path = tmp_path / "cast_edited.safetensors"
    _write_toy_checkpoint(original_path)

    pipeline1 = {
        "inputs": [f"base::{original_path}"],
        "transforms": [
            {"prefixes": {"mode": "add", "alias": "work"}},
            {"copy": {"from": r"base::(.+)", "to": r"work::\1"}},
            {"cast_": {"target": "work::wte.weight", "to": "float16"}},
            {"cast_": {"target": "work::ln_f.weight", "to": "float16"}},
            {"assert": {"dtype": {"of": "work::wte.weight", "is": "float16"}}},
            {"assert": {"dtype": {"of": "work::ln_f.weight", "is": "float16"}}},
            {"cast_": {"target": "work::wte.weight", "to": "float32"}},
            {"cast_": {"target": "work::ln_f.weight", "to": "float32"}},
            {"assert": {"dtype": {"of": "work::wte.weight", "is": "float32"}}},
            {"assert": {"dtype": {"of": "work::ln_f.weight", "is": "float32"}}},
            {"save": {"path": str(edited_path), "alias": "work", "format": "safetensors"}},
        ],
    }
    provider = create_state_dict_provider(
        provider="inmemory",
        model_paths=compile_plan(pipeline1).inputs,
        max_io_workers=2,
        arena_root=tmp_path / "arena_cast_1",
        arena_segment_size="256MB",
    )
    try:
        should_continue, executed = _run_pipeline(pipeline1, provider)
        assert should_continue is True
        assert len(executed) == len(pipeline1["transforms"])
    finally:
        provider.close()

    pipeline2 = {
        "inputs": [],
        "transforms": [
            {"load": {"path": str(edited_path), "alias": "work"}},
            {"load": {"path": str(original_path), "alias": "base"}},
            {"diff": {"mode": "aliases", "left_alias": "base", "right_alias": "work"}},
        ],
    }
    provider = create_state_dict_provider(
        provider="arena",
        model_paths={},
        max_io_workers=2,
        arena_root=tmp_path / "arena_cast_2",
        arena_segment_size="256MB",
    )
    try:
        should_continue, executed = _run_pipeline(pipeline2, provider)
        assert should_continue is True
        assert len(executed) == len(pipeline2["transforms"])
    finally:
        provider.close()

    _assert_no_diff_output(capsys.readouterr().out)

def test_matmul_multiply_sidepath_roundtrip_has_no_diff(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    original_path = tmp_path / "mat_original.safetensors"
    edited_path = tmp_path / "mat_edited.safetensors"
    _write_toy_checkpoint(original_path)

    pipeline1 = {
        "inputs": [f"base::{original_path}"],
        "transforms": [
            {"prefixes": {"mode": "add", "alias": "work"}},
            {"copy": {"from": r"base::(.+)", "to": r"work::\1"}},
            {"matmul": {"from_a": "work::mat.left", "from_b": "work::mat.right", "to": "work::mat.tmp"}},
            {"add_": {"from": "work::mat.tmp", "to": "work::mat.target"}},
            {"subtract_": {"from": "work::mat.tmp", "to": "work::mat.target"}},
            {"multiply": {"from_a": "work::mat.target", "from_b": "work::ones.2x2", "to": "work::mat.target"}},
            {"delete": {"target": "work::mat.tmp"}},
            {"save": {"path": str(edited_path), "alias": "work", "format": "safetensors"}},
        ],
    }
    provider = create_state_dict_provider(
        provider="inmemory",
        model_paths=compile_plan(pipeline1).inputs,
        max_io_workers=2,
        arena_root=tmp_path / "arena_mat_1",
        arena_segment_size="256MB",
    )
    try:
        should_continue, executed = _run_pipeline(pipeline1, provider)
        assert should_continue is True
        assert len(executed) == len(pipeline1["transforms"])
    finally:
        provider.close()

    pipeline2 = {
        "inputs": [],
        "transforms": [
            {"load": {"path": str(edited_path), "alias": "work"}},
            {"load": {"path": str(original_path), "alias": "base"}},
            {"diff": {"mode": "aliases", "left_alias": "base", "right_alias": "work"}},
        ],
    }
    provider = create_state_dict_provider(
        provider="inmemory",
        model_paths={},
        max_io_workers=2,
        arena_root=tmp_path / "arena_mat_2",
        arena_segment_size="256MB",
    )
    try:
        should_continue, executed = _run_pipeline(pipeline2, provider)
        assert should_continue is True
        assert len(executed) == len(pipeline2["transforms"])
    finally:
        provider.close()

    _assert_no_diff_output(capsys.readouterr().out)

def test_assign_slice_restore_roundtrip_has_no_diff(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    original_path = tmp_path / "assign_original.safetensors"
    edited_path = tmp_path / "assign_edited.safetensors"
    _write_toy_checkpoint(original_path)

    pipeline1 = {
        "inputs": [f"base::{original_path}"],
        "transforms": [
            {"prefixes": {"mode": "add", "alias": "work"}},
            {"copy": {"from": r"base::(.+)", "to": r"work::\1"}},
            {"copy": {"from": "work::x.square", "to": "work::x.backup"}},
            {"fill_": {"target": "work::x.square::[:, :2]", "mode": "constant", "value": 0.0}},
            {"assign": {"from": "work::x.backup::[:, :2]", "to": "work::x.square::[:, :2]"}},
            {"delete": {"target": "work::x.backup"}},
            {"save": {"path": str(edited_path), "alias": "work", "format": "safetensors"}},
        ],
    }
    provider = create_state_dict_provider(
        provider="inmemory",
        model_paths=compile_plan(pipeline1).inputs,
        max_io_workers=2,
        arena_root=tmp_path / "arena_assign_1",
        arena_segment_size="256MB",
    )
    try:
        should_continue, executed = _run_pipeline(pipeline1, provider)
        assert should_continue is True
        assert len(executed) == len(pipeline1["transforms"])
    finally:
        provider.close()

    pipeline2 = {
        "inputs": [],
        "transforms": [
            {"load": {"path": str(edited_path), "alias": "work"}},
            {"load": {"path": str(original_path), "alias": "base"}},
            {"diff": {"mode": "aliases", "left_alias": "base", "right_alias": "work"}},
        ],
    }
    provider = create_state_dict_provider(
        provider="inmemory",
        model_paths={},
        max_io_workers=2,
        arena_root=tmp_path / "arena_assign_2",
        arena_segment_size="256MB",
    )
    try:
        should_continue, executed = _run_pipeline(pipeline2, provider)
        assert should_continue is True
        assert len(executed) == len(pipeline2["transforms"])
    finally:
        provider.close()

    _assert_no_diff_output(capsys.readouterr().out)

def test_tensor_save_load_reinject_roundtrip_has_no_diff(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    original_path = tmp_path / "tensor_io_original.safetensors"
    tensor_path = tmp_path / "wte.npy"
    edited_path = tmp_path / "tensor_io_edited.safetensors"
    _write_toy_checkpoint(original_path)

    pipeline1 = {
        "inputs": [f"base::{original_path}"],
        "transforms": [
            {"save": {"path": str(tensor_path), "target": "base::wte.weight", "format": "numpy"}},
            {"prefixes": {"mode": "add", "alias": "work"}},
            {"copy": {"from": r"base::(.+)", "to": r"work::\1"}},
            {"fill_": {"target": "work::wte.weight", "mode": "constant", "value": 0.0}},
            {"load": {"path": str(tensor_path), "to": "work::wte.restored", "format": "numpy"}},
            {"assign": {"from": "work::wte.restored", "to": "work::wte.weight"}},
            {"delete": {"target": "work::wte.restored"}},
            {"save": {"path": str(edited_path), "alias": "work", "format": "safetensors"}},
        ],
    }
    provider = create_state_dict_provider(
        provider="inmemory",
        model_paths=compile_plan(pipeline1).inputs,
        max_io_workers=2,
        arena_root=tmp_path / "arena_tensorio_1",
        arena_segment_size="256MB",
    )
    try:
        should_continue, executed = _run_pipeline(pipeline1, provider)
        assert should_continue is True
        assert len(executed) == len(pipeline1["transforms"])
    finally:
        provider.close()

    pipeline2 = {
        "inputs": [],
        "transforms": [
            {"load": {"path": str(edited_path), "alias": "work"}},
            {"load": {"path": str(original_path), "alias": "base"}},
            {"diff": {"mode": "aliases", "left_alias": "base", "right_alias": "work"}},
        ],
    }
    provider = create_state_dict_provider(
        provider="inmemory",
        model_paths={},
        max_io_workers=2,
        arena_root=tmp_path / "arena_tensorio_2",
        arena_segment_size="256MB",
    )
    try:
        should_continue, executed = _run_pipeline(pipeline2, provider)
        assert should_continue is True
        assert len(executed) == len(pipeline2["transforms"])
    finally:
        provider.close()

    _assert_no_diff_output(capsys.readouterr().out)

def test_expression_gated_pipeline_roundtrip_has_no_diff(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    original_path = tmp_path / "expr_original.safetensors"
    edited_path = tmp_path / "expr_edited.safetensors"
    _write_toy_checkpoint(original_path)

    pipeline1 = {
        "inputs": [f"base::{original_path}"],
        "transforms": [
            {"prefixes": {"mode": "add", "alias": "work"}},
            {"copy": {"from": r"base::(.+)", "to": r"work::\1"}},
            {"assert": {"all": [
                {"exists": "work::wte.weight"},
                {"dtype": {"of": "work::ln_f.weight", "is": "float32"}},
                {"shape": {"of": "work::h.0.attn.c_proj.weight", "is": [4, 4]}},
                {"dimensions": {"of": "work::x.square", "is": 2}},
            ]}},
            {"assert": {"any": [{"exists": "work::does_not_exist"}, {"exists": "work::wte.weight"}]}},
            {"assert": {"not": {"exists": "work::does_not_exist"}}},
            {"assert": {"count": {"of": "work::h\\.0\\..*", "is": 4}}},
            {"fill": {"from": "work::ln_f.weight", "to": "work::delta.ln_f", "mode": "constant", "value": 1.0}},
            {"add_": {"from": "work::delta.ln_f", "to": "work::ln_f.weight"}},
            {"subtract_": {"from": "work::delta.ln_f", "to": "work::ln_f.weight"}},
            {"delete": {"target": "work::delta.ln_f"}},
            {"save": {"path": str(edited_path), "alias": "work", "format": "safetensors"}},
        ],
    }
    provider = create_state_dict_provider(
        provider="inmemory",
        model_paths=compile_plan(pipeline1).inputs,
        max_io_workers=2,
        arena_root=tmp_path / "arena_expr_1",
        arena_segment_size="256MB",
    )
    try:
        should_continue, executed = _run_pipeline(pipeline1, provider)
        assert should_continue is True
        assert len(executed) == len(pipeline1["transforms"])
    finally:
        provider.close()

    pipeline2 = {
        "inputs": [],
        "transforms": [
            {"load": {"path": str(edited_path), "alias": "work"}},
            {"load": {"path": str(original_path), "alias": "base"}},
            {"diff": {"mode": "aliases", "left_alias": "base", "right_alias": "work"}},
        ],
    }
    provider = create_state_dict_provider(
        provider="inmemory",
        model_paths={},
        max_io_workers=2,
        arena_root=tmp_path / "arena_expr_2",
        arena_segment_size="256MB",
    )
    try:
        should_continue, executed = _run_pipeline(pipeline2, provider)
        assert should_continue is True
        assert len(executed) == len(pipeline2["transforms"])
    finally:
        provider.close()

    _assert_no_diff_output(capsys.readouterr().out)

def test_sharded_save_load_provider_mix_roundtrip_has_no_diff(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    original_path = tmp_path / "shardmix_original.safetensors"
    sharded_dir = tmp_path / "shardmix_out"
    _write_toy_checkpoint(original_path)

    pipeline1 = {
        "inputs": [f"base::{original_path}"],
        "transforms": [
            {"save": {"path": str(sharded_dir), "alias": "base", "format": "safetensors", "shard": "1KB"}},
        ],
    }
    provider = create_state_dict_provider(
        provider="arena",
        model_paths=compile_plan(pipeline1).inputs,
        max_io_workers=2,
        arena_root=tmp_path / "arena_shardmix_1",
        arena_segment_size="256MB",
    )
    try:
        should_continue, executed = _run_pipeline(pipeline1, provider)
        assert should_continue is True
        assert len(executed) == len(pipeline1["transforms"])
    finally:
        provider.close()

    assert (sharded_dir / "model.safetensors.index.json").exists()

    pipeline2 = {
        "inputs": [],
        "transforms": [
            {"load": {"path": str(sharded_dir), "alias": "work"}},
            {"load": {"path": str(original_path), "alias": "base"}},
            {"diff": {"mode": "aliases", "left_alias": "base", "right_alias": "work"}},
        ],
    }
    provider = create_state_dict_provider(
        provider="inmemory",
        model_paths={},
        max_io_workers=2,
        arena_root=tmp_path / "arena_shardmix_2",
        arena_segment_size="256MB",
    )
    try:
        should_continue, executed = _run_pipeline(pipeline2, provider)
        assert should_continue is True
        assert len(executed) == len(pipeline2["transforms"])
    finally:
        provider.close()

    _assert_no_diff_output(capsys.readouterr().out)
