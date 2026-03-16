from __future__ import annotations

from pathlib import Path

import torch

from brainsurgery.transforms.execute import ExecuteTransform, ExecuteTransformError


class _Provider:
    def __init__(self) -> None:
        self._models = {"model": {"a": torch.tensor([1.0], dtype=torch.float32)}}

    def get_state_dict(self, model: str):
        return self._models[model]

    def list_model_aliases(self) -> list[str]:
        return sorted(self._models)


def test_execute_transform_applies_inline_transform_batch() -> None:
    transform = ExecuteTransform()
    provider = _Provider()

    spec = transform.compile(
        {
            "transforms": [
                {"copy": {"from": "a", "to": "b"}},
                {"delete": {"target": "b"}},
            ]
        },
        default_model=None,
    )
    result = transform.apply(spec, provider)

    assert result.name == "execute"
    assert result.count == 2
    assert "a" in provider.get_state_dict("model")
    assert "b" not in provider.get_state_dict("model")


def test_execute_transform_supports_plan_yaml_with_inputs(tmp_path: Path) -> None:
    plan_path = tmp_path / "batch.yaml"
    plan_path.write_text(
        "inputs:\n  - model::/tmp/example.safetensors\ntransforms:\n  - help: {}\n",
        encoding="utf-8",
    )

    transform = ExecuteTransform()
    spec = transform.compile({"path": str(plan_path)}, default_model=None)

    first = spec.raw_transforms[0]
    second = spec.raw_transforms[1]
    assert spec.default_model_hint == "model"
    assert first == {"load": {"path": "/tmp/example.safetensors", "alias": "model"}}
    assert second == {"help": {}}


def test_execute_transform_converts_plan_output_to_save_transform(tmp_path: Path) -> None:
    plan_path = tmp_path / "batch_with_output.yaml"
    plan_path.write_text(
        "inputs:\n"
        "  - model::/tmp/example.safetensors\n"
        "transforms:\n"
        "  - help: {}\n"
        "output:\n"
        "  path: model::/tmp/out.safetensors\n"
        "  format: safetensors\n"
        "  shard: 100MB\n",
        encoding="utf-8",
    )

    transform = ExecuteTransform()
    spec = transform.compile({"path": str(plan_path)}, default_model=None)

    assert spec.raw_transforms[-1] == {
        "save": {
            "path": "/tmp/out.safetensors",
            "alias": "model",
            "format": "safetensors",
            "shard": "100MB",
        }
    }


def test_execute_transform_converts_string_output_to_save_transform() -> None:
    transform = ExecuteTransform()
    spec = transform.compile(
        {
            "plan": {
                "transforms": [{"help": {}}],
                "output": "/tmp/out.safetensors",
            }
        },
        default_model=None,
    )
    assert spec.raw_transforms[-1] == {"save": {"path": "/tmp/out.safetensors"}}


def test_execute_transform_rejects_empty_sources() -> None:
    transform = ExecuteTransform()
    try:
        transform.compile({}, default_model=None)
    except ExecuteTransformError as exc:
        assert "at least one transform source" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ExecuteTransformError")
