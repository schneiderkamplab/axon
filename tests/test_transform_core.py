from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

import brainsurgery.transform as transform_module
from brainsurgery.plan import SurgeryPlan
from brainsurgery.providers import InMemoryStateDict
from brainsurgery.transform import (
    BaseTransform,
    CompiledTransform,
    TransformError,
    TransformResult,
    ensure_mapping_payload,
    get_transform,
    infer_output_model,
    list_transforms,
    register_transform,
    require_expr,
    require_nonempty_string,
    require_numeric,
    validate_payload_keys,
)
from brainsurgery.transforms.copy import CopyTransform
from brainsurgery.transforms.help import HelpTransform
from brainsurgery.transforms.save import SaveTransform


@dataclass(frozen=True)
class _Spec:
    model: str

    def collect_models(self) -> set[str]:
        return {self.model, "other"}


class _Transform(BaseTransform):
    name = "dummy"

    def compile(self, payload: dict, default_model: str | None) -> object:
        del payload, default_model
        return _Spec(model="model")

    def apply(self, spec: object, provider: object) -> TransformResult:
        del spec, provider
        return TransformResult(name=self.name, count=1)

    def infer_output_model(self, spec: object) -> str:
        assert isinstance(spec, _Spec)
        return spec.model


class _FallbackTransform(_Transform):
    name = "fallback"

    def infer_output_model(self, spec: object) -> str:
        del spec
        raise TransformError("needs provider")


class _Provider:
    def __init__(self) -> None:
        self.state_dicts = {
            "model": InMemoryStateDict(),
            "other": InMemoryStateDict(),
        }
        self.state_dicts["model"]["w"] = torch.ones(1)

    def get_state_dict(self, model: str) -> InMemoryStateDict:
        return self.state_dicts[model]


def test_transform_registry_registers_lists_and_rejects_duplicates(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(transform_module, "_REGISTRY", {})
    transform = _Transform()
    register_transform(transform)

    assert get_transform("dummy") is transform
    assert list_transforms() == ["dummy"]

    with pytest.raises(TransformError, match="already registered"):
        register_transform(_Transform())


def test_infer_output_model_uses_provider_fallback_when_needed() -> None:
    plan = SurgeryPlan(
        inputs={},
        output=None,
        transforms=[CompiledTransform(_FallbackTransform(), _Spec("model"))],
    )
    assert infer_output_model(plan, _Provider()) == "model"


def test_infer_output_model_skips_non_contributing_transforms() -> None:
    plan = SurgeryPlan(
        inputs={},
        output=None,
        transforms=[
            CompiledTransform(HelpTransform(), HelpTransform().compile("copy", default_model=None)),
            CompiledTransform(
                SaveTransform(),
                SaveTransform().compile({"path": "/tmp/out.safetensors", "alias": "model"}, default_model=None),
            ),
            CompiledTransform(
                CopyTransform(),
                CopyTransform().compile({"from": "model::w", "to": "model::w_copy"}, default_model=None),
            ),
        ],
    )

    assert infer_output_model(plan, _Provider()) == "model"


def test_validate_payload_helpers_cover_required_unknown_and_type_errors() -> None:
    validate_payload_keys({"from": "a"}, op_name="copy", allowed_keys={"from"}, required_keys={"from"})
    assert ensure_mapping_payload({"x": 1}, "copy") == {"x": 1}
    assert require_nonempty_string({"alias": "base"}, op_name="load", key="alias") == "base"
    assert require_expr({"from": ["a", "b"]}, op_name="copy", key="from") == ["a", "b"]
    assert require_numeric({"factor": "1.5"}, op_name="scale", key="factor") == 1.5

    with pytest.raises(TransformError, match="unknown keys"):
        validate_payload_keys({"extra": 1}, op_name="copy", allowed_keys={"from"})

    with pytest.raises(TransformError, match="copy.to is required"):
        validate_payload_keys(
            {"from": "a"},
            op_name="copy",
            allowed_keys={"from", "to"},
            required_keys={"from", "to"},
        )

    with pytest.raises(TransformError, match="must be a mapping"):
        ensure_mapping_payload([], "copy")

    with pytest.raises(TransformError, match="must be a non-empty string"):
        require_nonempty_string({"alias": ""}, op_name="load", key="alias")

    with pytest.raises(TransformError, match="non-empty string or a non-empty list"):
        require_expr({"from": []}, op_name="copy", key="from")

    with pytest.raises(TransformError, match="must be numeric"):
        require_numeric({"factor": "nope"}, op_name="scale", key="factor")
