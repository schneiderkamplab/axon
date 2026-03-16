from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import brainsurgery.engine.config as config_module
import brainsurgery.engine.execution as execution_module
import brainsurgery.engine.summary as summary_module
import brainsurgery.transforms.help as help_module
import brainsurgery.web.http.server as http_server_module
import brainsurgery.web.ui.backend as webui_backend_module
from brainsurgery.cli.complete import _match_payload_candidates
from brainsurgery.core import CompiledTransform, TensorRef, TransformError
from brainsurgery.core.specs.validation import require_numeric
from brainsurgery.engine.plan import PlanStep, SurgeryPlan, _OutputSpec
from brainsurgery.engine.provider_utils import (
    _has_model_alias,
    get_or_create_alias_state_dict,
    list_loaded_tensor_names,
    list_model_aliases,
)
from brainsurgery.engine.providers import BaseStateDictProvider
from brainsurgery.transforms.execute import (
    ExecuteSpec,
    ExecuteTransform,
    ExecuteTransformError,
    _extract_plan_transforms,
    _inputs_to_load_transforms,
    _output_to_save_transform,
)


def test_serve_http_handles_interrupt_and_close_callback(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []

    class _FakeLogger:
        def info(self, msg: str, *args: object) -> None:
            events.append(msg % args if args else msg)

    class _FakeServer:
        def __init__(self, _addr, _handler) -> None:
            pass

        def serve_forever(self) -> None:
            raise KeyboardInterrupt

        def server_close(self) -> None:
            events.append("closed")

    monkeypatch.setattr(http_server_module, "ThreadingHTTPServer", _FakeServer)
    http_server_module.serve_http(
        host="127.0.0.1",
        port=8123,
        handler_factory=lambda: object,  # type: ignore[return-value]
        startup_message="up %s:%d",
        shutdown_message="down",
        logger=_FakeLogger(),  # type: ignore[arg-type]
        on_close=lambda: events.append("on_close"),
    )
    assert events == ["up 127.0.0.1:8123", "down", "closed", "on_close"]


def test_serve_http_uses_default_logger_when_none(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []

    class _FakeServer:
        def __init__(self, _addr, _handler) -> None:
            pass

        def serve_forever(self) -> None:
            events.append("served")

        def server_close(self) -> None:
            events.append("closed")

    monkeypatch.setattr(http_server_module, "ThreadingHTTPServer", _FakeServer)
    http_server_module.serve_http(
        host="127.0.0.1",
        port=8124,
        handler_factory=lambda: object,  # type: ignore[return-value]
        startup_message="up %s:%d",
        shutdown_message="down",
        logger=None,
    )
    assert events == ["served", "closed"]


def test_require_numeric_rejects_none() -> None:
    with pytest.raises(TransformError, match="x.y must be numeric"):
        require_numeric({"y": None}, op_name="x", key="y")


def test_engine_config_normalization_error_branches() -> None:
    assert config_module._normalize_single_transform_spec({"help": None}) == {"help": {}}
    assert config_module._normalize_single_transform_spec("help") == {"help": {}}

    with pytest.raises(ValueError, match="exactly one key"):
        config_module._normalize_single_transform_spec({"a": {}, "b": {}})
    with pytest.raises(ValueError, match="non-empty string"):
        config_module._normalize_single_transform_spec(" ")
    with pytest.raises(ValueError, match="either a YAML mapping or a bare transform name"):
        config_module._normalize_single_transform_spec(123)
    with pytest.raises(ValueError, match="plan must be a mapping"):
        config_module.normalize_raw_plan([])


def test_plan_helpers_cover_remaining_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    plan = SurgeryPlan(inputs={}, output=None)
    plan.append_raw_transforms([{"help": {}}])
    step = plan.record_executed_raw({"exit": {}})
    assert step.status == "done"

    plan.raw_inputs = ["model::/tmp/in.safetensors"]
    plan.raw_output = {"path": "/tmp/out.safetensors"}
    raw_all = plan.to_raw_plan(executed_only=False)
    assert raw_all["output"] == {"path": "/tmp/out.safetensors"}
    raw_done = plan.to_raw_plan(executed_only=True)
    assert raw_done["transforms"] == [{"exit": {}}]

    pending_plan = SurgeryPlan(
        inputs={},
        output=None,
        steps=[
            PlanStep(
                raw={"help": {}}, compiled=CompiledTransform(SimpleNamespace(name="help"), object())
            )
        ],  # type: ignore[arg-type]
    )

    monkeypatch.setattr(
        "brainsurgery.engine.plan._execute_transform_pairs",
        lambda _pairs, _provider, interactive: (True, []),
    )
    should_continue = pending_plan.execute_pending(object(), interactive=True)
    assert should_continue is True
    assert pending_plan.steps[0].status == "failed"
    assert pending_plan.steps[0].error is not None


class _ProviderForUtils(BaseStateDictProvider):
    def __init__(self) -> None:
        super().__init__({}, max_io_workers=1)

    def get_state_dict(self, model: str):  # type: ignore[override]
        return self.state_dicts[model]

    def create_state_dict(self):  # type: ignore[override]
        return {}


def test_provider_utils_base_provider_fallback_branches() -> None:
    provider = _ProviderForUtils()
    provider.state_dicts["a"] = {}

    provider.list_model_aliases = None  # type: ignore[method-assign]
    assert list_model_aliases(provider) == set()

    provider.list_model_aliases = lambda: {"a"}  # type: ignore[method-assign]
    provider.has_model_alias = None  # type: ignore[method-assign]
    assert _has_model_alias(provider, "a") is True

    provider.get_or_create_alias_state_dict = None  # type: ignore[method-assign]
    assert (
        get_or_create_alias_state_dict(
            provider,
            "a",
            error_type=TransformError,
            op_name="prefixes",
        )
        == {}
    )

    provider.state_dicts = []  # type: ignore[assignment]
    assert list_loaded_tensor_names(provider) == {}


def test_summary_private_serializers_cover_expression_branches() -> None:
    class _Mode(Enum):
        ONE = "one"

    @dataclass(frozen=True)
    class _DataclassExpr:
        value: int

    assert summary_module._normalize_summary_node(None) == {}
    assert (
        summary_module._serialize_tensor_ref(TensorRef(model="m", expr="a", slice_spec="[:1]"))
        == "m::a::[:1]"
    )
    assert summary_module._serialize_tensor_ref(TensorRef(model=None, expr="a")) == "a"
    assert summary_module._serialize_tensor_ref(TensorRef(model="m", expr=["a", "b"])) == ["a", "b"]
    assert summary_module._serialize_tensor_ref(TensorRef(model=None, expr=7)) == "7"
    assert summary_module._serialize_scalar(Path("/tmp/x")) == "/tmp/x"
    assert summary_module._serialize_scalar(TensorRef(model="m", expr="x")) == "m::x"
    assert summary_module._serialize_scalar(_Mode.ONE) == "one"
    assert summary_module._serialize_scalar((1, 2)) == [1, 2]
    assert summary_module._serialize_scalar({"k": 1}) == {"k": 1}
    assert summary_module._serialize_scalar(torch.float32) == "float32"
    assert summary_module._serialize_scalar(object()).startswith("<object object")
    assert summary_module._serialize_scalar(_DataclassExpr(3)) == {"value": 3}

    cmp = SimpleNamespace(exact=1, ge=2, gt=3, le=4, lt=5)
    assert summary_module._serialize_scalar_comparison(cmp) == {
        "is": 1,
        "ge": 2,
        "gt": 3,
        "le": 4,
        "lt": 5,
    }

    exists_expr = type("ExistsExpr", (), {"ref": TensorRef(model="m", expr="w")})()
    not_expr = type("NotExpr", (), {"expr": exists_expr})()
    all_expr = type("AllExpr", (), {"exprs": [not_expr]})()
    any_expr = type("AnyExpr", (), {"exprs": [exists_expr]})()
    assert "all" in summary_module._serialize_assert_expr(all_expr)
    assert "any" in summary_module._serialize_assert_expr(any_expr)

    equal_expr = type(
        "EqualExpr",
        (),
        {
            "left": TensorRef(model="m", expr="a"),
            "right": TensorRef(model="m", expr="b"),
            "eps": 0.1,
        },
    )()
    assert summary_module._serialize_assert_expr(equal_expr)["equal"]["eps"] == 0.1

    iszero_expr = type("IsZeroExpr", (), {"ref": TensorRef(model="m", expr="a"), "eps": None})()
    assert summary_module._serialize_assert_expr(iszero_expr) == {"iszero": "m::a"}
    iszero_eps_expr = type("IsZeroExpr", (), {"ref": TensorRef(model="m", expr="a"), "eps": 1e-3})()
    assert summary_module._serialize_assert_expr(iszero_eps_expr) == {
        "iszero": {"of": "m::a", "eps": 1e-3}
    }

    dtype_expr = type(
        "DtypeExpr",
        (),
        {"ref": TensorRef(model="m", expr="a"), "is_value": "float32"},
    )()
    shape_expr = type(
        "ShapeExpr",
        (),
        {"ref": TensorRef(model="m", expr="a"), "is_value": [1]},
    )()
    count_expr = type(
        "CountExpr",
        (),
        {"ref": TensorRef(model="m", expr="a"), "is_value": 3},
    )()
    dimensions_expr = type(
        "DimensionsExpr",
        (),
        {"ref": TensorRef(model="m", expr="a"), "comparison": cmp},
    )()
    access_expr = type(
        "TensorAccessExpr",
        (),
        {"field": "reads", "ref": TensorRef(model="m", expr="a"), "comparison": cmp},
    )()
    assert "dtype" in summary_module._serialize_assert_expr(dtype_expr)
    assert "shape" in summary_module._serialize_assert_expr(shape_expr)
    assert "count" in summary_module._serialize_assert_expr(count_expr)
    assert "dimensions" in summary_module._serialize_assert_expr(dimensions_expr)
    assert "reads" in summary_module._serialize_assert_expr(access_expr)
    assert summary_module._serialize_assert_expr(_DataclassExpr(2)) == {"_dataclass": {"value": 2}}
    assert summary_module._serialize_assert_expr("x") == {"expr": "x"}


def test_summary_fill_and_resolved_output_paths() -> None:
    constant_cfg = SimpleNamespace(mode="constant", seed=1, constant_value=2.0)
    tensor_cfg = SimpleNamespace(mode="tensor", seed=None, values_payload=[1, 2])
    normal_cfg = SimpleNamespace(mode="random", seed=2, distribution="normal", mean=0.0, std=1.0)
    uniform_cfg = SimpleNamespace(mode="random", seed=None, distribution="uniform", low=0, high=1)
    assert summary_module._serialize_fill_config(constant_cfg)["value"] == 2.0
    assert summary_module._serialize_fill_config(tensor_cfg)["values"] == [1, 2]
    assert "mean" in summary_module._serialize_fill_config(normal_cfg)
    assert "low" in summary_module._serialize_fill_config(uniform_cfg)

    fill_spec = SimpleNamespace(
        config=constant_cfg,
        from_ref=TensorRef(model="m", expr="a"),
        to_ref=TensorRef(model="m", expr="b"),
    )
    fillu_spec = SimpleNamespace(config=constant_cfg, target_ref=TensorRef(model="m", expr="c"))
    assert "from" in summary_module._serialize_fill_spec("fill", fill_spec)["fill"]
    assert "target" in summary_module._serialize_fill_spec("fill_", fillu_spec)["fill_"]

    plan = SurgeryPlan(
        inputs={"model": Path("/tmp/in.safetensors")},
        output=_OutputSpec(path=Path("/tmp/out.safetensors"), format="torch", shard="10MB"),
        steps=[PlanStep(raw={"help": {}}, status="done")],
    )
    resolved = summary_module.executed_plan_summary_doc(plan, mode="resolve")
    assert resolved["output"] == {
        "path": "/tmp/out.safetensors",
        "format": "torch",
        "shard": "10MB",
    }
    text = summary_module.executed_plan_summary_yaml(plan, mode="resolve")
    assert "output:" in text


def test_execute_transform_private_branches(tmp_path: Path) -> None:
    transform = ExecuteTransform()

    with pytest.raises(ExecuteTransformError, match="execute payload must be a mapping"):
        transform.compile("bad", default_model=None)
    with pytest.raises(ExecuteTransformError, match="execute.transforms invalid"):
        transform.compile({"transforms": {"a": {}, "b": {}}}, default_model=None)
    with pytest.raises(ExecuteTransformError, match="execute.plan-yaml must be a string"):
        transform.compile({"plan-yaml": 1}, default_model=None)
    with pytest.raises(ExecuteTransformError, match="execute.path must be a non-empty string"):
        transform.compile({"path": ""}, default_model=None)

    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ExecuteTransformError, match="at least one transform source"):
        transform.compile({"path": str(path)}, default_model=None)

    with pytest.raises(ExecuteTransformError, match="execute cannot recursively invoke execute"):
        transform.apply(ExecuteSpec(raw_transforms=[{"execute": {}}]), object())
    with pytest.raises(ExecuteTransformError, match="execute transform #0"):
        transform.apply(ExecuteSpec(raw_transforms=[{"copy": {}}]), object())
    assert transform.apply(ExecuteSpec(raw_transforms=[]), object()).count == 0

    with pytest.raises(ExecuteTransformError, match="does not infer output model for empty"):
        transform._infer_output_model(ExecuteSpec(raw_transforms=[]))
    with pytest.raises(ExecuteTransformError, match="does not infer an output model"):
        transform._infer_output_model(ExecuteSpec(raw_transforms=[{"help": {}}]))
    assert transform.contributes_output_model(object()) is False

    with pytest.raises(
        ExecuteTransformError, match="execute plan must be a mapping or transform list"
    ):
        _extract_plan_transforms(1)
    with pytest.raises(ExecuteTransformError, match="execute plan list invalid"):
        _extract_plan_transforms([{}])
    with pytest.raises(ExecuteTransformError, match="execute plan invalid"):
        _extract_plan_transforms({"transforms": {"a": {}, "b": {}}})
    with pytest.raises(ExecuteTransformError, match="execute plan.inputs must be a list"):
        _inputs_to_load_transforms("x")
    assert _inputs_to_load_transforms([]) == ([], None)
    with pytest.raises(ExecuteTransformError, match="entries must be non-empty strings"):
        _inputs_to_load_transforms([1])  # type: ignore[list-item]
    with pytest.raises(ExecuteTransformError, match="must not be empty"):
        _inputs_to_load_transforms(["a::"])
    with pytest.raises(ExecuteTransformError, match="requires explicit alias::path"):
        _inputs_to_load_transforms(["/a", "/b"])
    assert _inputs_to_load_transforms(["::/tmp/a.safetensors"]) == (
        [{"load": {"path": "/tmp/a.safetensors", "alias": "model"}}],
        "model",
    )
    assert _inputs_to_load_transforms(["/tmp/single.safetensors"]) == (
        [{"load": {"path": "/tmp/single.safetensors", "alias": "model"}}],
        "model",
    )
    assert ExecuteSpec(raw_transforms=[{"help": {}}]).collect_models() == set()

    with pytest.raises(ExecuteTransformError, match=r"execute\.plan-yaml must be a string"):
        transform.compile({"plan-yaml": 1}, default_model=None)
    plan_yaml_spec = transform.compile(
        {"plan-yaml": "inputs:\n  - model::/tmp/x.safetensors\ntransforms:\n  - help: {}\n"},
        default_model=None,
    )
    assert plan_yaml_spec.default_model_hint == "model"
    plan_spec = transform.compile(
        {
            "transforms": [{"help": {}}],
            "plan": {"inputs": ["model::/tmp/y.safetensors"], "transforms": [{"help": {}}]},
        },
        default_model=None,
    )
    assert plan_spec.default_model_hint == "model"
    assert _output_to_save_transform(None) is None
    assert _output_to_save_transform("") is None
    assert _output_to_save_transform("m::/tmp/out.safetensors") == {
        "save": {"path": "/tmp/out.safetensors", "alias": "m"}
    }
    with pytest.raises(ExecuteTransformError, match="plan.output.path must be a non-empty string"):
        _output_to_save_transform({"path": ""})
    with pytest.raises(
        ExecuteTransformError, match="plan.output must be either a string or a mapping"
    ):
        _output_to_save_transform(1)


def test_execution_helper_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    lines: list[str] = []
    monkeypatch.setattr(execution_module, "emit_line", lines.append)

    answers = iter(["maybe", "go"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))
    assert execution_module._confirm_preview_apply(
        transform_index=1, total=1, transform_name="copy"
    )
    assert "Please answer 'go' or 'no-go'." in lines[-1]

    monkeypatch.setattr("builtins.input", lambda _prompt: (_ for _ in ()).throw(EOFError))
    assert (
        execution_module._confirm_preview_apply(transform_index=1, total=1, transform_name="copy")
        is False
    )

    impact = execution_module._PreviewImpact(changed=set(), created=set(), deleted=set())
    assert execution_module._should_skip_apply_for_preview("help", impact) is False
    assert execution_module._should_skip_apply_for_preview("load", impact) is True
    assert execution_module.preview_requires_confirmation("load", impact) is True
    assert execution_module._format_preview_impact(impact) == "no tensor impact"
    assert execution_module._format_concrete_ref("m", "x", "[:1]") == "m::x::[:1]"

    class _Provider:
        def __init__(self) -> None:
            self._sds = {"m": {"x": 1}}

        def get_state_dict(self, model: str) -> dict[str, int]:
            return self._sds[model]

    provider = _Provider()
    changed: set[str] = set()
    created: set[str] = set()
    execution_module._classify_ref_write(
        ref=TensorRef(model=None, expr="x"),
        provider=provider,
        op_name="t",
        changed=changed,
        created=created,
    )
    assert changed == set() and created == set()
    execution_module._classify_ref_write(
        ref=TensorRef(model="m", expr="x"),
        provider=provider,
        op_name="t",
        changed=changed,
        created=created,
    )
    execution_module._classify_ref_write(
        ref=TensorRef(model="m", expr="y"),
        provider=provider,
        op_name="t",
        changed=changed,
        created=created,
    )
    assert "m::x" in changed
    assert "m::y" in created

    assert (
        execution_module._resolve_ref_names(
            ref=TensorRef(model=None, expr="x"),
            provider=provider,
            op_name="t",
            role="target",
        )
        == []
    )

    changed2: set[str] = set()
    created2: set[str] = set()
    execution_module._classify_ref_write(
        ref=TensorRef(model="m", expr=".*z"),
        provider=provider,
        op_name="t",
        changed=changed2,
        created=created2,
    )
    assert "m::.*z" in created2

    class _Map:
        def __init__(self, src_model: str, src_name: str, dst_model: str, dst_name: str) -> None:
            self.src_model = src_model
            self.src_name = src_name
            self.dst_model = dst_model
            self.dst_name = dst_name

    class _Provider2:
        def __init__(self) -> None:
            self._sds = {"m": {"x": 1, "fa": 1}}

        def get_state_dict(self, model: str) -> dict[str, int]:
            return self._sds[model]

    spec = SimpleNamespace(
        from_ref=None,
        to_ref=TensorRef(model="m", expr="x"),
        target_ref=TensorRef(model="m", expr="x"),
        from_a_ref=TensorRef(model="m", expr="x"),
        to_refs=[TensorRef(model="m", expr="x"), TensorRef(model="m", expr="new.*")],
        from_refs=[TensorRef(model="m", expr="x")],
        source_ref=TensorRef(model="m", expr="x"),
        factor_a_ref=TensorRef(model="m", expr="fa"),
        factor_b_ref=TensorRef(model="m", expr="fb"),
    )
    compiled = CompiledTransform(transform=SimpleNamespace(name="op"), spec=spec)  # type: ignore[arg-type]

    monkeypatch.setattr(
        execution_module,
        "resolve_name_mappings",
        lambda **_kwargs: [_Map("m", "x", "m", "x")],
    )
    monkeypatch.setattr(
        execution_module,
        "match_expr_names",
        lambda **kwargs: ["x"] if kwargs.get("expr") in {"x", "new.*"} else [],
    )
    impact2 = execution_module._preview_impact_for_transform(compiled, _Provider2())
    assert "m::x" in impact2.changed
    assert "changed[" in execution_module._format_preview_impact(impact2)


def test_execution_preview_exception_and_early_exit_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _T:
        name = "copy"

    compiled = CompiledTransform(transform=_T(), spec=object())

    monkeypatch.setattr(
        execution_module,
        "_preview_impact_for_transform",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    monkeypatch.setattr(
        execution_module,
        "apply_transform",
        lambda _compiled, _provider: SimpleNamespace(
            name="copy", count=1, control=execution_module.TransformControl.CONTINUE
        ),
    )
    monkeypatch.setattr(
        execution_module,
        "get_runtime_flags",
        lambda: SimpleNamespace(preview=True, dry_run=False),
    )
    lines: list[str] = []
    monkeypatch.setattr(execution_module, "emit_line", lines.append)
    should_continue, executed = execution_module._execute_transform_pairs(
        [({"copy": {}}, compiled)],
        object(),
        interactive=False,
    )
    assert should_continue is True
    assert executed == [{"copy": {}}]
    assert any("could not infer impact" in line for line in lines)

    monkeypatch.setattr(
        execution_module,
        "_preview_impact_for_transform",
        lambda *_args, **_kwargs: execution_module._PreviewImpact(
            changed={"m::x"}, created=set(), deleted=set()
        ),
    )
    monkeypatch.setattr(
        execution_module,
        "apply_transform",
        lambda _compiled, _provider: SimpleNamespace(
            name="exit", count=1, control=execution_module.TransformControl.EXIT
        ),
    )
    lines2: list[str] = []
    monkeypatch.setattr(execution_module, "emit_line", lines2.append)
    should_continue2, _executed2 = execution_module._execute_transform_pairs(
        [({"exit": {}}, compiled)],
        object(),
        interactive=False,
    )
    assert should_continue2 is False
    assert any("preview session:" in line for line in lines2)


def test_complete_collapses_large_reference_matches() -> None:
    payload_candidates = ["{ ", "}", ", ", "from: "]
    for i in range(10):
        payload_candidates.extend(
            [
                f"model::h.0.block{i}.attn.weight",
                f"model::h.0.block{i}.attn.bias",
                f"model::h.0.block{i}.mlp.weight",
            ]
        )
    matches = _match_payload_candidates(
        text="model::h.0.",
        line_buffer="copy: { from: model::h.0.",
        begidx=len("copy: { from: model::h.0."),
        payload_candidates=payload_candidates,
        active_transform="copy",
        model_aliases=["model"],
    )
    assert matches
    assert len(matches) < 30


def test_complete_collapse_handles_edge_cases() -> None:
    payload_candidates = ["{ ", "}", ", ", "from: ", "model::"]
    payload_candidates += [f"model::plain{i}" for i in range(26)]
    payload_candidates += [f"model::.x{i}.y" for i in range(26)]
    matches = _match_payload_candidates(
        text="model::",
        line_buffer="copy: { from: model::",
        begidx=len("copy: { from: model::"),
        payload_candidates=payload_candidates,
        active_transform="copy",
        model_aliases=["model"],
    )
    assert "model::" in matches
    assert any(item.startswith("model::.x") for item in matches)

    no_reduce_candidates = ["{ ", "}", ", ", "from: "] + [f"model::v{i}" for i in range(30)]
    no_reduce = _match_payload_candidates(
        text="model::",
        line_buffer="copy: { from: model::",
        begidx=len("copy: { from: model::"),
        payload_candidates=no_reduce_candidates,
        active_transform="copy",
        model_aliases=["model"],
    )
    assert len(no_reduce) == 30


def test_webui_backend_normalize_assert_payload_success() -> None:
    parsed = webui_backend_module._normalize_assert_payload("equal:\n  left: a\n  right: b\n")
    assert parsed == {"equal": {"left": "a", "right": "b"}}


def test_help_transform_handles_unknown_assert_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    transform = help_module.HelpTransform()
    monkeypatch.setattr(help_module, "get_assert_expr_names", lambda: ["odd"])
    monkeypatch.setattr(help_module, "get_assert_expr_help", lambda _name: {"bad": "shape"})
    emitted: list[str] = []
    monkeypatch.setattr(help_module, "emit_line", emitted.append)
    transform._print_assert_help()
    with pytest.raises(help_module.TransformError, match="unknown assert op"):
        transform._print_assert_expr_help("odd")
