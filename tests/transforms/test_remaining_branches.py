from __future__ import annotations

from importlib import import_module, reload
from types import SimpleNamespace

import pkgutil
import pytest
import torch

from brainsurgery.core import ResolvedMapping, TensorRef, TransformError

from brainsurgery.engine.state_dicts import _InMemoryStateDict
def _provider_for(*, model: str, state_dict) -> object:
    class _Provider:
        def get_state_dict(self, selected: str):
            assert selected == model
            return state_dict

    return _Provider()

def test_assign_validate_refs_parses_slices() -> None:
    module = import_module("brainsurgery.transforms.assign")
    transform = module.AssignTransform()
    transform.validate_refs(
        TensorRef(model="m", expr="a", slice_spec="[:1]"),
        TensorRef(model="m", expr="b", slice_spec="[:1]"),
    )

def test_assert_completion_payload_start_candidates_non_empty_prefix() -> None:
    module = import_module("brainsurgery.transforms.assert_")
    out = module.AssertTransform().completion_payload_start_candidates("e")
    assert out is not None
    assert "{ " not in out

def test_clamp_compile_rejects_min_gt_max() -> None:
    module = import_module("brainsurgery.transforms.clamp")
    with pytest.raises(TransformError, match="min must be <="):
        module.ClampTransform().compile(
            {"from": "x", "to": "y", "min": 2, "max": 1},
            default_model="m",
        )

def test_concat_error_branches_and_validation_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    module = import_module("brainsurgery.transforms.concat")
    transform = module.ConcatTransform()

    monkeypatch.setattr(
        module,
        "parse_model_expr",
        lambda value, default_model=None: (
            TensorRef(model=None, expr="x")
            if value == "missing"
            else TensorRef(model="m", expr="x")
        ),
    )
    with pytest.raises(module.ConcatTransformError, match="missing model alias"):
        transform.compile({"from": ["missing", "ok"], "to": "ok"}, default_model=None)
    with pytest.raises(module.ConcatTransformError, match="missing model alias"):
        transform.compile({"from": ["ok", "ok"], "to": "missing"}, default_model=None)

    with monkeypatch.context() as patch:
        patch.setattr(
            module,
            "parse_model_expr",
            lambda _value, default_model=None: TensorRef(model="m", expr=object()),
        )
        with pytest.raises(module.ConcatTransformError, match="single tensor name"):
            transform.compile({"from": ["m::x", "m::y"], "to": "m::z"}, default_model="m")

    with pytest.raises(module.ConcatTransformError, match="must be an integer"):
        transform.compile({"from": ["m::x", "m::y"], "to": "m::z", "dim": "0"}, default_model=None)

    with pytest.raises(module.ConcatTransformError, match="wrong spec type"):
        transform.require_spec(object())

    with pytest.raises(module.ConcatTransformError, match="at least one source"):
        transform._validate_sources([], dim=0)

    with pytest.raises(module.ConcatTransformError, match="out of range"):
        transform._validate_sources([torch.ones(2, 2)], dim=2)

    with pytest.raises(module.ConcatTransformError, match="rank mismatch"):
        transform._validate_sources([torch.ones(2, 2), torch.ones(2)], dim=0)

    with pytest.raises(module.ConcatTransformError, match="dtype mismatch"):
        transform._validate_sources(
            [torch.ones(2, 2, dtype=torch.float32), torch.ones(2, 2, dtype=torch.float16)],
            dim=0,
        )

    class _FakeTensor:
        def __init__(self, shape: tuple[int, ...], dtype: torch.dtype, device: str) -> None:
            self.shape = shape
            self.dtype = dtype
            self.device = torch.device(device)

        def dim(self) -> int:
            return len(self.shape)

    with pytest.raises(module.ConcatTransformError, match="device mismatch"):
        transform._validate_sources(
            [
                _FakeTensor((2, 2), torch.float32, "cpu"),
                _FakeTensor((2, 2), torch.float32, "meta"),
            ],
            dim=0,
        )

def test_concat_apply_rejects_existing_destination_and_source_match_count() -> None:
    module = import_module("brainsurgery.transforms.concat")
    transform = module.ConcatTransform()
    spec = module.ConcatSpec(
        from_refs=[TensorRef(model="m", expr="left"), TensorRef(model="m", expr="right")],
        to_ref=TensorRef(model="m", expr="dst"),
        dim=0,
    )
    state_dict = _InMemoryStateDict()
    state_dict["left"] = torch.ones(1, 2)
    state_dict["right"] = torch.ones(1, 2)
    state_dict["dst"] = torch.zeros(2, 2)
    provider = _provider_for(model="m", state_dict=state_dict)

    with pytest.raises(module.ConcatTransformError, match="destination already exists"):
        transform.apply(spec, provider)

    state_dict = _InMemoryStateDict()
    state_dict["left"] = torch.ones(1, 2)
    state_dict["right"] = torch.ones(1, 2)
    provider = _provider_for(model="m", state_dict=state_dict)
    with pytest.raises(module.ConcatTransformError, match="matched zero tensors"):
        transform._resolve_source_tensor(TensorRef(model="m", expr="missing"), provider)
    with pytest.raises(module.ConcatTransformError, match="exactly one tensor"):
        transform._resolve_source_tensor(TensorRef(model="m", expr=".*"), provider)

def test_copy_apply_slice_and_destination_exists_branch() -> None:
    module = import_module("brainsurgery.transforms.copy")
    state_dict = _InMemoryStateDict()
    state_dict["src"] = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    state_dict["dst"] = torch.tensor([0.0], dtype=torch.float32)
    provider = _provider_for(model="m", state_dict=state_dict)
    spec = module.BinaryMappingSpec(TensorRef("m", "src", slice_spec="[:1]"), TensorRef("m", "dst"))
    with pytest.raises(TransformError, match="destination already exists"):
        module._copy_apply(spec, "src", "dst", provider)

def test_diff_compile_and_helper_error_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    module = import_module("brainsurgery.transforms.diff")
    transform = module.DiffTransform()

    with pytest.raises(module.DiffTransformError, match="non-empty string"):
        transform.compile({"mode": ""}, default_model="m")
    with pytest.raises(module.DiffTransformError, match="one of: refs, aliases"):
        transform.compile({"mode": "wat"}, default_model="m")
    with pytest.raises(module.DiffTransformError, match="diff.left is required"):
        transform.compile({"mode": "refs", "right": "m::x"}, default_model=None)
    with pytest.raises(module.DiffTransformError, match="diff.right is required"):
        transform.compile({"mode": "refs", "left": "m::x"}, default_model=None)
    with pytest.raises(module.DiffTransformError, match="non-negative number"):
        transform.compile({"left": "m::x", "right": "m::x", "eps": True}, default_model=None)
    with pytest.raises(module.DiffTransformError, match="non-negative number"):
        transform.compile({"left": "m::x", "right": "m::x", "eps": -1}, default_model=None)

    with pytest.raises(module.DiffTransformError, match="does not infer an output model"):
        transform._infer_output_model(object())

    spec = transform.compile({"left": "m::x::[:1]", "right": "m::x::[:1]"}, default_model=None)
    assert spec.left_ref.slice_spec == "[:1]"

    with pytest.raises(module.DiffTransformError, match="boom"):
        monkeypatch.setattr(module, "match_expr_names", lambda **_: (_ for _ in ()).throw(TransformError("boom")))
        state_dict = _InMemoryStateDict()
        state_dict["x"] = torch.ones(1)
        module._resolve_names(TensorRef(model="m", expr="x"), _provider_for(model="m", state_dict=state_dict))

    class _FakeTensor:
        def __init__(self, *, shape, dtype, device, complex_: bool = False) -> None:
            self.shape = shape
            self.dtype = dtype
            self.device = torch.device(device)
            self._complex = complex_

        def is_complex(self) -> bool:
            return self._complex

    reason = module._difference_reason(
        _FakeTensor(shape=(1,), dtype=torch.float32, device="cpu"),
        _FakeTensor(shape=(1,), dtype=torch.float32, device="meta"),
        eps=None,
    )
    assert reason == "device cpu != meta"
    assert module._diff_mode("left: x") is None
    assert module._compile_eps(None) is None

def test_diff_complex_eps_branch() -> None:
    module = import_module("brainsurgery.transforms.diff")
    left = torch.tensor([1 + 1j], dtype=torch.complex64)
    right = torch.tensor([2 + 1j], dtype=torch.complex64)
    reason = module._difference_reason(left, right, eps=0.1)
    assert "max_abs_diff" in str(reason)

def test_diff_remaining_completion_and_spec_helpers() -> None:
    module = import_module("brainsurgery.transforms.diff")
    transform = module.DiffTransform()
    spec = module.DiffSpec(
        left_ref=TensorRef(model="a", expr="x"),
        right_ref=TensorRef(model="b", expr="x"),
        eps=None,
        mode="refs",
    )
    assert spec.collect_models() == {"a", "b"}
    assert transform.contributes_output_model(object()) is False
    assert transform.completion_key_candidates("mode: aliases", "") == [
        "left_alias: ",
        "right_alias: ",
        "eps: ",
    ]
    assert transform.completion_key_candidates("mode: refs left: x", "z") == []
    assert transform.completion_key_candidates("mode: refs left: x right: y eps: 1", "") == ["}"]
    assert module._diff_mode("mode: ALIASES left_alias: a") == "aliases"

def test_dump_remaining_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    module = import_module("brainsurgery.transforms.dump")
    transform = module.DumpTransform()
    with pytest.raises(module.DumpTransformError, match="non-empty string"):
        transform.compile({"target": "x", "format": ""}, default_model="m")
    with pytest.raises(module.DumpTransformError, match="non-empty string"):
        transform.compile({"target": "x", "verbosity": ""}, default_model="m")
    with pytest.raises(module.DumpTransformError, match="one of: shape, stat, full"):
        transform.compile({"target": "x", "verbosity": "loud"}, default_model="m")
    assert transform.contributes_output_model(object()) is False

    tree: dict[str, object] = {}
    module.insert_into_tree(tree, ["a", "0"], "leaf")
    assert tree == {"a": ["leaf"]}

    with pytest.raises(module.DumpTransformError, match="invalid tree structure"):
        module.insert_into_tree({"a": [1]}, ["a", "0", "x"], 2)
    with pytest.raises(module.DumpTransformError, match="invalid tree structure"):
        module.insert_into_tree({"a": 1}, ["a", "b"], 2)
    with pytest.raises(module.DumpTransformError, match="invalid tree structure"):
        module.insert_into_tree({}, ["0"], 2)

    assert module._maybe_get_access_counts(SimpleNamespace(access_counts={}), "x", verbosity="full") is None
    monkeypatch.setattr(module, "emit_line", lambda _line: None)

def test_exit_contributes_output_model_false() -> None:
    module = import_module("brainsurgery.transforms.exit")
    assert module.ExitTransform().contributes_output_model(object()) is False

def test_fill_parse_config_and_build_error_branches() -> None:
    module = import_module("brainsurgery.transforms.fill")
    with pytest.raises(TransformError, match="mode must be one of"):
        module.parse_fill_config({}, op_name="fill", error_type=TransformError)
    with pytest.raises(TransformError, match="mode must be one of"):
        module.parse_fill_config({"mode": "wat"}, op_name="fill", error_type=TransformError)
    with pytest.raises(TransformError, match="seed must be an integer"):
        module.parse_fill_config({"mode": "rand", "seed": 1.2}, op_name="fill", error_type=TransformError)
    with pytest.raises(TransformError, match="values is required"):
        module.parse_fill_config({"mode": "tensor"}, op_name="fill", error_type=TransformError)
    with pytest.raises(TransformError, match="distribution must be one of"):
        module.parse_fill_config(
            {"mode": "rand", "distribution": "lognormal"},
            op_name="fill",
            error_type=TransformError,
        )
    with pytest.raises(TransformError, match="low < high"):
        module.parse_fill_config(
            {"mode": "rand", "distribution": "uniform", "low": 1, "high": 1},
            op_name="fill",
            error_type=TransformError,
        )
    with pytest.raises(TransformError, match="std must be > 0"):
        module.parse_fill_config(
            {"mode": "rand", "distribution": "normal", "std": 0},
            op_name="fill",
            error_type=TransformError,
        )

    uniform = module.parse_fill_config(
        {"mode": "rand", "distribution": "uniform", "low": -1, "high": 2},
        op_name="fill",
        error_type=TransformError,
    )
    assert uniform.low == -1 and uniform.high == 2

    normal = module.parse_fill_config(
        {"mode": "rand", "distribution": "normal", "mean": 2.5, "std": 0.5},
        op_name="fill",
        error_type=TransformError,
    )
    template = torch.zeros((3,), dtype=torch.float32)
    out = module.build_filled_tensor_like(template, normal, TransformError)
    assert out.shape == (3,)

    bad = module.FillConfig(
        mode="tensor",
        constant_value=None,
        values_payload=[[1.0], [2.0], [3.0]],
        distribution="uniform",
        low=0.0,
        high=1.0,
        mean=0.0,
        std=1.0,
        seed=None,
    )
    with pytest.raises(TransformError, match="cannot broadcast"):
        module.build_filled_tensor_like(torch.zeros((2, 2), dtype=torch.float32), bad, TransformError)

def test_matmul_error_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    module = import_module("brainsurgery.transforms.matmul")
    state_dict = _InMemoryStateDict()
    state_dict["a"] = torch.tensor([[1.0]], dtype=torch.float32)
    state_dict["b"] = torch.tensor([[1.0]], dtype=torch.float16)
    provider = _provider_for(model="m", state_dict=state_dict)
    spec = module.TernaryMappingSpec(TensorRef("m", "a"), TensorRef("m", "b"), TensorRef("m", "c"))
    with pytest.raises(TransformError, match="dtype mismatch"):
        module._matmul_apply(spec, "a", "b", "c", provider)

    with monkeypatch.context() as patch:
        calls = [0]

        def _select(_tensor, _slice):
            calls[0] += 1
            if calls[0] == 1:
                return SimpleNamespace(dtype=torch.float32, device=torch.device("cpu"))
            return SimpleNamespace(dtype=torch.float32, device=torch.device("meta"))

        patch.setattr(module, "select_tensor", _select)
        with pytest.raises(TransformError, match="device mismatch"):
            module._matmul_apply(spec, "a", "b", "c", provider)

    state_dict = _InMemoryStateDict()
    state_dict["a"] = torch.ones((2, 3), dtype=torch.float32)
    state_dict["b"] = torch.ones((4, 2), dtype=torch.float32)
    provider = _provider_for(model="m", state_dict=state_dict)
    with pytest.raises(TransformError, match="shape mismatch"):
        module._matmul_apply(spec, "a", "b", "c", provider)

def test_move_apply_error_branches() -> None:
    module = import_module("brainsurgery.transforms.move")

    class _StateDict(dict):
        def slot(self, key):
            return self[key]

        def bind_slot(self, key, slot):
            self[key] = slot

    state_dict = _StateDict({"src": torch.ones(1), "dst": torch.ones(1)})
    spec = module.BinaryMappingSpec(TensorRef("m", "src"), TensorRef("m", "dst"))
    with pytest.raises(module.MoveTransformError, match="destination already exists"):
        module.MoveTransform().apply_mapping(
            spec,
            "src",
            "dst",
            _provider_for(model="m", state_dict=state_dict),
        )

    class _BrokenBindStateDict(_StateDict):
        def bind_slot(self, key, slot):
            del key, slot

    with pytest.raises(module.MoveTransformError, match="destination missing after move"):
        module.MoveTransform().apply_mapping(
            spec,
            "src",
            "dst",
            _provider_for(model="m", state_dict=_BrokenBindStateDict({"src": torch.ones(1)})),
        )

    class _StickyDeleteStateDict(_StateDict):
        def __delitem__(self, key):
            del key

    with pytest.raises(module.MoveTransformError, match="source still present after move"):
        module.MoveTransform().apply_mapping(
            spec,
            "src",
            "dst",
            _provider_for(model="m", state_dict=_StickyDeleteStateDict({"src": torch.ones(1)})),
        )

def test_permute_rank_permutation_and_order_type_errors() -> None:
    module = import_module("brainsurgery.transforms.permute")
    with pytest.raises(TransformError, match="non-empty list of integers"):
        module._parse_order([0, "1"])

    state_dict = _InMemoryStateDict()
    state_dict["x"] = torch.ones((2, 3))
    provider = _provider_for(model="m", state_dict=state_dict)
    with pytest.raises(TransformError, match="rank mismatch"):
        module._permute_apply(
            module.PermuteSpec(TensorRef("m", "x"), TensorRef("m", "y"), order=(0, 1, 2)),
            "x",
            "y",
            provider,
        )
    with pytest.raises(TransformError, match="must be a permutation"):
        module._permute_apply(
            module.PermuteSpec(TensorRef("m", "x"), TensorRef("m", "y"), order=(0, 0)),
            "x",
            "y",
            provider,
        )

def test_phlora_remaining_error_branches() -> None:
    module = import_module("brainsurgery.transforms.phlora")
    spec = module.PhloraSpec(
        source_ref=TensorRef(model="m1", expr="x"),
        factor_b_ref=TensorRef(model="m2", expr="x.b"),
        factor_a_ref=TensorRef(model="m3", expr="x.a"),
        rank=1,
        delete_source=True,
        require_missing_outputs=True,
    )
    assert spec.collect_models() == {"m1", "m2", "m3"}

    transform = module.PhloraTransform()
    with pytest.raises(module.PhloraTransformError, match="target must not be sliced"):
        transform.validate_refs(
            TensorRef(model="m", expr="x", slice_spec="[:]"),
            TensorRef(model="m", expr="x.b"),
            TensorRef(model="m", expr="x.a"),
        )
    with pytest.raises(module.PhloraTransformError, match="target_b must not be sliced"):
        transform.validate_refs(
            TensorRef(model="m", expr="x"),
            TensorRef(model="m", expr="x.b", slice_spec="[:]"),
            TensorRef(model="m", expr="x.a"),
        )
    with pytest.raises(module.PhloraTransformError, match="target_a must not be sliced"):
        transform.validate_refs(
            TensorRef(model="m", expr="x"),
            TensorRef(model="m", expr="x.b"),
            TensorRef(model="m", expr="x.a", slice_spec="[:]"),
        )

    with pytest.raises(module.PhloraTransformError, match="output model missing"):
        transform._infer_output_model(
            module.PhloraSpec(
                source_ref=TensorRef(model=None, expr="x"),
                factor_b_ref=TensorRef(model="m", expr="x.b"),
                factor_a_ref=TensorRef(model="m", expr="x.a"),
                rank=1,
                delete_source=True,
                require_missing_outputs=True,
            )
        )
    assert transform._infer_output_model(spec) == "m1"

    with pytest.raises(module.PhloraTransformError, match="must be a boolean"):
        module._require_boolean({"x": "nope"}, op_name="phlora", key="x", default=True)

    with pytest.raises(module.PhloraTransformError, match="do not match source set"):
        module._pair_mappings(
            [ResolvedMapping("m", "a", None, "m", "a.a", None)],
            [ResolvedMapping("m", "b", None, "m", "b.b", None)],
            op_name="phlora",
        )

    class _Provider:
        def __init__(self) -> None:
            self.src = _InMemoryStateDict()
            self.a = _InMemoryStateDict()
            self.b = _InMemoryStateDict()

        def get_state_dict(self, model: str):
            return {"src": self.src, "a": self.a, "b": self.b}[model]

    provider = _Provider()
    with pytest.raises(module.PhloraTransformError, match="source disappeared"):
        transform.apply_item(
            spec,
            module.ResolvedPhloraMapping("src", "missing", "a", "x.a", "b", "x.b"),
            provider,
        )

    provider.src["w"] = torch.ones((2, 2))
    provider.a["x.a"] = torch.ones((1, 2))
    with pytest.raises(module.PhloraTransformError, match="destination already exists"):
        transform.apply_item(
            spec,
            module.ResolvedPhloraMapping("src", "w", "a", "x.a", "b", "x.b"),
            provider,
        )

    provider = _Provider()
    provider.src["w"] = torch.ones((2, 2))
    provider.b["x.b"] = torch.ones((2, 1))
    with pytest.raises(module.PhloraTransformError, match="destination already exists"):
        transform.apply_item(
            spec,
            module.ResolvedPhloraMapping("src", "w", "a", "x.a", "b", "x.b"),
            provider,
        )

    provider = _Provider()
    provider.src["w"] = torch.ones((2, 2))
    with pytest.raises(module.PhloraTransformError, match="destination collision"):
        transform.apply_item(
            module.PhloraSpec(
                source_ref=TensorRef(model="src", expr="w"),
                factor_b_ref=TensorRef(model="a", expr="same"),
                factor_a_ref=TensorRef(model="a", expr="same"),
                rank=1,
                delete_source=False,
                require_missing_outputs=False,
            ),
            module.ResolvedPhloraMapping("src", "w", "a", "same", "a", "same"),
            provider,
        )

def test_reshape_parse_and_runtime_error_branches() -> None:
    module = import_module("brainsurgery.transforms.reshape")
    with pytest.raises(TransformError, match="non-empty list of integers"):
        module._parse_shape(None, op_name="reshape", error_type=TransformError)
    with pytest.raises(TransformError, match="non-empty list of integers"):
        module._parse_shape([1, "2"], op_name="reshape", error_type=TransformError)
    with pytest.raises(TransformError, match="positive integers or -1"):
        module._parse_shape([0, 2], op_name="reshape", error_type=TransformError)

    state_dict = _InMemoryStateDict()
    state_dict["x"] = torch.arange(5)
    provider = _provider_for(model="m", state_dict=state_dict)
    with pytest.raises(TransformError, match="reshape failed"):
        module._reshape_apply(
            module.ReshapeSpec(TensorRef("m", "x"), TensorRef("m", "y"), shape=(2, 3)),
            "x",
            "y",
            provider,
        )
    with pytest.raises(TransformError, match="reshape_ failed"):
        module._reshape_in_place_apply(
            module.ReshapeInPlaceSpec(TensorRef("m", "x"), shape=(2, 3)),
            "x",
            provider,
        )

def test_split_compile_and_apply_error_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    module = import_module("brainsurgery.transforms.split")
    transform = module.SplitTransform()

    spec = transform.compile({"from": "x::[:1]", "to": ["a", "b"], "sizes": [1, 1]}, default_model="m")
    assert spec.from_ref.slice_spec == "[:1]"
    with pytest.raises(module.SplitTransformError, match="at least two references"):
        transform.compile({"from": "x", "to": ["a"], "sizes": [1]}, default_model="m")

    with monkeypatch.context() as patch:
        patch.setattr(
            module,
            "parse_model_expr",
            lambda value, default_model=None: (
                TensorRef(model=None, expr="x")
                if value in {"src_missing", "dst_missing"}
                else TensorRef(model="m", expr="x")
            ),
        )
        with pytest.raises(module.SplitTransformError, match="missing model alias"):
            transform.compile(
                {"from": "src_missing", "to": ["ok", "ok"], "sizes": [1, 1]},
                default_model=None,
            )
        with pytest.raises(module.SplitTransformError, match="missing model alias"):
            transform.compile(
                {"from": "ok", "to": ["ok", "dst_missing"], "sizes": [1, 1]},
                default_model=None,
            )
    with pytest.raises(module.SplitTransformError, match="must not be sliced"):
        transform.compile({"from": "m::x", "to": ["m::a::[:]", "m::b"], "sizes": [1, 1]}, default_model=None)

    with monkeypatch.context() as patch:
        patch.setattr(
            module,
            "parse_model_expr",
            lambda _value, default_model=None: TensorRef(model="m", expr=object()),
        )
        with pytest.raises(module.SplitTransformError, match="single names"):
            transform.compile({"from": "m::x", "to": ["m::a", "m::b"], "sizes": [1, 1]}, default_model=None)

    with pytest.raises(module.SplitTransformError, match="must be an integer"):
        transform.compile({"from": "m::x", "to": ["m::a", "m::b"], "sizes": [1, 1], "dim": "0"}, default_model=None)

    with pytest.raises(module.SplitTransformError, match="wrong spec type"):
        transform.require_spec(object())

    with pytest.raises(module.SplitTransformError, match="non-empty list of positive integers"):
        module._parse_sizes([])
    with pytest.raises(module.SplitTransformError, match="non-empty list of positive integers"):
        module._parse_sizes([1, 0])

    state_dict = _InMemoryStateDict()
    state_dict["x"] = torch.ones((2, 2))
    state_dict["z"] = torch.ones((2, 2))
    provider = _provider_for(model="m", state_dict=state_dict)

    with pytest.raises(module.SplitTransformError, match="matched zero tensors"):
        transform.apply(
            module.SplitSpec(TensorRef("m", "missing"), [TensorRef("m", "a"), TensorRef("m", "b")], [1, 1], 0),
            provider,
        )
    with pytest.raises(module.SplitTransformError, match="exactly one tensor"):
        transform.apply(
            module.SplitSpec(TensorRef("m", ".*"), [TensorRef("m", "a"), TensorRef("m", "b")], [1, 1], 0),
            provider,
        )
    with pytest.raises(module.SplitTransformError, match="out of range"):
        transform.apply(
            module.SplitSpec(TensorRef("m", "x"), [TensorRef("m", "a"), TensorRef("m", "b")], [1, 1], 2),
            provider,
        )
    with pytest.raises(module.SplitTransformError, match="must sum to source size"):
        transform.apply(
            module.SplitSpec(TensorRef("m", "x"), [TensorRef("m", "a"), TensorRef("m", "b")], [1, 2], 0),
            provider,
        )

    state_dict = _InMemoryStateDict()
    state_dict["x"] = torch.tensor([1.0, 2.0, 3.0, 4.0])
    state_dict["a"] = torch.tensor([0.0, 0.0])
    provider = _provider_for(model="m", state_dict=state_dict)
    with pytest.raises(module.SplitTransformError, match="destination already exists"):
        transform.apply(
            module.SplitSpec(TensorRef("m", "x"), [TensorRef("m", "a"), TensorRef("m", "b")], [2, 2], 0),
            provider,
        )

def test_transforms_package_skips_private_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(pkgutil, "iter_modules", lambda _path: [SimpleNamespace(name="_p"), SimpleNamespace(name="x")])
    import importlib as _importlib

    monkeypatch.setattr(_importlib, "import_module", lambda name: calls.append(name))
    package = import_module("brainsurgery.transforms")
    reload(package)
    assert calls == ["brainsurgery.transforms.x"]

def test_help_print_all_commands(monkeypatch: pytest.MonkeyPatch) -> None:
    module = import_module("brainsurgery.transforms.help")
    lines: list[str] = []
    monkeypatch.setattr(module, "list_transforms", lambda: ["a", "b"])
    monkeypatch.setattr(module, "emit_line", lines.append)
    module.HelpTransform()._print_all_commands()
    output = "\n".join(lines)
    assert "Help for commands" in output
    assert "Available commands:" in output
    assert "a" in output
    assert "For help on a specific command:" in output
    assert "YAML: help: <command>" in output
    assert "OLY:  help: <command>" in output
