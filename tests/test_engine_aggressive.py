from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

import brainsurgery.engine.checkpoint_io as checkpoint_io_module
import brainsurgery.engine.output_model as output_model_module
import brainsurgery.engine.output_paths as output_paths_module
import brainsurgery.engine.provider_utils as provider_utils_module
import brainsurgery.engine.providers as providers_module
import brainsurgery.engine.arena as arena_module
from brainsurgery.core import CompiledTransform, TransformError

from brainsurgery.engine.arena import ArenaSegment, ProviderError, _SegmentedFileBackedArena, torch_element_size
from brainsurgery.engine.output_model import _infer_output_model
from brainsurgery.engine.plan import _OutputSpec, _SurgeryPlan, parse_output
from brainsurgery.engine.render import _shape_only, render_tree, summarize_tensor
from brainsurgery.engine.state_dicts import _ArenaStateDict

from brainsurgery.engine.state_dicts import _InMemoryStateDict
from brainsurgery.engine.providers import InMemoryStateDictProvider
class _OutputModelProvider:
    def __init__(self) -> None:
        self.state_dicts = {
            "model": _InMemoryStateDict(),
            "other": _InMemoryStateDict(),
        }
        self.state_dicts["model"]["w"] = torch.ones(1)

    def get_state_dict(self, model: str) -> _InMemoryStateDict:
        return self.state_dicts[model]

@dataclass(frozen=True)
class _Spec:
    model: str

    def collect_models(self) -> set[str]:
        return {self.model}

class _Transform:
    name = "x"

    def contributes_output_model(self, spec: object) -> bool:
        del spec
        return True

    def _infer_output_model(self, spec: object) -> str:
        assert isinstance(spec, _Spec)
        return spec.model

class _FallbackTransform(_Transform):
    def _infer_output_model(self, spec: object) -> str:
        del spec
        raise TransformError("fallback needed")

def test_engine_output_model_uncovered_paths() -> None:
    provider = _OutputModelProvider()
    plan = _SurgeryPlan(
        inputs={},
        output=None,
        transforms=[
            CompiledTransform(_Transform(), _Spec("model")),
            CompiledTransform(_Transform(), _Spec("other")),
        ],
    )
    assert _infer_output_model(plan, provider) == "model"

    class _BadSpec:
        collect_models = 1

    with pytest.raises(TransformError, match="fallback needed"):
        _infer_output_model(
            _SurgeryPlan(inputs={}, output=None, transforms=[CompiledTransform(_FallbackTransform(), _BadSpec())]),
            provider,
        )

    @dataclass(frozen=True)
    class _TwoModelsSpec:
        def collect_models(self) -> set[str]:
            return {"model", "other"}

    provider.state_dicts["other"]["w"] = torch.ones(1)
    with pytest.raises(TransformError, match="fallback needed"):
        _infer_output_model(
            _SurgeryPlan(
                inputs={},
                output=None,
                transforms=[CompiledTransform(_FallbackTransform(), _TwoModelsSpec())],
            ),
            provider,
        )

    class _BrokenProvider:
        def get_state_dict(self, model: str):
            del model
            raise RuntimeError("broken")

    assert output_model_module._has_any_tensor(_BrokenProvider(), "x") is False

def test_engine_state_dict_uncovered_paths(tmp_path: Path) -> None:
    state_dict = _InMemoryStateDict()
    state_dict["w"] = torch.ones(2)
    assert list(iter(state_dict)) == ["w"]
    assert list(state_dict.values())[0].shape == (2,)
    assert state_dict.access_counts("missing") == {"reads": 0, "writes": 0}

    with pytest.raises(ProviderError, match="non-negative"):
        state_dict.mark_write("w", count=-1)
    with pytest.raises(ProviderError, match="non-negative"):
        state_dict._mark_read("w", count=-1)  # noqa: SLF001
    with pytest.raises(KeyError, match="missing"):
        state_dict._ensure_access_counts("missing")  # noqa: SLF001
    with pytest.raises(ProviderError, match="not a tensor"):
        state_dict.bind_slot("w", object())  # type: ignore[arg-type]

    arena = _SegmentedFileBackedArena(tmp_path, segment_size_bytes=128, alignment=8)
    arena_sd = _ArenaStateDict(arena)
    with pytest.raises(ProviderError, match="not a tensor"):
        arena_sd["bad"] = object()  # type: ignore[assignment]
    with pytest.raises(ProviderError, match="not a _TensorSlot"):
        arena_sd.bind_slot("bad", object())  # type: ignore[arg-type]
    with pytest.raises(KeyError, match="missing"):
        arena_sd.slot("missing")
    arena.close()

def test_engine_render_uncovered_paths() -> None:
    empty = summarize_tensor(
        torch.empty(0),
        verbosity="stats",
        access_counts={"reads": 1, "writes": 2},
    )
    assert empty["min"] is None and empty["reads"] == 1

    stats = summarize_tensor(torch.tensor([1, 2, 3], dtype=torch.int32), verbosity="stats")
    assert stats["mean"] == 2.0

    assert _shape_only(7) == 7
    assert "min=None max=None mean=None reads=3 writes=4" in render_tree(
        {"x": {"shape": [0], "min": None, "max": None, "mean": None, "reads": 3, "writes": 4}},
        compact=True,
    )
    assert "leaf  3" in render_tree({"leaf": 3}, compact=True)
    assert render_tree({"items": [None, None]}, compact=True).splitlines() == ["└── items"]

def test_engine_arena_uncovered_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ProviderError, match="positive"):
        ArenaSegment(tmp_path / "bad.bin", 0)
    segment = ArenaSegment(tmp_path / "ok.bin", 32)
    segment.flush()
    segment.close()

    with pytest.raises(ProviderError, match="segment_size_bytes must be positive"):
        _SegmentedFileBackedArena(tmp_path, segment_size_bytes=0, alignment=8)
    with pytest.raises(ProviderError, match="alignment must be positive"):
        _SegmentedFileBackedArena(tmp_path, segment_size_bytes=64, alignment=0)

    with _SegmentedFileBackedArena(tmp_path, segment_size_bytes=64, alignment=8) as arena:
        arena.flush()
        with pytest.raises(ProviderError, match="negative bytes"):
            arena.allocate(-1)
        with pytest.raises(ProviderError, match="exceeds segment bounds"):
            arena.tensor_view(segment_id=0, offset=63, dtype=torch.float32, shape=(1,))

        non_contiguous = torch.arange(6, dtype=torch.float32).reshape(2, 3).T
        slot = arena.store_tensor(non_contiguous)
        assert arena.tensor_from_slot(slot).shape == (3, 2)

        class _FakeCudaTensor:
            def __init__(self) -> None:
                self.device = type("_D", (), {"type": "cuda"})()
                self.dtype = torch.float32
                self.shape = (1,)

            def cpu(self) -> torch.Tensor:
                return torch.tensor([1.0], dtype=torch.float32)

        fake_slot = arena.store_tensor(_FakeCudaTensor())  # type: ignore[arg-type]
        assert fake_slot.nbytes == 4

        monkeypatch.setattr(
            arena_module.torch,
            "frombuffer",
            lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        with pytest.raises(ProviderError, match="failed to create tensor view"):
            arena.tensor_view(segment_id=0, offset=0, dtype=torch.float32, shape=(1,))

    with pytest.raises(ProviderError, match="unsupported dtype"):
        torch_element_size(torch.complex64)

def test_engine_provider_utils_and_providers_uncovered_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _ImportBreakProvider:
        state_dicts = {}

    real_import = __import__

    def _import_breaker(name, globals=None, locals=None, fromlist=(), level=0):
        if name in {"providers", "brainsurgery.engine.providers"} or name.endswith(".providers"):
            raise RuntimeError("boom")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", _import_breaker)
    assert provider_utils_module._is_base_provider_instance(_ImportBreakProvider()) is False
    monkeypatch.setattr("builtins.__import__", real_import)

    base_provider = InMemoryStateDictProvider({}, max_io_workers=1)
    assert provider_utils_module._has_model_alias(base_provider, "x") is False
    assert provider_utils_module.list_loaded_tensor_names(base_provider) == {}

    class _ProviderWithBadStateDicts:
        state_dicts = []

    assert provider_utils_module.list_loaded_tensor_names(_ProviderWithBadStateDicts()) == {}

    class _ProviderWithNonCallableKeys:
        state_dicts = {"a": object()}

    assert provider_utils_module.list_loaded_tensor_names(_ProviderWithNonCallableKeys()) == {}

    base = providers_module.BaseStateDictProvider({}, max_io_workers=1)
    with pytest.raises(NotImplementedError):
        base.get_state_dict("x")
    with pytest.raises(NotImplementedError):
        base.create_state_dict()

    base.state_dicts["x"] = _InMemoryStateDict()
    with pytest.raises(ProviderError, match="already exists"):
        base.load_alias_from_path("x", tmp_path / "m.safetensors")

    with pytest.raises(ProviderError, match="save_output requires plan.output"):
        base.save_output(_SurgeryPlan(inputs={}, output=None, transforms=[]), default_shard_size="1MB", max_io_workers=1)

    class _P(providers_module.BaseStateDictProvider):
        def get_state_dict(self, model: str):
            return self.state_dicts[model]

        def create_state_dict(self):
            return _InMemoryStateDict()

    p = _P({}, max_io_workers=1)
    p.state_dicts["model"] = _InMemoryStateDict()
    p.state_dicts["model"]["w"] = torch.ones(1)
    monkeypatch.setattr(providers_module, "_infer_output_model", lambda plan, provider: "model")
    monkeypatch.setattr(
        providers_module,
        "_resolve_output_destination",
        lambda output, default_shard_size: (output.path, "safetensors", 1),
    )
    monkeypatch.setattr(
        providers_module,
        "persist_state_dict",
        lambda state_dict, **kwargs: kwargs["output_path"],
    )
    written = p.save_output(
        _SurgeryPlan(inputs={}, output=_OutputSpec(path=tmp_path / "out"), transforms=[]),
        default_shard_size="1MB",
        max_io_workers=1,
    )
    assert written == tmp_path / "out"

    monkeypatch.setattr(providers_module, "parse_shard_size", lambda raw: None)
    with pytest.raises(ProviderError, match="must not be 'none'"):
        providers_module.create_state_dict_provider(
            provider="arena",
            model_paths={},
            max_io_workers=1,
            arena_root=tmp_path,
            arena_segment_size="none",
        )

def test_engine_output_paths_plan_and_checkpoint_edges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, fmt = output_paths_module._resolve_explicit_safetensors_destination(tmp_path / "dirish")
    assert path == tmp_path / "dirish" / "model.safetensors"
    assert fmt == "safetensors"
    file_path, file_fmt = output_paths_module._resolve_explicit_safetensors_destination(
        tmp_path / "m.safetensors"
    )
    assert file_path == tmp_path / "m.safetensors"
    assert file_fmt == "safetensors"

    out = _OutputSpec(path=tmp_path / "folder", format="safetensors")
    (tmp_path / "folder").mkdir()
    assert output_paths_module._is_directory_style_output(out) is True
    assert output_paths_module._resolve_output_destination(
        _OutputSpec(path=tmp_path / "newdir"),
        default_shard_size="1MB",
    )[:2] == (tmp_path / "newdir" / "model.safetensors", "safetensors")

    explicit_dir = tmp_path / "explicit_dir"
    explicit_dir.mkdir()
    assert output_paths_module._resolve_explicit_safetensors_destination(explicit_dir) == (
        explicit_dir / "model.safetensors",
        "safetensors",
    )

    assert parse_output({"path": str(tmp_path / "x.safetensors")}) == _OutputSpec(path=tmp_path / "x.safetensors")

    with pytest.raises(RuntimeError, match="max_shard_size must be positive"):
        checkpoint_io_module.shard_state_dict({"a": torch.ones(1)}, 0)

    shards = checkpoint_io_module.shard_state_dict(
        {"a": torch.ones(1), "big": torch.ones(16)},
        max_shard_size=16,
    )
    assert len(shards) >= 2

    monkeypatch.setattr(checkpoint_io_module, "shard_state_dict", lambda state_dict, max_shard_size: [{}])
    with pytest.raises(RuntimeError, match="coverage mismatch"):
        checkpoint_io_module.save_sharded_safetensors(
            {"a": torch.ones(1)},
            tmp_path / "shards",
            16,
            max_io_workers=1,
        )

    with pytest.raises(RuntimeError, match="does not exist"):
        checkpoint_io_module._load_state_dict_from_path(
            tmp_path / "missing",
            _InMemoryStateDict(),
            max_io_workers=1,
        )

    called = {"dir": False}

    def _fake_dir_loader(path, global_state_dict, *, max_io_workers):
        del path, global_state_dict, max_io_workers
        called["dir"] = True

    real_dir_loader = checkpoint_io_module.load_state_dict_from_directory
    monkeypatch.setattr(checkpoint_io_module, "load_state_dict_from_directory", _fake_dir_loader)
    root_dir = tmp_path / "root_dir"
    root_dir.mkdir()
    checkpoint_io_module._load_state_dict_from_path(root_dir, _InMemoryStateDict(), max_io_workers=1)
    assert called["dir"] is True
    monkeypatch.setattr(checkpoint_io_module, "load_state_dict_from_directory", real_dir_loader)

    empty_dir = tmp_path / "empty_ckpt"
    empty_dir.mkdir()
    with pytest.raises(RuntimeError, match="no supported checkpoint files"):
        checkpoint_io_module.load_state_dict_from_directory(
            empty_dir,
            _InMemoryStateDict(),
            max_io_workers=1,
        )

    index_file = tmp_path / "model.safetensors.index.json"
    index_file.write_text('{"weight_map":{"a":""}}', encoding="utf-8")
    with pytest.raises(RuntimeError, match="non-empty string"):
        checkpoint_io_module.resolve_safetensor_shards_from_index(index_file, tmp_path)

    wrapped_path = tmp_path / "wrapped.pt"
    torch.save({"state_dict": {"w": torch.ones(1)}}, wrapped_path)
    global_sd = _InMemoryStateDict()
    loaded_count = checkpoint_io_module.load_state_dict_from_file(wrapped_path, global_sd)
    assert loaded_count == 1

    mismatch_dir = tmp_path / "mismatch_ckpt"
    mismatch_dir.mkdir()
    torch.save({"w": torch.ones(1)}, mismatch_dir / "one.pt")
    monkeypatch.setattr(checkpoint_io_module, "_choose_num_io_workers", lambda n, max_io_workers: 1)
    monkeypatch.setattr(checkpoint_io_module, "_run_threadpool_tasks_with_progress", lambda **kwargs: None)
    with pytest.raises(RuntimeError, match="progress count mismatch"):
        checkpoint_io_module.load_state_dict_from_directory(
            mismatch_dir,
            _InMemoryStateDict(),
            max_io_workers=1,
        )

    def _fake_progress(**kwargs):
        files = kwargs["items"]
        on_result = kwargs["on_result"]
        for file_path in files:
            on_result(file_path, 1)

    monkeypatch.setattr(checkpoint_io_module, "_run_threadpool_tasks_with_progress", _fake_progress)
    with pytest.raises(RuntimeError, match="merged tensor count mismatch"):
        checkpoint_io_module.load_state_dict_from_directory(
            mismatch_dir,
            _InMemoryStateDict(),
            max_io_workers=1,
        )
