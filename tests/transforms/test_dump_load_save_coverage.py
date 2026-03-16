from __future__ import annotations

from pathlib import Path

import pytest
import torch

from brainsurgery.core import TensorRef
from brainsurgery.engine import (
    RuntimeFlagLifecycleScope,
    reset_runtime_flags_for_scope,
    set_runtime_flag,
)
from brainsurgery.engine.providers import InMemoryStateDictProvider
from brainsurgery.engine.state_dicts import _InMemoryStateDict
from brainsurgery.transforms.dump import (
    DumpTransform,
    DumpTransformError,
    _maybe_get_access_counts,
    _resolve_model_aliases,
    insert_into_tree,
)
from brainsurgery.transforms.load import LoadSpec, LoadTransform
from brainsurgery.transforms.save import SaveSpec, SaveTransform, SaveTransformError


def test_dump_apply_all_formats_and_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    reset_runtime_flags_for_scope(RuntimeFlagLifecycleScope.CLI_RUN)
    provider = InMemoryStateDictProvider({}, max_io_workers=1)
    a = provider.get_or_create_alias_state_dict("a")
    b = provider.get_or_create_alias_state_dict("b")
    a["x"] = torch.ones(2)
    b["y"] = torch.zeros(2)

    lines: list[str] = []
    monkeypatch.setattr("brainsurgery.transforms.dump.emit_line", lines.append)
    monkeypatch.setattr("brainsurgery.transforms.dump.tqdm", lambda seq, **kwargs: seq)

    for fmt in ("json", "tree", "compact"):
        spec = DumpTransform().compile({"format": fmt, "verbosity": "stat"}, default_model="a")
        result = DumpTransform().apply(spec, provider)
        assert result.count == 2
    assert lines

    ref = TensorRef(model="a", expr="x", slice_spec="[:1]")
    sliced = DumpTransform().build_spec(
        ref, {"format": "tree", "verbosity": "full"}, dump_all_models=False
    )
    sliced_result = DumpTransform().apply(sliced, provider)
    assert sliced_result.count == 1

    sd = provider.get_state_dict("a")
    assert _maybe_get_access_counts(sd, "x", verbosity="shape") is None
    assert isinstance(_maybe_get_access_counts(sd, "x", verbosity="full"), dict)

    class _NoAliasProvider:
        pass

    assert _resolve_model_aliases(_NoAliasProvider(), "fallback") == ["fallback"]
    assert _resolve_model_aliases(_NoAliasProvider(), None) == []

    with pytest.raises(DumpTransformError, match="dump.format must be a non-empty string"):
        DumpTransform().build_spec(None, {"format": 1, "verbosity": "shape"})  # type: ignore[arg-type]
    with pytest.raises(DumpTransformError, match="dump.verbosity must be a non-empty string"):
        DumpTransform().build_spec(None, {"format": "tree", "verbosity": 1})  # type: ignore[arg-type]


def test_insert_into_tree_numeric_paths_and_error_branches() -> None:
    root: dict[str, object] = {}
    insert_into_tree(root, ["h", "0", "weight"], {"shape": [1]})
    assert isinstance(root["h"], list)

    with pytest.raises(DumpTransformError, match="invalid tree structure"):
        insert_into_tree({"h": {}}, ["h", "0", "x"], 1)
    with pytest.raises(DumpTransformError, match="invalid tree structure"):
        insert_into_tree({"h": [1]}, ["h", "0", "x"], 1)


def test_load_transform_completion_and_success_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    tr = LoadTransform()
    assert tr.completion_reference_keys() == ["to"]
    assert tr.completion_value_candidates("alias", "m", ["model", "other"]) == ["model"]
    assert tr.completion_value_candidates("format", "s", []) == ["safetensors"]
    assert tr.completion_value_candidates("unknown", "x", []) is None

    provider = InMemoryStateDictProvider({}, max_io_workers=1)
    called = {"alias": False}
    sd = _InMemoryStateDict()
    sd["x"] = torch.ones(1)
    monkeypatch.setattr(
        provider,
        "load_alias_from_path",
        lambda alias, path: (called.__setitem__("alias", True), sd)[1],
    )
    state_spec = LoadSpec(
        path=Path("/tmp/m.safetensors"), alias="model", tensor_name=None, format="auto"
    )
    result = tr.apply(state_spec, provider)
    assert called["alias"] is True
    assert result.count == 1
    assert tr._infer_output_model(state_spec) == "model"

    reset_runtime_flags_for_scope(RuntimeFlagLifecycleScope.CLI_RUN)
    set_runtime_flag("dry_run", True)
    tensor_spec = LoadSpec(
        path=Path("/tmp/t.npy"), alias="model", tensor_name="new_x", format="numpy"
    )
    monkeypatch.setattr(
        "brainsurgery.transforms.load.load_tensor_from_path", lambda path, format: torch.zeros(1)
    )
    result = tr.apply(tensor_spec, provider)
    assert result.count == 1
    reset_runtime_flags_for_scope(RuntimeFlagLifecycleScope.CLI_RUN)


def test_save_transform_completion_and_success_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    tr = SaveTransform()
    assert tr.completion_reference_keys() == ["target"]
    assert tr.completion_value_candidates("alias", "m", ["model", "other"]) == ["model"]
    assert tr.completion_value_candidates("format", "d", []) == ["dcp"]
    assert tr.completion_value_candidates("unknown", "x", []) is None
    assert tr.contributes_output_model(object()) is False

    provider = InMemoryStateDictProvider({}, max_io_workers=2)
    sd = provider.get_or_create_alias_state_dict("model")
    sd["x"] = torch.ones(1)

    persisted: list[Path] = []
    monkeypatch.setattr(
        "brainsurgery.transforms.save.persist_state_dict",
        lambda *args, **kwargs: persisted.append(kwargs["output_path"]),
    )
    state_spec = SaveSpec(
        path=Path("/tmp/out.safetensors"),
        alias="model",
        tensor_name=None,
        format=None,
        shard_size=None,
    )
    result = tr.apply(state_spec, provider)
    assert result.count == 1
    assert persisted == [Path("/tmp/out.safetensors")]

    saved: list[Path] = []
    monkeypatch.setattr(
        "brainsurgery.transforms.save.save_tensor_to_path",
        lambda name, tensor, path, format: saved.append(path),
    )
    tensor_spec = SaveSpec(
        path=Path("/tmp/out.npy"), alias="model", tensor_name="x", format="numpy", shard_size=None
    )
    result = tr.apply(tensor_spec, provider)
    assert result.count == 1
    assert saved == [Path("/tmp/out.npy")]

    with pytest.raises(SaveTransformError, match="cannot infer output model"):
        tr._infer_output_model(
            SaveSpec(
                path=Path("/tmp/o"), alias=None, tensor_name=None, format=None, shard_size=None
            )
        )
