from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

import brainsurgery.transforms.dump as dump_module
from brainsurgery.engine.state_dicts import _InMemoryStateDict
from brainsurgery.transforms.dump import DumpTransform


class _Provider:
    def __init__(self, state_dict) -> None:
        self._state_dict = state_dict

    def get_state_dict(self, model: str):
        assert model == "model"
        return self._state_dict


class _MultiProvider:
    def __init__(self, state_dicts: dict[str, _InMemoryStateDict]) -> None:
        self.state_dicts = state_dicts
        self.model_paths: dict[str, Path] = {}

    def get_state_dict(self, model: str):
        return self.state_dicts[model]

    def list_model_aliases(self) -> set[str]:
        return set(self.state_dicts)


class _NoAliasProvider:
    def get_state_dict(self, model: str):  # pragma: no cover - should never be called
        raise AssertionError(f"unexpected state_dict access for {model!r}")


@pytest.fixture
def sample_state_dict() -> _InMemoryStateDict:
    # 0 and 1 are intentionally identical; 2 is different so compact list grouping
    # should group [0-1] but keep [2] separate.
    state_dict = _InMemoryStateDict()
    state_dict["block.0.weight"] = torch.tensor([1.0, 2.0])
    state_dict["block.1.weight"] = torch.tensor([1.0, 2.0])
    state_dict["block.2.weight"] = torch.tensor([9.0, 8.0, 7.0])
    return state_dict


def _run_dump(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    state_dict: _InMemoryStateDict,
    *,
    fmt: str,
    verbosity: str,
) -> str:
    transform = DumpTransform()
    spec = transform.compile(
        {
            "target": "block\\..*\\.weight",
            "format": fmt,
            "verbosity": verbosity,
        },
        default_model="model",
    )
    # Make output deterministic and avoid progress-bar artifacts.
    monkeypatch.setattr(dump_module, "tqdm", lambda iterable, **_: iterable)
    result = transform.apply(spec, _Provider(state_dict))
    assert result.count == 3
    return capsys.readouterr().out.strip()


@pytest.mark.parametrize("verbosity", ["shape", "stat", "full"])
def test_dump_tree_does_not_group_list_entries(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    sample_state_dict: _InMemoryStateDict,
    verbosity: str,
) -> None:
    output = _run_dump(
        monkeypatch,
        capsys,
        sample_state_dict,
        fmt="tree",
        verbosity=verbosity,
    )
    assert "[0-1]" not in output
    assert "[0]" in output
    assert "[1]" in output
    assert "[2]" in output


@pytest.mark.parametrize("verbosity", ["shape", "stat", "full"])
def test_dump_compact_groups_list_entries_with_same_structure(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    sample_state_dict: _InMemoryStateDict,
    verbosity: str,
) -> None:
    output = _run_dump(
        monkeypatch,
        capsys,
        sample_state_dict,
        fmt="compact",
        verbosity=verbosity,
    )
    assert "[0-1]" in output
    assert "[2]" in output
    assert "[0]\n" not in output
    assert "[1]\n" not in output


@pytest.mark.parametrize("fmt", ["tree", "compact"])
@pytest.mark.parametrize("verbosity", ["shape", "stat", "full"])
def test_dump_text_output_respects_verbosity(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    sample_state_dict: _InMemoryStateDict,
    fmt: str,
    verbosity: str,
) -> None:
    output = _run_dump(
        monkeypatch,
        capsys,
        sample_state_dict,
        fmt=fmt,
        verbosity=verbosity,
    )
    assert "shape=[" in output
    if verbosity == "shape":
        assert "min=" not in output
        assert "max=" not in output
        assert "mean=" not in output
        assert "reads=" not in output
        assert "writes=" not in output
        assert "dtype=" not in output
        assert "device=" not in output
        assert "values=" not in output
    elif verbosity == "stat":
        assert "min=" in output
        assert "max=" in output
        assert "mean=" in output
        assert "reads=" in output
        assert "writes=" in output
        assert "dtype=" not in output
        assert "device=" not in output
        assert "values=" not in output
        # First tensor is [1, 2] -> min=1 max=2 mean=1.5
        assert "min=1" in output
        assert "max=2" in output
        assert "mean=1.5" in output
    else:
        assert "dtype=torch.float32" in output
        assert "device=cpu" in output
        assert "values=" in output
        assert "reads=" in output
        assert "writes=" in output
        assert "min=" not in output
        assert "max=" not in output
        assert "mean=" not in output


@pytest.mark.parametrize("verbosity", ["shape", "stat", "full"])
def test_dump_json_output_respects_verbosity(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    sample_state_dict: _InMemoryStateDict,
    verbosity: str,
) -> None:
    output = _run_dump(
        monkeypatch,
        capsys,
        sample_state_dict,
        fmt="json",
        verbosity=verbosity,
    )
    payload = json.loads(output)
    leaf0 = payload["block"][0]["weight"]
    leaf1 = payload["block"][1]["weight"]
    leaf2 = payload["block"][2]["weight"]

    if verbosity == "shape":
        assert set(leaf0.keys()) == {"shape"}
        assert leaf0["shape"] == [2]
        assert leaf1["shape"] == [2]
        assert leaf2["shape"] == [3]
    elif verbosity == "stat":
        assert set(leaf0.keys()) == {"shape", "min", "max", "mean", "reads", "writes"}
        assert leaf0["shape"] == [2]
        assert leaf0["min"] == 1.0
        assert leaf0["max"] == 2.0
        assert leaf0["mean"] == 1.5
        assert leaf0["reads"] >= 1
        assert leaf0["writes"] == 1
    else:
        assert set(leaf0.keys()) == {"shape", "dtype", "device", "values", "reads", "writes"}
        assert leaf0["shape"] == [2]
        assert leaf0["dtype"] == "torch.float32"
        assert leaf0["device"] == "cpu"
        assert leaf0["values"] == [1.0, 2.0]
        assert leaf0["reads"] >= 1
        assert leaf0["writes"] == 1


def test_dump_without_target_dumps_all_models_as_root_nodes(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    base = _InMemoryStateDict()
    base["ln_f.weight"] = torch.ones(2)
    edited = _InMemoryStateDict()
    edited["ln_f.weight"] = torch.zeros(2)

    provider = _MultiProvider({"base": base, "edited": edited})
    transform = DumpTransform()
    spec = transform.compile(
        {"format": "json", "verbosity": "shape"},
        default_model=None,
    )

    monkeypatch.setattr(dump_module, "tqdm", lambda iterable, **_: iterable)
    result = transform.apply(spec, provider)
    assert result.count == 2

    lines = [line for line in capsys.readouterr().out.strip().splitlines() if line]
    assert len(lines) == 2
    payloads = [json.loads(line) for line in lines]
    merged: dict[str, object] = {}
    for payload in payloads:
        assert len(payload) == 1
        merged.update(payload)
    assert set(merged.keys()) == {"base", "edited"}
    assert merged["base"]["ln_f"]["weight"]["shape"] == [2]
    assert merged["edited"]["ln_f"]["weight"]["shape"] == [2]


def test_dump_without_target_and_no_aliases_is_empty_output(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    transform = DumpTransform()
    spec = transform.compile(
        {"format": "json", "verbosity": "shape"},
        default_model=None,
    )
    monkeypatch.setattr(dump_module, "tqdm", lambda iterable, **_: iterable)
    result = transform.apply(spec, _NoAliasProvider())
    assert result.count == 0
    assert json.loads(capsys.readouterr().out.strip()) == {}


def test_dump_without_target_text_does_not_connect_model_roots(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    base = _InMemoryStateDict()
    base["w"] = torch.ones(1)
    edited = _InMemoryStateDict()
    edited["w"] = torch.zeros(1)
    provider = _MultiProvider({"model": base, "orig": edited})

    transform = DumpTransform()
    spec = transform.compile(
        {"format": "compact", "verbosity": "shape"},
        default_model=None,
    )
    monkeypatch.setattr(dump_module, "tqdm", lambda iterable, **_: iterable)
    result = transform.apply(spec, provider)
    assert result.count == 2

    output = capsys.readouterr().out.strip()
    assert "└── model" in output
    assert "└── orig" in output
    assert "├── orig" not in output
    assert "\n\n└── orig" in output
