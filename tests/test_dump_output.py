from __future__ import annotations

import json

import pytest
import torch

import brainsurgery.transforms.dump as dump_module
from brainsurgery.transforms.dump import DumpTransform


class _Provider:
    def __init__(self, state_dict: dict[str, torch.Tensor]) -> None:
        self._state_dict = state_dict

    def get_state_dict(self, model: str):
        assert model == "model"
        return self._state_dict


@pytest.fixture
def sample_state_dict() -> dict[str, torch.Tensor]:
    # 0 and 1 are intentionally identical; 2 is different so compact list grouping
    # should group [0-1] but keep [2] separate.
    return {
        "block.0.weight": torch.tensor([1.0, 2.0]),
        "block.1.weight": torch.tensor([1.0, 2.0]),
        "block.2.weight": torch.tensor([9.0, 8.0, 7.0]),
    }


def _run_dump(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    state_dict: dict[str, torch.Tensor],
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
    sample_state_dict: dict[str, torch.Tensor],
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
    sample_state_dict: dict[str, torch.Tensor],
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
    sample_state_dict: dict[str, torch.Tensor],
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
        assert "dtype=" not in output
        assert "device=" not in output
        assert "values=" not in output
    elif verbosity == "stat":
        assert "min=" in output
        assert "max=" in output
        assert "mean=" in output
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
        assert "min=" not in output
        assert "max=" not in output
        assert "mean=" not in output


@pytest.mark.parametrize("verbosity", ["shape", "stat", "full"])
def test_dump_json_output_respects_verbosity(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    sample_state_dict: dict[str, torch.Tensor],
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
        assert set(leaf0.keys()) == {"shape", "min", "max", "mean"}
        assert leaf0["shape"] == [2]
        assert leaf0["min"] == 1.0
        assert leaf0["max"] == 2.0
        assert leaf0["mean"] == 1.5
    else:
        assert set(leaf0.keys()) == {"shape", "dtype", "device", "values"}
        assert leaf0["shape"] == [2]
        assert leaf0["dtype"] == "torch.float32"
        assert leaf0["device"] == "cpu"
        assert leaf0["values"] == [1.0, 2.0]
