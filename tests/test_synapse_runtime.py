from __future__ import annotations

from pathlib import Path

import pytest
import torch

from brainsurgery.synapse.runtime import SynapseProgramModel


def _tiny_linear_spec() -> dict[str, object]:
    return {
        "synapse": 1,
        "model": {
            "symbols": {"B": None, "T": None, "V": 8, "D": 4},
            "inputs": {"input_ids": {"shape": ["B", "T"], "dtype": "int64"}},
            "graph": [
                {
                    "embed_tokens": {
                        "op": "embedding",
                        "in": "input_ids",
                        "out": "x",
                        "num_embeddings": "V",
                        "embedding_dim": "D",
                    }
                },
                {
                    "lm_head": {
                        "op": "linear",
                        "in": "x",
                        "out": "logits",
                        "out_features": "V",
                        "bias": False,
                        "tie_weight": "embed_tokens.weight",
                    }
                },
            ],
            "outputs": {"logits": "logits"},
        },
    }


def _reshape_triplet_spec(
    *, heads: int | None = None, head_dim: int | None = None
) -> dict[str, object]:
    node: dict[str, object] = {
        "op": "reshape_heads_triplet",
        "in": ["q", "k", "v"],
        "out": ["qh", "kh", "vh"],
    }
    if heads is not None:
        node["heads"] = heads
    if head_dim is not None:
        node["head_dim"] = head_dim
    return {
        "synapse": 1,
        "model": {
            "inputs": {"q": {}, "k": {}, "v": {}},
            "graph": [{"r": node}],
            "outputs": {"qh": "qh", "kh": "kh", "vh": "vh"},
        },
    }


def _reshape_heads_spec(
    *, heads: int | None = None, head_dim: int | None = None
) -> dict[str, object]:
    node: dict[str, object] = {
        "op": "reshape_heads",
        "in": "x",
        "out": "xh",
    }
    if heads is not None:
        node["heads"] = heads
    if head_dim is not None:
        node["head_dim"] = head_dim
    return {
        "synapse": 1,
        "model": {
            "inputs": {"x": {}},
            "graph": [{"r": node}],
            "outputs": {"xh": "xh"},
        },
    }


def test_runtime_from_spec_and_from_yaml(tmp_path: Path) -> None:
    spec = _tiny_linear_spec()
    model = SynapseProgramModel.from_spec(spec)

    state_dict = {
        "embed_tokens.weight": torch.randn(8, 4),
    }
    model.load_state_dict_tensors(state_dict)

    input_ids = torch.randint(low=0, high=8, size=(2, 3), dtype=torch.long)
    logits = model(input_ids)
    assert logits.shape == (2, 3, 8)

    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(
        """synapse: 1
model:
  symbols:
    B: null
    T: null
    V: 8
    D: 4
  graph:
    - embed_tokens:
        op: embedding
        in: input_ids
        out: x
        num_embeddings: V
        embedding_dim: D
    - lm_head:
        op: linear
        in: x
        out: logits
        out_features: V
        bias: false
        tie_weight: embed_tokens.weight
  outputs:
    logits: logits
""",
        encoding="utf-8",
    )
    model_from_yaml = SynapseProgramModel.from_yaml(spec_path, state_dict=state_dict)
    logits_yaml = model_from_yaml(input_ids)
    assert logits_yaml.shape == (2, 3, 8)


def test_runtime_reshape_heads_triplet_infers_head_dim_from_heads() -> None:
    spec = _reshape_triplet_spec(heads=12)
    model = SynapseProgramModel.from_spec(spec)
    q = torch.randn(2, 5, 768)
    out = model(q=q, k=q, v=q)
    assert out["qh"].shape == (2, 12, 5, 64)
    assert out["kh"].shape == (2, 12, 5, 64)
    assert out["vh"].shape == (2, 12, 5, 64)


def test_runtime_reshape_heads_triplet_infers_heads_from_head_dim() -> None:
    spec = _reshape_triplet_spec(head_dim=64)
    model = SynapseProgramModel.from_spec(spec)
    q = torch.randn(2, 5, 768)
    out = model(q=q, k=q, v=q)
    assert out["qh"].shape == (2, 12, 5, 64)
    assert out["kh"].shape == (2, 12, 5, 64)
    assert out["vh"].shape == (2, 12, 5, 64)


def test_runtime_reshape_heads_triplet_requires_heads_or_head_dim() -> None:
    spec = _reshape_triplet_spec()
    model = SynapseProgramModel.from_spec(spec)
    q = torch.randn(2, 5, 768)
    with pytest.raises(ValueError, match="requires heads or head_dim"):
        model(q=q, k=q, v=q)


def test_runtime_reshape_heads_infers_head_dim_from_heads() -> None:
    spec = _reshape_heads_spec(heads=12)
    model = SynapseProgramModel.from_spec(spec)
    x = torch.randn(2, 5, 768)
    out = model(x=x)
    assert out["xh"].shape == (2, 12, 5, 64)


def test_runtime_reshape_heads_infers_heads_from_head_dim() -> None:
    spec = _reshape_heads_spec(head_dim=64)
    model = SynapseProgramModel.from_spec(spec)
    x = torch.randn(2, 5, 768)
    out = model(x=x)
    assert out["xh"].shape == (2, 12, 5, 64)


def test_runtime_reshape_heads_requires_heads_or_head_dim() -> None:
    spec = _reshape_heads_spec()
    model = SynapseProgramModel.from_spec(spec)
    x = torch.randn(2, 5, 768)
    with pytest.raises(ValueError, match="requires heads or head_dim"):
        model(x=x)
