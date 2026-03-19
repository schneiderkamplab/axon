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
                        "dim": "V",
                        "bias": False,
                        "weight": "embed_tokens.weight",
                    }
                },
            ],
            "outputs": {"logits": "logits"},
        },
    }


def _reshape_triplet_lowered_spec(
    *, heads: int | None = None, head_dim: int | None = None
) -> dict[str, object]:
    q_node: dict[str, object] = {"op": "reshape_heads", "in": "q", "out": "qh"}
    k_node: dict[str, object] = {"op": "reshape_heads", "in": "k", "out": "kh"}
    v_node: dict[str, object] = {"op": "reshape_heads", "in": "v", "out": "vh"}
    if heads is not None:
        q_node["heads"] = heads
        k_node["heads"] = heads
        v_node["heads"] = heads
    if head_dim is not None:
        q_node["head_dim"] = head_dim
        k_node["head_dim"] = head_dim
        v_node["head_dim"] = head_dim
    return {
        "synapse": 1,
        "model": {
            "inputs": {"q": {}, "k": {}, "v": {}},
            "graph": [{"q": q_node}, {"k": k_node}, {"v": v_node}],
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


def _causal_mask_with_padding_spec() -> dict[str, object]:
    return {
        "synapse": 1,
        "model": {
            "inputs": {"q": {}, "k": {}, "padding_mask": {}},
            "graph": [
                {
                    "m": {
                        "op": "causal_mask",
                        "in": "q",
                        "key": "k",
                        "padding_mask": "padding_mask",
                        "window": 8,
                        "out": "mask",
                    }
                }
            ],
            "outputs": {"mask": "mask"},
        },
    }


def _arange_positions_with_mask_spec() -> dict[str, object]:
    return {
        "synapse": 1,
        "model": {
            "inputs": {"input_ids": {}, "attention_mask": {"optional": True}},
            "graph": [
                {
                    "p": {
                        "op": "arange_positions",
                        "in": "input_ids",
                        "attention_mask": "attention_mask",
                        "out": "pos",
                    }
                }
            ],
            "outputs": {"pos": "pos"},
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
        dim: V
        bias: false
        weight: embed_tokens.weight
  outputs:
    logits: logits
""",
        encoding="utf-8",
    )
    model_from_yaml = SynapseProgramModel.from_yaml(spec_path, state_dict=state_dict)
    logits_yaml = model_from_yaml(input_ids)
    assert logits_yaml.shape == (2, 3, 8)


def test_runtime_reshape_heads_triplet_lowering_equivalent_infers_head_dim_from_heads() -> None:
    spec = _reshape_triplet_lowered_spec(heads=12)
    model = SynapseProgramModel.from_spec(spec)
    q = torch.randn(2, 5, 768)
    out = model(q=q, k=q, v=q)
    assert out["qh"].shape == (2, 12, 5, 64)
    assert out["kh"].shape == (2, 12, 5, 64)
    assert out["vh"].shape == (2, 12, 5, 64)


def test_runtime_reshape_heads_triplet_lowering_equivalent_infers_heads_from_head_dim() -> None:
    spec = _reshape_triplet_lowered_spec(head_dim=64)
    model = SynapseProgramModel.from_spec(spec)
    q = torch.randn(2, 5, 768)
    out = model(q=q, k=q, v=q)
    assert out["qh"].shape == (2, 12, 5, 64)
    assert out["kh"].shape == (2, 12, 5, 64)
    assert out["vh"].shape == (2, 12, 5, 64)


def test_runtime_reshape_heads_triplet_lowering_equivalent_requires_heads_or_head_dim() -> None:
    spec = _reshape_triplet_lowered_spec()
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


def test_runtime_causal_mask_combines_padding_mask() -> None:
    spec = _causal_mask_with_padding_spec()
    model = SynapseProgramModel.from_spec(spec)
    q = torch.randn(2, 12, 4, 8)
    k = torch.randn(2, 12, 4, 8)
    padding_mask = torch.tensor(
        [[1, 1, 1, 1], [0, 0, 1, 1]],
        dtype=torch.long,
    )
    out = model(q=q, k=k, padding_mask=padding_mask)
    mask = out["mask"]
    assert mask.shape == (2, 1, 4, 4)
    assert torch.isfinite(mask[0]).all()
    assert torch.isfinite(mask[1, :, :, 2:]).all()
    assert torch.all(mask[1, :, :, :2] < -1.0e20)


def test_runtime_arange_positions_uses_attention_mask_for_left_padding() -> None:
    spec = _arange_positions_with_mask_spec()
    model = SynapseProgramModel.from_spec(spec)
    input_ids = torch.tensor([[10, 11, 12, 13], [0, 0, 20, 21]], dtype=torch.long)
    attention_mask = torch.tensor([[1, 1, 1, 1], [0, 0, 1, 1]], dtype=torch.long)
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    pos = out["pos"]
    assert torch.equal(pos[0], torch.tensor([0, 1, 2, 3], dtype=torch.long))
    assert torch.equal(pos[1], torch.tensor([0, 0, 0, 1], dtype=torch.long))


def test_runtime_linear_handles_empty_batch_without_kernel_work() -> None:
    spec = {
        "synapse": 1,
        "model": {
            "inputs": {"x": {}},
            "graph": [{"n": {"op": "linear", "in": "x", "out": "y", "bias": False}}],
            "outputs": {"y": "y"},
        },
    }
    model = SynapseProgramModel.from_spec(spec)
    model.load_state_dict_tensors({"n.weight": torch.randn(8, 4)})
    x = torch.empty((0, 4), dtype=torch.float32)
    out = model(x=x)
    assert out["y"].shape == (0, 8)
