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
                        "_op": "embedding",
                        "_args": "input_ids",
                        "_bind": "x",
                        "dim": "D",
                    }
                },
                {
                    "lm_head": {
                        "_op": "linear",
                        "_args": "x",
                        "_bind": "logits",
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
    q_node: dict[str, object] = {"_op": "reshape_heads", "_args": "q", "_bind": "qh"}
    k_node: dict[str, object] = {"_op": "reshape_heads", "_args": "k", "_bind": "kh"}
    v_node: dict[str, object] = {"_op": "reshape_heads", "_args": "v", "_bind": "vh"}
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
        "_op": "reshape_heads",
        "_args": "x",
        "_bind": "xh",
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
                        "_op": "causal_mask",
                        "_args": "q",
                        "key": "k",
                        "padding_mask": "padding_mask",
                        "window": 8,
                        "_bind": "mask",
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
                        "_op": "position_ids",
                        "_args": ["input_ids", "attention_mask"],
                        "_bind": "pos",
                    }
                }
            ],
            "outputs": {"pos": "pos"},
        },
    }


def _coalesce_spec() -> dict[str, object]:
    return {
        "synapse": 1,
        "model": {
            "inputs": {"a": {}, "b": {}, "c": {}, "d": {}},
            "graph": [
                {
                    "n": {
                        "_op": "coalesce",
                        "_args": ["a", "b", "c", "d"],
                        "_bind": ["o1", "o2"],
                    }
                }
            ],
            "outputs": {"o1": "o1", "o2": "o2"},
        },
    }


def _moe_select_tokens_spec(*, expert: object = 1) -> dict[str, object]:
    return {
        "synapse": 1,
        "model": {
            "inputs": {"x": {}, "scores": {}, "idx": {}},
            "graph": [
                {
                    "sel": {
                        "_op": "moe_select_tokens",
                        "_args": ["x", "scores", "idx"],
                        "_bind": ["x_sel", "token_idx", "topk_pos", "sel_scores"],
                        "expert": expert,
                    }
                }
            ],
            "outputs": {
                "x_sel": "x_sel",
                "token_idx": "token_idx",
                "topk_pos": "topk_pos",
                "sel_scores": "sel_scores",
            },
        },
    }


def _moe_scatter_add_spec() -> dict[str, object]:
    return {
        "synapse": 1,
        "model": {
            "inputs": {"m": {}, "token_idx": {}, "upd": {}, "scores": {}},
            "graph": [
                {
                    "scatter": {
                        "_op": "moe_scatter_add",
                        "_args": ["m", "token_idx", "upd", "scores"],
                        "_bind": "m_out",
                    }
                }
            ],
            "outputs": {"m_out": "m_out"},
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
        _op: embedding
        _args: input_ids
        _bind: x
        dim: D
    - lm_head:
        _op: linear
        _args: x
        _bind: logits
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


def test_runtime_coalesce_uses_grouped_fallback() -> None:
    spec = _coalesce_spec()
    model = SynapseProgramModel.from_spec(spec)
    out = model(a=None, b=torch.tensor(2), c=torch.tensor(3), d=None)
    assert torch.equal(out["o1"], torch.tensor(3))
    assert torch.equal(out["o2"], torch.tensor(2))


def test_runtime_coalesce_raises_for_missing_candidate() -> None:
    spec = _coalesce_spec()
    graph = spec["model"]["graph"]
    assert isinstance(graph, list)
    node = graph[0]["n"]
    assert isinstance(node, dict)
    node["_args"] = ["a", "b", "missing", "d"]
    model = SynapseProgramModel.from_spec(spec)
    with pytest.raises(ValueError, match="coalesce candidate 'missing' missing in env"):
        model(a=None, b=torch.tensor(2), c=None, d=None)


def test_runtime_moe_select_tokens_selects_routed_rows() -> None:
    spec = _moe_select_tokens_spec(expert=1)
    model = SynapseProgramModel.from_spec(spec)
    x = torch.tensor([[[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]]])
    scores = torch.tensor([[[0.7, 0.3], [0.2, 0.8], [0.6, 0.4]]])
    idx = torch.tensor([[[1, 0], [2, 1], [1, 2]]], dtype=torch.long)

    out = model(x=x, scores=scores, idx=idx)
    assert torch.equal(out["token_idx"], torch.tensor([0, 1, 2], dtype=torch.long))
    assert torch.equal(out["topk_pos"], torch.tensor([0, 1, 0], dtype=torch.long))
    assert torch.equal(out["x_sel"], torch.tensor([[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]]))
    assert torch.allclose(out["sel_scores"], torch.tensor([0.7, 0.8, 0.6], dtype=torch.float32))


def test_runtime_moe_select_tokens_allows_empty_selection() -> None:
    spec = _moe_select_tokens_spec(expert=9)
    model = SynapseProgramModel.from_spec(spec)
    x = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    scores = torch.tensor([[[0.6, 0.4], [0.2, 0.8]]])
    idx = torch.tensor([[[1, 2], [2, 1]]], dtype=torch.long)

    out = model(x=x, scores=scores, idx=idx)
    assert out["x_sel"].shape == (0, 2)
    assert out["token_idx"].numel() == 0
    assert out["topk_pos"].numel() == 0
    assert out["sel_scores"].numel() == 0


def test_runtime_moe_select_tokens_validates_flattened_token_alignment() -> None:
    spec = _moe_select_tokens_spec(expert=1)
    model = SynapseProgramModel.from_spec(spec)
    x = torch.randn(1, 3, 4)
    scores = torch.randn(1, 2, 2)
    idx = torch.tensor([[[1, 0], [0, 1]]], dtype=torch.long)
    with pytest.raises(
        ValueError,
        match="moe_select_tokens hidden and topk tensors must align on flattened token count",
    ):
        model(x=x, scores=scores, idx=idx)


def test_runtime_moe_select_tokens_validates_index_dtype() -> None:
    spec = _moe_select_tokens_spec(expert=1)
    model = SynapseProgramModel.from_spec(spec)
    x = torch.randn(1, 2, 4)
    scores = torch.randn(1, 2, 2)
    idx = torch.randn(1, 2, 2)
    with pytest.raises(
        ValueError, match=r"moe_select_tokens topk_indices must be an integer tensor"
    ):
        model(x=x, scores=scores, idx=idx)


def test_runtime_moe_select_tokens_requires_integral_expert() -> None:
    spec = _moe_select_tokens_spec(expert=1.5)
    model = SynapseProgramModel.from_spec(spec)
    x = torch.randn(1, 2, 4)
    scores = torch.randn(1, 2, 2)
    idx = torch.tensor([[[1, 0], [0, 1]]], dtype=torch.long)
    with pytest.raises(ValueError, match=r"moe_select_tokens expert must evaluate to an integer"):
        model(x=x, scores=scores, idx=idx)


def test_runtime_moe_scatter_add_accumulates_weighted_updates() -> None:
    spec = _moe_scatter_add_spec()
    model = SynapseProgramModel.from_spec(spec)
    m = torch.zeros(1, 3, 2)
    token_idx = torch.tensor([0, 2, 2], dtype=torch.long)
    upd = torch.tensor([[1.0, 2.0], [10.0, 20.0], [30.0, 40.0]])
    scores = torch.tensor([1.0, 0.5, 0.25])

    out = model(m=m.clone(), token_idx=token_idx, upd=upd, scores=scores)
    expected = m.reshape(-1, 2)
    expected[0] += torch.tensor([1.0, 2.0])
    expected[2] += torch.tensor([5.0, 10.0])
    expected[2] += torch.tensor([7.5, 10.0])
    assert torch.allclose(out["m_out"].reshape(-1, 2), expected)


def test_runtime_moe_scatter_add_empty_indices_is_noop() -> None:
    spec = _moe_scatter_add_spec()
    model = SynapseProgramModel.from_spec(spec)
    m = torch.randn(1, 2, 3)
    token_idx = torch.zeros((0,), dtype=torch.long)
    upd = torch.zeros((0, 3))
    scores = torch.zeros((0,))
    out = model(m=m.clone(), token_idx=token_idx, upd=upd, scores=scores)
    assert torch.equal(out["m_out"], m)


def test_runtime_moe_scatter_add_validates_alignment_and_dtypes() -> None:
    spec = _moe_scatter_add_spec()
    model = SynapseProgramModel.from_spec(spec)
    m = torch.zeros(1, 2, 4)
    with pytest.raises(ValueError, match=r"moe_scatter_add token_idx must be an integer tensor"):
        model(
            m=m.clone(),
            token_idx=torch.tensor([0.0, 1.0]),
            upd=torch.randn(2, 4),
            scores=torch.ones(2),
        )
    with pytest.raises(
        ValueError,
        match="moe_scatter_add token_idx, updates, and scores must align on row count",
    ):
        model(
            m=m.clone(),
            token_idx=torch.tensor([0, 1], dtype=torch.long),
            upd=torch.randn(3, 4),
            scores=torch.ones(2),
        )


def test_runtime_moe_scatter_add_validates_token_index_bounds() -> None:
    spec = _moe_scatter_add_spec()
    model = SynapseProgramModel.from_spec(spec)
    with pytest.raises(ValueError, match="moe_scatter_add token_idx contains out-of-range values"):
        model(
            m=torch.zeros(1, 2, 3),
            token_idx=torch.tensor([3], dtype=torch.long),
            upd=torch.randn(1, 3),
            scores=torch.ones(1),
        )


def test_runtime_validates_input_rank_from_shape_spec() -> None:
    spec = {
        "synapse": 1,
        "model": {
            "symbols": {},
            "inputs": {"x": {"shape": ["B", "T", "D"]}},
            "graph": [],
            "outputs": {"x": "x"},
        },
    }
    model = SynapseProgramModel.from_spec(spec)
    with pytest.raises(ValueError, match="rank mismatch"):
        model(x=torch.randn(2, 3))


def test_runtime_validates_symbolic_shape_consistency_across_inputs() -> None:
    spec = {
        "synapse": 1,
        "model": {
            "symbols": {},
            "inputs": {
                "x": {"shape": ["B", "T", "D"]},
                "mask": {"shape": ["B", "T"]},
            },
            "graph": [],
            "outputs": {"x": "x"},
        },
    }
    model = SynapseProgramModel.from_spec(spec)
    with pytest.raises(ValueError, match="symbol T was previously bound"):
        model(x=torch.randn(2, 3, 4), mask=torch.ones(2, 5, dtype=torch.long))


def test_runtime_validates_numeric_symbol_dims_in_input_specs() -> None:
    spec = {
        "synapse": 1,
        "model": {
            "symbols": {"D": 4},
            "inputs": {"x": {"shape": ["B", "T", "D"]}},
            "graph": [],
            "outputs": {"x": "x"},
        },
    }
    model = SynapseProgramModel.from_spec(spec)
    with pytest.raises(ValueError, match="expected symbol D=4"):
        model(x=torch.randn(2, 3, 5))


def test_runtime_linear_handles_empty_batch_without_kernel_work() -> None:
    spec = {
        "synapse": 1,
        "model": {
            "inputs": {"x": {}},
            "graph": [{"n": {"_op": "linear", "_args": "x", "_bind": "y", "bias": False}}],
            "outputs": {"y": "y"},
        },
    }
    model = SynapseProgramModel.from_spec(spec)
    model.load_state_dict_tensors({"n.weight": torch.randn(8, 4)})
    x = torch.empty((0, 4), dtype=torch.float32)
    out = model(x=x)
    assert out["y"].shape == (0, 8)
