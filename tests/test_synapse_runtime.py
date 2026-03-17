from __future__ import annotations

from pathlib import Path

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
