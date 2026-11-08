from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import AutoTokenizer, Qwen3Config, Qwen3Model
from transformers.masking_utils import create_bidirectional_mask

from synapse.axon.codegen2_mlx import emit_model_code_from_graph_ir, torch_state_dict_to_mlx
from tests.test_pplx_embed_support import GENERIC_EMBED, _mlx_graph

mx = pytest.importorskip("mlx.core", exc_type=ImportError)


def test_public_pplx_embeddings_match_reference(pplx_embed_local_path: Path) -> None:
    """Check the public 0.6B checkpoint against bidirectional Qwen3 + masked mean."""
    payload = json.loads((pplx_embed_local_path / "config.json").read_text())
    config = Qwen3Config.from_dict({
        key: value for key, value in payload.items()
        if key not in {"model_type", "architectures", "auto_map"}
    })
    reference = Qwen3Model.from_pretrained(
        pplx_embed_local_path, config=config, dtype=torch.float32, local_files_only=True
    ).eval()
    for layer in reference.layers:
        layer.self_attn.is_causal = False
    tokenizer = AutoTokenizer.from_pretrained(pplx_embed_local_path, local_files_only=True)

    namespace = {}
    exec(emit_model_code_from_graph_ir(_mlx_graph(GENERIC_EMBED)), namespace)
    model = namespace["AxonMlxModel"].from_state_dict(
        torch_state_dict_to_mlx(dict(reference.state_dict())), model_config=payload
    )
    texts = [
        "The cat sits on the mat.",
        "Paris is the capital of France, and the Seine flows through the city.",
    ]
    for batch, padding_side in ((texts[:1], "right"), (texts, "right"), (texts, "left")):
        tokenizer.padding_side = padding_side
        tokens = tokenizer(batch, padding=True, return_tensors="pt")
        ids, mask = tokens["input_ids"], tokens["attention_mask"]
        with torch.no_grad():
            inputs_embeds = reference.embed_tokens(ids)
            attention_mask = create_bidirectional_mask(
                config=reference.config, inputs_embeds=inputs_embeds,
                attention_mask=mask, allow_is_bidirectional_skip=False,
            )
            hidden = reference(
                inputs_embeds=inputs_embeds, attention_mask=attention_mask, use_cache=False
            ).last_hidden_state
            keep = mask.unsqueeze(-1).float()
            expected = (hidden * keep).sum(dim=1) / keep.sum(dim=1)
        actual = model.forward(input_ids=mx.array(ids.numpy()), attn_mask=mx.array(mask.numpy()))
        mx.eval(actual)
        assert actual.shape == (len(batch), config.hidden_size)
        np.testing.assert_allclose(np.array(actual), expected.numpy(), atol=1e-4, rtol=1e-4)
