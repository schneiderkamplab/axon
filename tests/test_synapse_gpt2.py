from __future__ import annotations

import torch

from brainsurgery.synapse import SynapseGPT2LMHeadModel, build_gpt2_from_synapse_spec


def _tiny_spec() -> dict[str, object]:
    return {
        "synapse": 1,
        "model": {
            "symbols": {
                "D": 16,
                "V": 32,
                "C": 12,
                "L": 2,
                "H": 4,
                "M": 64,
            },
            "params": {
                "activation": "gelu_new",
                "layer_norm_epsilon": 1e-5,
                "attn_backend": "sdpa",
            },
        },
    }


def test_build_from_synapse_spec_has_gpt2_tensor_names() -> None:
    model = build_gpt2_from_synapse_spec(_tiny_spec())
    keys = set(model.state_dict().keys())

    assert "wte.weight" in keys
    assert "wpe.weight" in keys
    assert "h.0.attn.bias" in keys
    assert "h.0.attn.c_attn.weight" in keys
    assert "h.0.attn.c_attn.bias" in keys
    assert "h.0.attn.c_proj.weight" in keys
    assert "h.0.mlp.c_fc.weight" in keys
    assert "h.0.mlp.c_proj.weight" in keys
    assert "ln_f.weight" in keys
    assert "ln_f.bias" in keys


def test_from_state_dict_and_forward() -> None:
    spec = _tiny_spec()
    base = build_gpt2_from_synapse_spec(spec)
    state_dict = {
        key: torch.randn_like(value) if value.is_floating_point() else value.clone()
        for key, value in base.state_dict().items()
    }

    model = SynapseGPT2LMHeadModel.from_state_dict(state_dict, spec)

    input_ids = torch.randint(low=0, high=32, size=(2, 5), dtype=torch.long)
    logits = model(input_ids)
    assert logits.shape == (2, 5, 32)


def test_generate_stops_on_eos_and_max_len() -> None:
    spec = _tiny_spec()
    model = build_gpt2_from_synapse_spec(spec)
    for value in model.state_dict().values():
        if value.is_floating_point():
            value.zero_()

    start = torch.tensor([[1, 2]], dtype=torch.long)
    generated_eos = model.generate(start, eos_token_id=0, max_len=10)
    assert generated_eos.shape == (1, 3)
    assert generated_eos[0, -1].item() == 0

    generated_max = model.generate(start, eos_token_id=31, max_len=5)
    assert generated_max.shape == (1, 5)


def test_forward_with_past_updates_kv_cache() -> None:
    model = build_gpt2_from_synapse_spec(_tiny_spec())
    input_ids = torch.randint(low=0, high=32, size=(1, 4), dtype=torch.long)

    logits_full, past = model.forward_with_past(input_ids)
    assert logits_full.shape == (1, 4, 32)
    assert past is not None
    assert len(past) == 2
    assert past[0][0].shape[-2] == 4
    assert past[0][1].shape[-2] == 4

    next_token = torch.randint(low=0, high=32, size=(1, 1), dtype=torch.long)
    logits_step, past2 = model.forward_with_past(next_token, past_key_values=past)
    assert logits_step.shape == (1, 1, 32)
    assert past2 is not None
    assert past2[0][0].shape[-2] == 5
    assert past2[0][1].shape[-2] == 5
