"""The agnosticism proof: the identical Session engine over a torch backend.

These tests mirror the core engine tests in ``test_engine.py`` but the mock
speaks torch tensors instead of numpy-via-mlx. No engine changes are allowed
to be needed — that is the assertion this file exists to make.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from server.backends import TorchBackend
from server.engine import DecodeState, GenerateParams, Session


class TorchMockModel(torch.nn.Module):
    """The 'increment' model on torch tensors: favored next = last + 1 (mod vocab).

    A real ``nn.Module`` because generated torch models are: the ``TorchBackend``
    calls ``.to(device).eval()`` like it does on any generated model.
    """

    def __init__(self, vocab: int = 6, eos_token_id: int | None = 5) -> None:
        super().__init__()
        self.vocab = vocab
        self.config: dict = {}
        if eos_token_id is not None:
            self.config["eos_token_id"] = eos_token_id
        self.forward_inputs: list[torch.Tensor] = []

    def forward(self, input_ids=None, past_kv=None, use_cache=True):  # noqa: ANN001
        ids = torch.as_tensor(input_ids)
        self.forward_inputs.append(ids.clone())
        last = int(ids.reshape(-1)[-1])
        logits = torch.full((1, int(ids.shape[1]), self.vocab), -10.0)
        logits[0, -1, (last + 1) % self.vocab] = 10.0
        return {"logits": logits, "new_kv": ("cache", len(self.forward_inputs))}


def make_torch_session(model) -> Session:
    return Session(TorchBackend(model, device="cpu"), tokenizer=None)


def test_torch_backend_prefill_and_step_chain() -> None:
    model = TorchMockModel(vocab=6, eos_token_id=None)
    session = make_torch_session(model)
    state = session.prefill([0])
    assert state.prompt_len == 1
    assert state.token_ids == [0]
    assert state.cache is not None

    params = GenerateParams(max_new_tokens=3, temperature=0.0)
    tokens = []
    for _ in range(params.max_new_tokens):
        token = session.step(state, params)
        if token is None:
            break
        tokens.append(token)
    assert tokens == [1, 2, 3]
    assert [t.reshape(-1).tolist() for t in model.forward_inputs] == [[0], [1], [2]]


def test_torch_backend_eos_and_max_new_tokens() -> None:
    model = TorchMockModel(vocab=6, eos_token_id=3)
    session = make_torch_session(model)
    tokens = list(session.generate_ids([0], GenerateParams(max_new_tokens=10, temperature=0.0)))
    assert tokens == [1, 2]
    assert len(model.forward_inputs) == 3

    model = TorchMockModel(vocab=6, eos_token_id=None)
    session = make_torch_session(model)
    tokens = list(session.generate_ids([0], GenerateParams(max_new_tokens=4, temperature=0.0)))
    assert tokens == [1, 2, 3, 4]


def test_torch_backend_stats_match_mlx_accounting() -> None:
    model = TorchMockModel(vocab=6, eos_token_id=3)
    session = make_torch_session(model)
    stats: dict = {}
    list(session.generate_ids([0], GenerateParams(max_new_tokens=10, temperature=0.0), stats))
    # Same accounting as the MLX path: EOS charged as a generated token.
    assert stats == {"prompt_tokens": 1, "completion_tokens": 3, "finish_reason": "stop"}


def test_torch_backend_decode_state_defaults() -> None:
    assert DecodeState().finished is False
