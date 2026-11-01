"""Engine loop tests against a mock model (no MLX or checkpoints needed).

The mock mirrors the generated-model contract: ``forward(input_ids, past_kv=,
use_cache=)`` returns ``{"logits": ..., "new_kv": ...}``, so the Session's
cache threading, EOS handling, and stop truncation are exercised exactly as
they will run against real generated MLX models.
"""

from __future__ import annotations

import numpy as np
import pytest

from server.backends import MlxBackend
from server.engine import DecodeState, GenerateParams, Session


class _FakeMx:
    """Minimal stand-in for mlx.core used by the Session internals."""

    int64 = np.int64

    @staticmethod
    def array(data, dtype=None):  # noqa: ANN001 - mirrors mx.array signature
        return np.asarray(data, dtype=np.int64)


class MockModel:
    """Deterministic 'increment' model: favored next token = last + 1 (mod vocab)."""

    def __init__(self, vocab: int = 6, eos_token_id: int | None = 5) -> None:
        self.vocab = vocab
        self.config: dict = {}
        if eos_token_id is not None:
            self.config["eos_token_id"] = eos_token_id
        self.forward_inputs: list[np.ndarray] = []

    def forward(self, input_ids, past_kv=None, use_cache=True):  # noqa: ANN001
        ids = np.asarray(input_ids)
        self.forward_inputs.append(ids.copy())
        last = int(ids.reshape(-1)[-1])
        logits = np.full((1, int(ids.shape[1]), self.vocab), -10.0)
        logits[0, -1, (last + 1) % self.vocab] = 10.0
        return {"logits": logits, "new_kv": ("cache", len(self.forward_inputs))}


class FakeTokenizer:
    """Maps token ids to letters so stop truncation is observable."""

    eos_token_id = None

    def encode(self, text: str) -> list[int]:
        return [ord(ch) - 65 for ch in text.upper()]

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(65 + int(i)) for i in ids)


@pytest.fixture()
def patched_mx(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(MlxBackend, "_mx", lambda self: _FakeMx)


def make_mlx_session(model, tokenizer=None) -> Session:
    return Session(MlxBackend(model, warm_compile=False), tokenizer=tokenizer)


def test_prefill_and_step_thread_cache_and_greedy_chain(patched_mx) -> None:
    model = MockModel(vocab=6, eos_token_id=None)  # no EOS: runs to max_new_tokens
    session = make_mlx_session(model)
    state = session.prefill([0])
    assert state.prompt_len == 1
    assert state.token_ids == [0]
    assert state.cache is not None

    params = GenerateParams(max_new_tokens=3, temperature=0.0)
    tokens = []
    # step() terminates only on EOS; the caller bounds the loop (server semantics).
    for _ in range(params.max_new_tokens):
        token = session.step(state, params)
        if token is None:
            break
        tokens.append(token)
    assert tokens == [1, 2, 3]
    # Prefill consumed the full prompt; every step consumed exactly one token.
    assert [np.asarray(x).reshape(-1).tolist() for x in model.forward_inputs] == [
        [0],
        [1],
        [2],
    ]


def test_generate_ids_stops_at_eos(patched_mx) -> None:
    # Chain 0 -> 1 -> 2 -> 3, with 3 as EOS: yields [1, 2] and stops.
    model = MockModel(vocab=6, eos_token_id=3)
    session = make_mlx_session(model)
    tokens = list(session.generate_ids([0], GenerateParams(max_new_tokens=10, temperature=0.0)))
    assert tokens == [1, 2]
    # Three forwards: prefill + two steps. EOS is discovered from the logits of
    # the final forward; the EOS token itself is consumed, never yielded.
    assert len(model.forward_inputs) == 3


def test_generate_respects_max_new_tokens_without_eos(patched_mx) -> None:
    model = MockModel(vocab=6, eos_token_id=None)
    session = make_mlx_session(model)
    tokens = list(session.generate_ids([0], GenerateParams(max_new_tokens=4, temperature=0.0)))
    assert tokens == [1, 2, 3, 4]


def test_generate_streams_text_and_truncates_at_stop(patched_mx) -> None:
    # vocab large enough that (last+1)%vocab walks A->B->C->D->E->F
    model = MockModel(vocab=8, eos_token_id=None)
    session = make_mlx_session(model, FakeTokenizer())
    # Prompt "A" (id 0) -> generated B, C, D, ...
    text = "".join(
        session.generate("A", GenerateParams(max_new_tokens=4, temperature=0.0, stop=("D",)))
    )
    assert text == "BC"  # truncated before the stop sequence "D"


def test_generate_ids_stats_report_length_without_eos(patched_mx) -> None:
    model = MockModel(vocab=6, eos_token_id=None)
    session = make_mlx_session(model)
    stats: dict = {}
    tokens = list(
        session.generate_ids([0], GenerateParams(max_new_tokens=4, temperature=0.0), stats)
    )
    assert tokens == [1, 2, 3, 4]
    assert stats == {"prompt_tokens": 1, "completion_tokens": 4, "finish_reason": "length"}


def test_generate_ids_stats_report_stop_on_eos(patched_mx) -> None:
    model = MockModel(vocab=6, eos_token_id=3)
    session = make_mlx_session(model)
    stats: dict = {}
    tokens = list(
        session.generate_ids([0], GenerateParams(max_new_tokens=10, temperature=0.0), stats)
    )
    assert tokens == [1, 2]
    # EOS is charged as a generated token (vLLM accounting) though not yielded.
    assert stats == {"prompt_tokens": 1, "completion_tokens": 3, "finish_reason": "stop"}


def test_generate_stats_stop_string_overrides_finish_reason(patched_mx) -> None:
    # No EOS and the budget not exhausted, yet a stop-string hit must report
    # "stop", overriding the "length" that generate_ids' finally wrote.
    model = MockModel(vocab=8, eos_token_id=None)
    session = make_mlx_session(model, FakeTokenizer())
    stats: dict = {}
    text = "".join(
        session.generate(
            "A", GenerateParams(max_new_tokens=4, temperature=0.0, stop=("D",)), stats
        )
    )
    assert text == "BC"
    assert stats == {"prompt_tokens": 1, "completion_tokens": 3, "finish_reason": "stop"}


def test_decode_state_defaults(patched_mx) -> None:
    state = DecodeState()
    assert state.token_ids == []
    assert state.finished is False
    assert state.cache is None
    assert state.pending_logits is None
