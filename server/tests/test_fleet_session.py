"""Fleet adapter tests: the per-request streaming facade over the engine.

Exercises ``FleetSession`` — the Session-contract shim (``generate`` /
``generate_ids`` + stats) over the shared continuous-batching engine thread:
single-stream parity, concurrent streams that actually batch together,
abort-on-close, EOS/stop semantics, and loud failures.
"""

from __future__ import annotations

import time

import pytest

torch = pytest.importorskip("torch")

from server.engine import GenerateParams
from server.fleet.session import FleetSession
from server.tests.test_fleet_runner import MODEL_CONFIG, MockContextModel

VOCAB = 8


class MockTokenizer:
    """Id i <-> letter i of the alphabet; single-char tokens decode cleanly."""

    alphabet = "abcdefgh"

    def encode(self, text: str) -> list[int]:
        return [self.alphabet.index(ch) for ch in text]

    def decode(self, ids) -> str:  # noqa: ANN001
        return "".join(self.alphabet[i] for i in ids)


def make_session(**kwargs) -> FleetSession:
    model = MockContextModel(vocab=VOCAB)
    session = FleetSession(
        model,
        model_config=dict(MODEL_CONFIG),
        tokenizer=MockTokenizer(),
        num_kvcache_blocks=16,
        max_num_batched_tokens=4096,
        **kwargs,
    )
    return session


def drain(iterator) -> list:  # noqa: ANN001
    return list(iterator)


def test_single_stream_matches_greedy_cycle_and_fills_stats() -> None:
    with make_session() as session:
        stats: dict = {}
        tokens = drain(session.generate_ids([0], GenerateParams(max_new_tokens=4, temperature=0.0), stats))
        assert tokens == [1, 2, 3, 4]
        assert stats == {
            "prompt_tokens": 1,
            "completion_tokens": 4,
            "finish_reason": "length",
        }


def test_generate_streams_text_deltas() -> None:
    with make_session() as session:
        stats: dict = {}
        text = "".join(session.generate("abc", GenerateParams(max_new_tokens=3, temperature=0.0), stats))
        assert text == "def"
        assert stats["finish_reason"] == "length"
        assert stats["completion_tokens"] == 3


def test_eos_is_consumed_but_counted() -> None:
    with make_session(eos=5) as session:
        stats: dict = {}
        tokens = drain(session.generate_ids([0], GenerateParams(max_new_tokens=16, temperature=0.0), stats))
        assert tokens == [1, 2, 3, 4]  # 5 is EOS: consumed, never yielded
        assert stats["completion_tokens"] == 5  # EOS counted (usage accounting)
        assert stats["finish_reason"] == "stop"


def test_stop_string_ends_stream_and_overrides_finish_reason() -> None:
    with make_session() as session:
        stats: dict = {}
        chunks = list(session.generate("a", GenerateParams(max_new_tokens=8, temperature=0.0, stop=("c",)), stats))
        assert chunks == ["b"]  # "c" is the stop string: truncated away, stream ends
        assert stats["finish_reason"] == "stop"


def test_zero_token_budget_yields_nothing() -> None:
    with make_session() as session:
        stats: dict = {}
        tokens = drain(session.generate_ids([0], GenerateParams(max_new_tokens=0, temperature=0.0), stats))
        assert tokens == []
        assert stats["completion_tokens"] == 0
        assert stats["finish_reason"] == "length"


def test_seed_raises_loudly() -> None:
    with make_session() as session:
        with pytest.raises(ValueError, match="seed"):
            drain(session.generate_ids([0], GenerateParams(max_new_tokens=2, seed=7)))


def test_close_aborts_an_in_flight_request() -> None:
    session = make_session()
    stream = session.generate_ids([0], GenerateParams(max_new_tokens=4096, temperature=0.0))
    assert next(stream) == 1
    stream.close()  # consumer gone -> abort inside the shared engine
    deadline = time.monotonic() + 5.0
    while not session.engine.is_finished() and time.monotonic() < deadline:
        time.sleep(0.001)
    assert session.engine.is_finished(), "aborted request still scheduled"
    session.close()


def test_two_concurrent_streams_batch_and_match_single_stream() -> None:
    prompts = [[0], [3]]
    with make_session() as session:
        # Single-stream references (sequential, batch of one).
        references = [
            drain(session.generate_ids(prompt, GenerateParams(max_new_tokens=6, temperature=0.0)))
            for prompt in prompts
        ]
        # Now two streams alive at once: alternate pulls so both stay open.
        streams = [
            session.generate_ids(prompt, GenerateParams(max_new_tokens=6, temperature=0.0))
            for prompt in prompts
        ]
        seen_batch_of_two = False
        calls_before = len(session.engine.model_runner.model.calls)
        outputs: list[list[int]] = [[], []]
        alive = {0: streams[0], 1: streams[1]}
        while alive:
            for index in list(alive):
                try:
                    outputs[index].append(next(alive[index]))
                except StopIteration:
                    del alive[index]
            batched = any(
                not call["is_prefill"] and len(call["ids"]) == 2
                for call in session.engine.model_runner.model.calls[calls_before:]
            )
            seen_batch_of_two = seen_batch_of_two or batched
        assert outputs == references, "concurrent streams drifted from single-stream parity"
        assert seen_batch_of_two, "streams never decoded together — no continuous batching"


@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
def test_engine_failure_reaches_subscribers_as_error() -> None:
    model = MockContextModel(vocab=VOCAB)
    session = FleetSession(
        model,
        model_config=dict(MODEL_CONFIG),
        tokenizer=MockTokenizer(),
        num_kvcache_blocks=16,
        max_num_batched_tokens=4096,
    )
    try:
        # Sabotage the model so the next forward raises inside the engine thread.
        def boom(input_ids, positions):  # noqa: ANN001
            raise RuntimeError("kaboom")

        model.forward = boom
        stream = session.generate_ids([0], GenerateParams(max_new_tokens=2, temperature=0.0))
        with pytest.raises(RuntimeError, match="kaboom"):
            drain(stream)
    finally:
        session.close()