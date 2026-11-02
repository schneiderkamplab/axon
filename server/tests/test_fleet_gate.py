"""Fleet gate: the acceptance criteria for engine integration.

The fleet earns its keep only if, over ONE shared engine:
1. concurrent streams produce the same tokens as each stream run alone
   (batched decode must not change semantics),
2. concurrent streams actually share decode steps (batch width > 1 observed),
3. a later request with an identical prompt prefix hits the prefix cache
   (its prefill starts at the cached boundary) and matches the first
   request's continuation,
4. aggregate throughput beats the Session shim: the fleet carries every
   scheduled sequence in one step, the shim runs one sequence per step.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from server.engine import GenerateParams, Session
from server.fleet.session import FleetSession
from server.tests.test_engine import (  # noqa: F401 - patched_mx is a fixture
    FakeTokenizer,
    MockModel,
    make_mlx_session,
    patched_mx,
)
from server.tests.test_fleet_runner import MODEL_CONFIG, MockContextModel
from server.tests.test_fleet_session import MockTokenizer

VOCAB = 8


class _PricedFleetModel(MockContextModel):
    """Fleet mock with a fixed per-forward matmul cost.

    At zero forward cost the fleet's per-token cross-thread handoff dominates
    and the batching gain is invisible; real serving runs where one step's
    model cost dwarfs the handoff. The gate prices both mocks into that
    regime (~1-2 ms per forward) so the comparison measures batching, not
    queue overhead at a degenerate scale.
    """

    def __init__(self, vocab: int = VOCAB, num_layers: int = 2) -> None:
        super().__init__(vocab=vocab, num_layers=num_layers)
        self._burn = torch.rand(1024, 1024)

    def forward(self, input_ids, positions):  # noqa: ANN001
        self._burn @ self._burn
        return super().forward(input_ids, positions)


class _PricedShimModel(MockModel):
    """Session-shim mock priced into the same regime as ``_PricedFleetModel``."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._burn = np.random.rand(1024, 1024).astype(np.float32)

    def forward(self, input_ids, past_kv=None, use_cache=True):  # noqa: ANN001
        self._burn @ self._burn
        return super().forward(input_ids, past_kv, use_cache)


def make_session(**kwargs) -> FleetSession:
    model = _PricedFleetModel(vocab=VOCAB)
    return FleetSession(
        model,
        model_config=dict(MODEL_CONFIG),
        tokenizer=MockTokenizer(),
        num_kvcache_blocks=16,
        max_num_batched_tokens=4096,
        **kwargs,
    )


PROMPTS = ([0], [3], [6], [2])


def test_gate_concurrent_streams_match_solo_and_batch() -> None:
    params = GenerateParams(max_new_tokens=8, temperature=0.0)
    with make_session() as session:
        solo = {
            tuple(prompt): list(session.generate_ids(list(prompt), params))
            for prompt in PROMPTS
        }
        model = session.engine.model_runner.model
        calls_before = len(model.calls)

        results: dict[tuple[int, ...], list[int]] = {}

        def run(prompt: tuple[int, ...]) -> None:
            results[prompt] = list(session.generate_ids(list(prompt), params))

        threads = [threading.Thread(target=run, args=(tuple(prompt),)) for prompt in PROMPTS]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # Batched decode must not change semantics: same tokens as solo.
        assert results == solo
        # And the streams really shared steps: some decode batch carried
        # more than one sequence.
        decode_widths = [
            len(call["ids"])
            for call in model.calls[calls_before:]
            if not call["is_prefill"]
        ]
        assert max(decode_widths) >= 2


def test_gate_identical_prefix_hits_cache_and_matches_continuation() -> None:
    params = GenerateParams(max_new_tokens=2, temperature=0.0)
    prompt = list(range(300))  # > one 256-token block so a full block can hash
    with make_session() as session:
        stats_a: dict = {}
        stream_a = session.generate_ids(list(prompt), params, stats_a)
        first_a = next(stream_a)  # A has prefilled and hashed its blocks; keep it decoding

        stats_b: dict = {}
        tokens_b = list(session.generate_ids(list(prompt), params, stats_b))
        tokens_a = [first_a, *stream_a]

    assert tokens_b == tokens_a  # cache-correct: same prompt, same continuation
    assert stats_a["completion_tokens"] == stats_b["completion_tokens"] == 2

    prefills = [
        call for call in session.engine.model_runner.model.calls if call["is_prefill"]
    ]
    assert len(prefills) == 2
    assert prefills[0]["positions"][0] == 0
    # B's prefill starts at the cached boundary: only the tail is computed.
    assert prefills[1]["positions"][0] == 256
    assert prefills[1]["ids"] == prompt[256:]


def test_gate_fleet_outperforms_session_shim(patched_mx) -> None:
    max_new = 48

    # Session shim under the server's own concurrency model: a bounded
    # semaphore, one decode step per sequence at a time.
    shim_model = _PricedShimModel(vocab=VOCAB, eos_token_id=None)
    shim = make_mlx_session(shim_model, tokenizer=FakeTokenizer())
    slots = threading.BoundedSemaphore(2)
    shim_out: dict[tuple[int, ...], list[int]] = {}

    def run_shim(prompt: tuple[int, ...]) -> None:
        with slots:
            shim_out[prompt] = list(
                shim.generate_ids(list(prompt), GenerateParams(max_new_tokens=max_new, temperature=0.0))
            )

    threads = [threading.Thread(target=run_shim, args=(tuple(p),)) for p in PROMPTS]
    start = time.perf_counter()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    shim_seconds = time.perf_counter() - start
    shim_steps = len(shim_model.forward_inputs)  # one forward per step per request

    with make_session() as session:
        fleet_out: dict[tuple[int, ...], list[int]] = {}

        def run_fleet(prompt: tuple[int, ...]) -> None:
            fleet_out[prompt] = list(
                session.generate_ids(list(prompt), GenerateParams(max_new_tokens=max_new, temperature=0.0))
            )

        threads = [threading.Thread(target=run_fleet, args=(tuple(p),)) for p in PROMPTS]
        start = time.perf_counter()
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        fleet_seconds = time.perf_counter() - start
        fleet_steps = len(session.engine.model_runner.model.calls)

    # Parity across engines first: the shim and the fleet must agree token
    # for token on the same greedy chains.
    assert fleet_out == shim_out
    # The mechanism: the fleet batches concurrent sequences into one step,
    # the shim spends one forward per sequence per step. 4 requests x 48
    # tokens: the shim does ~4x48 forwards, the fleet ~48 batched steps.
    assert fleet_steps < shim_steps, (fleet_steps, shim_steps)
    # And the claim the gate exists for: strictly faster wall clock.
    assert fleet_seconds < shim_seconds, (fleet_seconds, shim_seconds)
