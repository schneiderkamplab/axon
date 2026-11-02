"""Fleet streaming adapter: per-request streams over one shared engine.

``FleetSession`` mirrors ``Session``'s per-request streaming contract
(``generate`` / ``generate_ids`` with a stats dict) on top of the vendored
continuous-batching engine (``LLMEngine``), so the HTTP layer drives either
engine through the same shape and the OpenAI surface is identical.

Concurrency model: ONE engine thread owns all scheduler state. Requests
submit ``("add", ...)`` commands through a queue and consume their own token
events from a per-request subscriber queue; the engine thread publishes the
newly sampled tokens after every step, so concurrent streams batch
continuously (the 2c gate). Aborts — the consumer closed the stream — are
commands too, and a ``_Ticket`` carries the request identity so an abort
that arrives before the scheduler has even seen the request still cancels
it (abort-before-schedule).

No silent fallbacks: unsupported request features raise loudly (``seed``),
and an engine-thread failure is pushed to every live subscriber as an error
event before the thread dies.
"""

from __future__ import annotations

import queue
import threading
from typing import Any, Iterator

from ..detokenize import stream_text
from ..engine import GenerateParams
from .engine import LLMEngine
from .sampling_params import SamplingParams

__all__ = ["FleetSession"]


class _Ticket:
    """Request identity handed to the engine thread; resolved lazily.

    ``seq_id`` is set by the engine thread when the add command runs; an
    abort that arrives earlier marks the ticket cancelled, and the add is
    skipped entirely when the engine thread reaches it.
    """

    __slots__ = ("seq_id", "cancelled", "ignore_eos")

    def __init__(self, ignore_eos: bool) -> None:
        self.seq_id: int | None = None
        self.cancelled = False
        self.ignore_eos = ignore_eos


class FleetSession:
    """Per-request streaming facade over one shared, batched ``LLMEngine``."""

    def __init__(
        self,
        model: Any,
        *,
        model_config: dict[str, Any],
        tokenizer: Any | None = None,
        **config_kwargs: Any,
    ) -> None:
        self.engine = LLMEngine(model, model_config=model_config, tokenizer=tokenizer, **config_kwargs)
        self._commands: queue.SimpleQueue = queue.SimpleQueue()
        # Subscriber queues and tickets live ONLY on the engine thread
        # (registered in _add, removed at finish/abort) — no lock needed.
        self._subscribers: dict[int, queue.Queue] = {}
        self._tickets: dict[int, _Ticket] = {}
        self._pending = 0  # commands enqueued but not yet drained
        self._stopping = False
        self._cond = threading.Condition()
        self._thread = threading.Thread(target=self._loop, name="fleet-engine", daemon=True)
        self._thread.start()

    @property
    def tokenizer(self) -> Any | None:
        """Uniform surface with ``Session``: the HTTP layer reads it for chat templates."""
        return self.engine.tokenizer

    # ── per-request streaming contract (mirrors Session) ────────────────────

    def generate(
        self,
        prompt: str | list[int],
        params: GenerateParams | None = None,
        stats: dict[str, Any] | None = None,
    ) -> Iterator[str]:
        """Stream decoded completion text chunks as generation proceeds.

        Same contract as ``Session.generate``: deltas come from full
        re-decode, a stop-string truncation ends the stream (overriding
        ``finish_reason`` to ``"stop"``), and closing the generator early
        aborts the request inside the shared engine.
        """
        if self.engine.tokenizer is None:
            raise RuntimeError("FleetSession requires a tokenizer to stream text")
        params = params or GenerateParams()
        token_stream = self.generate_ids(prompt, params, stats)
        yield from stream_text(self.engine.tokenizer, token_stream, params.stop, stats)

    def generate_ids(
        self,
        prompt: str | list[int],
        params: GenerateParams | None = None,
        stats: dict[str, Any] | None = None,
    ) -> Iterator[int]:
        """Yield sampled token ids one at a time (excludes the prompt).

        Fills ``stats`` exactly like ``Session.generate_ids``:
        ``prompt_tokens`` up front; ``completion_tokens`` (including a
        terminating EOS, which is consumed, never yielded) and
        ``finish_reason`` (``"stop"`` on EOS, ``"length"`` when the budget
        is exhausted) as the request completes.
        """
        params = params or GenerateParams()
        sampling = self._sampling_params(params)
        prompt_ids = self._encode(prompt)
        if stats is not None:
            stats["prompt_tokens"] = len(prompt_ids)
        ticket, events = self._submit(prompt_ids, sampling)
        completion_tokens = 0
        try:
            while True:
                kind, payload = events.get()
                if kind == "error":
                    raise RuntimeError(f"fleet engine failed: {payload}")
                if kind == "done":
                    if stats is not None and payload != "aborted":
                        stats["completion_tokens"] = completion_tokens
                        stats["finish_reason"] = payload
                    return
                if completion_tokens >= sampling.max_tokens:
                    continue  # zero-token budget: the prefill still yields one spurious token
                completion_tokens += 1
                if payload == self.eos and not sampling.ignore_eos:
                    continue  # EOS consumed, never yielded (counted above)
                yield payload
        finally:
            self._cancel(ticket)

    # ── lifecycle ───────────────────────────────────────────────────────────

    def close(self) -> None:
        """Stop the engine thread. In-flight requests are abandoned."""
        with self._cond:
            self._stopping = True
            self._cond.notify()
        self._thread.join()

    def __enter__(self) -> FleetSession:
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    @property
    def eos(self) -> int:
        return self.engine.config.eos

    # ── internals: request plumbing ─────────────────────────────────────────

    def _sampling_params(self, params: GenerateParams) -> SamplingParams:
        if params.seed is not None:
            raise ValueError("seed is not supported by the fleet engine yet")
        return SamplingParams(
            temperature=params.temperature,
            max_tokens=params.max_new_tokens,
            top_k=params.top_k,
            top_p=params.top_p,
        )

    def _encode(self, prompt: str | list[int]) -> list[int]:
        if isinstance(prompt, str):
            if self.engine.tokenizer is None:
                raise RuntimeError("String prompts require a tokenizer")
            return list(self.engine.tokenizer.encode(prompt))
        return [int(token) for token in prompt]

    def _submit(
        self, prompt_ids: list[int], sampling: SamplingParams
    ) -> tuple[_Ticket, queue.Queue]:
        events: queue.Queue = queue.Queue()
        ticket = _Ticket(sampling.ignore_eos)
        with self._cond:
            self._pending += 1
            self._commands.put(("add", ticket, prompt_ids, sampling, events))
            self._cond.notify()
        return ticket, events

    def _cancel(self, ticket: _Ticket) -> None:
        with self._cond:
            self._pending += 1
            self._commands.put(("abort", ticket))
            self._cond.notify()

    # ── internals: the engine thread ────────────────────────────────────────

    def _loop(self) -> None:
        while True:
            self._drain()
            if self._stopping:
                break
            if self.engine.is_finished():
                with self._cond:
                    if self._pending > 0 or self._stopping:
                        continue
                    self._cond.wait()
                continue
            try:
                finished, new_tokens, _ = self.engine.step_stream()
            except Exception as error:  # noqa: BLE001 — fail every live stream loudly
                self._fail_all(error)
                raise
            self._publish(finished, new_tokens)

    def _drain(self) -> None:
        while True:
            try:
                command = self._commands.get_nowait()
            except queue.Empty:
                return
            with self._cond:
                self._pending -= 1
            kind = command[0]
            if kind == "add":
                self._add(command)
            elif kind == "abort":
                self._abort(command)
            else:  # pragma: no cover — commands are internal
                raise RuntimeError(f"unknown fleet command {kind!r}")

    def _add(self, command: tuple[Any, ...]) -> None:
        _, ticket, prompt_ids, sampling, events = command
        if ticket.cancelled:
            events.put(("done", "aborted"))
            return
        seq = self.engine.add_request(prompt_ids, sampling)
        ticket.seq_id = seq.seq_id
        self._subscribers[seq.seq_id] = events
        self._tickets[seq.seq_id] = ticket

    def _abort(self, command: tuple[Any, ...]) -> None:
        _, ticket = command
        if ticket.cancelled:
            return
        if ticket.seq_id is None:
            ticket.cancelled = True  # add not drained yet; it will no-op
            return
        if self.engine.scheduler.abort(ticket.seq_id):
            self._finish(ticket.seq_id, "aborted")

    def _publish(
        self,
        finished: list[tuple[int, list[int]]],
        new_tokens: list[tuple[int, int]],
    ) -> None:
        for seq_id, token in new_tokens:
            events = self._subscribers.get(seq_id)
            if events is not None:
                events.put(("token", token))
        for seq_id, completion in finished:
            ticket = self._tickets.get(seq_id)
            stopped_on_eos = (
                ticket is not None
                and not ticket.ignore_eos
                and completion[-1] == self.eos
            )
            self._finish(seq_id, "stop" if stopped_on_eos else "length")

    def _finish(self, seq_id: int, reason: str) -> None:
        events = self._subscribers.pop(seq_id, None)
        self._tickets.pop(seq_id, None)
        if events is not None:
            events.put(("done", reason))

    def _fail_all(self, error: Exception) -> None:
        for seq_id in list(self._subscribers):
            events = self._subscribers.pop(seq_id)
            events.put(("error", f"{type(error).__name__}: {error}"))
        self._tickets.clear()
