"""Shared incremental detokenization for the two engine shims.

The Session shim (Track 1) and the fleet adapter (Track 2c) stream text
deltas with the same contract — full re-decode, safe-prefix emission,
stop-string truncation — so this module is the one source of truth for that
behavior and the OpenAI surface sees identical output shape from either
engine.
"""

from __future__ import annotations

from typing import Any, Iterator


def truncate_at_stop(text: str, stops: tuple[str, ...]) -> str | None:
    """Return text truncated before the first stop sequence, or None."""
    for stop in stops:
        if stop and stop in text:
            return text.split(stop, 1)[0]
    return None


def stream_text(
    tokenizer: Any,
    token_stream: Iterator[int],
    stop: tuple[str, ...],
    stats: dict[str, Any] | None = None,
) -> Iterator[str]:
    """Consume a token-id iterator, yield decoded text deltas.

    The shared streaming contract: full re-decode each token, emit only the
    not-yet-emitted suffix, and end generation at the first stop string (the
    truncated tail is still yielded). When ``stats`` is given, a stop-string
    truncation overrides ``finish_reason`` to ``"stop"``. The token stream is
    closed deterministically so its cleanup ``finally`` runs even when a
    stop string ends generation early.
    """
    stream = TextStream(tokenizer, stop)
    for token in token_stream:
        delta = stream.feed(token)
        if delta:
            yield delta
        if stream.stopped:
            break
    token_stream.close()
    if stats is not None and stream.stopped:
        stats["finish_reason"] = "stop"


def truncate_at_stop(text: str, stops: tuple[str, ...]) -> str | None:
    """Return text truncated before the first stop sequence, or None."""
    for stop in stops:
        if stop and stop in text:
            return text.split(stop, 1)[0]
    return None


class TextStream:
    """Token ids in, safe text deltas out.

    ``feed`` decodes everything generated so far and returns only the
    not-yet-emitted suffix ("" when the tokenizer hasn't produced new safe
    text yet, e.g. mid multi-byte character). A ``None`` return signals a
    stop-string hit: the delta was truncated and generation must end — the
    same semantics Session's generate loop has always had.
    """

    def __init__(self, tokenizer: Any, stop: tuple[str, ...] = ()) -> None:
        self._tokenizer = tokenizer
        self._stop = tuple(stop)
        self._generated: list[int] = []
        self._emitted = 0
        self._stopped = False

    def feed(self, token: int) -> str:
        self._generated.append(int(token))
        text = self._tokenizer.decode(self._generated)
        delta = text[self._emitted :]
        self._emitted += len(delta)
        cut = truncate_at_stop(delta, self._stop)
        if cut is not None:
            self._stopped = True
            return cut
        return delta

    @property
    def stopped(self) -> bool:
        """A stop string truncated the stream; the caller must end generation."""
        return self._stopped
