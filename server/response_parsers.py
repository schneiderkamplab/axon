"""Shared response parsers: reasoning + tool-call extraction from model output.

Reference implementation for wiki/response-parsers.md. The parser sits
downstream of the incremental detokenizer: it consumes decoded text deltas
and emits an ordered event stream (reasoning_delta / content_delta /
tool_call_begin / tool_args_delta / tool_call_end) that the HTTP layer maps
mechanically onto OpenAI ``reasoning_content`` / ``tool_calls`` fields.

Design rules (from the wiki page):

- Formats, not models: ``config.json`` declares ``reasoning_parser`` /
  ``tool_call_parser`` keys selecting generic implementations from
  ``REASONING_PARSERS`` / ``TOOL_PARSERS``; absent keys mean raw text
  passthrough. No model- or family-name branches anywhere.
- Parse decoded text, not token ids: Bifrost declares zero special tokens,
  so every marker is plain text split arbitrarily across BPE tokens. Single
  token specials (harmony) flow through the same text matcher.
- Streaming uses holdback: never emit a suffix that could be the prefix of
  any declared marker; flush on a full match or when continuation becomes
  impossible.

The think parser contract (Step 1): the open tag is what classifies
reasoning. Output that never contains the open tag is all content (Qwen3
non-thinking mode); open-without-close at EOS is all reasoning (truncated
think); a close tag without a prior open is stripped, not reclassified
(bare-reasoning-then-close, DeepSeek-R1 style, is a distinct format deferred
until such a checkpoint is actually hosted).

Tool parsers (Step 2) own their own marker segmenter (tool open/close) and
route everything outside a tool block to the reasoning parser, so a
(reasoning, tools) combo behaves as one coherent state machine in stream
order rather than two parsers racing over the same text. Hermes blocks wrap
a JSON object: the head is buffered until the ``"arguments"`` key completes
so the name is extractable, then the arguments value streams as deltas by
JSON-aware brace counting (string/escape aware). gemma-4 blocks use the
chat template's exact markers; the args region is gemma-style serialization
(``key:<|"|>value<|"|>``), streamed raw, with the terminating brace found by
brace counting that treats the ``<|"|>`` quote token as a string toggle.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Iterator

# ── event model ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ReasoningDelta:
    text: str


@dataclass(frozen=True)
class ContentDelta:
    text: str


@dataclass(frozen=True)
class ToolCallBegin:
    index: int
    name: str


@dataclass(frozen=True)
class ToolArgsDelta:
    index: int
    text: str


@dataclass(frozen=True)
class ToolCallEnd:
    index: int


Event = ReasoningDelta | ContentDelta | ToolCallBegin | ToolArgsDelta | ToolCallEnd


def event_to_json(event: Event) -> list[Any]:
    """Serialize an event as ``[kind, payload]`` (vectors + parity dumps)."""
    if isinstance(event, ReasoningDelta):
        return ["reasoning_delta", event.text]
    if isinstance(event, ContentDelta):
        return ["content_delta", event.text]
    if isinstance(event, ToolCallBegin):
        return ["tool_call_begin", event.index, event.name]
    if isinstance(event, ToolArgsDelta):
        return ["tool_args_delta", event.index, event.text]
    if isinstance(event, ToolCallEnd):
        return ["tool_call_end", event.index]
    raise TypeError(f"unknown event: {event!r}")


# ── marker segmenter (the holdback engine) ───────────────────────────────────


@dataclass(frozen=True)
class Segment:
    kind: str  # "text" or "marker"
    value: str  # text payload, or the marker's declared name


class MarkerSegmenter:
    """Split an incremental text stream on literal markers with holdback.

    Feed text deltas; receive ordered segments. Text segments carry everything
    that is provably not part of a marker yet; a suffix that could still grow
    into any marker is held back until a full match occurs or the suffix is
    ruled out by later text. ``finish`` flushes the holdback as plain text.
    """

    def __init__(self, markers: dict[str, str]) -> None:
        if not markers:
            raise ValueError("MarkerSegmenter needs at least one marker")
        self._markers = dict(markers)
        self._buf = ""

    def feed(self, text: str) -> list[Segment]:
        self._buf += text
        out: list[Segment] = []
        while True:
            hit = self._earliest_match()
            if hit is not None:
                name, start, length = hit
                if start > 0:
                    out.append(Segment("text", self._buf[:start]))
                out.append(Segment("marker", name))
                self._buf = self._buf[start + length :]
                continue
            hold = self._holdback_len()
            if hold:
                emit = self._buf[: len(self._buf) - hold]
                if emit:
                    out.append(Segment("text", emit))
                self._buf = self._buf[len(self._buf) - hold :]
            else:
                if self._buf:
                    out.append(Segment("text", self._buf))
                self._buf = ""
            return out

    def finish(self) -> list[Segment]:
        out: list[Segment] = []
        if self._buf:
            out.append(Segment("text", self._buf))
            self._buf = ""
        return out

    def flush_text(self) -> Segment | None:
        """Drain the holdback as plain text (a boundary proved it not a marker)."""
        if not self._buf:
            return None
        segment, self._buf = Segment("text", self._buf), ""
        return segment

    def _earliest_match(self) -> tuple[str, int, int] | None:
        best: tuple[str, int, int] | None = None
        for name, marker in self._markers.items():
            start = self._buf.find(marker)
            if start < 0:
                continue
            length = len(marker)
            if best is None or start < best[1] or (start == best[1] and length > best[2]):
                best = (name, start, length)
        return best

    def _holdback_len(self) -> int:
        """Length of the longest buffer suffix that is a proper marker prefix."""
        best = 0
        for marker in self._markers.values():
            top = min(len(marker) - 1, len(self._buf))
            for size in range(top, best, -1):
                if self._buf.endswith(marker[:size]):
                    best = size
                    break
        return best


# ── reasoning parsers ────────────────────────────────────────────────────────

THINK_OPEN = "\u003cthink\u003e"
THINK_CLOSE = "\u003c/think\u003e"

# The gemma-4 chat template renders the thought channel as
# '<|channel>thought\n' + text + '\n<channel|>'. The open marker stays bare
# (robust to truncated output) and the framing newlines are consumed by the
# close marker plus a one-shot leading-newline trim, so parsed
# reasoning_content re-renders byte-identically through the template.
GEMMA4_THOUGHT_OPEN = "\u003c|channel\u003ethought"
GEMMA4_THOUGHT_CLOSE = "\n\u003cchannel|\u003e"


class _ToggleReasoningParser:
    """Open/close markers toggle between content and reasoning text.

    Text before the first open marker is content; the open marker switches to
    reasoning; the close marker returns to content. A close marker with no
    prior open is stripped rather than reclassifying already-streamed text.
    Undelivered holdback at EOS flushes into the current state.
    """

    _MARKERS: dict[str, str]
    _OPEN: str
    _CLOSE: str
    _LEAD_TRIM: str | None = None  # dropped once from the first text after open

    def __init__(self) -> None:
        self._segmenter = MarkerSegmenter(self._MARKERS)
        self._state = "content"
        self._trim_pending = False

    def feed(self, text: str) -> list[Event]:
        return self._consume(self._segmenter.feed(text))

    def finish(self) -> list[Event]:
        return self._consume(self._segmenter.finish())

    def flush(self) -> list[Event]:
        """Force the holdback out as text in the current state.

        Used at tool-block boundaries: the raw stream continues with a tool
        marker, which proves any held suffix is not this format's marker and
        must not join across the removed block.
        """
        segment = self._segmenter.flush_text()
        return self._consume([segment]) if segment is not None else []

    def _consume(self, segments: list[Segment]) -> list[Event]:
        events: list[Event] = []
        for segment in segments:
            if segment.kind == "marker":
                if segment.value == self._OPEN:
                    self._state = "reasoning"
                    self._trim_pending = self._LEAD_TRIM is not None
                else:
                    self._state = "content"
                continue
            text = segment.value
            if not text:
                continue
            if self._state != "reasoning":
                events.append(ContentDelta(text))
                continue
            if self._trim_pending:
                self._trim_pending = False
                if text.startswith(self._LEAD_TRIM or ""):
                    text = text[len(self._LEAD_TRIM or "") :]
            if text:
                events.append(ReasoningDelta(text))
        return events


class ThinkReasoningParser(_ToggleReasoningParser):
    """HTML-style think tags (Qwen3 family): ``<think\u003e ... </think\u003e``."""

    _MARKERS = {"think_open": THINK_OPEN, "think_close": THINK_CLOSE}
    _OPEN = "think_open"
    _CLOSE = "think_close"


class Gemma4ThoughtParser(_ToggleReasoningParser):
    """gemma-4 thought channel: ``<|channel\u003ethought ... <channel|\u003e``."""

    _MARKERS = {
        "thought_open": GEMMA4_THOUGHT_OPEN,
        "thought_close": GEMMA4_THOUGHT_CLOSE,
    }
    _OPEN = "thought_open"
    _CLOSE = "thought_close"
    _LEAD_TRIM = "\n"


# ── tool-call parsers ────────────────────────────────────────────────────────

HERMES_OPEN = "\u003ctool_call\u003e"
HERMES_CLOSE = "\u003c/tool_call\u003e"

# chat_template.jinja renders exactly these (escape_keys=False serialization).
GEMMA4_CALL_OPEN = "\u003c|tool_call\u003ecall:"
GEMMA4_CALL_CLOSE = "\u003ctool_call|\u003e"
GEMMA4_QUOTE = "\u003c|\"|\u003e"

_JSON_ESCAPES = {
    '"': '"',
    "\\": "\\",
    "/": "/",
    "b": "\b",
    "f": "\f",
    "n": "\n",
    "r": "\r",
    "t": "\t",
}


def _json_unescape(text: str) -> str:
    """Best-effort JSON string unescape for short extracted fields."""
    out: list[str] = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == "\\" and i + 1 < len(text):
            nxt = text[i + 1]
            if nxt in _JSON_ESCAPES:
                out.append(_JSON_ESCAPES[nxt])
                i += 2
                continue
            if nxt == "u" and i + 6 <= len(text):
                try:
                    out.append(chr(int(text[i + 2 : i + 6], 16)))
                    i += 6
                    continue
                except ValueError:
                    pass
        out.append(ch)
        i += 1
    return "".join(out)


def _json_string_field(text: str, key: str) -> str:
    """Best-effort value of ``"key": "..."`` in a raw JSON-ish region."""
    match = re.search(r'"' + key + r'"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
    return _json_unescape(match.group(1)) if match else ""


class _HermesToolMachine:
    """Hermes-style tool calls: XML-ish tags wrapping one JSON object each.

    Block shape: ``<tool_call\u003e\\n{"name": ..., "arguments": {...}}\\n</tool_call\u003e``;
    multiple blocks per turn, one call each. Body text outside blocks routes
    to the reasoning parser via ``route_text``. Inside a block the head is
    buffered until the ``"arguments"`` key and its colon complete (so the
    name is extractable), the arguments value streams as ToolArgsDelta by
    JSON-aware brace counting (string/escape aware, ending at the brace that
    closes the arguments value), and anything after it up to the close marker
    is ignored. Non-object arguments fall back to streaming everything until
    the close marker. Known limits: ``name`` must precede ``arguments``, one
    object per block (no arrays), and a literal close-marker string inside
    the arguments would split the block.
    """

    _ARGS_KEY = '"arguments"'

    def __init__(
        self,
        route_text: Callable[[str], list[Event]],
        route_flush: Callable[[], list[Event]],
    ) -> None:
        self._route = route_text
        self._route_flush = route_flush
        self._segmenter = MarkerSegmenter({"tool_open": HERMES_OPEN, "tool_close": HERMES_CLOSE})
        self._index = -1
        self._state = "body"
        self._head = ""
        self._depth = 0
        self._in_string = False
        self._escape = False
        self._args_started = False
        self._args_done = False
        self._scalar = False

    def feed(self, text: str) -> list[Event]:
        events: list[Event] = []
        for segment in self._segmenter.feed(text):
            if segment.kind == "marker":
                events += self._on_marker(segment.value)
            else:
                events += self._on_text(segment.value)
        return events

    def finish(self) -> list[Event]:
        events: list[Event] = []
        for segment in self._segmenter.finish():
            if segment.kind == "marker":
                events += self._on_marker(segment.value)
            else:
                events += self._on_text(segment.value)
        if self._state == "head" and self._head.strip():
            self._index += 1
            events.append(ToolCallBegin(self._index, _json_string_field(self._head, "name")))
            events.append(ToolCallEnd(self._index))
        elif self._state in ("args", "tail"):
            events.append(ToolCallEnd(self._index))
        return events

    def _on_marker(self, name: str) -> list[Event]:
        if name == "tool_open":
            if self._state == "body":
                events = self._route_flush()  # block boundary: held suffix is not a marker
                self._state = "head"
                self._head = ""
                return events
            return []
        if self._state == "head":
            head, self._head, self._state = self._head, "", "body"
            if not head.strip():
                return []
            self._index += 1
            return [
                ToolCallBegin(self._index, _json_string_field(head, "name")),
                ToolCallEnd(self._index),
            ]
        if self._state in ("args", "tail"):
            self._state = "body"
            self._depth = 0
            self._in_string = False
            self._escape = False
            self._args_started = False
            self._args_done = False
            self._scalar = False
            return [ToolCallEnd(self._index)]
        return []  # stray close in body: stripped, not an error

    def _on_text(self, text: str) -> list[Event]:
        if self._state == "body":
            return self._route(text)
        if self._state == "head":
            self._head += text
            return self._complete_head()
        if self._state == "args":
            return self._scan_args(text)
        return []  # tail: between the arguments value and the close marker

    def _complete_head(self) -> list[Event]:
        idx = self._head.find(self._ARGS_KEY)
        if idx < 0:
            return []
        after = self._head[idx + len(self._ARGS_KEY) :]
        colon = re.match(r"\s*:", after)
        if colon is None:
            return []
        name = _json_string_field(self._head[:idx], "name")
        self._index += 1
        begin = ToolCallBegin(self._index, name)
        self._state = "args"
        return [begin] + self._scan_args(after[colon.end() :])

    def _scan_args(self, text: str) -> list[Event]:
        if self._args_done:
            self._state = "tail"
            return []
        chars: list[str] = []
        for ch in text:
            if not self._args_started:
                if ch.isspace():
                    continue
                if ch in "{[":
                    self._args_started = True
                    self._depth = 1
                    chars.append(ch)
                    continue
                # Scalar arguments (rare): stream everything to the close marker.
                self._args_started = True
                self._scalar = True
                chars.append(ch)
                continue
            if self._scalar:
                chars.append(ch)
                continue
            if self._in_string:
                if self._escape:
                    self._escape = False
                elif ch == "\\":
                    self._escape = True
                elif ch == '"':
                    self._in_string = False
                chars.append(ch)
                continue
            if ch == '"':
                self._in_string = True
            elif ch in "{[":
                self._depth += 1
            elif ch in "}]":
                self._depth -= 1
            chars.append(ch)
            if ch in "}]" and self._depth == 0:
                self._args_done = True
                self._state = "tail"
                break
        if not chars:
            return []
        return [ToolArgsDelta(self._index, "".join(chars))]


class _Gemma4ToolMachine:
    """gemma-4 tool calls: ``<|tool_call\u003ecall:NAME{args}<tool_call|\u003e``.

    Template-exact (chat_template.jinja): the name is the text between the
    open marker and the first brace; the args region is gemma-style
    serialization (``key:<|"|>value<|"|>,...``), not JSON, so the terminating
    brace is found by brace counting that treats the ``<|"|>`` quote token as
    a string toggle (braces inside quoted values are literal). Args stream
    raw; conversion to OpenAI JSON arguments is future work (wiki non-goal
    for this step). One call per block; the segmenter owns block boundaries,
    so a literal close-marker string inside a value would split the block.
    """

    def __init__(
        self,
        route_text: Callable[[str], list[Event]],
        route_flush: Callable[[], list[Event]],
    ) -> None:
        self._route = route_text
        self._route_flush = route_flush
        self._segmenter = MarkerSegmenter(
            {"call_open": GEMMA4_CALL_OPEN, "call_close": GEMMA4_CALL_CLOSE}
        )
        self._index = -1
        self._state = "body"
        self._head = ""
        self._depth = 0
        self._in_string = False
        self._qbuf = ""
        self._args_done = False

    def feed(self, text: str) -> list[Event]:
        events: list[Event] = []
        for segment in self._segmenter.feed(text):
            if segment.kind == "marker":
                events += self._on_marker(segment.value)
            else:
                events += self._on_text(segment.value)
        return events

    def finish(self) -> list[Event]:
        events: list[Event] = []
        for segment in self._segmenter.finish():
            if segment.kind == "marker":
                events += self._on_marker(segment.value)
            else:
                events += self._on_text(segment.value)
        if self._state == "head":
            head, self._head, self._state = self._head, "", "body"
            name = head.split("{", 1)[0].rstrip()
            if name:
                self._index += 1
                events.append(ToolCallBegin(self._index, name))
                events.append(ToolCallEnd(self._index))
        elif self._state == "args":
            if self._qbuf:
                events.append(ToolArgsDelta(self._index, self._qbuf))
                self._qbuf = ""
            events.append(ToolCallEnd(self._index))
        elif self._state == "tail":
            events.append(ToolCallEnd(self._index))
        return events

    def _on_marker(self, name: str) -> list[Event]:
        if name == "call_open":
            if self._state == "body":
                events = self._route_flush()  # block boundary: held suffix is not a marker
                self._state = "head"
                self._head = ""
                return events
            return []
        if self._state == "head":
            head, self._head, self._state = self._head, "", "body"
            name = head.split("{", 1)[0].rstrip()
            if not name:
                return []
            self._index += 1
            return [ToolCallBegin(self._index, name), ToolCallEnd(self._index)]
        if self._state in ("args", "tail"):
            self._state = "body"
            self._head = ""
            self._depth = 0
            self._in_string = False
            self._qbuf = ""
            self._args_done = False
            return [ToolCallEnd(self._index)]
        return []  # stray close in body: stripped, not an error

    def _on_text(self, text: str) -> list[Event]:
        if self._state == "body":
            return self._route(text)
        if self._state == "head":
            self._head += text
            return self._complete_head()
        if self._state == "args":
            return self._scan_args(text)
        return []

    def _complete_head(self) -> list[Event]:
        idx = self._head.find("{")
        if idx < 0:
            return []
        name = self._head[:idx].rstrip()
        self._index += 1
        begin = ToolCallBegin(self._index, name)
        self._state = "args"
        return [begin] + self._scan_args(self._head[idx:])

    def _scan_args(self, text: str) -> list[Event]:
        if self._args_done:
            self._state = "tail"
            return []
        chars: list[str] = []
        for ch in text:
            if self._args_done:
                break
            if self._qbuf:
                self._qbuf += ch
                if GEMMA4_QUOTE.startswith(self._qbuf):
                    if len(self._qbuf) == len(GEMMA4_QUOTE):
                        self._in_string = not self._in_string
                        chars.append(self._qbuf)
                        self._qbuf = ""
                    continue
                # Mismatch: flush the buffered run, then retry a trailing '<'.
                retry, self._qbuf = self._qbuf, ""
                if retry.endswith("<"):
                    chars.append(retry[:-1])
                    self._qbuf = "<"
                else:
                    chars.append(retry)
                continue
            if ch == "<":
                self._qbuf = ch  # could start the quote token
                continue
            if self._in_string:
                chars.append(ch)
                continue
            if ch == "{":
                self._depth += 1
            elif ch == "}":
                self._depth -= 1
            chars.append(ch)
            if ch == "}" and self._depth == 0:
                self._args_done = True
                self._state = "tail"
                break
        if not chars:
            return []
        return [ToolArgsDelta(self._index, "".join(chars))]


# ── registry + facade ────────────────────────────────────────────────────────

REASONING_PARSERS: dict[str, Callable[[], Any]] = {
    "think": ThinkReasoningParser,
    "gemma4_thought": Gemma4ThoughtParser,
}

TOOL_PARSERS: dict[
    str, Callable[[Callable[[str], list[Event]], Callable[[], list[Event]]], Any]
] = {
    "hermes": _HermesToolMachine,
    "gemma4_calls": _Gemma4ToolMachine,
}


def resolve_reasoning_parser(name: str | None) -> Any | None:
    """Build a per-request reasoning parser, or None for raw passthrough."""
    if not name or name == "none":
        return None
    factory = REASONING_PARSERS.get(name)
    if factory is None:
        raise ValueError(f"unknown reasoning parser: {name!r}")
    return factory()


def resolve_tool_machine(name: str | None) -> Any | None:
    """Tool state-machine class for a format name, or None for no tool parsing."""
    if not name or name == "none":
        return None
    factory = TOOL_PARSERS.get(name)
    if factory is None:
        raise ValueError(f"unknown tool parser: {name!r}")
    return factory


class ParserSpec:
    """Checkpoint-level parser configuration; builds one parser per request.

    Served from ``config.json``'s ``reasoning_parser`` / ``tool_call_parser``
    keys (Step 4 wires the resolution) or explicit CLI overrides.
    """

    def __init__(self, reasoning: str | None = None, tools: str | None = None) -> None:
        self.reasoning = reasoning
        self.tools = tools

    def build(self) -> ResponseParser:
        return ResponseParser.from_spec(self.reasoning, self.tools)


class ResponseParser:
    """Per-request facade: text in, events out.

    With a tool parser configured, the tool machine owns the stream: it
    routes body text to the reasoning parser (or straight through as
    content) and turns tool markers into tool events, all in stream order.
    Without one, text flows to the reasoning parser alone; with neither, it
    passes through as content events so the HTTP mapping code has exactly one
    shape to handle.
    """

    def __init__(self, reasoning: Any | None = None, tool_machine: Any | None = None) -> None:
        self._reasoning = reasoning
        if tool_machine is not None:
            self._machine = tool_machine(self._route_text, self._route_flush)
        else:
            self._machine = None

    @classmethod
    def from_spec(cls, reasoning: str | None, tools: str | None) -> "ResponseParser":
        return cls(
            reasoning=resolve_reasoning_parser(reasoning),
            tool_machine=resolve_tool_machine(tools),
        )

    def _route_text(self, text: str) -> list[Event]:
        if self._reasoning is not None:
            return self._reasoning.feed(text)
        return [ContentDelta(text)] if text else []

    def _route_flush(self) -> list[Event]:
        if self._reasoning is not None:
            return self._reasoning.flush()
        return []

    def feed(self, text: str) -> list[Event]:
        if self._machine is not None:
            return self._machine.feed(text)
        if self._reasoning is not None:
            return self._reasoning.feed(text)
        return [ContentDelta(text)] if text else []

    def finish(self) -> list[Event]:
        events: list[Event] = []
        if self._machine is not None:
            events += self._machine.finish()
        if self._reasoning is not None:
            events += self._reasoning.finish()
        return events


# ── OpenAI mapping helpers ───────────────────────────────────────────────────


def events_to_stream_deltas(events: list[Event]) -> list[dict[str, Any]]:
    """Map parser events to OpenAI streaming delta fragments.

    Reasoning/content deltas become single-key fragments; tool events group
    into ``tool_calls`` entries keyed by index (OpenAI accumulate semantics).
    """
    deltas: list[dict[str, Any]] = []
    for event in events:
        if isinstance(event, ReasoningDelta):
            deltas.append({"reasoning_content": event.text})
        elif isinstance(event, ContentDelta):
            deltas.append({"content": event.text})
        elif isinstance(event, ToolCallBegin):
            deltas.append(
                {
                    "tool_calls": [
                        {"index": event.index, "function": {"name": event.name, "arguments": ""}}
                    ]
                }
            )
        elif isinstance(event, ToolArgsDelta):
            deltas.append(
                {"tool_calls": [{"index": event.index, "function": {"arguments": event.text}}]}
            )
        elif isinstance(event, ToolCallEnd):
            continue  # nothing to stream; validation happens on the full message
    return deltas


def events_to_message(events: list[Event]) -> dict[str, Any]:
    """Aggregate parser events into a non-streaming assistant message body."""
    message: dict[str, Any] = {"role": "assistant", "content": ""}
    tool_calls: list[dict[str, Any]] | None = None
    for event in events:
        if isinstance(event, ReasoningDelta):
            message["reasoning_content"] = message.get("reasoning_content", "") + event.text
        elif isinstance(event, ContentDelta):
            message["content"] += event.text
        elif isinstance(event, ToolCallBegin):
            tool_calls = tool_calls or []
            tool_calls.append(
                {
                    "id": f"call_{event.index}",
                    "type": "function",
                    "function": {"name": event.name, "arguments": ""},
                }
            )
        elif isinstance(event, ToolArgsDelta):
            tool_calls[event.index]["function"]["arguments"] += event.text
        elif isinstance(event, ToolCallEnd):
            # Validate the accumulated arguments are parseable JSON; malformed
            # arguments stay client-visible (wiki design: stream raw, validate
            # at close).
            try:
                json.loads(tool_calls[event.index]["function"]["arguments"])
            except (json.JSONDecodeError, TypeError):
                pass
    if tool_calls:
        message["tool_calls"] = tool_calls
    return message


def chat_stream_events(
    parser: ResponseParser | None, text: str, *, finish: bool = False
) -> Iterator[dict[str, Any]]:
    """Convenience for SSE loops: feed (or finish) and yield OpenAI deltas."""
    if parser is None:
        if text:
            yield {"content": text}
        return
    events = parser.finish() if finish else parser.feed(text)
    yield from events_to_stream_deltas(events)


def parse_full_text(parser: ResponseParser | None, text: str) -> dict[str, Any]:
    """Non-stream path: run the whole text through the parser, map once."""
    if parser is None:
        return {"role": "assistant", "content": text}
    return events_to_message(parser.feed(text) + parser.finish())
