"""Vector-driven tests for the shared response parsers (wiki/response-parsers.md).

The JSON vectors under ``tests/parsers/vectors/`` are the cross-implementation
contract: the Python reference below and the later Rust port must both satisfy
them exactly (the tokref/chatref parity pattern applied to text processing).
Marker strings in the vectors use ``\\u003c`` JSON escapes so the exact marker
bytes are unambiguous in source.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from server.response_parsers import (
    ContentDelta,
    ParserSpec,
    ReasoningDelta,
    ResponseParser,
    ToolArgsDelta,
    ToolCallBegin,
    ToolCallEnd,
    event_to_json,
    events_to_message,
    events_to_stream_deltas,
    parse_full_text,
)

VECTORS = Path(__file__).parent / "parsers" / "vectors"


def _vector_files():
    return sorted(VECTORS.glob("*.json"))


def _run_chunks(parser: ResponseParser, chunks: list[str]):
    events = []
    for chunk in chunks:
        events.extend(parser.feed(chunk))
    events.extend(parser.finish())
    return events


def _grouped_text(events) -> tuple[str, str]:
    reasoning = "".join(e.text for e in events if isinstance(e, ReasoningDelta))
    content = "".join(e.text for e in events if isinstance(e, ContentDelta))
    return reasoning, content


@pytest.mark.parametrize("path", _vector_files(), ids=lambda p: p.stem)
def test_vectors(path: Path) -> None:
    spec = json.loads(path.read_text(encoding="utf-8"))
    for case in spec["cases"]:
        parser = ResponseParser.from_spec(spec.get("reasoning_parser"), spec.get("tool_call_parser"))
        got = [event_to_json(event) for event in _run_chunks(parser, case["chunks"])]
        assert got == case["events"], f"{path.stem}/{case['name']}"


@pytest.mark.parametrize("path", _vector_files(), ids=lambda p: p.stem)
def test_vector_chunking_invariance(path: Path) -> None:
    """Chunk boundaries may move deltas around but never the text per channel."""
    spec = json.loads(path.read_text(encoding="utf-8"))
    for case in spec["cases"]:
        whole = "".join(case["chunks"])
        batched = _run_chunks(
            ResponseParser.from_spec(spec.get("reasoning_parser"), spec.get("tool_call_parser")),
            [whole],
        )
        incremental = _run_chunks(
            ResponseParser.from_spec(spec.get("reasoning_parser"), spec.get("tool_call_parser")),
            case["chunks"],
        )
        assert _grouped_text(batched) == _grouped_text(incremental), case["name"]


def test_default_spec_is_passthrough() -> None:
    parser = ResponseParser.from_spec(None, None)
    assert [event_to_json(event) for event in parser.feed("hi")] == [["content_delta", "hi"]]
    assert parser.finish() == []
    assert parse_full_text(None, "raw") == {"role": "assistant", "content": "raw"}


def test_unknown_reasoning_parser_raises() -> None:
    with pytest.raises(ValueError):
        ResponseParser.from_spec("nope", None)


def test_full_stream_parser_rejects_reasoning_parser() -> None:
    """harmony classifies channels itself; a reasoning parser has nothing to do."""
    with pytest.raises(ValueError, match="full-stream"):
        ResponseParser.from_spec("think", "harmony")
    # Alone it builds and parses the full protocol.
    parser = ResponseParser.from_spec(None, "harmony")
    events = parser.feed("\u003c|channel|>analysis\u003c|message|>r\u003c|end|>")
    assert [event_to_json(event) for event in events] == [["reasoning_delta", "r"]]


def test_stream_delta_mapping() -> None:
    deltas = events_to_stream_deltas([ReasoningDelta("a"), ContentDelta("b")])
    assert deltas == [{"reasoning_content": "a"}, {"content": "b"}]


def test_tool_event_stream_mapping() -> None:
    events = [
        ToolCallBegin(0, "get_weather"),
        ToolArgsDelta(0, '{"city"'),
        ToolArgsDelta(0, ': "Oslo"}'),
        ToolCallEnd(0),
    ]
    assert events_to_stream_deltas(events) == [
        {"tool_calls": [{"index": 0, "function": {"name": "get_weather", "arguments": ""}}]},
        {"tool_calls": [{"index": 0, "function": {"arguments": '{"city"'}}]},
        {"tool_calls": [{"index": 0, "function": {"arguments": ': "Oslo"}'}}]},
    ]


def test_tool_event_message_mapping() -> None:
    events = [
        ToolCallBegin(0, "get_weather"),
        ToolArgsDelta(0, '{"city": "Oslo"}'),
        ToolCallEnd(0),
        ContentDelta("Done"),
    ]
    message = events_to_message(events)
    assert message["tool_calls"][0]["function"] == {
        "name": "get_weather",
        "arguments": '{"city": "Oslo"}',
    }
    assert message["content"] == "Done"


def test_full_message_mapping_groups_reasoning_and_content() -> None:
    parser = ResponseParser.from_spec("think", None)
    message = parse_full_text(parser, "\u003cthink\u003er \u003c/think\u003ec")
    assert message == {"role": "assistant", "content": "c", "reasoning_content": "r "}


def test_parser_spec_builds_per_request_state() -> None:
    spec = ParserSpec("think", None)
    first = spec.build().feed("\u003cthink\u003eone")
    second = spec.build().feed("\u003cthink\u003etwo")
    assert _grouped_text(first)[0] == "one"
    assert _grouped_text(second)[0] == "two"


def test_sse_chat_wiring_parses_reasoning() -> None:
    from server.server import _ServeState, _sse_chunks

    class _FakeSession:
        def generate(self, prompt, params, stats):
            stats["finish_reason"] = "stop"
            stats["prompt_tokens"] = 1
            stats["completion_tokens"] = 3
            yield "\u003cth"
            yield "ink\u003eWhy"
            yield " 42"

    state = _ServeState(_FakeSession(), "m", 2, parsers=ParserSpec("think", None))
    raw = "".join(_sse_chunks(state, "chat", "p", {}))
    payloads = [
        json.loads(line[len("data: ") :])
        for line in raw.split("\n\n")
        if line.startswith("data: ") and line != "data: [DONE]"
    ]
    deltas = [payload["choices"][0]["delta"] for payload in payloads]
    assert deltas[0] == {"role": "assistant"}
    assert {"reasoning_content": "Why"} in deltas
    assert {"reasoning_content": " 42"} in deltas
    assert not any("content" in delta for delta in deltas)
    assert payloads[-1]["choices"][0]["finish_reason"] == "stop"
