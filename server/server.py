"""Minimal OpenAI-compatible HTTP server over the ``Session`` engine.

Deliberately thin: the substance lives in ``Session`` (decode loop, cache,
sampling). This module only speaks HTTP/SSE and caps concurrency. Uses only
the standard library so serving adds zero dependencies.

Endpoints:
- ``GET  /health``               liveness
- ``GET  /v1/models``            minimal model listing
- ``POST /v1/completions``       text completion (``stream`` supported, SSE)
- ``POST /v1/chat/completions``  chat completion (``stream`` supported, SSE)

Dynamic batching: v1 interleaves concurrent requests through a bounded
semaphore (each request streams its own decode over shared weights). True
cross-request batched decode is deferred until the cache contract supports
batched per-sequence caches; the scheduler seam lives behind ``Session``.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterator

from .engine import GenerateParams, Session

__all__ = ["serve", "main"]


def _render_chat(session: Session, messages: list[dict[str, Any]]) -> str:
    """Apply the checkpoint's chat template when available, else a naive join."""
    tokenizer = session.tokenizer
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        try:
            rendered = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            if isinstance(rendered, str) and rendered:
                return rendered
        except Exception:
            pass
    parts = [f"{message.get('role', 'user')}: {message.get('content', '')}" for message in messages]
    return "\n".join(parts) + "\nassistant:"


def _params_from_body(body: dict[str, Any]) -> GenerateParams:
    stop = body.get("stop") or ()
    if isinstance(stop, str):
        stop = (stop,)
    return GenerateParams(
        max_new_tokens=int(body.get("max_tokens") or body.get("max_new_tokens") or 64),
        temperature=float(body.get("temperature", 1.0)),
        top_k=int(body["top_k"]) if body.get("top_k") is not None else None,
        top_p=float(body["top_p"]) if body.get("top_p") is not None else None,
        seed=int(body["seed"]) if body.get("seed") is not None else None,
        stop=tuple(str(value) for value in stop),
    )


class _ServeState:
    """Shared server state: one Session, a concurrency cap, request bookkeeping."""

    def __init__(self, session: Session, model_name: str, max_concurrency: int) -> None:
        self.session = session
        self.model_name = model_name
        self.slots = threading.BoundedSemaphore(max_concurrency)


def _sse_chunks(state: _ServeState, kind: str, prompt: str, body: dict[str, Any]) -> Iterator[str]:
    """Stream a completion as OpenAI-style SSE chunks (``kind``: chat or completion)."""
    request_id = f"cmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())
    object_name = "chat.completion.chunk" if kind == "chat" else "text_completion.chunk"
    params = _params_from_body(body)
    stream_options = body.get("stream_options")
    include_usage = isinstance(stream_options, dict) and bool(stream_options.get("include_usage"))

    def chunk(delta: dict[str, Any], finish_reason: str | None = None) -> str:
        payload = {
            "id": request_id,
            "object": object_name,
            "created": created,
            "model": state.model_name,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }
        return f"data: {json.dumps(payload)}\n\n"

    stats: dict[str, Any] = {}
    yield chunk({"role": "assistant"} if kind == "chat" else {})
    for text in state.session.generate(prompt, params, stats):
        yield chunk({"content": text})
    yield chunk({}, finish_reason=stats.get("finish_reason", "stop"))
    if include_usage:
        prompt_tokens = int(stats.get("prompt_tokens", 0))
        completion_tokens = int(stats.get("completion_tokens", 0))
        usage_payload = {
            "id": request_id,
            "object": object_name,
            "created": created,
            "model": state.model_name,
            "choices": [],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }
        yield f"data: {json.dumps(usage_payload)}\n\n"
    yield "data: [DONE]\n\n"


def _full_completion(state: _ServeState, prompt: str, body: dict[str, Any]) -> dict[str, Any]:
    params = _params_from_body(body)
    stats: dict[str, Any] = {}
    pieces: list[str] = []
    for text in state.session.generate(prompt, params, stats):
        pieces.append(text)
    prompt_tokens = int(stats.get("prompt_tokens", 0))
    completion_tokens = int(stats.get("completion_tokens", 0))
    return {
        "object": "text_completion",
        "model": state.model_name,
        "choices": [
            {
                "index": 0,
                "text": "".join(pieces),
                "finish_reason": stats.get("finish_reason", "stop"),
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


class _Handler(BaseHTTPRequestHandler):
    server_version = "axon-serve/0.1"
    state: _ServeState

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - stdlib signature
        return  # keep stdout clean; override for access logs

    # ── routing ─────────────────────────────────────────────────────────────

    def do_GET(self) -> None:  # noqa: N802 - stdlib name
        if self.path in {"/health", "/healthz"}:
            self._send_json({"ok": True, "model": self.state.model_name})
            return
        if self.path == "/v1/models":
            self._send_json(
                {
                    "object": "list",
                    "data": [{"id": self.state.model_name, "object": "model", "owned_by": "axon"}],
                }
            )
            return
        self._send_error(404, f"not found: {self.path}")

    def do_POST(self) -> None:  # noqa: N802 - stdlib name
        try:
            body = self._read_json_body()
        except ValueError as error:
            self._send_error(400, str(error))
            return
        try:
            if self.path == "/v1/completions":
                self._completion(body)
                return
            if self.path == "/v1/chat/completions":
                self._chat_completion(body)
                return
        except Exception as error:  # noqa: BLE001 - convert to HTTP error
            self._send_error(500, f"{type(error).__name__}: {error}")
            return
        self._send_error(404, f"not found: {self.path}")

    # ── endpoint bodies ─────────────────────────────────────────────────────

    def _completion(self, body: dict[str, Any]) -> None:
        prompt = body.get("prompt")
        if not isinstance(prompt, str):
            self._send_error(400, "'prompt' must be a string")
            return
        self._generate_response(prompt, body, kind="text_completion")

    def _chat_completion(self, body: dict[str, Any]) -> None:
        messages = body.get("messages")
        if not isinstance(messages, list):
            self._send_error(400, "'messages' must be a list")
            return
        prompt = _render_chat(self.state.session, messages)
        self._generate_response(prompt, body, kind="chat")

    def _generate_response(self, prompt: str, body: dict[str, Any], *, kind: str) -> None:
        if body.get("stream"):
            self._respond_sse(prompt, body, kind=kind)
            return
        with self.state.slots:
            payload = _full_completion(self.state, prompt, body)
        if kind == "chat":
            text = payload["choices"][0]["text"]
            payload = {
                "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
                "object": "chat.completion",
                "model": self.state.model_name,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": text},
                        "finish_reason": payload["choices"][0]["finish_reason"],
                    }
                ],
                "usage": payload["usage"],
            }
        self._send_json(payload)

    def _respond_sse(self, prompt: str, body: dict[str, Any], *, kind: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        with self.state.slots:
            for payload in _sse_chunks(self.state, kind, prompt, body):
                self.wfile.write(payload.encode("utf-8"))
                self.wfile.flush()

    # ── plumbing ────────────────────────────────────────────────────────────

    def _read_json_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            raise ValueError("request body required")
        raw = self.rfile.read(length)
        parsed = json.loads(raw.decode("utf-8"))
        if not isinstance(parsed, dict):
            raise ValueError("request body must be a JSON object")
        return parsed

    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _send_error(self, status: int, message: str) -> None:
        self._send_json({"error": {"message": message, "code": status}}, status=status)


def serve(
    axon_path: Path | str,
    *,
    checkpoint_dir: Path | str | None = None,
    host: str = "127.0.0.1",
    port: int = 8000,
    max_concurrency: int = 2,
    dtype: str = "float16",
    warm_compile: bool = True,
) -> None:
    """Load the model and block serving requests (Ctrl+C to stop)."""
    session = Session.from_axon(
        axon_path,
        checkpoint_dir=checkpoint_dir,
        dtype=dtype,
        emit_profile="inference",
        warm_compile=warm_compile,
    )
    model_name = Path(axon_path).stem
    state = _ServeState(session, model_name, max_concurrency=max(1, int(max_concurrency)))

    handler = type("_BoundHandler", (_Handler,), {"state": state})
    httpd = ThreadingHTTPServer((host, port), handler)
    print(f"axon-serve: {model_name} on http://{host}:{port} (max_concurrency={max_concurrency})")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m server")
    parser.add_argument("--axon", required=True, help="Path to a materialized .axon model file")
    parser.add_argument("--checkpoint", default=None, help="HF checkpoint dir (default: models/<family>)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-concurrency", type=int, default=2)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--no-warm-compile", action="store_true")
    args = parser.parse_args(argv)
    serve(
        args.axon,
        checkpoint_dir=args.checkpoint,
        host=args.host,
        port=args.port,
        max_concurrency=args.max_concurrency,
        dtype=args.dtype,
        warm_compile=not args.no_warm_compile,
    )


if __name__ == "__main__":
    main()
