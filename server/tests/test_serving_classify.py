"""HTTP routing for the two served kinds, over fakes (no MLX, no model).

A ``/v1/classify`` request must reach the classify session only on a
token-classification server, and the generation routes must reject it with a
400 that names the right route — and vice versa on a generation server.
"""

from __future__ import annotations

import http.client
import json
import threading
from http.server import ThreadingHTTPServer

import pytest

from server.engine import GenerateParams
from server.server import _Handler, _ServeState


class FakeClassifySession:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def classify(self, texts, *, decoder="viterbi", max_length=None):
        self.calls.append({"texts": list(texts), "decoder": decoder, "max_length": max_length})
        return [
            {
                "tokens": list(text.split()),
                "labels": ["O"] * len(text.split()),
                "entities": [],
                "usage": {"input_tokens": len(text.split())},
            }
            for text in texts
        ]


class FakeGenerateSession:
    tokenizer = None

    def generate(self, prompt, params: GenerateParams, stats: dict):
        stats["prompt_tokens"] = 1
        stats["completion_tokens"] = 2
        stats["finish_reason"] = "stop"
        yield "ok"


def _serve(state: _ServeState):
    handler = type("_BoundHandler", (_Handler,), {"state": state})
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd, thread


def _post(port: int, path: str, payload: dict) -> tuple[int, dict]:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    connection.request(
        "POST", path, json.dumps(payload), {"Content-Type": "application/json"}
    )
    response = connection.getresponse()
    body = json.loads(response.read().decode("utf-8"))
    connection.close()
    return response.status, body


def test_classify_server_serves_classify_and_rejects_generation() -> None:
    fake = FakeClassifySession()
    state = _ServeState(
        None, "classifier", 2, kind="token_classification", classify=fake
    )
    httpd, thread = _serve(state)
    try:
        status, body = _post(
            httpd.server_address[1], "/v1/classify", {"input": ["hello world"]}
        )
        assert status == 200
        assert body["object"] == "list"
        assert body["data"][0]["labels"] == ["O", "O"]
        assert body["usage"] == {"input_tokens": 2}
        assert fake.calls == [{"texts": ["hello world"], "decoder": "viterbi", "max_length": None}]

        status, body = _post(
            httpd.server_address[1], "/v1/completions", {"prompt": "hi"}
        )
        assert status == 400
        assert "/v1/classify" in body["error"]["message"]

        status, body = _post(
            httpd.server_address[1], "/v1/classify", {"input": "hi", "decoder": "greedy"}
        )
        assert status == 400

        status, body = _post(httpd.server_address[1], "/v1/classify", {"input": 3})
        assert status == 400
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_generation_server_rejects_classify() -> None:
    state = _ServeState(FakeGenerateSession(), "generator", 2)
    httpd, thread = _serve(state)
    try:
        status, body = _post(httpd.server_address[1], "/v1/classify", {"input": "hi"})
        assert status == 400
        assert "/v1/completions" in body["error"]["message"]

        status, body = _post(
            httpd.server_address[1], "/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}]}
        )
        assert status == 200
        assert body["choices"][0]["message"]["content"] == "ok"
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_models_listing_reports_task() -> None:
    for kind, task in (
        ("token_classification", "token-classification"),
        ("generation", "text-generation"),
    ):
        state = _ServeState(
            FakeGenerateSession() if kind == "generation" else None,
            "m",
            1,
            kind=kind,
            classify=FakeClassifySession() if kind == "token_classification" else None,
        )
        assert state.task == task
