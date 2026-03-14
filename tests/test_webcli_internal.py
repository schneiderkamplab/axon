from __future__ import annotations

import io
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
import typer

from brainsurgery.engine import ProviderError
import brainsurgery.webcli.cli as webcli_cli
import brainsurgery.webcli.handler as webcli_handler
from brainsurgery.webcli.models import WebRunResult
import brainsurgery.webcli.runner as webcli_runner
import brainsurgery.webcli.server as webcli_server


def test_webcli_configure_logging_and_bad_level() -> None:
    webcli_cli.configure_logging("debug")
    with pytest.raises(typer.BadParameter):
        webcli_cli.configure_logging("bogus")


def test_webcli_callback_handles_browser_error_and_wildcard_host(monkeypatch: pytest.MonkeyPatch) -> None:
    served: list[tuple[str, int]] = []

    monkeypatch.setattr(webcli_cli, "configure_logging", lambda _lvl: None)

    def _boom(_url: str) -> None:
        raise RuntimeError("browser failed")

    monkeypatch.setattr(webcli_cli.webbrowser, "open", _boom)
    monkeypatch.setattr(webcli_cli, "serve_webcli", lambda *, host, port: served.append((host, port)))
    webcli_cli.webcli(host="0.0.0.0", port=8123, log_level="info", open_browser=True)
    assert served == [("0.0.0.0", 8123)]


def _make_handler_for_read() -> type:
    return webcli_handler.handler_factory()


def test_webcli_handler_read_json_body_validation() -> None:
    handler_cls = _make_handler_for_read()

    missing_len = object.__new__(handler_cls)
    missing_len.headers = {}
    missing_len.rfile = io.BytesIO(b"{}")
    with pytest.raises(ValueError, match="Missing Content-Length"):
        handler_cls._read_json_body(missing_len)

    non_dict = object.__new__(handler_cls)
    non_dict.headers = {"Content-Length": "2"}
    non_dict.rfile = io.BytesIO(b"[]")
    with pytest.raises(ValueError, match="JSON object"):
        handler_cls._read_json_body(non_dict)

    valid = object.__new__(handler_cls)
    valid.headers = {"Content-Length": "8"}
    valid.rfile = io.BytesIO(b'{"x": 1}')
    assert handler_cls._read_json_body(valid) == {"x": 1}


def test_webcli_handler_send_helpers_and_log_message(monkeypatch: pytest.MonkeyPatch) -> None:
    handler_cls = _make_handler_for_read()
    h = object.__new__(handler_cls)
    h.wfile = io.BytesIO()
    h._responses = []
    h._headers = []
    h.end_headers = lambda: None
    h.send_response = lambda code: h._responses.append(code)
    h.send_header = lambda key, val: h._headers.append((key, val))
    handler_cls._send_html(h, "<b>x</b>")
    assert h._responses[-1] == 200
    assert any(k == "Content-Type" and "text/html" in v for k, v in h._headers)

    handler_cls._send_json(h, {"ok": True})
    assert h._responses[-1] == 200
    assert any(k == "Content-Type" and "application/json" in v for k, v in h._headers)

    seen: list[str] = []
    monkeypatch.setattr(webcli_handler.logger, "debug", lambda msg, *a: seen.append(msg % a))
    handler_cls.log_message(h, "hello %s", "world")
    assert seen and seen[0] == "webcli request: hello world"


def test_webcli_handler_do_get_and_post_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    handler_cls = _make_handler_for_read()
    h = object.__new__(handler_cls)
    h._html = []
    h._json = []
    h._errors = []
    h._payload = {"plan_yaml": "{}", "num_workers": 2, "provider": "inmemory", "log_level": "info"}
    h._send_html = lambda body: h._html.append(body)
    h._send_json = lambda payload, status=200: h._json.append((status, payload))
    h.send_error = lambda code, message: h._errors.append((code, message))
    h._read_json_body = lambda: h._payload

    monkeypatch.setattr(
        webcli_handler,
        "run_web_plan",
        lambda **_kwargs: WebRunResult(
            ok=True,
            logs=["a"],
            output_lines=["b"],
            executed_transforms=[{"x": {}}],
            summary_yaml="transforms: []\n",
            written_path=None,
            error=None,
        ),
    )

    h.path = "/"
    handler_cls.do_GET(h)
    assert h._html

    h.path = "/missing"
    handler_cls.do_GET(h)
    assert h._errors[-1][0] == 404

    h.path = "/api/other"
    handler_cls.do_POST(h)
    assert h._errors[-1][0] == 404

    h.path = "/api/run"
    handler_cls.do_POST(h)
    assert h._json[-1][1]["ok"] is True


@pytest.mark.parametrize(
    ("exc", "needle"),
    [
        (ProviderError("p"), "Provider error"),
        (ValueError("v"), "v"),
        (RuntimeError("r"), "r"),
    ],
)
def test_webcli_handler_do_post_error_branches(
    monkeypatch: pytest.MonkeyPatch,
    exc: Exception,
    needle: str,
) -> None:
    handler_cls = _make_handler_for_read()
    h = object.__new__(handler_cls)
    h._json = []
    h.path = "/api/run"
    h._send_json = lambda payload, status=200: h._json.append((status, payload))
    h._read_json_body = lambda: {"plan_yaml": "{}"}
    monkeypatch.setattr(webcli_handler, "run_web_plan", lambda **_kwargs: (_ for _ in ()).throw(exc))
    handler_cls.do_POST(h)
    assert h._json
    payload = h._json[-1][1]
    assert payload["ok"] is False
    assert needle in payload["error"]


def test_webcli_handler_cast_helpers() -> None:
    assert webcli_handler._as_string("x", "k") == "x"
    with pytest.raises(ValueError):
        webcli_handler._as_string(1, "k")
    assert webcli_handler._as_int(3, "k") == 3
    with pytest.raises(ValueError):
        webcli_handler._as_int(True, "k")


def test_webcli_runner_configure_logging_and_list_log_handler() -> None:
    webcli_runner.configure_logging(log_level="info")
    with pytest.raises(ValueError):
        webcli_runner.configure_logging(log_level="bad")

    sink: list[str] = []
    h = webcli_runner._ListLogHandler(sink)
    h.setFormatter(logging.Formatter("%(message)s"))
    h.emit(logging.LogRecord("x", logging.INFO, __file__, 1, "hello", (), None))
    assert sink == ["hello"]


def test_webcli_runner_rejects_non_mapping_yaml() -> None:
    with pytest.raises(ValueError, match="root must be a mapping"):
        webcli_runner.run_web_plan(
            plan_yaml="- x\n",
            shard_size="5GB",
            num_workers=1,
            provider="inmemory",
            arena_root=Path(".brainsurgery"),
            arena_segment_size="1GB",
            summarize=True,
            log_level="info",
        )


def test_webcli_runner_branches(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class _FakeProvider:
        def __init__(self) -> None:
            self.closed = False
            self.saved = []

        def save_output(self, plan, *, default_shard_size: str, max_io_workers: int):
            self.saved.append((plan, default_shard_size, max_io_workers))
            return tmp_path / "out.safetensors"

        def close(self) -> None:
            self.closed = True

    class _FakePlan:
        def __init__(self, *, output):
            self.inputs = {}
            self.transforms = []
            self.output = output

    fake_provider = _FakeProvider()
    monkeypatch.setattr(webcli_runner, "configure_logging", lambda *, log_level: None)
    monkeypatch.setattr(webcli_runner, "compile_plan", lambda raw: _FakePlan(output=raw.get("output")))
    monkeypatch.setattr(webcli_runner, "create_state_dict_provider", lambda **_kwargs: fake_provider)
    monkeypatch.setattr(webcli_runner, "normalize_transform_specs", lambda _t: [])
    monkeypatch.setattr(webcli_runner, "execute_transform_pairs", lambda *_a, **_k: (None, [{"dump": {}}]))
    monkeypatch.setattr(webcli_runner, "build_raw_plan", lambda **_kwargs: {"transforms": [{"dump": {}}]})

    class _Ctx:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(webcli_runner, "use_output_emitter", lambda _sink: _Ctx())
    monkeypatch.setattr(webcli_runner, "get_runtime_flags", lambda: SimpleNamespace(dry_run=True))

    res_no_output = webcli_runner.run_web_plan(
        plan_yaml="{}",
        shard_size="5GB",
        num_workers=2,
        provider="inmemory",
        arena_root=Path(".brainsurgery"),
        arena_segment_size="1GB",
        summarize=False,
        log_level="info",
    )
    assert res_no_output.ok is True
    assert res_no_output.summary_yaml is None
    assert fake_provider.closed is True

    fake_provider2 = _FakeProvider()
    monkeypatch.setattr(webcli_runner, "create_state_dict_provider", lambda **_kwargs: fake_provider2)
    monkeypatch.setattr(webcli_runner, "get_runtime_flags", lambda: SimpleNamespace(dry_run=False))
    res_save = webcli_runner.run_web_plan(
        plan_yaml="output: model::/tmp/out\n",
        shard_size="1GB",
        num_workers=3,
        provider="inmemory",
        arena_root=Path(".brainsurgery"),
        arena_segment_size="1GB",
        summarize=True,
        log_level="info",
    )
    assert res_save.ok is True
    assert res_save.written_path is not None
    assert "transforms:" in (res_save.summary_yaml or "")

    fake_provider3 = _FakeProvider()
    monkeypatch.setattr(webcli_runner, "create_state_dict_provider", lambda **_kwargs: fake_provider3)
    monkeypatch.setattr(webcli_runner, "get_runtime_flags", lambda: SimpleNamespace(dry_run=True))
    res_dry = webcli_runner.run_web_plan(
        plan_yaml="output: model::/tmp/out\n",
        shard_size="1GB",
        num_workers=3,
        provider="inmemory",
        arena_root=Path(".brainsurgery"),
        arena_segment_size="1GB",
        summarize=False,
        log_level="info",
    )
    assert res_dry.ok is True

    fake_provider4 = _FakeProvider()
    monkeypatch.setattr(webcli_runner, "create_state_dict_provider", lambda **_kwargs: fake_provider4)
    res_null = webcli_runner.run_web_plan(
        plan_yaml="null\n",
        shard_size="1GB",
        num_workers=3,
        provider="inmemory",
        arena_root=Path(".brainsurgery"),
        arena_segment_size="1GB",
        summarize=False,
        log_level="info",
    )
    assert res_null.ok is True


def test_webcli_server_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []

    class _Server:
        def __init__(self, addr, handler) -> None:
            assert addr == ("127.0.0.1", 9001)
            assert handler is not None

        def serve_forever(self) -> None:
            events.append("serve")
            raise KeyboardInterrupt

        def server_close(self) -> None:
            events.append("close")

    monkeypatch.setattr(webcli_server, "ThreadingHTTPServer", _Server)
    monkeypatch.setattr(webcli_server, "handler_factory", lambda: object)
    webcli_server.serve_webcli(host="127.0.0.1", port=9001)
    assert events == ["serve", "close"]
