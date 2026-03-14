from __future__ import annotations

import brainsurgery.webcli.cli as webcli_module


def test_webcli_opens_browser_and_serves(monkeypatch) -> None:
    opened: list[str] = []
    served: list[tuple[str, int]] = []

    monkeypatch.setattr(webcli_module, "configure_logging", lambda _: None)
    monkeypatch.setattr(webcli_module.webbrowser, "open", lambda url: opened.append(url))
    monkeypatch.setattr(webcli_module, "serve_webcli", lambda *, host, port: served.append((host, port)))

    webcli_module.webcli(host="127.0.0.1", port=8765, log_level="info", open_browser=True)

    assert opened == ["http://127.0.0.1:8765"]
    assert served == [("127.0.0.1", 8765)]
