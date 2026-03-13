from __future__ import annotations

import brainsurgery.webui.cli as webui_module


def test_webui_opens_browser_and_serves(monkeypatch) -> None:
    opened: list[str] = []
    served: list[tuple[str, int]] = []

    monkeypatch.setattr(webui_module, "configure_logging", lambda _: None)
    monkeypatch.setattr(webui_module.webbrowser, "open", lambda url: opened.append(url))
    monkeypatch.setattr(webui_module, "serve_webui", lambda *, host, port: served.append((host, port)))

    webui_module.webui(host="127.0.0.1", port=8765, log_level="info", open_browser=True)

    assert opened == ["http://127.0.0.1:8765"]
    assert served == [("127.0.0.1", 8765)]
