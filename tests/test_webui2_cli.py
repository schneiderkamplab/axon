from __future__ import annotations

import brainsurgery.webui2.cli as webui2_module


def test_webui2_opens_browser_and_serves(monkeypatch) -> None:
    opened: list[str] = []
    served: list[tuple[str, int]] = []

    monkeypatch.setattr(webui2_module, "configure_logging", lambda _: None)
    monkeypatch.setattr(webui2_module.webbrowser, "open", lambda url: opened.append(url))
    monkeypatch.setattr(webui2_module, "serve_webui2", lambda *, host, port: served.append((host, port)))

    webui2_module.webui2(host="127.0.0.1", port=8766, log_level="info", open_browser=True)

    assert opened == ["http://127.0.0.1:8766"]
    assert served == [("127.0.0.1", 8766)]
