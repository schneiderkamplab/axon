from __future__ import annotations

from contextlib import contextmanager
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import threading
from typing import Any, Iterator
from urllib.error import HTTPError
from urllib import request

from omegaconf import OmegaConf

import brainsurgery  # noqa: F401

from brainsurgery.engine import create_state_dict_provider
from brainsurgery.webui.handler import handler_factory
from brainsurgery.webui.session import SessionState


REPO_ROOT = Path(__file__).resolve().parents[1]


def _post_json(base_url: str, path: str, payload: dict[str, Any]) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url=base_url + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=300) as resp:  # noqa: S310
            body = resp.read().decode("utf-8")
    except HTTPError as exc:
        body = exc.read().decode("utf-8")
    decoded = json.loads(body)
    assert isinstance(decoded, dict)
    return decoded


def _get_json(base_url: str, path: str) -> dict[str, Any]:
    req = request.Request(
        url=base_url + path,
        method="GET",
    )
    with request.urlopen(req, timeout=300) as resp:  # noqa: S310
        body = resp.read().decode("utf-8")
    decoded = json.loads(body)
    assert isinstance(decoded, dict)
    return decoded


@contextmanager
def _running_webui(tmp_path: Path) -> Iterator[str]:
    provider = create_state_dict_provider(
        provider="inmemory",
        model_paths={},
        max_io_workers=8,
        arena_root=Path(".brainsurgery"),
        arena_segment_size="1GB",
    )
    session = SessionState(
        provider=provider,
        lock=threading.Lock(),
        upload_root=tmp_path / "uploads",
    )
    session.upload_root.mkdir(parents=True, exist_ok=True)

    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_factory(session))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}"
    try:
        yield base_url
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        provider.close()


def test_webui_e2e_replays_gpt2_plan_and_exit_summary_contains_full_transforms(
    tmp_path: Path,
) -> None:
    plan_path = REPO_ROOT / "examples" / "gpt2.yaml"
    plan_obj = OmegaConf.to_container(OmegaConf.load(plan_path), resolve=True)
    assert isinstance(plan_obj, dict)

    inputs = plan_obj.get("inputs")
    transforms = plan_obj.get("transforms")
    assert isinstance(inputs, list) and inputs
    assert isinstance(transforms, list) and transforms

    model_input = inputs[0]
    assert isinstance(model_input, str)

    with _running_webui(tmp_path) as base_url:
        load_resp = _post_json(
            base_url,
            "/api/load",
            {
                "server_path": model_input,
                "alias": "model",
            },
        )
        assert load_resp.get("ok") is True

        for raw in transforms:
            assert isinstance(raw, dict) and len(raw) == 1
            transform_name, payload = next(iter(raw.items()))
            assert isinstance(transform_name, str)
            apply_resp = _post_json(
                base_url,
                "/api/apply_transform",
                {
                    "transform": transform_name,
                    "payload": payload,
                },
            )
            assert apply_resp.get("ok") is True, f"{transform_name}: {apply_resp.get('error')}"

        exit_resp = _post_json(
            base_url,
            "/api/apply_transform",
            {"transform": "exit", "payload": {}},
        )
        assert exit_resp.get("ok") is True, exit_resp.get("error")

    output = exit_resp.get("output")
    assert isinstance(output, str) and output.strip()
    summary_obj = OmegaConf.to_container(OmegaConf.create(output), resolve=True)
    assert isinstance(summary_obj, dict)

    expected_transforms = [{"load": {"path": model_input, "alias": "model"}}, *transforms, {"exit": {}}]
    assert summary_obj == {"transforms": expected_transforms}


def test_webui_state_exposes_runtime_flags_and_set_updates_them(tmp_path: Path) -> None:
    with _running_webui(tmp_path) as base_url:
        state_before = _get_json(base_url, "/api/state")
        assert state_before.get("ok") is True
        flags_before = state_before.get("runtime_flags")
        assert isinstance(flags_before, dict)
        assert isinstance(flags_before.get("dry_run"), bool)
        assert isinstance(flags_before.get("verbose"), bool)

        set_resp = _post_json(
            base_url,
            "/api/apply_transform",
            {
                "transform": "set",
                "payload": {"dry-run": True, "verbose": True},
            },
        )
        assert set_resp.get("ok") is True, set_resp.get("error")
        flags_after_set = set_resp.get("runtime_flags")
        assert isinstance(flags_after_set, dict)
        assert flags_after_set.get("dry_run") is True
        assert flags_after_set.get("verbose") is True

        state_after = _get_json(base_url, "/api/state")
        flags_after_state = state_after.get("runtime_flags")
        assert isinstance(flags_after_state, dict)
        assert flags_after_state.get("dry_run") is True
        assert flags_after_state.get("verbose") is True
