from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
import typer

import brainsurgery.cli as cli
import brainsurgery.cli.cli as cli_module


def test_configure_logging_sets_root_and_named_logger_levels() -> None:
    cli.configure_logging("debug")

    assert logging.getLogger().level == logging.DEBUG
    assert cli.logger.level == logging.DEBUG


def test_configure_logging_rejects_unknown_level() -> None:
    try:
        cli.configure_logging("verbose")
    except typer.BadParameter as exc:
        assert "log-level must be one of" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected invalid log-level error")


class _Provider:
    def __init__(self) -> None:
        self.closed = False
        self.save_calls: list[tuple[object, str, int]] = []

    def save_output(self, plan: object, *, default_shard_size: str, max_io_workers: int) -> Path:
        self.save_calls.append((plan, default_shard_size, max_io_workers))
        return Path("/tmp/out.safetensors")

    def close(self) -> None:
        self.closed = True


def test_run_executes_configured_and_interactive_and_writes_summary(monkeypatch: pytest.MonkeyPatch) -> None:
    raw_plan = {
        "inputs": ["model::/tmp/in.safetensors"],
        "output": {"path": "/tmp/out.safetensors"},
        "transforms": [{"copy": {"from": "x", "to": "y"}}],
    }
    surgery_plan = SimpleNamespace(
        inputs={"model": Path("/tmp/in.safetensors")},
        transforms=[object()],
        output=SimpleNamespace(path=Path("/tmp/out.safetensors")),
    )
    provider = _Provider()
    summary_calls: list[dict[str, object]] = []

    monkeypatch.setattr(cli_module, "load_cli_config", lambda _: raw_plan)
    monkeypatch.setattr(cli_module, "compile_plan", lambda _: surgery_plan)
    monkeypatch.setattr(cli_module, "create_state_dict_provider", lambda **_: provider)
    monkeypatch.setattr(cli_module, "configure_history", lambda: None)
    monkeypatch.setattr(
        cli_module,
        "_execute_configured_transforms",
        lambda **_: (True, [{"copy": {"from": "x", "to": "y"}}]),
    )
    monkeypatch.setattr(cli_module, "_run_interactive_session", lambda **_: (False, [{"exit": {}}]))
    monkeypatch.setattr(
        cli_module,
        "write_executed_plan_summary",
        lambda **kwargs: summary_calls.append(kwargs),
    )

    cli.run(
        config_items=["plan.yaml"],
        interactive=True,
        summarize=True,
        summarize_path=Path("/tmp/summary.yaml"),
        log_level="info",
    )

    assert provider.closed is True
    assert len(provider.save_calls) == 1
    assert len(summary_calls) == 1
    assert summary_calls[0]["transforms"] == [{"copy": {"from": "x", "to": "y"}}, {"exit": {}}]


def test_run_skips_save_when_no_output(monkeypatch: pytest.MonkeyPatch) -> None:
    raw_plan = {"inputs": ["model::/tmp/in.safetensors"], "transforms": [{"help": {}}]}
    surgery_plan = SimpleNamespace(
        inputs={"model": Path("/tmp/in.safetensors")},
        transforms=[object()],
        output=None,
    )
    provider = _Provider()

    monkeypatch.setattr(cli_module, "load_cli_config", lambda _: raw_plan)
    monkeypatch.setattr(cli_module, "compile_plan", lambda _: surgery_plan)
    monkeypatch.setattr(cli_module, "create_state_dict_provider", lambda **_: provider)
    monkeypatch.setattr(cli_module, "configure_history", lambda: None)
    monkeypatch.setattr(cli_module, "_execute_configured_transforms", lambda **_: (True, [{"help": {}}]))
    monkeypatch.setattr(
        cli_module,
        "write_executed_plan_summary",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError(f"unexpected summary call {kwargs}")),
    )

    cli.run(
        config_items=["plan.yaml"],
        interactive=False,
        summarize=False,
        log_level="info",
    )

    assert provider.closed is True
    assert provider.save_calls == []


def test_run_wraps_provider_error_as_bad_parameter(monkeypatch: pytest.MonkeyPatch) -> None:
    raw_plan = {
        "inputs": ["model::/tmp/in.safetensors"],
        "output": {"path": "/tmp/out.safetensors"},
        "transforms": [{"help": {}}],
    }
    surgery_plan = SimpleNamespace(
        inputs={"model": Path("/tmp/in.safetensors")},
        transforms=[object()],
        output=SimpleNamespace(path=Path("/tmp/out.safetensors")),
    )

    monkeypatch.setattr(cli_module, "load_cli_config", lambda _: raw_plan)
    monkeypatch.setattr(cli_module, "compile_plan", lambda _: surgery_plan)
    monkeypatch.setattr(cli_module, "configure_history", lambda: None)

    def _raise_provider_error(**_: object) -> None:
        raise cli_module.ProviderError("bad provider config")

    monkeypatch.setattr(cli_module, "create_state_dict_provider", _raise_provider_error)

    with pytest.raises(typer.BadParameter, match="bad provider config"):
        cli.run(config_items=["plan.yaml"], log_level="info")
