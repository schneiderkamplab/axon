from __future__ import annotations

import logging
from pathlib import Path
import runpy
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
    reset_calls: list[bool] = []

    monkeypatch.setattr(cli_module, "load_cli_config", lambda _: raw_plan)
    monkeypatch.setattr(cli_module, "compile_plan", lambda _: surgery_plan)
    monkeypatch.setattr(cli_module, "create_state_dict_provider", lambda **_: provider)
    monkeypatch.setattr(cli_module, "configure_history", lambda: None)
    monkeypatch.setattr(cli_module, "reset_runtime_flags", lambda: reset_calls.append(True))
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
    assert reset_calls == [True]
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


def test_run_skips_output_save_when_dry_run(monkeypatch: pytest.MonkeyPatch) -> None:
    raw_plan = {
        "inputs": ["model::/tmp/in.safetensors"],
        "output": {"path": "/tmp/out.safetensors"},
        "transforms": [{"set": {"dry-run": True}}],
    }
    surgery_plan = SimpleNamespace(
        inputs={"model": Path("/tmp/in.safetensors")},
        transforms=[object()],
        output=SimpleNamespace(path=Path("/tmp/out.safetensors")),
    )
    provider = _Provider()

    monkeypatch.setattr(cli_module, "load_cli_config", lambda _: raw_plan)
    monkeypatch.setattr(cli_module, "compile_plan", lambda _: surgery_plan)
    monkeypatch.setattr(cli_module, "create_state_dict_provider", lambda **_: provider)
    monkeypatch.setattr(cli_module, "configure_history", lambda: None)
    monkeypatch.setattr(cli_module, "reset_runtime_flags", lambda: None)
    monkeypatch.setattr(
        cli_module,
        "_execute_configured_transforms",
        lambda **_: (True, [{"set": {"dry-run": True}}]),
    )
    monkeypatch.setattr(
        cli_module,
        "get_runtime_flags",
        lambda: SimpleNamespace(dry_run=True, verbose=False),
    )

    cli.run(
        config_items=["plan.yaml"],
        interactive=False,
        summarize=False,
        log_level="info",
    )

    assert provider.closed is True
    assert provider.save_calls == []


def test_run_skips_summary_file_write_when_dry_run(monkeypatch: pytest.MonkeyPatch) -> None:
    raw_plan = {
        "inputs": ["model::/tmp/in.safetensors"],
        "output": None,
        "transforms": [{"set": {"dry-run": True}}],
    }
    surgery_plan = SimpleNamespace(
        inputs={"model": Path("/tmp/in.safetensors")},
        transforms=[object()],
        output=None,
    )
    provider = _Provider()
    summary_calls: list[dict[str, object]] = []

    monkeypatch.setattr(cli_module, "load_cli_config", lambda _: raw_plan)
    monkeypatch.setattr(cli_module, "compile_plan", lambda _: surgery_plan)
    monkeypatch.setattr(cli_module, "create_state_dict_provider", lambda **_: provider)
    monkeypatch.setattr(cli_module, "configure_history", lambda: None)
    monkeypatch.setattr(cli_module, "reset_runtime_flags", lambda: None)
    monkeypatch.setattr(
        cli_module,
        "_execute_configured_transforms",
        lambda **_: (True, [{"set": {"dry-run": True}}]),
    )
    monkeypatch.setattr(
        cli_module,
        "get_runtime_flags",
        lambda: SimpleNamespace(dry_run=True, verbose=False),
    )
    monkeypatch.setattr(
        cli_module,
        "write_executed_plan_summary",
        lambda **kwargs: summary_calls.append(kwargs),
    )

    cli.run(
        config_items=["plan.yaml"],
        interactive=False,
        summarize=True,
        summarize_path=Path("/tmp/summary.yaml"),
        log_level="info",
    )

    assert provider.closed is True
    assert summary_calls == []


def test_execute_configured_transforms_normalizes_specs(monkeypatch: pytest.MonkeyPatch) -> None:
    called: dict[str, object] = {}

    def _fake_normalize(specs: object) -> list[dict[str, object]]:
        called["specs"] = specs
        return [{"help": {}}]

    def _fake_execute(pairs: object, provider: object, interactive: bool) -> tuple[bool, list[dict[str, object]]]:
        called["provider"] = provider
        called["interactive"] = interactive
        called["pairs"] = list(pairs)  # type: ignore[arg-type]
        return True, [{"help": {}}]

    monkeypatch.setattr(cli_module, "normalize_transform_specs", _fake_normalize)
    monkeypatch.setattr(cli_module, "execute_transform_pairs", _fake_execute)

    surgery_plan = SimpleNamespace(transforms=[object()])
    should_continue, executed = cli_module._execute_configured_transforms(
        raw_plan={"transforms": [{"help": {}}]},
        surgery_plan=surgery_plan,
        state_dict_provider="provider",
    )

    assert should_continue is True
    assert executed == [{"help": {}}]
    assert called["specs"] == [{"help": {}}]
    assert called["interactive"] is False
    assert called["provider"] == "provider"
    assert len(called["pairs"]) == 1


def test_build_interactive_raw_plan_prefers_alias_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli_module, "list_model_aliases", lambda _: {"z", "a"})

    built = cli_module._build_interactive_raw_plan(
        raw_plan={"inputs": ["orig::/tmp/in.safetensors"], "output": {"path": "/tmp/out"}},
        state_dict_provider=object(),
        extra_specs=[{"help": {}}],
    )

    assert built["inputs"] == ["a::/dev/null", "z::/dev/null"]
    assert built["output"] == {"path": "/tmp/out"}
    assert built["transforms"] == [{"help": {}}]


def test_run_interactive_session_retries_on_compile_error_then_exits(monkeypatch: pytest.MonkeyPatch) -> None:
    prompts = iter([[{"help": {}}], [{"exit": {}}]])
    calls: list[tuple[object, bool]] = []

    monkeypatch.setattr(cli_module, "prompt_interactive_transform", lambda **_: next(prompts))
    monkeypatch.setattr(
        cli_module,
        "_build_interactive_raw_plan",
        lambda **kwargs: {"transforms": kwargs["extra_specs"]},
    )

    def _fake_compile(raw_plan: dict[str, object]) -> object:
        if raw_plan["transforms"] == [{"help": {}}]:
            raise RuntimeError("bad interactive input")
        return SimpleNamespace(transforms=[object()])

    monkeypatch.setattr(cli_module, "compile_plan", _fake_compile)

    def _fake_execute(pairs: object, provider: object, interactive: bool) -> tuple[bool, list[dict[str, object]]]:
        calls.append((list(pairs), interactive))  # type: ignore[arg-type]
        return False, [{"exit": {}}]

    monkeypatch.setattr(cli_module, "execute_transform_pairs", _fake_execute)

    should_continue, executed = cli_module._run_interactive_session(
        raw_plan={"inputs": [], "output": None},
        state_dict_provider=object(),
    )

    assert should_continue is False
    assert executed == [{"exit": {}}]
    assert len(calls) == 1
    assert calls[0][1] is True


def test_run_interactive_session_returns_when_prompt_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli_module, "prompt_interactive_transform", lambda **_: None)
    should_continue, executed = cli_module._run_interactive_session(
        raw_plan={"inputs": [], "output": None},
        state_dict_provider=object(),
    )
    assert should_continue is True
    assert executed == []


def test_cli_module_main_guard_executes_app(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[bool] = []
    import sys
    import typer

    class _FakeTyper:
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs

        def command(self, *args, **kwargs):
            del args, kwargs
            return lambda fn: fn

        def callback(self, *args, **kwargs):
            del args, kwargs
            return lambda fn: fn

        def __call__(self, *args, **kwargs):
            del args, kwargs
            calls.append(True)

    monkeypatch.setattr(typer, "Typer", _FakeTyper)
    prior = sys.modules.pop("brainsurgery.cli.cli", None)
    try:
        runpy.run_module("brainsurgery.cli.cli", run_name="__main__")
    finally:
        if prior is not None:
            sys.modules["brainsurgery.cli.cli"] = prior
    assert calls == [True]
