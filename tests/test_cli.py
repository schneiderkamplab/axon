from __future__ import annotations

import logging
from pathlib import Path

import typer

import brainsurgery.cli as cli


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
