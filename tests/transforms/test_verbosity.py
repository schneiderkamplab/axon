from importlib import import_module

from brainsurgery.engine import (
    RuntimeFlagLifecycleScope,
    reset_runtime_flags_for_scope,
    set_runtime_flag,
)

module = import_module("brainsurgery.engine.verbosity")


def test_emit_verbose_ternary_and_event_without_detail(capsys) -> None:
    reset_runtime_flags_for_scope(RuntimeFlagLifecycleScope.CLI_RUN)
    set_runtime_flag("verbose", True)

    module.emit_verbose_ternary_activity(
        "add",
        "a",
        "b",
        "c",
    )
    module.emit_verbose_event("exit")

    out = capsys.readouterr().out
    assert "add: a, b -> c" in out
    assert "exit" in out

    reset_runtime_flags_for_scope(RuntimeFlagLifecycleScope.CLI_RUN)


def test_emit_verbose_helpers_prefix_dry_run(capsys) -> None:
    reset_runtime_flags_for_scope(RuntimeFlagLifecycleScope.CLI_RUN)
    set_runtime_flag("dry_run", True)
    set_runtime_flag("verbose", True)

    module.emit_verbose_binary_activity(
        "copy",
        "x",
        "y",
    )
    assert "dry-run copy: x -> y" in capsys.readouterr().out

    reset_runtime_flags_for_scope(RuntimeFlagLifecycleScope.CLI_RUN)
