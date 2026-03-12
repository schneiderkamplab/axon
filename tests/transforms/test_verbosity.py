from importlib import import_module

from brainsurgery.core import ResolvedMapping, ResolvedTernaryMapping
from brainsurgery.engine import reset_runtime_flags, set_runtime_flag

module = import_module("brainsurgery.engine.verbosity")


def test_emit_verbose_ternary_and_event_without_detail(capsys) -> None:
    reset_runtime_flags()
    set_runtime_flag("verbose", True)

    module.emit_verbose_ternary_activity(
        "add",
        ResolvedTernaryMapping(
            src_a_model="m",
            src_a_name="a",
            src_a_slice=None,
            src_b_model="m",
            src_b_name="b",
            src_b_slice=None,
            dst_model="m",
            dst_name="c",
            dst_slice=None,
        ),
    )
    module.emit_verbose_event("exit")

    out = capsys.readouterr().out
    assert "add: a, b -> c" in out
    assert "exit" in out

    reset_runtime_flags()


def test_emit_verbose_helpers_prefix_dry_run(capsys) -> None:
    reset_runtime_flags()
    set_runtime_flag("dry_run", True)
    set_runtime_flag("verbose", True)

    module.emit_verbose_binary_activity(
        "copy",
        ResolvedMapping(
            src_model="m",
            src_name="x",
            src_slice=None,
            dst_model="m",
            dst_name="y",
            dst_slice=None,
        ),
    )
    assert "dry-run copy: x -> y" in capsys.readouterr().out

    reset_runtime_flags()
