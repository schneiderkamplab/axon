import pytest

from brainsurgery.engine import get_runtime_flags, reset_runtime_flags, set_runtime_flag


def test_engine_flags_defaults_and_updates() -> None:
    reset_runtime_flags()
    flags = get_runtime_flags()
    assert flags.dry_run is False
    assert flags.verbose is False

    set_runtime_flag("dry_run", True)
    set_runtime_flag("verbose", True)
    flags = get_runtime_flags()
    assert flags.dry_run is True
    assert flags.verbose is True

    reset_runtime_flags()


def test_engine_flags_reject_unknown_flag() -> None:
    with pytest.raises(ValueError, match="unknown runtime flag"):
        set_runtime_flag("unknown", True)
