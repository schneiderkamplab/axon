from __future__ import annotations

import brainsurgery.engine.runtime_flags_policy as policy


def test_reset_runtime_flags_for_scope_delegates_to_reset(monkeypatch) -> None:
    sentinel = object()
    calls: list[policy.RuntimeFlagLifecycleScope] = []
    set_calls: list[tuple[str, bool]] = []

    def _fake_get():
        return sentinel

    monkeypatch.setattr(policy, "get_runtime_flags", _fake_get)
    monkeypatch.setattr(
        policy, "set_runtime_flag", lambda name, value: set_calls.append((name, value))
    )
    for scope in policy.RuntimeFlagLifecycleScope:
        calls.append(scope)
        assert policy.reset_runtime_flags_for_scope(scope) is sentinel

    assert calls == list(policy.RuntimeFlagLifecycleScope)
    assert set_calls == [
        ("dry_run", False),
        ("preview", False),
        ("verbose", False),
        ("dry_run", False),
        ("preview", False),
        ("verbose", False),
        ("dry_run", False),
        ("preview", False),
        ("verbose", False),
    ]
