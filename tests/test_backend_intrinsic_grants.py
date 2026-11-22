"""Policy guard: opt-in-only backend intrinsics are never granted by a bare target.

Root AGENTS.md special-casing policy: intrinsics that change the artifact
contract (serving-specific state, kernel replacement) must be granted only
when named explicitly — a bare ``target`` or ``target:*`` must exclude them.
See ``_OPT_IN_ONLY_BACKEND_INTRINSICS`` in ``synapse/axon/graph_ir/optimize.py``.

Currently opt-in-only:
- ``__torch_paged_attention`` — replaces attention with the paged runtime and
  requires the serving emit profile.
- ``__mlx_rope`` — emits per-instance frequency-table caches and S==1
  decode-offset machinery that are only correct under serving execution;
  the remaining MLX intrinsics are stateless function rewrites safe for
  training and inference alike.
"""

import pytest

from synapse.axon.graph_ir.optimize import (
    _OPT_IN_ONLY_BACKEND_INTRINSICS,
    _default_backend_intrinsics,
    _parse_backend_intrinsics,
)


def test_opt_in_only_set_contents() -> None:
    assert _OPT_IN_ONLY_BACKEND_INTRINSICS == frozenset(
        {"__torch_paged_attention", "__mlx_rope"}
    )


def test_bare_mlx_grant_excludes_rope() -> None:
    target, enabled = _parse_backend_intrinsics("codegen2-mlx")
    assert target == "codegen2-mlx"
    assert "__mlx_rope" not in enabled
    assert "__mlx_sdpa" in enabled
    assert "__mlx_rmsnorm_scaled" in enabled


def test_bare_mlx_star_matches_bare_grant() -> None:
    assert _parse_backend_intrinsics("codegen2-mlx") == _parse_backend_intrinsics(
        "codegen2-mlx:*"
    )


def test_named_mlx_rope_is_grantable() -> None:
    _, enabled = _parse_backend_intrinsics("codegen2-mlx:__mlx_rope")
    assert enabled == frozenset({"__mlx_rope"})


def test_named_short_form_resolves_to_rope() -> None:
    _, enabled = _parse_backend_intrinsics("codegen2-mlx:rope")
    assert enabled == frozenset({"__mlx_rope"})


def test_bare_torch_grant_excludes_paged_attention() -> None:
    _, enabled = _parse_backend_intrinsics("codegen2-torch")
    assert "__torch_paged_attention" not in enabled
    assert "__torch_sdpa" in enabled


def test_named_torch_paged_attention_is_grantable() -> None:
    _, enabled = _parse_backend_intrinsics("codegen2-torch:__torch_paged_attention")
    assert enabled == frozenset({"__torch_paged_attention"})


def test_every_opt_in_only_intrinsic_is_excluded_but_named_grantable() -> None:
    for target in ("codegen2-torch", "codegen2-mlx"):
        prefix = {"codegen2-torch": "__torch_", "codegen2-mlx": "__mlx_"}[target]
        default = _default_backend_intrinsics(target)
        opt_in_only = {name for name in _OPT_IN_ONLY_BACKEND_INTRINSICS if name.startswith(prefix)}
        assert opt_in_only, f"no opt-in-only intrinsics tracked for {target}"
        for name in opt_in_only:
            assert name not in default
            _, enabled = _parse_backend_intrinsics(f"{target}:{name}")
            assert enabled == frozenset({name})


def test_unknown_intrinsic_still_rejected() -> None:
    with pytest.raises(ValueError, match="unknown codegen2-mlx graph intrinsic"):
        _parse_backend_intrinsics("codegen2-mlx:__mlx_rope_typo")


def test_wrong_target_intrinsic_still_rejected() -> None:
    with pytest.raises(ValueError, match="unknown codegen2-mlx graph intrinsic"):
        _parse_backend_intrinsics("codegen2-mlx:__torch_paged_attention")
