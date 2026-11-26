"""Device→host sync hygiene for the serving paths.

A stray ``.item()`` / ``.tolist()`` / ``.cpu()`` / ``.numpy()`` in a decode
path is a full pipeline sync per call — the GPU drains while Python waits.
This pins the two contracts that keep the serving hot path clean:

1. The fleet engine materializes tensors on host exactly once per decode
   step: the sampler's token ids, which continuous batching needs for
   request bookkeeping (sequence append, detokenization, finish detection).
   Everything else host-known must travel through the Context ``*_host``
   fields (``cu_seqlens_q_host`` / ``cu_seqlens_k_host`` /
   ``context_lens_max``), computed from host lists before the step's H2D
   copies — see ``model_runner.prepare_prefill`` / ``prepare_decode``.

2. The paged kernels' ``*_host`` fast-path fields reject device tensors:
   a tensor smuggled through them degrades into per-layer element extractions
   (``int(x[i])``), which bypass ``.item()``-style sync instrumentation and
   once hid from the sync counters until an aten-level profiler ran.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from synapse.axon.codegen2_torch.paged_attention import varlen_paged_sdpa

H, H_KV, D = 4, 2, 8

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SYNC_PATTERN = re.compile(r"\.(?:item|tolist|cpu|numpy)\(")

# The one sanctioned device→host materialization per decode step.
_SAMPLER_SITE = (
    "server/fleet/engine/model_runner.py",
    "token_ids = self.sampler(logits, **temperatures).tolist()",
)


def test_fleet_engine_has_a_single_sync_site() -> None:
    found: list[tuple[str, str]] = []
    for path in sorted((_REPO_ROOT / "server" / "fleet").rglob("*.py")):
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue  # prose may mention .item(); only code syncs
            if _SYNC_PATTERN.search(stripped):
                found.append(
                    (
                        path.relative_to(_REPO_ROOT).as_posix(),
                        stripped,
                    )
                )
    expected = [_SAMPLER_SITE]
    assert found == expected, (
        "unexpected device→host sync in the fleet engine (each call is a "
        "per-step pipeline sync): "
        f"{found}. Continuous batching materializes data on host only for "
        "request bookkeeping (the sampler's token ids); anything else "
        "host-known must be computed from host lists and carried through "
        "the Context *_host fields."
    )


@pytest.mark.parametrize(
    "host_kwargs",
    [
        {"cu_seqlens_q_host": torch.tensor([0, 4])},
        {"cu_seqlens_k_host": torch.tensor([0, 4])},
    ],
    ids=["q_host", "k_host"],
)
def test_varlen_paged_sdpa_rejects_device_host_fields(host_kwargs: dict) -> None:
    # Shapes are irrelevant: the guard raises before the kernels run.
    with pytest.raises(TypeError, match="host-side Python ints"):
        varlen_paged_sdpa(
            torch.randn(4, H, D),
            torch.zeros(2, 2, H_KV, D),
            torch.zeros(2, 2, H_KV, D),
            num_heads=H,
            num_kv_heads=H_KV,
            head_dim=D,
            cu_seqlens_q=torch.tensor([0, 4]),
            cu_seqlens_k=torch.tensor([0, 4]),
            block_tables=torch.zeros(1, 1, dtype=torch.long),
            **host_kwargs,
        )
