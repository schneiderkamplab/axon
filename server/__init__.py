"""Serving layer for compiled Axon models.

Modular layout (each part is independently replaceable):

- ``sampling``  – backend-agnostic sampler chain (greedy/temperature/top-k/top-p).
- ``loading``   – materialized ``.axon`` file → generated model instance.
- ``engine``    – ``Session``: prefill/step/generate loop over the generated model's
                  ``forward(..., past_kv=..., use_cache=...)`` interface, holding the
                  KV cache outside the model.
- ``server``    – minimal stdlib OpenAI-compatible HTTP server (SSE streaming) that
                  is a thin adapter over ``Session``.

The server is deliberately a thin adapter: the real integration point is the
``Session`` engine API. True cross-request batched decoding is deferred until the
cache contract supports batched per-sequence caches; v1 interleaves concurrent
requests through a bounded semaphore.
"""

from .engine import DecodeState, GenerateParams, Session
from .sampling import SamplerConfig, sample_next_token
from .server import serve

__all__ = [
    "DecodeState",
    "GenerateParams",
    "SamplerConfig",
    "Session",
    "sample_next_token",
    "serve",
]
