"""Session engine: owns the decode loop and the KV cache, model stays pure.

The generated model exposes ``forward(input_ids, ..., past_kv=..., use_cache=...)``
returning ``logits`` plus an updated cache. The ``Session`` drives that interface
step by step so the server can stream, sample, and stop *between* tokens — the
generated ``generate()`` greedy loop is bypassed entirely.

Backends: tensor-facing details (id tensors, result normalization, warm-up)
live in ``server.backends`` behind the ``ModelBackend`` protocol; the Session
itself touches no tensor library, so the same loop runs over MLX and torch.

Concurrency model: one ``DecodeState`` per generation (caches are per-state),
one shared model instance; the serving layer caps concurrent decodes explicitly
and thread-safety of shared-weight forwards is each backend's concern.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from .backends import ModelBackend
from .detokenize import stream_text
from .sampling import SamplerConfig, sample_next_token

__all__ = ["DecodeState", "GenerateParams", "Session"]


@dataclass(frozen=True)
class GenerateParams:
    max_new_tokens: int = 64
    temperature: float = 1.0
    top_k: int | None = None
    top_p: float | None = None
    seed: int | None = None
    stop: tuple[str, ...] = ()

    def sampler(self) -> SamplerConfig:
        return SamplerConfig(
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            seed=self.seed,
        )

    def rng(self):
        return self.sampler().rng()


@dataclass
class DecodeState:
    """One in-flight generation: the cache plus decode bookkeeping."""

    cache: Any = None
    token_ids: list[int] = field(default_factory=list)
    prompt_len: int = 0
    finished: bool = False
    # Logits from the prefill's last position; consumed by the first step()
    # so no extra forward is needed to produce token one.
    pending_logits: Any = None


class Session:
    """Drives a generated model's forward interface for streaming generation."""

    def __init__(
        self,
        backend: ModelBackend,
        tokenizer: Any | None = None,
        *,
        eos_token_id: int | tuple[int, ...] | None = None,
    ) -> None:
        """``backend`` wraps one generated model instance (warm-up happens at
        backend construction); ``Session`` owns only the decode loop."""
        self.backend = backend
        self.model = backend.model
        self.tokenizer = tokenizer
        config = getattr(self.model, "config", None)
        self.config: dict[str, Any] = config if isinstance(config, dict) else {}
        if eos_token_id is None:
            eos_token_id = self.config.get("eos_token_id")
        if eos_token_id is None and tokenizer is not None:
            eos_token_id = getattr(tokenizer, "eos_token_id", None)
        self.eos_token_id = eos_token_id

    # ── construction ────────────────────────────────────────────────────────

    @classmethod
    def from_axon(
        cls,
        axon_path: Path | str,
        *,
        checkpoint_dir: Path | str | None = None,
        backend: str = "mlx",
        dtype: str | None = None,
        emit_profile: str = "inference",
        warm_compile: bool = True,
        tokenizer_dir: Path | str | None = None,
        device: str = "cpu",
    ) -> "Session":
        """Compile a materialized ``.axon`` file and wrap it in a backend.

        ``backend`` selects the adapter (``"mlx"`` or ``"torch"``); ``dtype``
        defaults to ``float16`` on MLX and ``float32`` on torch (CPU-safe).
        """
        path = Path(axon_path)
        tokenizer = cls._load_tokenizer(tokenizer_dir, checkpoint_dir, path)
        if backend == "mlx":
            from .backends import MlxBackend
            from .loading import load_mlx_model

            model = load_mlx_model(
                path,
                checkpoint_dir=checkpoint_dir,
                dtype=dtype or "float16",
                emit_profile=emit_profile,
            )
            return cls(MlxBackend(model, warm_compile=warm_compile), tokenizer)
        if backend == "torch":
            from .backends import TorchBackend
            from .loading import load_torch_model

            model = load_torch_model(
                path,
                checkpoint_dir=checkpoint_dir,
                dtype=dtype or "float32",
                emit_profile=emit_profile,
                device=device,
            )
            return cls(TorchBackend(model, device=device), tokenizer)
        raise ValueError(f"Unknown backend {backend!r}: expected 'mlx' or 'torch'")

    @staticmethod
    def _load_tokenizer(
        tokenizer_dir: Path | str | None, checkpoint_dir: Path | str | None, axon_path: Path
    ) -> Any:
        target = tokenizer_dir or checkpoint_dir
        if target is None:
            from .loading import default_checkpoint_dir

            target = default_checkpoint_dir(axon_path)
        try:
            from transformers import AutoTokenizer

            return AutoTokenizer.from_pretrained(str(target))
        except Exception:
            return None

    # ── generation ──────────────────────────────────────────────────────────

    def prefill(self, input_ids: list[int] | list[list[int]]) -> DecodeState:
        """Run the prompt through the model once; returns a DecodeState with
        the KV cache and the last-position logits pending sampling."""
        backend = self.backend
        ids = backend.ids_2d(input_ids)
        result = backend.forward(ids, past_kv=None, use_cache=True)
        state = DecodeState(
            cache=backend.extract_cache(result),
            token_ids=backend.flat_ids(ids),
            prompt_len=backend.seq_len(ids),
            pending_logits=backend.last_logits(backend.extract_logits(result)),
        )
        return state

    def step(self, state: DecodeState, params: GenerateParams | None = None) -> int | None:
        """Advance one token. Returns the sampled id, or ``None`` on EOS.

        ``step`` does not enforce ``max_new_tokens`` — the caller bounds the
        loop (the HTTP layer does this; see ``server.py``).
        """
        params = params or GenerateParams()
        if state.finished:
            return None
        if state.pending_logits is not None:
            last_logits = state.pending_logits
            state.pending_logits = None
        else:
            last_logits, new_cache = self._forward_step(state)
            if new_cache is not None:
                state.cache = new_cache
        token = sample_next_token(
            self.backend.to_numpy(last_logits), params.sampler(), params.rng()
        )
        state.token_ids.append(token)
        if self._is_eos(token):
            state.finished = True
            return None
        return token

    def generate_ids(
        self,
        prompt: str | list[int],
        params: GenerateParams | None = None,
        stats: dict[str, Any] | None = None,
    ) -> Iterator[int]:
        """Yield sampled token ids one at a time (excludes the prompt).

        When ``stats`` is given, fills ``prompt_tokens``, ``completion_tokens``
        and ``finish_reason`` (``"stop"`` on EOS, ``"length"`` when the token
        budget is exhausted) as generation proceeds. ``completion_tokens``
        counts every generated token, including a terminating EOS — the EOS
        token itself is consumed, never yielded (vLLM/OpenAI usage accounting).
        """
        params = params or GenerateParams()
        prompt_ids = self._encode(prompt)
        if stats is not None:
            stats["prompt_tokens"] = len(prompt_ids)
        state = self.prefill(prompt_ids)
        sampler = params.sampler()
        rng = params.rng()
        completion_tokens = 0
        hit_eos = False
        try:
            for _ in range(max(0, int(params.max_new_tokens))):
                if state.pending_logits is not None:
                    last_logits = state.pending_logits
                    state.pending_logits = None
                else:
                    last_logits, new_cache = self._forward_step(state)
                    if new_cache is not None:
                        state.cache = new_cache
                token = sample_next_token(self.backend.to_numpy(last_logits), sampler, rng)
                state.token_ids.append(token)
                completion_tokens += 1
                if self._is_eos(token):
                    state.finished = True
                    hit_eos = True
                    return
                yield token
            state.finished = True
        finally:
            if stats is not None:
                stats["completion_tokens"] = completion_tokens
                stats["finish_reason"] = "stop" if hit_eos else "length"

    def generate(
        self,
        prompt: str | list[int],
        params: GenerateParams | None = None,
        stats: dict[str, Any] | None = None,
    ) -> Iterator[str]:
        """Stream decoded completion text chunks as generation proceeds.

        When ``stats`` is given it is filled through ``generate_ids``; a
        stop-string truncation overrides ``finish_reason`` to ``"stop"``.
        """
        if self.tokenizer is None:
            raise RuntimeError("Session requires a tokenizer to stream text")
        params = params or GenerateParams()
        token_stream = self.generate_ids(prompt, params, stats)
        yield from stream_text(self.tokenizer, token_stream, params.stop, stats)

    # ── internals ───────────────────────────────────────────────────────────

    def _forward_step(self, state: DecodeState) -> tuple[Any, Any]:
        """One S=1 forward; returns (last-position logits, updated cache)."""
        backend = self.backend
        result = backend.forward(
            backend.step_2d(state.token_ids[-1]),
            past_kv=state.cache,
            use_cache=True,
        )
        return (
            backend.last_logits(backend.extract_logits(result)),
            backend.extract_cache(result),
        )

    def _encode(self, prompt: str | list[int]) -> list[int]:
        if isinstance(prompt, str):
            if self.tokenizer is None:
                raise RuntimeError("String prompts require a tokenizer")
            return list(self.tokenizer.encode(prompt))
        return [int(token) for token in prompt]

    def _is_eos(self, token: int) -> bool:
        eos = self.eos_token_id
        if eos is None:
            return False
        if isinstance(eos, int):
            return token == eos
        return token in eos
