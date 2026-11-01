"""Session engine: owns the decode loop and the KV cache, model stays pure.

The generated model exposes ``forward(input_ids, ..., past_kv=..., use_cache=...)``
returning ``logits`` plus an updated cache. The ``Session`` drives that interface
step by step so the server can stream, sample, and stop *between* tokens — the
generated ``generate()`` greedy loop is bypassed entirely.

Concurrency model: one ``DecodeState`` per generation (caches are per-state), one
shared model instance. MLX forwards are thread-safe over shared weights; the
serving layer caps concurrent decodes explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

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
        model: Any,
        tokenizer: Any | None = None,
        *,
        eos_token_id: int | tuple[int, ...] | None = None,
        warm_compile: bool = True,
        max_kv_length: int = 2048,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        config = getattr(model, "config", None)
        self.config: dict[str, Any] = config if isinstance(config, dict) else {}
        if eos_token_id is None:
            eos_token_id = self.config.get("eos_token_id")
        if eos_token_id is None and tokenizer is not None:
            eos_token_id = getattr(tokenizer, "eos_token_id", None)
        self.eos_token_id = eos_token_id
        if warm_compile and hasattr(model, "compile"):
            model.compile(max_kv_length=max_kv_length)

    # ── construction ────────────────────────────────────────────────────────

    @classmethod
    def from_axon(
        cls,
        axon_path: Path | str,
        *,
        checkpoint_dir: Path | str | None = None,
        dtype: str = "float16",
        emit_profile: str = "inference",
        warm_compile: bool = True,
        tokenizer_dir: Path | str | None = None,
    ) -> "Session":
        from .loading import default_checkpoint_dir, load_mlx_model

        path = Path(axon_path)
        model = load_mlx_model(
            path,
            checkpoint_dir=checkpoint_dir,
            dtype=dtype,
            emit_profile=emit_profile,
        )
        tokenizer = cls._load_tokenizer(tokenizer_dir, checkpoint_dir, path)
        return cls(model, tokenizer, warm_compile=warm_compile)

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
        mx = self._mx()
        ids = self._as_2d(input_ids, mx)
        result = self.model.forward(ids, past_kv=None, use_cache=True)
        logits = self._extract_logits(result)
        state = DecodeState(
            cache=self._extract_cache(result),
            token_ids=[int(v) for v in list(ids.reshape(-1))],
            prompt_len=int(ids.shape[1]),
            pending_logits=self._last_logits(logits),
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
            self._to_numpy(last_logits), params.sampler(), params.rng()
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
                token = sample_next_token(self._to_numpy(last_logits), sampler, rng)
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
        generated: list[int] = []
        emitted = 0
        stop_hit = False
        token_stream = self.generate_ids(prompt, params, stats)
        for token in token_stream:
            generated.append(token)
            text = self.tokenizer.decode(generated)
            new_text = text[emitted:]
            emitted += len(new_text)
            cut = self._truncate_at_stop(new_text, params.stop)
            if cut is not None:
                stop_hit = True
                if cut:
                    yield cut
                break
            if new_text:
                yield new_text
        token_stream.close()  # run generate_ids' finally deterministically
        if stats is not None and stop_hit:
            stats["finish_reason"] = "stop"

    # ── internals ───────────────────────────────────────────────────────────

    def _forward_step(self, state: DecodeState) -> tuple[Any, Any]:
        """One S=1 forward; returns (last-position logits, updated cache)."""
        mx = self._mx()
        result = self.model.forward(
            mx.array([[state.token_ids[-1]]], dtype=mx.int64),
            past_kv=state.cache,
            use_cache=True,
        )
        return self._last_logits(self._extract_logits(result)), self._extract_cache(result)

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

    @staticmethod
    def _truncate_at_stop(text: str, stops: tuple[str, ...]) -> str | None:
        """Return text truncated before the first stop sequence, or None."""
        for stop in stops:
            if stop and stop in text:
                return text.split(stop, 1)[0]
        return None

    def _mx(self) -> Any:
        import mlx.core as mx

        return mx

    @staticmethod
    def _as_2d(input_ids: list[int] | list[list[int]], mx: Any) -> Any:
        if input_ids and isinstance(input_ids[0], (list, tuple)):
            return mx.array([list(row) for row in input_ids], dtype=mx.int64)
        return mx.array([list(input_ids)], dtype=mx.int64)

    @staticmethod
    def _extract_logits(result: Any) -> Any:
        if isinstance(result, dict):
            return result.get("logits")
        if isinstance(result, (list, tuple)):
            return result[0]
        return result

    @staticmethod
    def _extract_cache(result: Any, previous: Any = None) -> Any:
        if isinstance(result, dict):
            for key in ("new_kv", "past_kv", "cache"):
                if key in result and result[key] is not None:
                    return result[key]
            return previous
        if isinstance(result, (list, tuple)) and len(result) > 1:
            return result[1]
        return previous

    @staticmethod
    def _last_logits(logits: Any) -> Any:
        if logits is None:
            raise RuntimeError("Model forward returned no logits")
        return logits[:, -1, :]

    @staticmethod
    def _to_numpy(values: Any) -> Any:
        import numpy as np

        return np.asarray(values)
