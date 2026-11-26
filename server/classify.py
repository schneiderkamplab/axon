"""Token-classification serving: the ``/v1/classify`` substance.

The generation engines (``engine``, ``fleet``) decode autoregressively; a
token-classification checkpoint instead labels every input token in a single
forward pass. This module wraps such a model — the same compiled MLX artifact
the generation path loads — with the pieces classification needs: per-token
logits, a constrained Viterbi decoder over the label scheme's transition
structure, and span grouping into entities.

The label scheme is read from the checkpoint's ``id2label`` (background label
``O`` plus ``B-``/``I-``/``E-``/``S-`` variants of each category), so the
decoder is generic BIOES, not tied to one taxonomy. Optional transition biases
come from the checkpoint's ``viterbi_calibration.json`` when present.
"""

from __future__ import annotations

import json
import re
import threading
from pathlib import Path
from typing import Any

import numpy as np

try:  # MLX is Apple-Silicon-only; the classify session needs it, the rest of
    import mlx.core as mx  # the server (generation on torch/CUDA) does not.
except ImportError:  # Guarded here so Linux hosts can import the server and
    mx = None  # run the generation path; classify fails fast at load.

__all__ = [
    "is_token_classifier",
    "constrained_viterbi_decode",
    "group_entities",
    "ClassifySession",
]

_ARCH_SUFFIXES = ("ForTokenClassification",)
_NEG = float("-inf")
_BIOES_TAG = re.compile(r"^(B|I|E|S)-(.+)$")


def is_token_classifier(config: dict[str, Any]) -> bool:
    """Dispatch on checkpoint metadata: a label map marks a token classifier.

    Generation checkpoints name a vocabulary; classification checkpoints name
    labels. ``id2label`` with more than the background label is the
    HF-standard signal, with the architecture suffix as a fallback.
    """
    id2label = config.get("id2label")
    if isinstance(id2label, dict) and len(id2label) > 1:
        return True
    architectures = config.get("architectures") or []
    return any(
        str(name).endswith(_ARCH_SUFFIXES)
        for name in architectures
        if isinstance(name, str)
    )


def _parse_label(label: str) -> tuple[str, str]:
    """Split a label into (tag, category); background is ("O", "")."""
    match = _BIOES_TAG.match(label)
    if match:
        return match.group(1), match.group(2)
    return "O", ""


def _bias(biases: dict[str, float], name: str) -> float:
    value = biases.get(name, 0.0)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def constrained_viterbi_decode(
    logits: np.ndarray,
    labels: list[str],
    biases: dict[str, float] | None = None,
) -> list[int]:
    """Decode one sequence of per-token logits under BIOES constraints.

    Linear-chain Viterbi over the label set with start, transition, and end
    terms: a span may only begin where the scheme allows (background or after
    a closed span), only continue within its own category, and must close
    before the sequence ends. The six calibration biases shift whole
    transition classes — background persistence, span entry, span
    continuation, span closure, and boundary-to-boundary handoff:

    - ``transition_bias_background_stay``      O -> O
    - ``transition_bias_background_to_start``  O -> B-*, S-*   (span entry)
    - ``transition_bias_end_to_background``    E-*, S-* -> O   (span closure)
    - ``transition_bias_end_to_start``         E-*, S-* -> B-*, S-*  (handoff)
    - ``transition_bias_inside_to_continue``   B-*, I-* -> I-* (continuation)
    - ``transition_bias_inside_to_end``        B-*, I-* -> E-* (closure)

    Scores are raw logits: per-position constants cancel in the path argmax,
    so no softmax is needed. Returns the best valid path as label indices.
    """
    biases = biases or {}
    count = len(labels)
    tags = [_parse_label(label) for label in labels]

    def is_background(index: int) -> bool:
        return tags[index][0] == "O"

    def opens_span(index: int) -> bool:
        return tags[index][0] in ("B", "S")

    def is_open_span(index: int) -> bool:
        return tags[index][0] in ("B", "I")

    def is_closed_span(index: int) -> bool:
        return tags[index][0] in ("E", "S")

    def same_category(a: int, b: int) -> bool:
        return tags[a][1] == tags[b][1]

    def transition_bias(prev: int, cur: int) -> float:
        tag_prev, tag_cur = tags[prev][0], tags[cur][0]
        if tag_prev == "O" and tag_cur == "O":
            return _bias(biases, "transition_bias_background_stay")
        if tag_prev == "O" and tag_cur in ("B", "S"):
            return _bias(biases, "transition_bias_background_to_start")
        if tag_prev in ("E", "S") and tag_cur == "O":
            return _bias(biases, "transition_bias_end_to_background")
        if tag_prev in ("E", "S") and tag_cur in ("B", "S"):
            return _bias(biases, "transition_bias_end_to_start")
        if tag_prev in ("B", "I") and tag_cur == "I":
            return _bias(biases, "transition_bias_inside_to_continue")
        if tag_prev in ("B", "I") and tag_cur == "E":
            return _bias(biases, "transition_bias_inside_to_end")
        return 0.0

    def allowed(prev: int, cur: int) -> bool:
        tag_prev, cat_prev = tags[prev]
        tag_cur, cat_cur = tags[cur]
        if tag_prev == "O":
            return tag_cur in ("O", "B", "S")
        if tag_prev == "B" or tag_prev == "I":
            return tag_cur in ("I", "E") and cat_cur == cat_prev
        # E / S close a span: background or a new span may follow.
        return tag_cur in ("O", "B", "S")

    emissions = np.asarray(logits, dtype=np.float64)
    if emissions.ndim != 2 or emissions.shape[1] != count:
        raise ValueError(
            f"logits must be [sequence, {count}] for {count} labels, got {emissions.shape}"
        )

    trans = np.full((count, count), _NEG, dtype=np.float64)
    for prev in range(count):
        for cur in range(count):
            if allowed(prev, cur):
                trans[prev, cur] = transition_bias(prev, cur)

    start = np.full(count, _NEG, dtype=np.float64)
    for cur in range(count):
        if opens_span(cur) or is_background(cur):
            start[cur] = 0.0

    end_ok = np.array([is_background(i) or is_closed_span(i) for i in range(count)])

    length = emissions.shape[0]
    if length == 0:
        return []
    dp = start + emissions[0]
    backpointers = np.zeros((length, count), dtype=np.int32)
    for step in range(1, length):
        candidates = dp[:, None] + trans  # [prev, cur]
        backpointers[step] = np.argmax(candidates, axis=0)
        dp = candidates[backpointers[step], np.arange(count)] + emissions[step]
    final = np.where(end_ok, dp, _NEG)
    path = [int(np.argmax(final))]
    for step in range(length - 1, 0, -1):
        path.append(int(backpointers[step, path[-1]]))
    path.reverse()
    return path


def group_entities(
    labels: list[str],
    offsets: list[tuple[int, int]] | None,
    text: str | None,
) -> list[dict[str, Any]]:
    """Group a decoded label path into entity spans.

    Well-formed BIOES paths group directly (B opens, I continues, E closes,
    S stands alone). Raw-argmax paths can violate the scheme; stray ``I``/``
    E`` tags then salvage into their own span rather than being dropped.
    ``offsets`` are the tokenizer's ``(start, end)`` character pairs; when
    unavailable the entities keep token indices only.
    """
    entities: list[dict[str, Any]] = []
    open_start: int | None = None
    open_category: str = ""

    def close(open_token: int | None, end_token: int, category: str) -> None:
        if open_token is None:
            return
        entry: dict[str, Any] = {
            "label": category,
            "token_start": open_token,
            "token_end": end_token,
        }
        if offsets is not None and text is not None:
            char_start = int(offsets[open_token][0])
            char_end = int(offsets[end_token][1])
            entry["start"] = char_start
            entry["end"] = char_end
            entry["text"] = text[char_start:char_end]
        entities.append(entry)

    for index, label in enumerate(labels):
        tag, category = _parse_label(label)
        if tag == "O":
            close(open_start, index - 1, open_category)
            open_start = None
        elif tag == "B":
            close(open_start, index - 1, open_category)
            open_start, open_category = index, category
        elif tag == "S":
            close(open_start, index - 1, open_category)
            close(index, index, category)
            open_start = None
        elif tag == "I":
            if open_start is not None and category == open_category:
                continue
            close(open_start, index - 1, open_category)
            open_start, open_category = index, category
        elif tag == "E":
            if open_start is not None and category == open_category:
                close(open_start, index, category)
            else:
                close(open_start, index - 1, open_category)
                close(index, index, category)
            open_start = None
    close(open_start, len(labels) - 1, open_category)
    return entities


def _load_viterbi_biases(checkpoint: Path) -> dict[str, float]:
    """Read the default operating point's transition biases, when calibrated."""
    path = checkpoint / "viterbi_calibration.json"
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            calibration = json.load(handle)
        operating_points = calibration.get("operating_points") or {}
        default = operating_points.get("default") or {}
        raw = default.get("biases") or {}
        return {str(key): float(value) for key, value in raw.items()}
    except (OSError, ValueError, TypeError):
        return {}


class ClassifySession:
    """A loaded token-classification model plus its decode configuration.

    Straight forward passes (no KV cache, no decode loop): each request
    tokenizes, runs one forward, and decodes the label path. The forward is
    serialized behind a lock because the compiled MLX model is shared.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        id2label: dict[int, str],
        biases: dict[str, float],
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.id2label = dict(id2label)
        self.biases = dict(biases)
        self._forward_lock = threading.Lock()

    @classmethod
    def from_axon(
        cls,
        axon_path: Path | str,
        *,
        checkpoint_dir: Path | str | None = None,
        dtype: str | None = None,
        warm_compile: bool = True,
    ) -> "ClassifySession":
        from .loading import default_checkpoint_dir, load_mlx_model, load_model_config

        if mx is None:
            raise RuntimeError(
                "the /v1/classify path requires mlx (Apple Silicon);"
                " this host has no MLX, so token classification cannot load here"
            )
        path = Path(axon_path)
        checkpoint = (
            Path(checkpoint_dir).resolve() if checkpoint_dir else default_checkpoint_dir(path)
        )
        config = load_model_config(checkpoint)
        resolved_dtype = dtype or _declared_dtype(config) or "float16"
        model = load_mlx_model(
            path,
            checkpoint_dir=checkpoint,
            dtype=resolved_dtype,
            emit_profile="inference",
            # Loader defaults (optimize_ast=True + MLX_INTRINSICS_SPEC naming
            # the full serving set) emit the same artifact the harness parity
            # path produces for this checkpoint — verified byte-identical by
            # log/probe-intrinsics-grant-parity.py (optimize_ast has no effect
            # on the emitted code; the named spec reproduces the pre-flag
            # bare-grant set exactly). No ad-hoc overrides here: the serving
            # path must not drift from the parity-tested loader path.
        )
        id2label = {
            int(key): str(value)
            for key, value in (config.get("id2label") or {}).items()
        }
        session = cls(model, _load_tokenizer(checkpoint), id2label, _load_viterbi_biases(checkpoint))
        if warm_compile:
            session.warmup()
        return session

    def warmup(self) -> None:
        """Prime kernels with a tiny forward; never fails startup."""
        try:
            ids = mx.array([[0, 1, 2, 3]], dtype=mx.int32)
            mask = mx.array([[1, 1, 1, 1]], dtype=mx.int32)
            with self._forward_lock:
                out = self._forward(ids, mask)
                mx.eval(out)
        except Exception:  # noqa: BLE001 - warmup is best effort
            pass

    def _forward(self, input_ids: mx.array, attn_mask: mx.array) -> Any:
        out = self.model(input_ids, attn_mask=attn_mask)
        logits = out if not isinstance(out, dict) else out.get("logits", out)
        if isinstance(logits, (list, tuple)):
            logits = logits[0]
        return logits

    def classify(
        self,
        texts: list[str],
        *,
        decoder: str = "viterbi",
        max_length: int | None = None,
    ) -> list[dict[str, Any]]:
        """Label each text; returns one result per text with tokens, labels,
        and grouped entities."""
        if decoder not in ("viterbi", "argmax"):
            raise ValueError(f"unknown decoder {decoder!r}: expected 'viterbi' or 'argmax'")
        # Labels aligned to the logits' column order (checkpoint id order).
        max_id = max(self.id2label) if self.id2label else -1
        labels = [self.id2label.get(index, f"LABEL_{index}") for index in range(max_id + 1)]
        results: list[dict[str, Any]] = []
        total_tokens = 0
        for text in texts:
            ids, offsets = self._encode(text, max_length)
            count = len(ids)
            mask = mx.array([[1] * count], dtype=mx.int32)
            with self._forward_lock:
                logits = self._forward(mx.array([ids], dtype=mx.int32), mask)
                mx.eval(logits)
            matrix = np.asarray(logits[0].astype(mx.float32).__array__())
            if matrix.shape[-1] != len(labels):
                raise ValueError(
                    f"model emitted {matrix.shape[-1]} classes but id2label has {len(labels)}"
                )
            if decoder == "argmax":
                path = [int(index) for index in matrix.argmax(axis=-1)]
            else:
                path = constrained_viterbi_decode(matrix, labels, self.biases)
            decoded = [labels[index] for index in path]
            tokens = (
                self.tokenizer.convert_ids_to_tokens(ids) if self.tokenizer is not None else ids
            )
            results.append(
                {
                    "tokens": tokens,
                    "labels": decoded,
                    "entities": group_entities(decoded, offsets, text),
                    "usage": {"input_tokens": count},
                }
            )
            total_tokens += count
        return results

    def _encode(self, text: str, max_length: int | None) -> tuple[list[int], list[tuple[int, int]] | None]:
        if self.tokenizer is None:
            raise RuntimeError("classify requires a tokenizer; none loaded for this checkpoint")
        kwargs: dict[str, Any] = {}
        if max_length is not None:
            kwargs["truncation"] = True
            kwargs["max_length"] = int(max_length)
        try:
            encoded = self.tokenizer(text, return_offsets_mapping=True, **kwargs)
            offsets = [tuple(pair) for pair in encoded["offset_mapping"]]
        except (KeyError, TypeError, NotImplementedError):
            encoded = self.tokenizer(text, **kwargs)
            offsets = None
        return [int(value) for value in encoded["input_ids"]], offsets


def _declared_dtype(config: dict[str, Any]) -> str | None:
    dtype = config.get("dtype")
    return dtype if dtype in ("float16", "bfloat16", "float32") else None


def _load_tokenizer(checkpoint: Path) -> Any:
    try:
        from transformers import AutoTokenizer

        try:
            return AutoTokenizer.from_pretrained(str(checkpoint))
        except Exception:
            return AutoTokenizer.from_pretrained(str(checkpoint), trust_remote_code=True)
    except Exception:  # noqa: BLE001 - serving classifies only with a tokenizer
        return None
