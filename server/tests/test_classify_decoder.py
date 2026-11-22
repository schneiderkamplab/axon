"""Constrained Viterbi decoder + span grouping + kind dispatch unit tests.

Pure-numpy tests: no model, no tokenizer, no MLX forward. The decoder is the
same code the ``/v1/classify`` route runs over real logits (see
``log/probe-serve-classify.py`` for the end-to-end check against HF).
"""

from __future__ import annotations

import numpy as np
import pytest

from server.classify import (
    constrained_viterbi_decode,
    group_entities,
    is_token_classifier,
)

# A 5-label BIOES slice: O plus one category's B/I/E/S.
LABELS = ["O", "B-x", "I-x", "E-x", "S-x"]
O, B, I, E, S = range(5)


def _one_hot(path: list[int], width: int = 5) -> np.ndarray:
    """Logits that argmax exactly to `path` (large margin, no ties)."""
    logits = np.full((len(path), width), -1.0)
    for step, label in enumerate(path):
        logits[step, label] = 10.0
    return logits


def test_dispatch_on_checkpoint_metadata() -> None:
    assert is_token_classifier({"id2label": {"0": "O", "1": "B-x"}})
    assert is_token_classifier(
        {"architectures": ["SomeFamilyForTokenClassification"]}
    )
    assert not is_token_classifier({"architectures": ["LlamaForCausalLM"]})
    assert not is_token_classifier({})


def test_viterbi_returns_valid_bioes_path() -> None:
    # Raw argmax would start with I (invalid) and end mid-span; the
    # constrained decoder must return a path that starts with O/B/S and
    # ends with O/E/S, with spans continuing only inside their category.
    logits = np.array(
        [
            [0.0, 0.1, 5.0, -1.0, -1.0],  # argmax: I (invalid start)
            [0.0, 0.2, 5.0, -1.0, -1.0],  # argmax: I
            [0.0, 0.3, 5.0, -1.0, -1.0],  # argmax: I
        ]
    )
    path = constrained_viterbi_decode(logits, LABELS)
    assert len(path) == 3
    decoded = [LABELS[index] for index in path]
    assert decoded[0] in ("O", "B-x", "S-x")
    assert decoded[-1] in ("O", "E-x", "S-x")
    for prev, cur in zip(path, path[1:]):
        tag_prev, tag_cur = LABELS[prev][0], LABELS[cur][0]
        if tag_prev in ("B", "I"):
            assert tag_cur in ("I", "E")  # open spans must close in-category
        else:
            assert tag_cur in ("O", "B", "S")


def test_viterbi_matches_argmax_when_path_is_valid() -> None:
    logits = _one_hot([O, B, I, E, O, S])
    assert constrained_viterbi_decode(logits, LABELS) == [O, B, I, E, O, S]


def test_viterbi_single_token_allows_only_background_or_single() -> None:
    # A lone B cannot close; a lone I can neither start nor close.
    for forced, valid in ((B, False), (I, False), (O, True), (S, True), (E, False)):
        logits = _one_hot([forced])
        path = constrained_viterbi_decode(logits, LABELS)
        assert (path == [forced]) == valid, f"label {LABELS[forced]} -> {path}"


def test_viterbi_span_entry_bias_flips_bordercase() -> None:
    # Position 1 is a borderline O-vs-B decision behind an O predecessor, and
    # position 2 closes the span weakly. Neutral scoring keeps background;
    # the span-entry bias on the O->B transition flips the path to the span.
    logits = np.array(
        [
            [1.0, 0.2, -9.0, -9.0, -9.0],
            [0.5, 0.45, -9.0, -9.0, -9.0],
            [0.9, -9.0, -9.0, 0.5, -9.0],
        ]
    )
    neutral = constrained_viterbi_decode(logits, LABELS, biases={})
    assert neutral == [O, O, O]
    entry = constrained_viterbi_decode(
        logits, LABELS, biases={"transition_bias_background_to_start": 0.5}
    )
    assert entry == [O, B, E]
    against = constrained_viterbi_decode(
        logits, LABELS, biases={"transition_bias_background_to_start": -1.0}
    )
    assert against == [O, O, O]


def test_viterbi_rejects_mismatched_width() -> None:
    with pytest.raises(ValueError, match="5 labels"):
        constrained_viterbi_decode(np.zeros((3, 4)), LABELS)


def test_group_entities_wellformed_bioes() -> None:
    labels = ["O", "B-x", "I-x", "E-x", "O", "S-y"]
    # token chars: "aa bb cc dd ee ff" -> offsets per token
    text = "aa bb cc dd ee ff"
    offsets = [(0, 2), (3, 5), (6, 8), (9, 11), (12, 14), (15, 17)]
    entities = group_entities(labels, offsets, text)
    assert entities == [
        {"label": "x", "token_start": 1, "token_end": 3, "start": 3, "end": 11, "text": "bb cc dd"},
        {"label": "y", "token_start": 5, "token_end": 5, "start": 15, "end": 17, "text": "ff"},
    ]


def test_group_entities_salvages_stray_inside_tag() -> None:
    # Raw argmax can emit I without B; grouping keeps the span instead of
    # dropping it.
    labels = ["O", "I-x", "I-x", "O"]
    entities = group_entities(labels, None, None)
    assert entities == [{"label": "x", "token_start": 1, "token_end": 2}]


def test_group_entities_flags_unclosed_span() -> None:
    labels = ["B-x", "I-x"]
    entities = group_entities(labels, None, None)
    assert entities == [{"label": "x", "token_start": 0, "token_end": 1}]
