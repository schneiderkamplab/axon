import pytest

from scripts.ner_common import NerScores, encode_words, io_spans


def test_io_spans_keep_adjacent_types_and_sentence_end():
    assert io_spans([0, 1, 1, 2, 0, 2]) == {(1, 3, 1), (3, 4, 2), (5, 6, 2)}


def test_scoring_reassembles_an_entity_across_windows():
    scorer = NerScores(["O", "person", "location"])
    scorer.add(
        [
            {"sentence_id": 0, "word_ids": [-1, 0, 1, -1], "labels": [-100, 0, 1, -100]},
            {"sentence_id": 0, "word_ids": [-1, 2, 3, -1], "labels": [-100, 1, 2, -100]},
        ],
        [[0, 0, 1, 0], [0, 1, 0, 0]],
    )
    result = scorer.result()
    assert result["gold_spans"] == 2
    assert result["true_positive_spans"] == 1
    assert result["recall"] == 0.5
    assert result["precision"] == 1.0
    assert result["words"] == 4


def test_word_boundary_windows_preserve_empty_words_and_subtoken_alignment():
    class Encoded(dict):
        def word_ids(self, index):
            return [0, 1, 1, 3]

    class Tokenizer:
        cls_token_id, sep_token_id, unk_token_id = 1, 2, 3

        def __call__(self, *args, **kwargs):
            return Encoded(input_ids=[[10, 11, 12, 13]])

        def num_special_tokens_to_add(self, pair):
            return 2

    result = encode_words(
        {"tokens": [["a", "subword", "", "b"]], "fine_ner_tags": [[0, 1, 1, 2]]},
        [5],
        tokenizer=Tokenizer(),
        max_length=5,
    )
    assert result["input_ids"] == [[1, 10, 11, 12, 2], [1, 3, 13, 2]]
    assert result["labels"] == [[-100, 0, 1, -100, -100], [-100, 1, 2, -100]]
    assert result["word_ids"] == [[-1, 0, 1, -1, -1], [-1, 2, 3, -1]]
    assert result["empty_token_words"] == [0, 1]
    assert result["sentence_id"] == [5, 5]


def test_duplicate_word_predictions_are_rejected():
    scorer = NerScores(["O", "person"])
    rows = [{"sentence_id": 0, "word_ids": [0], "labels": [1]}]
    scorer.add(rows, [[1]])
    with pytest.raises(ValueError, match="more than once"):
        scorer.add(rows, [[1]])
