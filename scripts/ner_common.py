"""Shared word alignment and exact-span scoring for the public NER experiment."""

from __future__ import annotations

from collections import Counter

import numpy as np


def encode_words(batch, indices, *, tokenizer, max_length=512):
    """Split at word boundaries; supervise the first subtoken of every word once."""
    encoded = tokenizer(
        batch["tokens"],
        is_split_into_words=True,
        add_special_tokens=False,
        truncation=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    out = {
        key: [] for key in ("input_ids", "labels", "word_ids", "sentence_id", "empty_token_words")
    }
    capacity = max_length - tokenizer.num_special_tokens_to_add(pair=False)
    for row, sentence_id in enumerate(indices):
        ids = encoded["input_ids"][row]
        word_ids = encoded.word_ids(row)
        groups = []
        for token, word in zip(ids, word_ids, strict=True):
            if not groups or groups[-1][0] != word:
                groups.append((word, []))
            groups[-1][1].append(token)
        # Normalization can erase an annotated word (e.g. a combining mark).
        # Represent it explicitly as UNK, preserving its gold label and position.
        present = dict(groups)
        missing = set(range(len(batch["tokens"][row]))) - present.keys()
        if missing and tokenizer.unk_token_id is None:
            raise ValueError("Tokenizer needs UNK to preserve zero-subtoken words")
        groups = [
            (word, present.get(word, [tokenizer.unk_token_id]))
            for word in range(len(batch["tokens"][row]))
        ]
        chunks, chunk, size = [], [], 0
        for word, pieces in groups:
            if len(pieces) > capacity:
                raise ValueError(f"Single word exceeds context in sentence {sentence_id}")
            if size + len(pieces) > capacity:
                chunks.append(chunk)
                chunk, size = [], 0
            chunk.append((word, pieces))
            size += len(pieces)
        if chunk:
            chunks.append(chunk)
        for chunk in chunks:
            content, labels, mapping = [], [], []
            for word, pieces in chunk:
                content.extend(pieces)
                labels.extend([batch["fine_ner_tags"][row][word]] + [-100] * (len(pieces) - 1))
                mapping.extend([word] + [-1] * (len(pieces) - 1))
            # This experiment uses the standard BERT single-sequence tokenizer.
            full_ids = [tokenizer.cls_token_id, *content, tokenizer.sep_token_id]
            if any(token is None for token in full_ids) or len(full_ids) > max_length:
                raise ValueError("Expected a BERT tokenizer with CLS and SEP tokens")
            out["input_ids"].append(full_ids)
            out["labels"].append([-100, *labels, -100])
            out["word_ids"].append([-1, *mapping, -1])
            out["sentence_id"].append(sentence_id)
            out["empty_token_words"].append(sum(word in missing for word, _ in chunk))
    return out


def collate_windows(rows, *, pad_token_id):
    import torch

    length = max(len(row["input_ids"]) for row in rows)
    ids = torch.full((len(rows), length), pad_token_id, dtype=torch.long)
    mask = torch.zeros_like(ids)
    labels = torch.full_like(ids, -100)
    for index, row in enumerate(rows):
        n = len(row["input_ids"])
        ids[index, :n] = torch.tensor(row["input_ids"])
        mask[index, :n] = 1
        labels[index, :n] = torch.tensor(row["labels"])
    return {
        "input_ids": ids,
        "attention_mask": mask,
        "token_type_ids": torch.zeros_like(ids),
        "labels": labels,
    }


def io_spans(labels, outside=0):
    """Few-NERD uses IO: each maximal run of an entity type is one span."""
    spans = set()
    start, previous = 0, outside
    for index, label in enumerate([*labels, outside]):
        if label != previous:
            if previous != outside:
                spans.add((start, index, previous))
            start, previous = index, label
    return spans


class NerScores:
    def __init__(self, label_names):
        self.label_names = label_names
        self.outside = label_names.index("O")
        self.sentences = {}

    def add(self, rows, predictions):
        for row, prediction in zip(rows, predictions, strict=True):
            sentence = self.sentences.setdefault(row["sentence_id"], {})
            for word, gold, pred in zip(row["word_ids"], row["labels"], prediction):
                if word >= 0:
                    if word in sentence:
                        raise ValueError("A word was scored more than once")
                    sentence[word] = (gold, int(pred))

    def result(self):
        gold_count, pred_count, tp_count = Counter(), Counter(), Counter()
        correct, total = 0, 0
        for sentence in self.sentences.values():
            if sorted(sentence) != list(range(len(sentence))):
                raise ValueError("Incomplete word coverage in evaluation")
            gold, pred = zip(*(sentence[i] for i in range(len(sentence))), strict=True)
            gold_spans, pred_spans = io_spans(gold, self.outside), io_spans(pred, self.outside)
            gold_count.update(span[2] for span in gold_spans)
            pred_count.update(span[2] for span in pred_spans)
            tp_count.update(span[2] for span in gold_spans & pred_spans)
            total += len(gold)
            correct += sum(a == b for a, b in zip(gold, pred, strict=True))

        def metrics(tp, predicted, gold):
            p, r = tp / predicted if predicted else 0.0, tp / gold if gold else 0.0
            return {
                "precision": p,
                "recall": r,
                "f1": 2 * p * r / (p + r) if p + r else 0.0,
                "gold_spans": gold,
                "predicted_spans": predicted,
                "true_positive_spans": tp,
            }

        per_label = {
            name: metrics(tp_count[i], pred_count[i], gold_count[i])
            for i, name in enumerate(self.label_names)
            if i != self.outside
        }
        return {
            **metrics(sum(tp_count.values()), sum(pred_count.values()), sum(gold_count.values())),
            "macro_f1": float(np.mean([row["f1"] for row in per_label.values()])),
            "word_accuracy": correct / total if total else 0.0,
            "words": total,
            "sentences": len(self.sentences),
            "per_label": per_label,
        }
