"""Thin wrapper around the checkpoint's ``tokenizer.json`` (``tokenizers`` library).

The Ettin ``tokenizer.json`` post-processor wraps every input as ``[CLS] text [SEP]``,
for the encoders *and* the decoders, and transformers 5.x applies it unchanged; ``encode``
reproduces that. For decoders ``[CLS]``/``[SEP]`` are BOS/EOS, so a prompt encoded that
way already ends in EOS; ``encode_prompt`` drops the trailing EOS for generation.
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import tokenizers

SPECIAL_TOKENS = ("bos_token", "eos_token", "unk_token", "sep_token", "pad_token", "cls_token", "mask_token")


def _clean_up_tokenization(text: str) -> str:
    # Same replacements as transformers' PreTrainedTokenizerBase.clean_up_tokenization.
    for a, b in ((" .", "."), (" ?", "?"), (" !", "!"), (" ,", ","), (" ' ", "'"), (" n't", "n't"),
                 (" 'm", "'m"), (" 's", "'s"), (" 've", "'ve"), (" 're", "'re")):
        text = text.replace(a, b)
    return text


class Tokenizer:
    def __init__(self, tokenizer: tokenizers.Tokenizer, special: dict[str, str], clean_up_spaces: bool = False):
        self.tokenizer = tokenizer
        self.special = special
        self.clean_up_spaces = clean_up_spaces

    @classmethod
    def from_dir(cls, path: str | Path) -> Tokenizer:
        path = Path(path)
        special: dict[str, str] = {}
        config: dict = {}
        for name in ("special_tokens_map.json", "tokenizer_config.json"):
            if not (path / name).exists():
                continue
            data = json.loads((path / name).read_text())
            if name == "tokenizer_config.json":
                config = data
            for key in SPECIAL_TOKENS:
                value = data.get(key)
                if isinstance(value, dict):
                    value = value.get("content")
                if value:
                    special[key] = value
        tok = tokenizers.Tokenizer.from_file(str(path / "tokenizer.json"))
        return cls(tok, special, bool(config.get("clean_up_tokenization_spaces", False)))

    def token_id(self, key: str) -> int | None:
        token = self.special.get(key)
        return None if token is None else self.tokenizer.token_to_id(token)

    @property
    def mask_token(self) -> str | None:
        return self.special.get("mask_token")

    @property
    def mask_token_id(self) -> int | None:
        return self.token_id("mask_token")

    @property
    def pad_token_id(self) -> int | None:
        return self.token_id("pad_token")

    @property
    def bos_token_id(self) -> int | None:
        return self.token_id("bos_token")

    @property
    def eos_token_id(self) -> int | None:
        return self.token_id("eos_token")

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=add_special_tokens).ids

    def encode_prompt(self, text: str) -> list[int]:
        """Prompt ids for generation: special tokens as in ``encode`` minus a trailing EOS."""
        ids = self.encode(text)
        eos = self.eos_token_id
        return ids[:-1] if eos is not None and ids and ids[-1] == eos else ids

    def batch(self, texts: list[str], add_special_tokens: bool = True) -> tuple[mx.array, mx.array]:
        """Right-padded ``(input_ids, attention_mask)``, both ``(B, L)`` int32."""
        seqs = [self.encode(t, add_special_tokens) for t in texts]
        return pad_batch(seqs, self.pad_token_id or 0)

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        text = self.tokenizer.decode(list(ids), skip_special_tokens=skip_special_tokens)
        return _clean_up_tokenization(text) if self.clean_up_spaces else text


def pad_batch(seqs: list[list[int]], pad_id: int) -> tuple[mx.array, mx.array]:
    length = max(len(s) for s in seqs)
    ids = [s + [pad_id] * (length - len(s)) for s in seqs]
    mask = [[1] * len(s) + [0] * (length - len(s)) for s in seqs]
    return mx.array(ids, dtype=mx.int32), mx.array(mask, dtype=mx.int32)
