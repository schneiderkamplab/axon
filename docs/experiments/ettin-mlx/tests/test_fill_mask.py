"""Test 9: fill-mask smoke test with the tokenizer's real mask token."""

import mlx.core as mx
import pytest

from ettin_mlx.loading import snapshot
from ettin_mlx.tokenizer import Tokenizer
from helpers import model, models, reference


@pytest.mark.parametrize("name", models("encoder"))
def test_paris_in_top5(name, report):
    tok = Tokenizer.from_dir(snapshot(name))
    text = f"The capital of France is {tok.mask_token}."
    ids = tok.encode(text)
    pos = ids.index(tok.mask_token_id)
    m = model(name)
    logits = m(mx.array([ids]))[0, pos]
    top5 = mx.argsort(-logits)[:5].tolist()
    words = [tok.decode([t]).strip() for t in top5]
    report.add(name, "fill_mask", {"top5": words})
    assert "Paris" in words, words
    ref = reference(name)
    assert ids == ref["fill_mask/input_ids"][0].tolist()  # same tokenization as HF
    assert set(top5) == set(ref["fill_mask/top5"].tolist())  # same top-5 as HF
