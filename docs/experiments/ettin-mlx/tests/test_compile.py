"""``mx.compile`` paths give the same results as eager execution."""

import mlx.core as mx
import numpy as np
import pytest

from ettin_mlx.loading import load_model
from helpers import reference


@pytest.mark.parametrize("name", ["ettin-encoder-17m", "ettin-decoder-17m"])
@pytest.mark.parametrize("case", ["medium", "long", "padded"])
def test_compiled_matches_eager(name, case):
    ref = reference(name)
    ids, mask = mx.array(ref[f"{case}/input_ids"]), ref[f"{case}/attention_mask"]
    am = mx.array(mask) if not mask.all() else None
    eager = load_model(name)
    compiled = load_model(name, compile=True)
    a, b = np.array(eager(ids, am)), np.array(compiled(ids, am))
    valid = mask.astype(bool)
    assert np.abs(a - b)[valid].max() < 1e-3  # fused kernels round differently
    assert np.array_equal(a.argmax(-1)[valid], b.argmax(-1)[valid])
    if not eager.config.is_decoder:
        assert np.abs(np.array(eager.encode(ids, am)) - np.array(compiled.encode(ids, am)))[valid].max() < 1e-3


def test_dropping_a_compiled_model_frees_its_weights():
    mx.clear_cache()
    before = mx.get_active_memory()
    m = load_model("ettin-encoder-17m", compile=True)
    mx.eval(m.encode(mx.array([[1, 2, 3]])))
    assert mx.get_active_memory() > before + 50 * 2**20
    del m  # no gc.collect(): the compiled closures must not keep the model alive
    mx.clear_cache()
    assert mx.get_active_memory() < before + 5 * 2**20
