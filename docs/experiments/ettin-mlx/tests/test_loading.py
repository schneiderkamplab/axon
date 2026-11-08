"""Weight loading, config parsing and tokenizer behaviour against HF."""

import collections
import io
import os
import pickle
import zipfile

import mlx.core as mx
import numpy as np
import pytest
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoConfig, AutoTokenizer

from ettin_mlx import EttinConfig, EttinDecoder, EttinEncoder
from ettin_mlx.config import FULL, SLIDING
from ettin_mlx.loading import load_weights, read_checkpoint, read_torch_bin, snapshot, to_mlx_weights
from ettin_mlx.revisions import HF_ORG, PINNED_REVISIONS
from ettin_mlx.tokenizer import Tokenizer


@pytest.mark.parametrize("name", ["ettin-encoder-17m", "ettin-decoder-17m"])
def test_torch_free_bin_reader_matches_torch_load(name):
    path = snapshot(name) / "pytorch_model.bin"
    ours = read_torch_bin(path)
    theirs = torch.load(path, map_location="cpu", weights_only=True)
    assert ours.keys() == theirs.keys()
    for k, v in theirs.items():
        assert np.array_equal(np.array(ours[k]), v.numpy()), k


def test_conversion_cache_can_be_disabled(monkeypatch):
    path = snapshot("ettin-encoder-17m")
    cached = read_checkpoint(path)
    monkeypatch.setenv("ETTIN_MLX_CACHE", "off")
    direct = read_checkpoint(path)
    assert cached.keys() == direct.keys()
    assert all(mx.array_equal(cached[k], direct[k]).item() for k in cached)


def _crafted_bin(path, offset, size, stride, storage_numel=4, byteorder=True):
    """A torch-format zip whose single tensor has the given layout over a 4-float storage."""

    class Tensor:
        def __reduce__(self):
            args = ("STORAGE", offset, size, stride, False, collections.OrderedDict())
            return (torch._utils._rebuild_tensor_v2, args)

    class Pickler(pickle.Pickler):
        def persistent_id(self, obj):
            if isinstance(obj, str) and obj == "STORAGE":
                return ("storage", torch.FloatStorage, "0", "cpu", storage_numel)
            return None

    buf = io.BytesIO()
    Pickler(buf, protocol=2).dump(collections.OrderedDict(w=Tensor()))
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("archive/data.pkl", buf.getvalue())
        if byteorder:
            z.writestr("archive/byteorder", "little")
        z.writestr("archive/data/0", np.arange(storage_numel, dtype=np.float32).tobytes())


def test_reader_accepts_checkpoints_without_byteorder_record(tmp_path):
    """torch < 2.x zip checkpoints have no byteorder record (little-endian)."""
    _crafted_bin(tmp_path / "old.bin", 0, (4,), (1,), byteorder=False)
    assert np.array(read_torch_bin(tmp_path / "old.bin")["w"]).tolist() == [0, 1, 2, 3]


def test_reader_rejects_zip_without_data_pkl(tmp_path):
    with zipfile.ZipFile(tmp_path / "x.bin", "w") as z:
        z.writestr("archive/other", "x")
    with pytest.raises(pickle.UnpicklingError, match="no data.pkl"):
        read_torch_bin(tmp_path / "x.bin")


@pytest.mark.parametrize("offset,size,stride,expected", [
    (0, (4,), (1,), [0, 1, 2, 3]),
    (1, (3,), (1,), [1, 2, 3]),
    (0, (2, 2), (2, 1), [[0, 1], [2, 3]]),
    (0, (2, 2), (1, 2), [[0, 2], [1, 3]]),
    (0, (0, 5), (5, 1), np.zeros((0, 5))),
])
def test_unpickler_reads_valid_layouts(tmp_path, offset, size, stride, expected):
    _crafted_bin(tmp_path / "ok.bin", offset, size, stride)
    assert np.array_equal(np.array(read_torch_bin(tmp_path / "ok.bin")["w"]), np.array(expected, np.float32))


@pytest.mark.parametrize("offset,size,stride", [
    (0, (10,), (1,)),          # longer than the storage
    (1, (4,), (1,)),           # offset pushes the end out
    (0, (2, 2), (3, 1)),       # stride reaches element 4
    (0, (2,), (1 << 40,)),     # far out-of-bounds read
    (-1, (2,), (1,)),          # negative offset
    (0, (2,), (-1,)),          # negative stride
    (0, (2, 2), (1,)),         # rank mismatch
])
def test_unpickler_rejects_out_of_bounds_layouts(tmp_path, offset, size, stride):
    _crafted_bin(tmp_path / "bad.bin", offset, size, stride)
    with pytest.raises(pickle.UnpicklingError, match="tensor layout"):
        read_torch_bin(tmp_path / "bad.bin")


def test_conversion_cache_sees_in_place_rewrites(tmp_path, monkeypatch):
    """Same path, same size, same mtime, different weights: the cache must not go stale."""
    monkeypatch.setenv("ETTIN_MLX_CACHE", str(tmp_path / "cache"))
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    bin_path = model_dir / "pytorch_model.bin"
    torch.save({"w": torch.zeros(8)}, bin_path)
    stat = bin_path.stat()
    assert np.array(read_checkpoint(model_dir)["w"]).tolist() == [0.0] * 8
    torch.save({"w": torch.ones(8)}, bin_path)
    os.utime(bin_path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert bin_path.stat().st_size == stat.st_size
    assert np.array(read_checkpoint(model_dir)["w"]).tolist() == [1.0] * 8
    assert not list((tmp_path / "cache").rglob("*.tmp.*"))  # temp files are cleaned up


def test_unpickler_refuses_arbitrary_globals(tmp_path):
    class Evil:
        def __reduce__(self):
            return (print, ("pwned",))

    path = tmp_path / "evil.bin"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("archive/data.pkl", pickle.dumps({"x": Evil()}))
        z.writestr("archive/byteorder", "little")
    with pytest.raises(pickle.UnpicklingError, match="builtins.print"):
        read_torch_bin(path)


def _fresh(name):
    path = snapshot(name)
    cfg = EttinConfig.from_file(path / "config.json")
    m = (EttinDecoder if cfg.is_decoder else EttinEncoder)(cfg)
    return m, to_mlx_weights(read_checkpoint(path), cfg), cfg


@pytest.mark.parametrize("name", ["ettin-encoder-17m", "ettin-decoder-17m"])
def test_strict_loading_fails_loudly(name):
    m, weights, cfg = _fresh(name)
    load_weights(m, dict(weights))  # the real checkpoint loads

    missing = dict(weights)
    del missing["model.layers.1.mlp_norm.weight"]
    with pytest.raises(ValueError, match=r"missing: \['model.layers.1.mlp_norm.weight'\]"):
        load_weights(_fresh(name)[0], missing)

    unexpected = dict(weights) | {"model.layers.0.attn_norm.weight": mx.ones(cfg.hidden_size)}
    with pytest.raises(ValueError, match=r"unexpected: \['model.layers.0.attn_norm.weight'\]"):
        load_weights(_fresh(name)[0], unexpected)

    bad_shape = dict(weights) | {"head.dense.weight" if not cfg.is_decoder else "lm_head.dense.weight": mx.zeros((3, 3))}
    with pytest.raises(ValueError, match="shape mismatch"):
        load_weights(_fresh(name)[0], bad_shape)


def test_split_qkv_must_be_complete():
    raw = read_checkpoint(snapshot("ettin-decoder-17m"))
    cfg = EttinConfig.from_file(snapshot("ettin-decoder-17m") / "config.json")
    del raw["model.layers.3.attn.k_proj.weight"]
    with pytest.raises(ValueError, match=r"model\.layers\.3\.attn\.k_proj\.weight"):
        to_mlx_weights(raw, cfg)


@pytest.mark.parametrize("name", ["ettin-encoder-17m", "ettin-decoder-17m"])
def test_decoder_weight_is_tied_by_reference(name):
    m, weights, cfg = _fresh(name)
    load_weights(m, weights)
    assert "decoder.weight" not in weights and "weight" not in m.decoder
    assert set(m.decoder) == {"bias"}  # decoder_bias=True in every Ettin config
    h = mx.random.normal((1, 2, cfg.hidden_size))
    before = m.logits(h)
    emb = m.model.embeddings.tok_embeddings
    emb.weight = emb.weight * 2  # the vocab projection must follow the embedding
    assert not mx.allclose(before, m.logits(h)).item()


@pytest.mark.parametrize("name", sorted(PINNED_REVISIONS))
def test_config_matches_transformers(name):
    """Our parse of every checkpoint's config.json agrees with transformers' config class."""
    repo, rev = f"{HF_ORG}/{name}", PINNED_REVISIONS[name]
    hf = AutoConfig.from_pretrained(repo, revision=rev)
    ours = EttinConfig.from_file(hf_hub_download(repo, "config.json", revision=rev))
    assert list(ours.layer_types) == list(hf.layer_types)
    for kind in set(hf.layer_types):
        assert ours.rope_theta[kind] == hf.rope_parameters[kind]["rope_theta"]
    assert ours.sliding_window == hf.sliding_window == 64
    for key in ["hidden_size", "num_attention_heads", "num_hidden_layers", "intermediate_size", "vocab_size",
                "norm_eps", "norm_bias", "attention_bias", "mlp_bias", "classifier_bias", "decoder_bias",
                "hidden_activation", "classifier_activation", "tie_word_embeddings"]:
        assert getattr(ours, key) == getattr(hf, key), key

    # The transformers-5 serialisation (layer_types + rope_parameters) parses identically.
    new = EttinConfig.from_dict(hf.to_dict())
    assert new.layer_types == ours.layer_types and new.rope_theta == ours.rope_theta


def test_config_legacy_defaults():
    base = {"model_type": "modernbert", "num_hidden_layers": 7, "hidden_size": 64, "num_attention_heads": 4}
    cfg = EttinConfig.from_dict(base)
    assert cfg.layer_types == (FULL, SLIDING, SLIDING, FULL, SLIDING, SLIDING, FULL)
    assert cfg.rope_theta == {FULL: 160000.0, SLIDING: 10000.0}  # HF default_theta
    cfg = EttinConfig.from_dict(base | {"global_attn_every_n_layers": 2, "local_rope_theta": 5.0})
    assert cfg.layer_types[:3] == (FULL, SLIDING, FULL) and cfg.rope_theta[SLIDING] == 5.0
    with pytest.raises(NotImplementedError):
        EttinConfig.from_dict(base | {"rope_parameters": {FULL: {"rope_type": "yarn", "factor": 2.0}}})
    with pytest.raises(ValueError, match="local_attention >= 2"):
        EttinConfig.from_dict(base | {"local_attention": 1})


TEXTS = [
    "The capital of France is [MASK].",
    "Hello world",
    "  leading spaces,\ttabs and\nnewlines  ",
    "Ünïcödé — emoji 🦙 and 中文",
    "def f(x):\n    return x ** 2  # code",
    "[CLS] literal special tokens [SEP] [PAD]",
]


@pytest.mark.parametrize("name", ["ettin-encoder-17m", "ettin-decoder-17m"])
def test_tokenizer_matches_hf(name):
    path = snapshot(name)
    ours = Tokenizer.from_dir(path)
    hf = AutoTokenizer.from_pretrained(f"{HF_ORG}/{name}", revision=PINNED_REVISIONS[name])
    assert ours.mask_token == hf.mask_token and ours.mask_token_id == hf.mask_token_id
    assert ours.pad_token_id == hf.pad_token_id
    assert ours.bos_token_id == hf.bos_token_id and ours.eos_token_id == hf.eos_token_id
    for text in TEXTS:
        ids = hf(text)["input_ids"]
        assert ours.encode(text) == ids, text
        assert ours.encode(text, add_special_tokens=False) == hf(text, add_special_tokens=False)["input_ids"]
        assert ours.decode(ids) == hf.decode(ids, skip_special_tokens=True), text
        assert ours.decode(ids, skip_special_tokens=False) == hf.decode(ids), text


def test_decoder_prompt_drops_trailing_eos():
    tok = Tokenizer.from_dir(snapshot("ettin-decoder-17m"))
    ids = tok.encode("The capital of France is")
    assert ids[0] == tok.bos_token_id and ids[-1] == tok.eos_token_id  # [CLS] ... [SEP], as in transformers 5
    assert tok.encode_prompt("The capital of France is") == ids[:-1]


def test_padded_batch_encoding():
    tok = Tokenizer.from_dir(snapshot("ettin-encoder-17m"))
    ids, mask = tok.batch(["a b c d", "a"])
    assert ids.shape == mask.shape == (2, 6)
    assert mask.tolist() == [[1] * 6, [1, 1, 1, 0, 0, 0]]
    assert ids[1, 3:].tolist() == [tok.pad_token_id] * 3
