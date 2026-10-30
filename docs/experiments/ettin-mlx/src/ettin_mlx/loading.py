"""Checkpoint download, conversion, name mapping, tying, dtype casting and strict loading.

The official Ettin repos ship only ``pytorch_model.bin`` (a torch zip/pickle), which
``mx.load`` cannot read and which we must not read with torch. ``read_torch_bin`` is a
restricted unpickler that accepts nothing but tensor-rebuild records and maps the raw
storages through numpy. The result is cached once as ``model.safetensors`` under
``$ETTIN_MLX_CACHE`` (default ``~/.cache/ettin_mlx``); every later load is a plain
``mx.load``. ``ETTIN_MLX_CACHE=off`` skips the cache (no second copy on disk; each load
re-reads the ``.bin``). Repos that do ship safetensors are loaded directly.
"""

from __future__ import annotations

import collections
import hashlib
import json
import os
import pickle
import re
import tempfile
import zipfile
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from huggingface_hub import snapshot_download
from mlx.utils import tree_flatten

from .config import EttinConfig
from .decoder import EttinDecoder
from .encoder import EttinEncoder
from .revisions import resolve
from .tokenizer import Tokenizer

METADATA_FILES = ["config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]

DTYPES = {
    "float32": mx.float32, "fp32": mx.float32,
    "bfloat16": mx.bfloat16, "bf16": mx.bfloat16,
    "float16": mx.float16, "fp16": mx.float16,
}


def snapshot(name_or_path: str, revision: str | None = None) -> Path:
    """Local directory holding config, tokenizer and weights for a model."""
    if Path(name_or_path).is_dir():
        return Path(name_or_path)
    repo, rev = resolve(name_or_path, revision)
    path = Path(snapshot_download(repo, revision=rev, allow_patterns=METADATA_FILES + ["*.safetensors*"]))
    if not any(path.glob("*.safetensors")):
        path = Path(snapshot_download(repo, revision=rev, allow_patterns=METADATA_FILES + ["pytorch_model*.bin*"]))
    return path


# --------------------------------------------------------------------------- torch .bin

_STORAGE_DTYPES = {
    "FloatStorage": np.float32,
    "DoubleStorage": np.float64,
    "HalfStorage": np.float16,
    "BFloat16Storage": np.uint16,  # reinterpreted as bfloat16 in MLX
    "LongStorage": np.int64,
    "IntStorage": np.int32,
    "ShortStorage": np.int16,
    "CharStorage": np.int8,
    "ByteStorage": np.uint8,
    "BoolStorage": np.bool_,
}


class _StorageType:
    def __init__(self, name: str):
        self.name = name


class _Tensor:
    def __init__(self, storage, offset, size, stride):
        self.storage, self.offset, self.size, self.stride = storage, offset, tuple(size), tuple(stride)

    def to_mlx(self) -> mx.array:
        data, is_bf16 = self.storage
        offset, size, stride = self.offset, self.size, self.stride
        # as_strided does no bounds checking, so a crafted file could otherwise read
        # memory outside the storage buffer.
        ints = all(isinstance(v, int) and v >= 0 for v in (offset, *size, *stride))
        if not ints or len(size) != len(stride):
            raise pickle.UnpicklingError(f"invalid tensor layout: offset={offset} size={size} stride={stride}")
        if 0 in size:
            arr = np.empty(size, dtype=data.dtype)
        else:
            last = offset + sum((n - 1) * st for n, st in zip(size, stride))
            if last >= len(data):
                raise pickle.UnpicklingError(
                    f"tensor layout reaches element {last} of a {len(data)}-element storage"
                )
            arr = np.lib.stride_tricks.as_strided(
                data[offset:], shape=size, strides=[st * data.itemsize for st in stride]
            )
        out = mx.array(np.ascontiguousarray(arr))
        return out.view(mx.bfloat16) if is_bf16 else out


def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad=False, backward_hooks=None, metadata=None):
    return _Tensor(storage, storage_offset, size, stride)


def _rebuild_parameter(data, requires_grad=False, backward_hooks=None):
    return data


class _RestrictedUnpickler(pickle.Unpickler):
    _ALLOWED = {
        ("torch._utils", "_rebuild_tensor_v2"): _rebuild_tensor_v2,
        ("torch._utils", "_rebuild_parameter"): _rebuild_parameter,
        ("collections", "OrderedDict"): collections.OrderedDict,
    }

    def __init__(self, file, archive: zipfile.ZipFile, prefix: str):
        super().__init__(file)
        self.archive, self.prefix, self.storages = archive, prefix, {}

    def find_class(self, module, name):
        if (module, name) in self._ALLOWED:
            return self._ALLOWED[(module, name)]
        if module == "torch" and name in _STORAGE_DTYPES:
            return _StorageType(name)
        raise pickle.UnpicklingError(f"refusing to unpickle {module}.{name}")

    def persistent_load(self, pid):
        kind, storage_type, key, _location, _numel = pid
        if kind != "storage" or not isinstance(storage_type, _StorageType):
            raise pickle.UnpicklingError(f"unexpected persistent id {pid!r}")
        if key not in self.storages:
            raw = self.archive.read(f"{self.prefix}/data/{key}")
            data = np.frombuffer(raw, dtype=_STORAGE_DTYPES[storage_type.name])
            self.storages[key] = (data, storage_type.name == "BFloat16Storage")
        return self.storages[key]


def read_torch_bin(path: str | Path) -> dict[str, mx.array]:
    """Read a torch zip checkpoint (``torch.save`` >= 1.6) without importing torch."""
    if not zipfile.is_zipfile(path):
        raise NotImplementedError(f"{path}: legacy (non-zip) torch checkpoints are not supported")
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        pkl = next((n for n in names if n.endswith("/data.pkl")), None)
        if pkl is None:
            raise pickle.UnpicklingError(f"{path}: no data.pkl record, not a torch zip checkpoint")
        prefix = pkl[: -len("/data.pkl")]
        # torch < 2.x writes no byteorder record; those files are little-endian.
        order = archive.read(f"{prefix}/byteorder").strip() if f"{prefix}/byteorder" in names else b"little"
        if order not in (b"little", b""):
            raise NotImplementedError(f"{path}: big-endian checkpoints are not supported")
        with archive.open(pkl) as f:
            state = _RestrictedUnpickler(f, archive, prefix).load()
        return {k: v.to_mlx() for k, v in state.items()}


def _fingerprint(path: Path) -> str:
    """Identity of a checkpoint file: location, size, mtime and the CRC of every zip entry
    (read from the central directory, so cheap), so a local file rewritten in place with
    same-size weights does not hit a stale cache entry."""
    stat = path.stat()
    crcs = ""
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            crcs = ",".join(f"{i.filename}={i.CRC:08x}" for i in archive.infolist())
    return f"{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}:{crcs}"


def _converted_path(bins: list[Path]) -> Path | None:
    root = os.environ.get("ETTIN_MLX_CACHE", str(Path.home() / ".cache" / "ettin_mlx"))
    if root.lower() in ("", "0", "off", "none"):
        return None
    key = hashlib.sha256("\n".join(_fingerprint(b) for b in bins).encode()).hexdigest()[:24]
    return Path(root) / key / "model.safetensors"


def read_checkpoint(path: Path) -> dict[str, mx.array]:
    """All tensors of a checkpoint directory, keyed by their HF names."""
    safetensors = sorted(path.glob("*.safetensors"))
    if not safetensors:
        bins = sorted(path.glob("pytorch_model*.bin"))
        if not bins:
            raise FileNotFoundError(f"no *.safetensors or pytorch_model*.bin in {path}")
        converted = _converted_path(bins)
        if converted is None or not converted.exists():
            weights = {}
            for b in bins:
                weights.update(read_torch_bin(b))
            if converted is None:
                return weights
            converted.parent.mkdir(parents=True, exist_ok=True)
            # Unique temp name: concurrent conversions must not write the same file.
            fd, tmp = tempfile.mkstemp(dir=converted.parent, prefix="model.", suffix=".tmp.safetensors")
            os.close(fd)
            try:
                mx.save_safetensors(tmp, weights, metadata={"source": json.dumps([str(b) for b in bins])})
                os.replace(tmp, converted)
            finally:
                Path(tmp).unlink(missing_ok=True)
            del weights
        safetensors = [converted]
    weights = {}
    for f in safetensors:
        weights.update(mx.load(str(f)))
    return weights


# --------------------------------------------------------------------------- mapping

_QKV = re.compile(r"^(model\.layers\.\d+\.attn)\.([qkv])_proj\.(weight|bias)$")
TIED_KEY = "decoder.weight"


def to_mlx_weights(hf: dict[str, mx.array], config: EttinConfig) -> dict[str, mx.array]:
    """The one place where HF checkpoint names become MLX parameter names.

    MLX modules mirror HF names, so this is the identity except that

    * ``decoder.weight`` is dropped when embeddings are tied (it is read from
      ``model.embeddings.tok_embeddings.weight`` by reference at call time);
    * the decoder's separate ``q_proj``/``k_proj``/``v_proj`` are concatenated into
      the encoder-style fused ``Wqkv`` (same ``[q; k; v]`` row order as HF's
      ``Wqkv(...).view(..., 3, heads, head_dim)``).
    """
    out: dict[str, mx.array] = {}
    qkv: dict[tuple[str, str], dict[str, mx.array]] = {}
    for name, w in hf.items():
        if name == TIED_KEY and config.tie_word_embeddings:
            continue
        if m := _QKV.match(name):
            qkv.setdefault((m[1], m[3]), {})[m[2]] = w
            continue
        out[name] = w
    for (prefix, kind), parts in sorted(qkv.items()):
        missing = [f"{prefix}.{p}_proj.{kind}" for p in "qkv" if p not in parts]
        if missing:
            raise ValueError(f"checkpoint is missing {missing}")
        out[f"{prefix}.Wqkv.{kind}"] = mx.concatenate([parts["q"], parts["k"], parts["v"]], axis=0)
    return out


def load_weights(model: nn.Module, weights: dict[str, mx.array], dtype: mx.Dtype = mx.float32) -> None:
    """Strict load: any missing, unexpected or shape-mismatched key is an error."""
    expected = dict(tree_flatten(model.parameters()))
    missing = sorted(expected.keys() - weights.keys())
    unexpected = sorted(weights.keys() - expected.keys())
    mismatched = sorted(
        f"{k}: checkpoint {tuple(weights[k].shape)} vs model {tuple(expected[k].shape)}"
        for k in expected.keys() & weights.keys()
        if weights[k].shape != expected[k].shape
    )
    if missing or unexpected or mismatched:
        raise ValueError(
            "checkpoint does not match the model:\n"
            f"  missing: {missing}\n  unexpected: {unexpected}\n  shape mismatch: {mismatched}"
        )
    cast = [(k, v.astype(dtype) if mx.issubdtype(v.dtype, mx.floating) else v) for k, v in weights.items()]
    model.load_weights(cast, strict=True)
    mx.eval(model.parameters())


def load_model(
    name_or_path: str,
    *,
    revision: str | None = None,
    dtype: str | mx.Dtype = "float32",
    compile: bool = False,
    local_attention: str = "auto",
    rope: str = "precise",
) -> EttinEncoder | EttinDecoder:
    """Build an encoder or decoder from ``config.json`` and load its weights."""
    path = snapshot(name_or_path, revision)
    config = EttinConfig.from_file(path / "config.json")
    model_cls = EttinDecoder if config.is_decoder else EttinEncoder
    model = model_cls(config, local_attention=local_attention, rope=rope)
    dtype = DTYPES[dtype] if isinstance(dtype, str) else dtype
    load_weights(model, to_mlx_weights(read_checkpoint(path), config), dtype)
    model.revision = path.name  # the commit SHA for Hub snapshots
    if compile:
        model.compile()
    return model


def load(name_or_path: str, **kwargs) -> tuple[EttinEncoder | EttinDecoder, Tokenizer]:
    """``(model, tokenizer)`` for a short name (``ettin-encoder-17m``), repo id or directory."""
    path = snapshot(name_or_path, kwargs.get("revision"))
    return load_model(str(path), **{k: v for k, v in kwargs.items() if k != "revision"}), Tokenizer.from_dir(path)
