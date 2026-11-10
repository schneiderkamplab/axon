"""The PP-OCR detector and recognizer on generated cores, with small synthetic weights.

Torch, MLX with the optimizer and MLX without it compile the same two ``.axon`` files and
must agree with a hand-written PyTorch reference of the published graphs. No download:
``scripts/ppocr_mlx.py parity`` compares the real checkpoint with ONNX Runtime.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from synapse.axon.codegen2_mlx import non_obvious_mlx_ops
from tests.mlx_core_support import generate_core, load_core

# CUDA builds of MLX round convolutions to TF32 unless this is set before MLX starts.
os.environ.setdefault("MLX_ENABLE_TF32", "0")

ROOT = Path(__file__).parents[1]
DET = ROOT / "synapse/models/ppocr/generic-ppocr-det.axon"
REC = ROOT / "synapse/models/ppocr/generic-ppocr-rec.axon"

BACKBONE = {"expand_ratio": 2, "se_reduction": 4, "se_slope": 1 / 6}
DET_CONFIG = {
    **BACKBONE,
    "stem_dim": 8,
    "stage_dims": [8, 16, 16, 32],
    "stage_ends": [1, 3, 5, 7],
    "block_se": [True, False, True, False, False, False, True],
    "neck_dim": 16,
    "neck_se_reduction": 4,
    "neck_se_slope": 0.2,
    "smooth_dim": 8,
    "head_dim": 8,
}
REC_CONFIG = {
    **BACKBONE,
    "stem_dim": 8,
    "stage_dims": [16, 32, 24],
    "stage_ends": [2, 4, 5],
    "block_se": [True, False, False, True, False],
    "neck_dim": 24,
    "num_layers": 2,
    "num_heads": 4,
    "mlp_dim": 48,
    "norm_eps": 1e-5,
    "final_norm_eps": 1e-6,
    "vocab_size": 11,
}


# ---------------------------------------------------------------- synthetic weights


class _State(dict):
    def __init__(self, seed: int) -> None:
        super().__init__()
        self.rng = np.random.default_rng(seed)

    def conv(self, path: str, out: int, inp: int, kh: int, kw: int, *, bias: bool = True) -> None:
        self[f"{path}.weight"] = self.rng.standard_normal((out, inp, kh, kw)).astype(np.float32) * 0.3
        if bias:
            self[f"{path}.bias"] = self.rng.standard_normal(out).astype(np.float32) * 0.1

    def linear(self, path: str, out: int, inp: int) -> None:
        self[f"{path}.weight"] = self.rng.standard_normal((out, inp)).astype(np.float32) * 0.2
        self[f"{path}.bias"] = self.rng.standard_normal(out).astype(np.float32) * 0.1

    def norm(self, path: str, dim: int) -> None:
        self[f"{path}.weight"] = 1 + self.rng.standard_normal(dim).astype(np.float32) * 0.1
        self[f"{path}.bias"] = self.rng.standard_normal(dim).astype(np.float32) * 0.1

    def squeeze_excite(self, path: str, dim: int, reduction: int) -> None:
        self.conv(f"{path}.reduce", dim // reduction, dim, 1, 1)
        self.conv(f"{path}.expand", dim, dim // reduction, 1, 1)


def _backbone_state(state: _State, config: dict) -> None:
    stem, dims, ratio = config["stem_dim"], config["stage_dims"], config["expand_ratio"]
    state.conv("backbone.stem.conv1", stem, 3, 3, 3)
    state.conv("backbone.stem.conv2a", stem // 2, stem, 2, 2)
    state.conv("backbone.stem.conv2b", stem, stem // 2, 2, 2)
    state.conv("backbone.stem.conv3", stem, 2 * stem, 3, 3)
    state.conv("backbone.stem.conv4", dims[0], stem, 1, 1)
    start = 0
    for stage, end in enumerate(config["stage_ends"]):
        dim = dims[stage]
        for i in range(start, end):
            base = f"backbone.blocks.{i}"
            width = dims[stage - 1] if stage and i == start else dim
            state.conv(f"{base}.dw", width, 1, 3, 3)
            if config["block_se"][i]:
                state.squeeze_excite(f"{base}.se", width, config["se_reduction"])
            state.conv(f"{base}.expand", ratio * width, width, 1, 1)
            state.conv(f"{base}.project", dim, ratio * width, 1, 1)
        start = end


def _det_state(config: dict) -> dict[str, np.ndarray]:
    state = _State(3)
    _backbone_state(state, config)
    neck, smooth, head = config["neck_dim"], config["smooth_dim"], config["head_dim"]
    for level, dim in enumerate(config["stage_dims"]):
        state.conv(f"neck.lateral.{level}.conv", neck, dim, 1, 1, bias=False)
        state.squeeze_excite(f"neck.lateral.{level}.se", neck, config["neck_se_reduction"])
        state.conv(f"neck.smooth.{level}.dw", neck, 1, 7, 7)
        state.conv(f"neck.smooth.{level}.conv", smooth, neck, 1, 1, bias=False)
        state.squeeze_excite(f"neck.smooth.{level}.se", smooth, config["neck_se_reduction"])
    state.conv("head.conv", head, 4 * smooth, 3, 3)
    state.conv("head.up1", head, head, 2, 2)
    state["head.up1.bias"] = state["head.up1.bias"][:head]
    state.conv("head.up2", head, 1, 2, 2)
    state["head.up2.bias"] = state["head.up2.bias"][:1]
    return dict(state)


def _rec_state(config: dict) -> dict[str, np.ndarray]:
    state = _State(5)
    _backbone_state(state, config)
    neck = config["neck_dim"]
    state.conv("neck.shortcut", neck, config["stage_dims"][-1], 1, 1)
    state.conv("neck.reduce", neck, config["stage_dims"][-1], 1, 1)
    state.conv("neck.local", neck, 1, 1, 7)
    for layer in range(config["num_layers"]):
        base = f"encoder.layers.{layer}"
        state.norm(f"{base}.norm1", neck)
        state.linear(f"{base}.attn.qkv", 3 * neck, neck)
        state.linear(f"{base}.attn.proj", neck, neck)
        state.norm(f"{base}.norm2", neck)
        state.linear(f"{base}.mlp.fc1", config["mlp_dim"], neck)
        state.linear(f"{base}.mlp.fc2", neck, config["mlp_dim"])
    state.norm("encoder.norm", neck)
    state.linear("head.fc", config["vocab_size"], neck)
    return dict(state)


# ------------------------------------------------- PyTorch reference (channels-first)


def _conv(x, st, path, stride=1, padding=0, groups=1):
    return F.conv2d(x, st[f"{path}.weight"], st.get(f"{path}.bias"), stride=stride, padding=padding, groups=groups)


def _gate(x, st, path, slope):
    s = x.mean((2, 3), keepdim=True)
    s = _conv(F.relu(_conv(s, st, f"{path}.reduce")), st, f"{path}.expand")
    return (s * slope + 0.5).clamp(0, 1)


def _same_upper(x, value=0.0):
    return F.pad(x, (0, 1, 0, 1), value=value)


def _backbone(x, st, config, stride):
    stem = "backbone.stem"
    x = F.relu(_conv(x, st, f"{stem}.conv1", 2, 1))
    a = F.relu(_conv(_same_upper(x), st, f"{stem}.conv2a"))
    a = F.relu(_conv(_same_upper(a), st, f"{stem}.conv2b"))
    pooled = F.max_pool2d(_same_upper(x, float("-inf")), 2, 1)
    x = F.relu(_conv(torch.cat([pooled, a], 1), st, f"{stem}.conv3", 2, 1))
    x = F.relu(_conv(x, st, f"{stem}.conv4"))
    features, start = [], 0
    for stage, end in enumerate(config["stage_ends"]):
        for i in range(start, end):
            base = f"backbone.blocks.{i}"
            opens_stage = stage > 0 and i == start
            y = _conv(x, st, f"{base}.dw", stride if opens_stage else 1, 1, groups=x.shape[1])
            if config["block_se"][i]:
                y = y * _gate(y, st, f"{base}.se", config["se_slope"])
            f = _conv(F.gelu(_conv(y, st, f"{base}.expand")), st, f"{base}.project")
            x = f if opens_stage else y + f
        features.append(x)
        start = end
    return features


def _gated_project(x, st, path, slope):
    y = _conv(x, st, f"{path}.conv")
    return y + y * _gate(y, st, f"{path}.se", slope)


def _det_reference(x, st, config):
    slope = config["neck_se_slope"]
    features = _backbone(x, st, config, 2)
    tops = [_gated_project(f, st, f"neck.lateral.{level}", slope) for level, f in enumerate(features)]
    for level in (2, 1, 0):
        tops[level] = tops[level] + F.interpolate(tops[level + 1], size=tops[level].shape[2:])
    smooth = [
        _gated_project(_conv(t, st, f"neck.smooth.{level}.dw", 1, 3, groups=t.shape[1]), st, f"neck.smooth.{level}", slope)
        for level, t in enumerate(tops)
    ]
    size = smooth[0].shape[2:]
    fused = torch.cat([F.interpolate(s, size=size) for s in reversed(smooth[1:])] + [smooth[0]], 1)
    y = F.relu(_conv(fused, st, "head.conv", 1, 1))
    y = F.relu(F.conv_transpose2d(y, st["head.up1.weight"], st["head.up1.bias"], stride=2))
    return torch.sigmoid(F.conv_transpose2d(y, st["head.up2.weight"], st["head.up2.bias"], stride=2))


def _rec_reference(x, st, config):
    dim, heads = config["neck_dim"], config["num_heads"]
    x = F.avg_pool2d(_backbone(x, st, config, (2, 1))[-1], (3, 2))
    shortcut = F.silu(_conv(x, st, "neck.shortcut"))
    y = F.silu(_conv(x, st, "neck.reduce"))
    y = y + F.silu(_conv(y, st, "neck.local", 1, (0, 3), groups=dim))
    seq = y.flatten(2).transpose(1, 2)
    batch, steps, _ = seq.shape
    for layer in range(config["num_layers"]):
        base = f"encoder.layers.{layer}"
        h = F.layer_norm(seq, (dim,), st[f"{base}.norm1.weight"], st[f"{base}.norm1.bias"], config["norm_eps"])
        qkv = F.linear(h, st[f"{base}.attn.qkv.weight"], st[f"{base}.attn.qkv.bias"])
        q, k, v = qkv.reshape(batch, steps, 3, heads, dim // heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(batch, steps, dim)
        seq = seq + F.linear(a, st[f"{base}.attn.proj.weight"], st[f"{base}.attn.proj.bias"])
        h = F.layer_norm(seq, (dim,), st[f"{base}.norm2.weight"], st[f"{base}.norm2.bias"], config["norm_eps"])
        h = F.silu(F.linear(h, st[f"{base}.mlp.fc1.weight"], st[f"{base}.mlp.fc1.bias"]))
        seq = seq + F.linear(h, st[f"{base}.mlp.fc2.weight"], st[f"{base}.mlp.fc2.bias"])
    seq = F.layer_norm(seq, (dim,), st["encoder.norm.weight"], st["encoder.norm.bias"], config["final_norm_eps"])
    seq = seq + shortcut.flatten(2).transpose(1, 2)
    return F.softmax(F.linear(seq, st["head.fc.weight"], st["head.fc.bias"]), dim=-1)


CASES = {
    "det": (DET, DET_CONFIG, _det_state, _det_reference, (2, 64, 96, 3)),
    "rec": (REC, REC_CONFIG, _rec_state, _rec_reference, (2, 48, 64, 3)),
}


def _expected(kind: str) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    _, config, make_state, reference, shape = CASES[kind]
    state = make_state(config)
    x = np.random.default_rng(11).standard_normal(shape).astype(np.float32)
    tensors = {key: torch.from_numpy(value) for key, value in state.items()}
    with torch.no_grad():
        want = reference(torch.from_numpy(x).permute(0, 3, 1, 2), tensors, config)
    if kind == "det":
        want = want.permute(0, 2, 3, 1)
    return x, state, want.numpy()


@pytest.mark.parametrize("kind", ["det", "rec"])
def test_ppocr_torch_matches_reference(kind: str) -> None:
    source, config, *_ = CASES[kind]
    x, state, want = _expected(kind)
    core = load_core(generate_core(source, backend="torch", optimize=False), "AxonCore")
    model = core.from_state_dict({key: torch.from_numpy(value) for key, value in state.items()}, model_config=config)
    got = model.forward(pixel_values=torch.from_numpy(x)).numpy()
    assert got.shape == want.shape
    np.testing.assert_allclose(got, want, atol=2e-5, rtol=1e-4)


@pytest.mark.parametrize("optimize", [True, False], ids=["optimized", "plain"])
@pytest.mark.parametrize("kind", ["det", "rec"])
def test_ppocr_mlx_matches_reference(kind: str, optimize: bool) -> None:
    mx = pytest.importorskip("mlx.core", exc_type=ImportError)
    source, config, *_ = CASES[kind]
    x, state, want = _expected(kind)
    code = generate_core(source, backend="mlx", backend_intrinsics="codegen2-mlx", optimize=optimize)
    model = load_core(code, "AxonCore").from_state_dict(
        {key: mx.array(value) for key, value in state.items()}, model_config=config
    )
    got = np.array(model.forward(pixel_values=mx.array(x)))
    assert got.shape == want.shape
    np.testing.assert_allclose(got, want, atol=1e-4, rtol=1e-3)
    if kind == "rec":
        assert (got.argmax(-1) == want.argmax(-1)).all()


@pytest.mark.parametrize("source", [DET, REC], ids=["det", "rec"])
def test_ppocr_graphs_have_mlx_lowerings(source: Path) -> None:
    from synapse.axon import (
        elaborate_closed_axon_file,
        flatten_closed_axon_file,
        lower_axon_program_to_graph_ir,
        normalize_closed_axon_file,
        resolve_axon_program_from_path,
        typecheck2_flat_axon_file,
    )

    program = resolve_axon_program_from_path(source).ast
    for transform in (
        normalize_closed_axon_file,
        elaborate_closed_axon_file,
        flatten_closed_axon_file,
        typecheck2_flat_axon_file,
    ):
        program = transform(program)
    graph = lower_axon_program_to_graph_ir(program)
    assert non_obvious_mlx_ops(graph) == ()
    ops = {node.op.name for module in graph.modules for node in module.nodes}
    assert "_conv2d" in ops
