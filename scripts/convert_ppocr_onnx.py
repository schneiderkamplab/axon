"""Convert PP-OCR ONNX exports into Axon checkpoints (config.json + model.safetensors).

The PP-OCR detector and recognizer are published as ONNX graphs with anonymous
initializer names. This script walks each graph in node order, recovers the layer
structure (stem, depthwise blocks, neck, head), and writes

* ``model.safetensors`` with structured parameter names in the PyTorch layouts that
  ``synapse/models/ppocr/generic-ppocr-{det,rec}.axon`` reads (conv ``[O,I/g,KH,KW]``,
  transposed conv ``[I,O,KH,KW]``, linear ``[out,in]``), and
* ``config.json`` with the block table and layer sizes.

Batch normalisation is folded into the preceding convolution.

Usage:
    python scripts/convert_ppocr_onnx.py --det det.onnx --rec rec.onnx --out models/ppocrv6-small
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import numpy_helper
from safetensors.numpy import save_file


@dataclass
class Conv:
    weight: np.ndarray
    bias: np.ndarray | None
    stride: tuple[int, int]
    groups: int
    node: Any
    source_op: str | None  # op type producing the conv input


class Graph:
    def __init__(self, path: Path) -> None:
        graph = onnx.load(str(path)).graph
        self.inits = {init.name: numpy_helper.to_array(init) for init in graph.initializer}
        self.alias: dict[str, str] = {}
        self.nodes = []
        for node in graph.node:
            if node.op_type == "Identity":
                self.alias[node.output[0]] = node.input[0]
            else:
                self.nodes.append(node)
        self.producer = {out: node for node in self.nodes for out in node.output}
        self.consumers: dict[str, list[Any]] = {}
        for node in self.nodes:
            for name in node.input:
                self.consumers.setdefault(self.resolve(name), []).append(node)

    def resolve(self, name: str) -> str:
        while name in self.alias:
            name = self.alias[name]
        return name

    def const(self, name: str) -> np.ndarray | None:
        return self.inits.get(self.resolve(name))

    def inputs(self, node: Any) -> list[str]:
        return [self.resolve(name) for name in node.input]

    def only_consumer(self, node: Any) -> Any | None:
        users = self.consumers.get(node.output[0], [])
        return users[0] if len(users) == 1 else None

    def added_const(self, node: Any, size: int) -> np.ndarray | None:
        """The constant a lone following Add adds to this node's output, as a vector."""
        user = self.only_consumer(node)
        if user is None or user.op_type != "Add":
            return None
        for name in self.inputs(user):
            value = self.inits.get(name)
            if value is not None and value.size == size:
                return value.reshape(-1).astype(np.float32)
        return None

    def convs(self) -> list[Conv]:
        found = []
        for node in self.nodes:
            if node.op_type != "Conv":
                continue
            attrs = _attrs(node)
            names = self.inputs(node)
            weight = self.inits[names[1]].astype(np.float32)
            out_dim = weight.shape[0]
            bias = self.inits[names[2]].astype(np.float32) if len(names) > 2 else None
            if bias is None:
                bias = self.added_const(node, out_dim)
            user = self.only_consumer(node)
            if bias is None and user is not None and user.op_type == "BatchNormalization":
                gamma, beta, mean, var = (self.inits[name] for name in self.inputs(user)[1:5])
                scale = gamma / np.sqrt(var + _attrs(user).get("epsilon", 1e-5))
                weight = (weight * scale[:, None, None, None]).astype(np.float32)
                bias = (beta - mean * scale).astype(np.float32)
            source = self.producer.get(names[0])
            found.append(
                Conv(
                    weight=weight,
                    bias=bias,
                    stride=tuple(attrs.get("strides", (1, 1))),
                    groups=int(attrs.get("group", 1)),
                    node=node,
                    source_op=None if source is None else source.op_type,
                )
            )
        return found

    def conv_transposes(self) -> list[tuple[np.ndarray, np.ndarray | None]]:
        found = []
        for node in self.nodes:
            if node.op_type != "ConvTranspose":
                continue
            names = self.inputs(node)
            weight = self.inits[names[1]].astype(np.float32)
            bias = self.inits[names[2]].astype(np.float32) if len(names) > 2 else None
            if bias is None:
                bias = self.added_const(node, weight.shape[1])
            found.append((weight, bias))
        return found

    def linears(self) -> list[tuple[np.ndarray, np.ndarray | None]]:
        found = []
        for node in self.nodes:
            if node.op_type != "MatMul":
                continue
            weight = self.inits.get(self.inputs(node)[1])
            if weight is None or weight.ndim != 2:
                continue
            bias = self.added_const(node, weight.shape[1])
            found.append((np.ascontiguousarray(weight.T.astype(np.float32)), bias))
        return found

    def layer_norms(self) -> list[tuple[np.ndarray, np.ndarray, float]]:
        """Decomposed layer norms: (x - mean) / sqrt(var + eps) * weight + bias."""
        found = []
        for node in self.nodes:
            if node.op_type != "Sqrt":
                continue
            variance = self.producer[self.inputs(node)[0]]
            eps = next(
                float(self.inits[name].reshape(-1)[0])
                for name in self.inputs(variance)
                if name in self.inits
            )
            scaled = self.only_consumer(self.only_consumer(node))
            weight = next(self.inits[name] for name in self.inputs(scaled) if name in self.inits)
            bias = self.added_const(scaled, weight.size)
            found.append((weight.reshape(-1).astype(np.float32), bias, eps))
        return found


def _attrs(node: Any) -> dict[str, Any]:
    return {attr.name: onnx.helper.get_attribute_value(attr) for attr in node.attribute}


def _put(state: dict[str, np.ndarray], path: str, weight: np.ndarray, bias: np.ndarray | None) -> None:
    state[f"{path}.weight"] = np.ascontiguousarray(weight)
    if bias is not None:
        state[f"{path}.bias"] = np.ascontiguousarray(bias)


def _expect_shape(state: dict[str, np.ndarray], key: str, shape: tuple[int, ...]) -> None:
    if state[key].shape != shape:
        raise ValueError(f"{key}: expected shape {shape}, got {state[key].shape}")


def _put_conv(state: dict[str, np.ndarray], path: str, conv: Conv) -> None:
    _put(state, path, conv.weight, conv.bias)


def _hard_sigmoid_slope(graph: Graph, after: Any) -> float:
    """Slope of the first HardSigmoid at or after a node, in node order."""
    start = graph.nodes.index(after)
    node = next(node for node in graph.nodes[start:] if node.op_type == "HardSigmoid")
    return float(_attrs(node)["alpha"])


def convert_backbone(
    graph: Graph,
    convs: list[Conv],
    state: dict[str, np.ndarray],
    *,
    stages: int,
    stage_stride: tuple[int, int],
) -> tuple[dict[str, Any], int]:
    """Stem and depthwise blocks. Returns the config and the index of the first neck conv.

    The number of stages and the stride of the block that opens a stage are fixed in the
    model files, so they are checked here rather than written to the config.
    """
    for name, conv in zip(("conv1", "conv2a", "conv2b", "conv3", "conv4"), convs[:5], strict=True):
        _put_conv(state, f"backbone.stem.{name}", conv)
    dims, strides, has_se = [], [], []
    se_reduction = None
    cursor = 5
    while convs[cursor].groups > 1 and convs[cursor].source_op != "AveragePool":
        base = f"backbone.blocks.{len(dims)}"
        depthwise = convs[cursor]
        width = int(depthwise.weight.shape[0])
        if depthwise.weight.shape[2:] != (3, 3) or depthwise.groups != width:
            raise ValueError(f"{base}: expected a 3x3 depthwise conv, got {depthwise.weight.shape}")
        _put_conv(state, f"{base}.dw", depthwise)
        cursor += 1
        squeeze = convs[cursor].source_op in {"ReduceMean", "GlobalAveragePool"}
        if squeeze:
            _put_conv(state, f"{base}.se.reduce", convs[cursor])
            _put_conv(state, f"{base}.se.expand", convs[cursor + 1])
            se_reduction = width // convs[cursor].weight.shape[0]
            cursor += 2
        expand, project = convs[cursor], convs[cursor + 1]
        _put_conv(state, f"{base}.expand", expand)
        _put_conv(state, f"{base}.project", project)
        if expand.weight.shape[0] != 2 * width:
            raise ValueError(f"{base}: expected expansion ratio 2, got {expand.weight.shape[0]}/{width}")
        dims.append(int(project.weight.shape[0]))
        strides.append(tuple(int(s) for s in depthwise.stride))
        has_se.append(bool(squeeze))
        cursor += 2
    # A strided block opens each stage after the first; the other blocks keep their shape.
    stage_ends = [i for i, stride in enumerate(strides) if stride != (1, 1)]
    opening_strides = {strides[i] for i in stage_ends}
    if opening_strides != {stage_stride}:
        raise ValueError(f"expected stage stride {stage_stride}, got {sorted(opening_strides)}")
    for i, (stride, dim) in enumerate(zip(strides, dims, strict=True)):
        previous = dims[i - 1] if i else int(convs[4].weight.shape[0])
        if stride == (1, 1) and dim != previous:
            raise ValueError(f"backbone.blocks.{i}: width changes without a stride")
        if stride != (1, 1) and has_se[i]:
            raise ValueError(f"backbone.blocks.{i}: strided block with a channel gate")
    if len(stage_ends) + 1 != stages:
        raise ValueError(f"expected {stages} stages, got {len(stage_ends) + 1}")
    config = {
        "stem_dim": int(convs[0].weight.shape[0]),
        "stage_dims": [int(convs[4].weight.shape[0]), *(dims[i] for i in stage_ends)],
        "stage_ends": [*stage_ends, len(dims)],
        "block_se": has_se,
        "expand_ratio": 2,
        "se_reduction": se_reduction,
        "se_slope": _hard_sigmoid_slope(graph, convs[5].node),
    }
    return config, cursor


def convert_det(path: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    graph = Graph(path)
    convs = graph.convs()
    state: dict[str, np.ndarray] = {}
    config, cursor = convert_backbone(graph, convs, state, stages=4, stage_stride=(2, 2))
    levels = len(config["stage_ends"])
    neck_start = convs[cursor].node
    # Laterals then smoothing convs, each from the deepest level to the shallowest.
    for level in reversed(range(levels)):
        base = f"neck.lateral.{level}"
        _put_conv(state, f"{base}.conv", convs[cursor])
        _put_conv(state, f"{base}.se.reduce", convs[cursor + 1])
        _put_conv(state, f"{base}.se.expand", convs[cursor + 2])
        cursor += 3
    for level in reversed(range(levels)):
        base = f"neck.smooth.{level}"
        _put_conv(state, f"{base}.dw", convs[cursor])
        _put_conv(state, f"{base}.conv", convs[cursor + 1])
        _put_conv(state, f"{base}.se.reduce", convs[cursor + 2])
        _put_conv(state, f"{base}.se.expand", convs[cursor + 3])
        cursor += 4
    _put_conv(state, "head.conv", convs[cursor])
    if cursor + 1 != len(convs):
        raise ValueError(f"unexpected detector layout: {len(convs) - cursor - 1} unassigned convs")
    (up1_w, up1_b), (up2_w, up2_b) = graph.conv_transposes()
    _put(state, "head.up1", up1_w, up1_b)
    _put(state, "head.up2", up2_w, up2_b)
    neck_dim = int(state["neck.lateral.0.conv.weight"].shape[0])
    smooth_dim = int(state["neck.smooth.0.conv.weight"].shape[0])
    config.update(
        {
            "model_type": "ppocr_det",
            "neck_dim": neck_dim,
            "neck_se_reduction": neck_dim // int(state["neck.lateral.0.se.reduce.weight"].shape[0]),
            "neck_se_slope": _hard_sigmoid_slope(graph, neck_start),
            "smooth_dim": smooth_dim,
            "head_dim": int(state["head.conv.weight"].shape[0]),
        }
    )
    # Kernel sizes are fixed in generic-ppocr-det.axon.
    _expect_shape(state, "neck.smooth.0.dw.weight", (neck_dim, 1, 7, 7))
    _expect_shape(state, "head.conv.weight", (config["head_dim"], 4 * smooth_dim, 3, 3))
    _expect_shape(state, "head.up1.weight", (config["head_dim"], config["head_dim"], 2, 2))
    _expect_shape(state, "head.up2.weight", (config["head_dim"], 1, 2, 2))
    return config, state


def convert_rec(path: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    graph = Graph(path)
    convs = graph.convs()
    state: dict[str, np.ndarray] = {}
    config, cursor = convert_backbone(graph, convs, state, stages=3, stage_stride=(2, 1))
    shortcut, reduce, local = convs[cursor : cursor + 3]
    if cursor + 3 != len(convs):
        raise ValueError(f"unexpected recognizer layout: {len(convs) - cursor - 3} unassigned convs")
    _put_conv(state, "neck.shortcut", shortcut)
    _put_conv(state, "neck.reduce", reduce)
    _put_conv(state, "neck.local", local)
    linears = graph.linears()
    norms = graph.layer_norms()
    num_layers = (len(linears) - 1) // 4
    if len(linears) != 4 * num_layers + 1 or len(norms) != 2 * num_layers + 1:
        raise ValueError(f"unexpected encoder layout: {len(linears)} linears, {len(norms)} norms")
    for layer in range(num_layers):
        base = f"encoder.layers.{layer}"
        for name, (weight, bias) in zip(
            ("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"),
            linears[4 * layer : 4 * layer + 4],
            strict=True,
        ):
            _put(state, f"{base}.{name}", weight, bias)
        for name, (weight, bias, _) in zip(
            ("norm1", "norm2"), norms[2 * layer : 2 * layer + 2], strict=True
        ):
            _put(state, f"{base}.{name}", weight, bias)
    _put(state, "encoder.norm", norms[-1][0], norms[-1][1])
    _put(state, "head.fc", *linears[-1])
    pool = next(node for node in graph.nodes if node.op_type == "AveragePool")
    heads = next(
        int(shape[3])
        for node in graph.nodes
        if node.op_type == "Reshape"
        and (shape := graph.const(node.input[1])) is not None
        and shape.size == 5
    )
    config.update(
        {
            "model_type": "ppocr_rec",
            "neck_dim": int(reduce.weight.shape[0]),
            "num_layers": num_layers,
            "num_heads": heads,
            "mlp_dim": int(state["encoder.layers.0.mlp.fc1.weight"].shape[0]),
            "norm_eps": norms[0][2],
            "final_norm_eps": norms[-1][2],
            "vocab_size": int(state["head.fc.weight"].shape[0]),
        }
    )
    # The pooling window and the local kernel are fixed in generic-ppocr-rec.axon.
    if list(_attrs(pool)["kernel_shape"]) != [3, 2] or list(_attrs(pool)["strides"]) != [3, 2]:
        raise ValueError(f"unexpected recognizer pooling: {_attrs(pool)}")
    _expect_shape(state, "neck.local.weight", (config["neck_dim"], 1, 1, 7))
    return config, state


def write_checkpoint(out_dir: Path, config: dict[str, Any], state: dict[str, np.ndarray]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    save_file(state, str(out_dir / "model.safetensors"))
    params = sum(value.size for value in state.values())
    print(f"{out_dir}: {len(state)} tensors, {params:,} parameters")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--det", type=Path, help="detector ONNX file")
    parser.add_argument("--rec", type=Path, help="recognizer ONNX file")
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    args = parser.parse_args()
    if args.det is None and args.rec is None:
        parser.error("pass --det and/or --rec")
    if args.det is not None:
        write_checkpoint(args.out / "det", *convert_det(args.det))
    if args.rec is not None:
        write_checkpoint(args.out / "rec", *convert_rec(args.rec))


if __name__ == "__main__":
    main()
