#!/usr/bin/env python3
"""Generate Axon Torch/MLX source and provider-specific ONNX NER artifacts."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import onnx
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from onnxruntime.quantization.shape_inference import quant_pre_process
from onnxruntime.transformers.optimizer import optimize_model
from transformers import AutoModelForTokenClassification


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--axon",
        type=Path,
        default=Path("synapse/models/bert/generic-bert-token-classification.axon"),
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((args.checkpoint / "config.json").read_text())
    if (
        cfg.get("hidden_act", "gelu") != "gelu"
        or cfg.get("position_embedding_type", "absolute") != "absolute"
    ):
        raise ValueError(
            "This Axon definition supports GELU BERT with absolute position embeddings"
        )
    for backend in ("torch", "mlx"):
        command = [
            sys.executable,
            "-c",
            "from synapse.cli.synapse import app; app()",
            "axon-codegen-dump",
            str(args.axon),
            str(args.output / f"axon_{backend}.py"),
            "--backend",
            f"codegen2-{backend}",
            "--class-name",
            "NerModel",
            "--weights",
            str(args.checkpoint),
            "--optimize-graph",
            "--graph-backend-intrinsics",
            f"codegen2-{backend}:__{backend}_sdpa",
            "--force",
        ]
        subprocess.run(command, check=True)
        if not (args.output / f"axon_{backend}.py").exists():
            raise RuntimeError("Axon CLI did not create the generated model")
    model = AutoModelForTokenClassification.from_pretrained(
        args.checkpoint, local_files_only=True, attn_implementation="eager"
    ).eval()

    class Logits(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, input_ids, attention_mask, token_type_ids):
            return self.model(
                input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
            ).logits

    inputs = (
        torch.ones((2, 16), dtype=torch.long),
        torch.ones((2, 16), dtype=torch.long),
        torch.zeros((2, 16), dtype=torch.long),
    )
    original = args.output / "model.onnx"
    with torch.inference_mode():
        torch.onnx.export(
            Logits(model),
            inputs,
            original,
            input_names=["input_ids", "attention_mask", "token_type_ids"],
            output_names=["logits"],
            opset_version=17,
            dynamo=False,
            dynamic_axes={
                k: {0: "batch", 1: "sequence"}
                for k in ("input_ids", "attention_mask", "token_type_ids", "logits")
            },
        )
    onnx.checker.check_model(original)
    fusion_counts = {}
    for device in ("cpu", "cuda"):
        optimized = optimize_model(
            str(original),
            model_type="bert",
            num_heads=cfg["num_attention_heads"],
            hidden_size=cfg["hidden_size"],
            opt_level=1,
            use_gpu=device == "cuda",
        )
        fusion_counts[device] = optimized.get_fused_operator_statistics()
        optimized.save_model_to_file(str(args.output / f"model_{device}.onnx"))
        if device == "cuda":
            optimized.convert_float_to_float16(keep_io_types=True)
            optimized.save_model_to_file(str(args.output / "model_cuda_fp16.onnx"))
    quant_ready = args.output / "model_cpu_quant_ready.onnx"
    quant_pre_process(
        args.output / "model_cpu.onnx", quant_ready, skip_optimization=True, auto_merge=True
    )
    quantize_dynamic(
        str(quant_ready),
        str(args.output / "model_cpu_int8.onnx"),
        per_channel=True,
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["MatMul", "Attention"],
    )
    artifacts = {}
    for path in sorted(args.output.iterdir()):
        if path.suffix not in {".py", ".onnx"}:
            continue
        record = {
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        if path.suffix == ".onnx":
            graph = onnx.load(path)
            record["operators"] = dict(Counter(n.op_type for n in graph.graph.node))
        artifacts[path.name] = record
    versions = {}
    for package in ("torch", "transformers", "onnx", "onnxruntime-gpu", "onnxruntime"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    manifest = {
        "checkpoint_sha256": hashlib.sha256(
            (args.checkpoint / "model.safetensors").read_bytes()
        ).hexdigest(),
        "axon_source": str(args.axon),
        "fusion_counts": fusion_counts,
        "versions": versions,
        "artifacts": artifacts,
        "mlx_executed": False,
    }
    (args.output / "export.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
