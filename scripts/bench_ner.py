#!/usr/bin/env python3
"""Compare one frozen NER checkpoint across HF, Axon, ORT, and optional Mac MLX.

All timed model calls start with host token arrays and finish with host logits.
Use a separate process/output file for each runtime and precision.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import platform
import resource
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk
from ner_common import NerScores, collate_windows, encode_words, io_spans
from transformers import AutoModelForTokenClassification, AutoTokenizer


def generated_class(path):
    spec = importlib.util.spec_from_file_location("ner_generated", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.NerModel


class Backend:
    def __init__(self, args):
        self.args = args
        self.info = {}
        self.root = args.run_dir
        self.sessions = {}
        self.config = json.loads((self.root / "checkpoint/config.json").read_text())
        started = time.perf_counter()
        if args.backend in {"hf", "axon"}:
            dtype = torch.float16 if args.dtype == "fp16" else torch.float32
            if args.backend == "hf":
                self.model = (
                    AutoModelForTokenClassification.from_pretrained(
                        self.root / "checkpoint",
                        local_files_only=True,
                        attn_implementation="sdpa",
                        dtype=dtype,
                    )
                    .to(args.device)
                    .eval()
                )
            else:
                from safetensors.torch import load_file

                state = {
                    k: v.to(device=args.device, dtype=dtype)
                    for k, v in load_file(str(self.root / "checkpoint/model.safetensors")).items()
                }
                self.model = (
                    generated_class(self.root / "export/axon_torch.py")
                    .from_state_dict(state)
                    .eval()
                )
            if args.compile:
                self.model = torch.compile(self.model)
        elif args.backend == "mlx":
            import mlx.core as mx

            mx.set_default_device(mx.gpu)
            self.model = (
                generated_class(self.root / "export/axon_mlx.py")
                .from_safetensors(
                    [self.root / "checkpoint/model.safetensors"],
                    dtype="float16" if args.dtype == "fp16" else "float32",
                )
                .eval()
            )

            # Compile the encoder graph directly. No decoder/cache fast path.
            def forward(ids, mask, segments):
                return self.model(input_ids=ids, attn_mask=mask, token_type_ids=segments)

            self.mlx_forward = mx.compile(forward) if args.compile else forward
        else:
            import onnxruntime as ort

            ort.disable_telemetry_events()
            if args.device.startswith("cuda") and hasattr(ort, "preload_dlls"):
                ort.preload_dlls()
            self.provider = {
                "cpu": "CPUExecutionProvider",
                "cuda:0": "CUDAExecutionProvider",
                "coreml": "CoreMLExecutionProvider",
            }[args.device]
            if self.provider not in ort.get_available_providers():
                raise RuntimeError(f"Required provider unavailable: {self.provider}")
            name = (
                "model_cpu_int8.onnx"
                if args.dtype == "int8"
                else (
                    "model_cuda_fp16.onnx"
                    if args.dtype == "fp16"
                    else "model.onnx"
                    if args.device == "coreml"
                    else "model_cuda.onnx"
                    if args.device.startswith("cuda")
                    else "model_cpu.onnx"
                )
            )
            self.onnx_path = self.root / "export" / name
            self.info["graph"] = name
            if not args.coreml_static:
                self.sessions[None] = self.ort_session(self.onnx_path)
        self.info["load_seconds"] = time.perf_counter() - started

    def ort_session(self, path, profile=False):
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = self.args.threads
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.enable_profiling = profile
        options.profile_file_prefix = str(self.args.output.with_suffix(""))
        if self.args.device == "coreml":
            provider = (
                self.provider,
                {
                    "ModelFormat": "MLProgram",
                    "MLComputeUnits": self.args.compute_units,
                    "RequireStaticInputShapes": "1" if self.args.coreml_static else "0",
                    "ModelCacheDirectory": str(self.root / "coreml-cache"),
                },
            )
        elif self.args.device.startswith("cuda"):
            provider = (self.provider, {"use_tf32": "0"})
        else:
            provider = self.provider
        session = ort.InferenceSession(
            str(path),
            sess_options=options,
            providers=[provider, "CPUExecutionProvider"]
            if self.args.device != "cpu"
            else [provider],
        )
        if self.provider not in session.get_providers():
            raise RuntimeError(f"Session silently lost {self.provider}")
        return session

    def get_session(self, arrays):
        if not self.args.coreml_static:
            return self.sessions[None]
        import onnx
        from onnx.tools.update_model_dims import update_inputs_outputs_dims

        batch, length = arrays["input_ids"].shape
        bucket = next(n for n in (32, 128, 512) if n >= length)
        key = batch, bucket
        if key not in self.sessions:
            model = onnx.load(self.onnx_path)
            model = update_inputs_outputs_dims(
                model,
                {name: [batch, bucket] for name in arrays},
                {"logits": [batch, bucket, len(self.config["id2label"])]},
            )
            path = self.root / "export" / f"coreml_b{batch}_s{bucket}.onnx"
            onnx.save(model, path)
            self.sessions[key] = self.ort_session(path)
        return self.sessions[key]

    @torch.inference_mode()
    def predict(self, arrays):
        args = self.args
        if args.backend in {"hf", "axon"}:
            inputs = {k: torch.from_numpy(v).to(args.device) for k, v in arrays.items()}
            if args.backend == "axon":
                inputs["attn_mask"] = inputs.pop("attention_mask")
                output = self.model(**inputs)
            else:
                output = self.model(**inputs).logits
            return output.float().cpu().numpy()
        if args.backend == "mlx":
            import mlx.core as mx

            output = self.mlx_forward(
                *(mx.array(arrays[k]) for k in ("input_ids", "attention_mask", "token_type_ids"))
            )
            mx.eval(output)
            return np.asarray(output, dtype=np.float32)
        length = arrays["input_ids"].shape[1]
        session = self.get_session(arrays)
        if args.coreml_static:
            bucket = next(n for n in (32, 128, 512) if n >= length)
            arrays = {
                k: np.pad(
                    v,
                    ((0, 0), (0, bucket - length)),
                    constant_values=self.config["pad_token_id"] if k == "input_ids" else 0,
                )
                for k, v in arrays.items()
            }
        return session.run(["logits"], arrays)[0][:, :length].astype(np.float32)

    def profile_placement(self, arrays):
        if self.args.backend != "ort":
            return None
        # A separate session keeps profiling overhead out of measured latency.
        path = self.onnx_path
        if self.args.coreml_static:
            b, s = arrays["input_ids"].shape
            path = self.root / "export" / f"coreml_b{b}_s{s}.onnx"
        session = self.ort_session(path, profile=True)
        session.run(["logits"], arrays)
        events = json.loads(Path(session.end_profiling()).read_text())
        counts = Counter(
            e.get("args", {}).get("provider")
            for e in events
            if e.get("cat") == "Node" and e.get("args", {}).get("provider")
        )
        if self.provider != "CPUExecutionProvider" and not counts[self.provider]:
            raise RuntimeError(f"No graph nodes executed on {self.provider}: {dict(counts)}")
        return dict(counts)


def arrays_for(rows, tokenizer):
    return {
        k: v.numpy()
        for k, v in collate_windows(rows, pad_token_id=tokenizer.pad_token_id).items()
        if k != "labels"
    }


def quality(args, backend, tokenizer, manifest):
    data = load_from_disk(args.run_dir / "encoded")[args.split]
    scores = NerScores(manifest["labels"])
    reference_path = args.run_dir / f"reference-{args.split}.npy"
    if args.write_reference:
        if args.backend != "hf" or args.dtype != "fp32":
            raise ValueError("Only HF FP32 can write the reference")
        if reference_path.exists():
            raise FileExistsError(f"Reference already exists: {reference_path}")
        total = sum(sum(x >= 0 for x in row) for row in data["word_ids"])
        reference = np.lib.format.open_memmap(
            reference_path, mode="w+", dtype=np.float32, shape=(total, len(manifest["labels"]))
        )
    else:
        reference = np.load(reference_path, mmap_mode="r")
    offset, same, elements, abs_sum, max_abs = 0, 0, 0, 0.0, 0.0
    allclose = True
    started = time.perf_counter()
    for start in range(0, len(data), args.batch_size):
        rows = [data[i] for i in range(start, min(start + args.batch_size, len(data)))]
        arrays = arrays_for(rows, tokenizer)
        logits = backend.predict(arrays)
        expected_shape = (*arrays["input_ids"].shape, len(manifest["labels"]))
        if logits.shape != expected_shape:
            raise ValueError(f"Logit shape {logits.shape} does not match {expected_shape}")
        if not np.isfinite(logits).all():
            raise FloatingPointError(f"Non-finite logits in batch starting at {start}")
        scores.add(rows, logits.argmax(-1))
        words = np.concatenate(
            [
                logits[i, np.flatnonzero(np.asarray(row["word_ids"]) >= 0)]
                for i, row in enumerate(rows)
            ]
        )
        end = offset + len(words)
        if args.write_reference:
            reference[offset:end] = words
        expected = reference[offset:end]
        delta = np.abs(words - expected)
        max_abs = max(max_abs, float(delta.max()))
        abs_sum += float(delta.sum(dtype=np.float64))
        elements += delta.size
        same += int(np.sum(words.argmax(-1) == expected.argmax(-1)))
        allclose &= bool(np.allclose(words, expected, atol=1e-4, rtol=1e-4))
        offset = end
    elapsed = time.perf_counter() - started
    if offset != len(reference):
        raise ValueError("Reference word count does not match evaluation")
    result = scores.result()
    result.update(
        {
            "seconds": elapsed,
            "word_logit_max_abs_diff": max_abs,
            "word_logit_mean_abs_diff": abs_sum / elements,
            "word_prediction_agreement": same / offset,
            "fp32_allclose": allclose,
        }
    )
    reference_metrics = args.run_dir / f"reference-{args.split}.json"
    if args.write_reference:
        reference.flush()
        reference_metrics.write_text(json.dumps(result, indent=2) + "\n")
    baseline = json.loads(reference_metrics.read_text())
    result["f1_change"] = result["f1"] - baseline["f1"]
    result["macro_f1_change"] = result["macro_f1"] - baseline["macro_f1"]
    result["quality_gate_passed"] = (
        allclose
        if args.dtype == "fp32" and args.device != "coreml"
        else (result["f1_change"] >= -0.005 and result["macro_f1_change"] >= -0.005)
    )
    return result


def measurements(args, call):
    start = time.perf_counter()
    call()
    first = time.perf_counter() - start
    for _ in range(args.warmup):
        call()
    samples = []
    for _ in range(args.rounds):
        for _ in range(args.repeat):
            start = time.perf_counter_ns()
            call()
            samples.append((time.perf_counter_ns() - start) / 1e6)
    return {
        "first_call_ms": first * 1000,
        "p50_ms": float(np.median(samples)),
        "p95_ms": float(np.percentile(samples, 95)),
        "mean_ms": float(np.mean(samples)),
        "samples_ms": samples,
    }


def performance(args, backend, tokenizer):
    rng = np.random.default_rng(17)
    shape_results = []
    last = None
    for batch in (1, 8, 32):
        for length in (32, 128, 512):
            # Vary inputs to prevent a lazy/compiled backend timing a cached constant.
            ring = [
                {
                    "input_ids": rng.integers(
                        1, backend.config["vocab_size"], (batch, length), dtype=np.int64
                    ),
                    "attention_mask": np.ones((batch, length), dtype=np.int64),
                    "token_type_ids": np.zeros((batch, length), dtype=np.int64),
                }
                for _ in range(4)
            ]
            cursor = 0

            def call():
                nonlocal cursor
                value = backend.predict(ring[cursor % len(ring)])
                cursor += 1
                return value

            result = measurements(args, call)
            result.update(
                {
                    "batch": batch,
                    "length": length,
                    "sequences_per_second": batch * 1000 / result["mean_ms"],
                    "useful_tokens_per_second": batch * length * 1000 / result["mean_ms"],
                }
            )
            shape_results.append(result)
            print(json.dumps({k: v for k, v in result.items() if k != "samples_ms"}), flush=True)
            last = ring[0]
    placement = backend.profile_placement(last)
    raw = load_from_disk(args.run_dir / "raw")[args.split]
    # Deterministic evenly spaced sentences; no labels used to tune runtime settings.
    choices = np.linspace(0, len(raw) - 1, min(256, len(raw)), dtype=int)
    rows = [raw[int(i)] for i in choices]
    e2e = []
    for batch in (1, 8, 32):
        cursor = 0
        word_counts = []

        def call_e2e():
            nonlocal cursor
            selected = [rows[(cursor + i) % len(rows)] for i in range(batch)]
            cursor = (cursor + batch) % len(rows)
            source = {k: [row[k] for row in selected] for k in ("tokens", "fine_ner_tags")}
            encoded = encode_words(source, list(range(batch)), tokenizer=tokenizer)
            windows = [
                dict(zip(encoded, values, strict=True))
                for values in zip(*encoded.values(), strict=True)
            ]
            predictions = backend.predict(arrays_for(windows, tokenizer)).argmax(-1)
            sentences = [[] for _ in selected]
            for row, pred in zip(windows, predictions, strict=True):
                sentences[row["sentence_id"]].extend(
                    int(pred[j]) for j, word in enumerate(row["word_ids"]) if word >= 0
                )
            result = [io_spans(sentence) for sentence in sentences]
            word_counts.append(sum(len(row["tokens"]) for row in selected))
            return result

        result = measurements(args, call_e2e)
        result.update(
            {
                "batch": batch,
                "sentences_per_second": batch * 1000 / result["mean_ms"],
                "scope": "presegmented words -> tokenization/windows -> host inference -> IO spans",
            }
        )
        e2e.append(result)
    return {"shapes": shape_results, "end_to_end": e2e, "provider_node_events": placement}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=["hf", "axon", "ort", "mlx"], required=True)
    parser.add_argument("--device", choices=["cpu", "cuda:0", "coreml", "metal"], default="cpu")
    parser.add_argument("--dtype", choices=["fp32", "fp16", "int8"], default="fp32")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--coreml-static", action="store_true")
    parser.add_argument(
        "--compute-units", choices=["ALL", "CPUAndGPU", "CPUAndNeuralEngine"], default="ALL"
    )
    parser.add_argument("--stage", choices=["all", "quality", "performance"], default="all")
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--write-reference", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a fresh output path for every run")
    if args.dtype == "int8" and (args.backend != "ort" or args.device != "cpu"):
        parser.error("INT8 is the ORT CPU configuration")
    if args.backend == "mlx" and args.device != "metal":
        parser.error("Use --device metal for the Mac MLX run")
    if args.backend in {"hf", "axon"} and args.device not in {"cpu", "cuda:0"}:
        parser.error("HF and Axon Torch use --device cpu or cuda:0")
    if args.backend == "ort" and args.device == "metal":
        parser.error("ORT on Apple accelerators uses --device coreml")
    if args.coreml_static and (args.backend != "ort" or args.device != "coreml"):
        parser.error("--coreml-static requires ORT Core ML")
    if args.device == "coreml" and args.dtype != "fp32":
        parser.error(
            "Core ML uses the original FP32 ONNX graph; provider precision is recorded separately"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    expected_hash = json.loads((args.run_dir / "training-complete.json").read_text())[
        "checkpoint_sha256"
    ]
    actual_hash = hashlib.sha256(
        (args.run_dir / "checkpoint/model.safetensors").read_bytes()
    ).hexdigest()
    export_hash = json.loads((args.run_dir / "export/export.json").read_text())["checkpoint_sha256"]
    if actual_hash != expected_hash or export_hash != expected_hash:
        raise ValueError("Checkpoint and exported artifacts must belong to the same frozen model")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tokenizer = AutoTokenizer.from_pretrained(args.run_dir / "checkpoint", local_files_only=True)
    manifest = json.loads((args.run_dir / "manifest.json").read_text())
    backend = Backend(args)
    result = {
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "platform": platform.platform(),
        "processor": platform.processor(),
        "checkpoint_sha256": json.loads((args.run_dir / "training-complete.json").read_text())[
            "checkpoint_sha256"
        ],
        "dataset_revision": manifest["dataset_revision"],
        "versions": {},
        "timing_scope": "host token arrays -> completed host logits; transfers included",
        "load": backend.info,
    }
    for package in ("torch", "transformers", "numpy", "onnxruntime-gpu", "onnxruntime", "mlx"):
        try:
            result["versions"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    if args.device.startswith("cuda"):
        result["gpu"] = torch.cuda.get_device_name()
        torch.cuda.reset_peak_memory_stats()
    if args.stage in {"all", "performance"}:
        result["performance"] = performance(args, backend, tokenizer)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    if args.stage in {"all", "quality"}:
        result["quality"] = quality(args, backend, tokenizer, manifest)
        print(
            json.dumps(
                {"quality": {k: v for k, v in result["quality"].items() if k != "per_label"}}
            ),
            flush=True,
        )
    result["process_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (
        1 if platform.system() == "Darwin" else 1024
    )
    if args.device.startswith("cuda"):
        result["torch_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    if args.backend == "mlx":
        import mlx.core as mx

        result["mlx_peak_allocated_bytes"] = mx.get_peak_memory()
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    gc.collect()
    if "quality" in result and not result["quality"]["quality_gate_passed"]:
        raise RuntimeError(f"Quality gate failed; measurements retained at {args.output}")


if __name__ == "__main__":
    main()
