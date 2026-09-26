#!/usr/bin/env python3
"""Prepare and fine-tune public MiniLM on Few-NERD SUP; never select on test."""

from __future__ import annotations

import argparse
import functools
import hashlib
import importlib.metadata
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from datasets import DatasetDict, load_dataset, load_from_disk
from huggingface_hub import HfApi, snapshot_download
from ner_common import NerScores, collate_windows, encode_words
from torch.utils.data import DataLoader
from transformers import AutoModelForTokenClassification, AutoTokenizer


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def prepare(args):
    root = args.run_dir
    if (root / "manifest.json").exists() or (root / "checkpoint").exists():
        raise FileExistsError("Use a fresh run directory to preserve the pinned experiment")
    root.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    base_sha = api.model_info(args.base_model, revision=args.base_revision).sha
    data_sha = api.dataset_info(args.dataset, revision=args.dataset_revision).sha
    reference_repo = "dslim/bert-base-NER"
    reference_sha = api.model_info(reference_repo).sha
    patterns = ["*.json", "*.safetensors", "vocab.txt", "tokenizer*", "pytorch_model.bin"]
    snapshot_download(
        args.base_model, revision=base_sha, local_dir=root / "base", allow_patterns=patterns
    )
    snapshot_download(
        reference_repo,
        revision=reference_sha,
        local_dir=root / "public-reference",
        allow_patterns=[p for p in patterns if p != "pytorch_model.bin"],
    )
    tokenizer = AutoTokenizer.from_pretrained(root / "base", local_files_only=True)
    raw = load_dataset(
        args.dataset, "supervised", revision=data_sha, cache_dir=str(root / "dataset-cache")
    )
    raw.save_to_disk(root / "raw")
    labels = raw["train"].features["fine_ner_tags"].feature.names
    if "O" not in labels:
        raise ValueError("Few-NERD label names must include O")
    encoded = DatasetDict(
        {
            split: data.map(
                functools.partial(encode_words, tokenizer=tokenizer, max_length=512),
                batched=True,
                batch_size=512,
                with_indices=True,
                remove_columns=data.column_names,
                desc=f"Encode {split} with full word coverage",
            )
            for split, data in raw.items()
        }
    )
    encoded.save_to_disk(root / "encoded")
    manifest = {
        "base_model": args.base_model,
        "base_revision": base_sha,
        "dataset": args.dataset,
        "dataset_revision": data_sha,
        "dataset_config": "supervised",
        "public_reference": reference_repo,
        "public_reference_revision": reference_sha,
        "labels": labels,
        "tag_scheme": "IO",
        "max_length": 512,
        "alignment": "first subtoken; word-boundary windows; zero-subtoken words become UNK",
        "sentences": {k: len(v) for k, v in raw.items()},
        "windows": {k: len(v) for k, v in encoded.items()},
        "empty_token_words": {k: sum(v["empty_token_words"]) for k, v in encoded.items()},
        "versions": {
            p: importlib.metadata.version(p)
            for p in ("torch", "transformers", "datasets", "tokenizers", "safetensors")
        },
        "quality_gates": {
            "fp32_atol": 1e-4,
            "fp32_rtol": 1e-4,
            "reduced_precision_max_absolute_f1_drop": 0.005,
            "reduced_precision_max_absolute_macro_f1_drop": 0.005,
        },
    }
    write_json(root / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2), flush=True)


@torch.inference_mode()
def evaluate(model, dataset, labels, collate, device, batch_size):
    model.eval()
    scores = NerScores(labels)
    for start in range(0, len(dataset), batch_size):
        rows = [dataset[i] for i in range(start, min(start + batch_size, len(dataset)))]
        inputs = {k: v.to(device) for k, v in collate(rows).items() if k != "labels"}
        pred = model(**inputs).logits.argmax(-1).cpu().numpy()
        scores.add(rows, pred)
    return scores.result()


def train(args):
    root = args.run_dir
    manifest = json.loads((root / "manifest.json").read_text())
    if (root / "checkpoint").exists():
        raise FileExistsError("Use a fresh run directory to avoid overwriting a frozen checkpoint")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tokenizer = AutoTokenizer.from_pretrained(root / "base", local_files_only=True)
    labels = manifest["labels"]
    model = AutoModelForTokenClassification.from_pretrained(
        root / "base",
        local_files_only=True,
        num_labels=len(labels),
        id2label=dict(enumerate(labels)),
        label2id={v: i for i, v in enumerate(labels)},
        attn_implementation="sdpa",
    ).to(args.device)
    data = load_from_disk(root / "encoded")
    collate = functools.partial(collate_windows, pad_token_id=tokenizer.pad_token_id)
    loader = DataLoader(
        data["train"],
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
        generator=torch.Generator().manual_seed(args.seed),
    )
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        (no_decay if name.endswith("bias") or "LayerNorm.weight" in name else decay).append(
            parameter
        )
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.01}, {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr,
    )
    total_steps = len(loader) * args.epochs
    warmup = math.ceil(total_steps * 0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min(
            step / max(1, warmup), max(0.0, (total_steps - step) / max(1, total_steps - warmup))
        ),
    )
    recipe = {
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "weight_decay": 0.01,
        "warmup_fraction": 0.1,
        "max_grad_norm": 1.0,
        "training_precision": "bfloat16 autocast, FP32 parameters",
        "checkpoint_selection": "highest validation exact-span micro F1; earliest on ties",
        "device": torch.cuda.get_device_name() if args.device.startswith("cuda") else args.device,
        "total_steps": total_steps,
        "threads": args.threads,
    }
    write_json(root / "training-recipe.json", recipe)
    print(json.dumps(recipe), flush=True)
    best, step, history = -1.0, 0, []
    started = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        loss_sum = 0.0
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=torch.device(args.device).type, dtype=torch.bfloat16):
                loss = model(**{k: v.to(args.device) for k, v in batch.items()}).loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at step {step}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            loss_sum += loss.item()
            step += 1
            if step % 100 == 0:
                print(
                    json.dumps(
                        {
                            "epoch": epoch + 1,
                            "step": step,
                            "loss": loss.item(),
                            "elapsed_seconds": time.perf_counter() - started,
                        }
                    ),
                    flush=True,
                )
        metrics = evaluate(model, data["validation"], labels, collate, args.device, args.batch_size)
        history.append(
            {"epoch": epoch + 1, "train_loss": loss_sum / len(loader), "validation": metrics}
        )
        print(json.dumps({"epoch": epoch + 1, "validation_f1": metrics["f1"]}), flush=True)
        if metrics["f1"] > best:
            best = metrics["f1"]
            model.save_pretrained(root / "checkpoint", safe_serialization=True)
            tokenizer.save_pretrained(root / "checkpoint")
            write_json(root / "selection.json", {"epoch": epoch + 1, "validation": metrics})
        write_json(root / "training-history.json", history)
    checkpoint = root / "checkpoint" / "model.safetensors"
    write_json(
        root / "training-complete.json",
        {
            "seconds": time.perf_counter() - started,
            "steps": step,
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "test_evaluated": False,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "train"])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--base-model", default="nreimers/MiniLM-L6-H384-uncased")
    parser.add_argument("--base-revision", default="main")
    parser.add_argument("--dataset", default="DFKI-SLT/few-nerd")
    parser.add_argument("--dataset-revision", default="main")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-5)
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
