#!/usr/bin/env python3
"""Create a portable Mac evaluation directory and archive from a frozen NER run."""

import argparse
import hashlib
import json
import shutil
import subprocess
import tarfile
from pathlib import Path


def digest(path):
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Fresh directory under log/")
    args = parser.parse_args()
    root, target = args.run_dir, args.output
    archive = target.with_suffix(".tar.gz")
    if target.exists() or archive.exists():
        raise FileExistsError("Use a fresh output path to preserve earlier handoffs")
    frozen = json.loads((root / "training-complete.json").read_text())
    if digest(root / "checkpoint/model.safetensors") != frozen["checkpoint_sha256"]:
        raise ValueError("Frozen checkpoint hash changed")
    target.mkdir(parents=True)
    shutil.copytree(root / "checkpoint", target / "checkpoint")
    (target / "export").mkdir()
    for name in (
        "export.json",
        "axon_mlx.py",
        "axon_torch.py",
        "model.onnx",
        "model_cpu.onnx",
        "model_cpu_int8.onnx",
    ):
        shutil.copyfile(root / "export" / name, target / "export" / name)
    for dataset in ("raw", "encoded"):
        folder = target / dataset
        folder.mkdir()
        shutil.copytree(root / dataset / "test", folder / "test")
        (folder / "dataset_dict.json").write_text('{"splits": ["test"]}\n')
    for name in (
        "manifest.json",
        "training-recipe.json",
        "training-complete.json",
        "selection.json",
        "training-history.json",
        "reference-test.npy",
        "reference-test.json",
    ):
        shutil.copyfile(root / name, target / name)
    for name in ("public-results.json", "public-report.md"):
        if (root / name).exists():
            shutil.copyfile(root / name, target / ("linux-" + name))
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    (target / "MAC_RUN.md").write_text(
        f"""# Mac evaluation of the frozen MiniLM / Few-NERD checkpoint

Source commit: `{revision}`. Use that Axon checkout with Python 3.13 on Apple Silicon.
Extract this archive below its `log/` directory; the commands below assume
`log/{target.name}`. This bundle contains the official test split only.

```bash
uv venv --python 3.13
uv pip install --python .venv/bin/python -e '.[mlx,ner-benchmark]'
OMP_NUM_THREADS=4 .venv/bin/python scripts/run_ner_benchmarks.py --run-dir log/{target.name} --results-dir log/{target.name}-comparison/results --suite mac-compare --stage quality --keep-going
# Run performance only after the quality suite succeeds:
OMP_NUM_THREADS=4 .venv/bin/python scripts/run_ner_benchmarks.py --run-dir log/{target.name} --results-dir log/{target.name}-comparison/results --suite mac-compare --stage performance --repetitions 3 --keep-going
```

Each suite runs configurations sequentially. `--resume` skips completed results
after an interruption. `--keep-going` retains failures, tries the remaining
configurations, and exits nonzero if any failed. Existing logs are never overwritten.
Keep failed results; do not loosen a gate to obtain a performance comparison.
Use a fresh results directory for repeated runs. Return the comparison `results/` (JSON and logs),
plus Mac chip, GPU cores, memory, macOS, power mode, and thermal conditions.

The primary suite compares HF and Axon Torch CPU FP32, HF and Axon Torch MPS
FP32/FP16, and Axon MLX Metal FP32/FP16 with and without compilation. MPS CPU
fallback is disabled; missing support fails explicitly. Three performance
repetitions use fresh shuffled processes, each with 10 warmups and 50 samples per case.
MLX uses the generated encoder directly. Optional `--suite coreml` tests
three compute-unit settings and static/dynamic inputs. Provider placement is
recorded; CPU-only fallback cannot produce a successful accelerator timing row.
MPS, MLX and Core ML have not been executed on the Linux source machine.

The frozen checkpoint SHA-256 is `{frozen["checkpoint_sha256"]}`.
`bundle.json` contains file checksums. `export/export.json` also describes Linux
CUDA graphs that are intentionally omitted from this Mac bundle.
Linux results, when present, are named `linux-public-*`; Mac outputs start fresh.

## Sources and attribution

Weights were fine-tuned from [nreimers/MiniLM-L6-H384-uncased](https://huggingface.co/nreimers/MiniLM-L6-H384-uncased),
whose model card identifies an MIT license. Original MiniLM work:
[Microsoft UniLM / MiniLM](https://github.com/microsoft/unilm/tree/master/minilm).
The modified weights and training recipe are for this reproducible runtime experiment.

The included Few-NERD test data is from [Ding et al., Few-NERD](https://github.com/thunlp/Few-NERD),
distributed through [DFKI-SLT/few-nerd](https://huggingface.co/datasets/DFKI-SLT/few-nerd)
under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
`raw/` preserves the supplied word annotations; `encoded/` adds tokenizer-derived
subtokens, word alignment, and inference windows. Revisions and changes are in
`manifest.json`. Test data was not used for training or checkpoint selection.
"""
    )
    manifest = {
        "source_commit": revision,
        "checkpoint_sha256": frozen["checkpoint_sha256"],
        "included_splits": ["test"],
        "files": {
            str(path.relative_to(target)): {"bytes": path.stat().st_size, "sha256": digest(path)}
            for path in sorted(target.rglob("*"))
            if path.is_file()
        },
    }
    (target / "bundle.json").write_text(json.dumps(manifest, indent=2) + "\n")
    with tarfile.open(archive, "w:gz", compresslevel=3) as stream:
        stream.add(target, arcname=target.name)
    checksum = digest(archive)
    archive.with_name(archive.name + ".sha256").write_text(f"{checksum}  {archive.name}\n")
    print(
        json.dumps({"archive": str(archive), "bytes": archive.stat().st_size, "sha256": checksum})
    )


if __name__ == "__main__":
    main()
