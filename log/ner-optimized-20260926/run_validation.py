"""Sequential fresh-process checks of the production exports; no MLX execution."""
import argparse
import os
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
ENV = dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
           CUDA_VISIBLE_DEVICES="0", TOKENIZERS_PARALLELISM="false")
CONFIGS = [(backend, device, dtype)
           for device, dtype in (("cpu", "fp32"), ("cuda:0", "fp32"), ("cuda:0", "fp16"))
           for backend in ("hf", "axon")]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--results-dir', type=Path, default=ROOT / 'results')
parser.add_argument('--skip-cpu-quality', action='store_true')
args = parser.parse_args()

def run(stage, config, suffix=""):
    backend, device, dtype = config
    name = f"{backend}-{device.split(':')[0]}-{dtype}-{stage}{suffix}"
    output = args.results_dir / f"{name}.json"
    log = output.with_suffix(".log")
    if output.exists() or log.exists():
        raise FileExistsError(output)
    output.parent.mkdir(exist_ok=True)
    command = [sys.executable, str(REPO / "scripts/bench_ner.py"),
               "--run-dir", str(ROOT), "--backend", backend, "--device", device,
               "--dtype", dtype, "--stage", stage, "--rounds", "1", "--output", str(output)]
    print(f"Starting {name}", flush=True)
    with log.open("w") as stream:
        subprocess.run(command, env=ENV, stdout=stream, stderr=subprocess.STDOUT, check=True)
    print(f"Completed {name}", flush=True)

if __name__ == "__main__":
    # The frozen HF CPU reference already passed the full test set.
    for config in CONFIGS:
        if config != ("hf", "cpu", "fp32") and not (args.skip_cpu_quality and config[1] == 'cpu'):
            run("quality", config)
    for repetition in range(3):
        configs = list(CONFIGS)
        random.Random(20260926 + repetition).shuffle(configs)
        for config in configs:
            run("performance", config, f"-r{repetition}")
