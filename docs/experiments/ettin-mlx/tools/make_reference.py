"""Reference outputs from HF transformers (FP32, CPU, eager attention) -> tests/data/<model>.npz.

Usage:
    uv run python tools/make_reference.py                       # default test models
    uv run python tools/make_reference.py --models ettin-encoder-150m ettin-decoder-1b

Full logits for a 2048-token case are ~400 MB, so per case we store

* ``hidden``  - ``model.final_norm`` output (last hidden state),
* ``head``    - prediction-head output, i.e. the input of the vocab projection,
* ``top1``/``margin`` - argmax of HF's full logits and the top-1/top-2 gap,
* ``sample_logits`` at ``sample_pos`` - HF logits verbatim for a few positions,
* ``head64`` - the head output of the same HF model run in float64 (ground truth for
  the fp32 weights), and ``meta.fp32_vs_fp64[case]`` = max |HF fp32 - HF fp64| logits,
  i.e. how far HF's own fp32 result is from exact. Some inputs carry massive
  activations (|h| ~ 5e3) and are ill-conditioned enough that this exceeds 1e-3.
* ``meta.fp32_spread[case]`` - the worst max |HF fp32 - HF fp64| over HF's fp32
  variants: eager, sdpa, and ``NOISE_RUNS`` eager runs in which the embedding and every
  layer output get +-1 ulp of random relative noise. This is how far fp32
  implementations with HF's per-op accuracy but different rounding land from exact.

Tests rebuild reference logits as ``head @ W.T + b`` from the checkpoint tensors; the
tool checks that this reconstruction matches HF's own logits (``meta.reconstruction``).
This file uses torch/transformers and is never imported by the runtime package.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# transformers 5 starts a background "auto_conversion" thread after loading a .bin: it asks
# the Hub for (and may trigger the creation of) a safetensors PR and downloads it. We want
# the pinned .bin only and no side effects on the Hub.
os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")

import numpy as np
import torch
import transformers
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer

from ettin_mlx.revisions import HF_ORG, PINNED_REVISIONS

DATA_DIR = Path(__file__).resolve().parent.parent / "tests" / "data"
DEFAULT_MODELS = ["ettin-encoder-17m", "ettin-decoder-17m", "ettin-encoder-32m", "ettin-decoder-32m"]

# (name, row lengths, corpus offsets). "padded" is a right-padded batch of mixed lengths.
CASES = [
    ("short", [8], [0]),
    ("medium", [300], [100]),
    ("long", [2048], [500]),
    ("padded", [300, 129, 64, 7], [3000, 3400, 3800, 4200]),
]
GEN_PROMPTS = ["The capital of France is", "def fibonacci(n):"]
GEN_LONG_PROMPT = (4600, 150)  # corpus offset, length in tokens
GEN_NEW_TOKENS = 64
FILL_MASK_TEXT = "The capital of France is {mask}."
MASK_RATE = 0.15
NOISE_RUNS = 3


def corpus_ids(tokenizer) -> list[int]:
    """Natural text: the Ettin model card at the pinned encoder-17m revision."""
    path = hf_hub_download(f"{HF_ORG}/ettin-encoder-17m", "README.md", revision=PINNED_REVISIONS["ettin-encoder-17m"])
    return tokenizer(Path(path).read_text(), add_special_tokens=False)["input_ids"]


def build_case(name, lengths, offsets, corpus, tok, is_decoder, rng):
    rows = []
    for n, off in zip(lengths, offsets):
        row = [tok.cls_token_id] + corpus[off : off + n - 2] + [tok.sep_token_id]
        assert len(row) == n, (name, n, len(row))
        if not is_decoder:  # MLM-style inputs: mask ~15% of the non-special tokens
            idx = rng.choice(np.arange(1, n - 1), size=max(1, int(MASK_RATE * (n - 2))), replace=False)
            for i in idx:
                row[i] = tok.mask_token_id
        rows.append(row)
    width = max(lengths)
    ids = np.full((len(rows), width), tok.pad_token_id, dtype=np.int64)
    mask = np.zeros((len(rows), width), dtype=np.int64)
    for i, row in enumerate(rows):
        ids[i, : len(row)] = row
        mask[i, : len(row)] = 1
    return ids, mask


def sample_positions(mask: np.ndarray, rng, window: int) -> np.ndarray:
    """A few (row, pos) pairs: sequence ends, window edges, and random positions."""
    picks = set()
    for r, n in enumerate(mask.sum(1)):
        for p in [0, 1, n - 1, n - 2, window, window + 1, 2 * window, 2 * window + 1, n // 2]:
            if 0 <= p < n:
                picks.add((r, int(p)))
        for p in rng.choice(n, size=min(4, n), replace=False):
            picks.add((r, int(p)))
    return np.array(sorted(picks), dtype=np.int32)


def add_ulp_noise(model, seed: int) -> list:
    """Hooks multiplying the embedding and every layer output by (1 + u * 2**-23), u ~ U(-1, 1)."""
    gen = torch.Generator().manual_seed(seed)

    def hook(module, args, output):
        return output * (1 + (torch.rand(output.shape, generator=gen, dtype=torch.float64) * 2 - 1) * 2**-23).float()

    return [m.register_forward_hook(hook) for m in [model.model.embeddings, *model.model.layers]]


@torch.no_grad()
def make_reference(short_name: str, out_dir: Path = DATA_DIR) -> Path:
    repo, revision = f"{HF_ORG}/{short_name}", PINNED_REVISIONS[short_name]
    tok = AutoTokenizer.from_pretrained(repo, revision=revision)
    config = json.loads(Path(hf_hub_download(repo, "config.json", revision=revision)).read_text())
    is_decoder = config["model_type"] == "modernbert-decoder"
    cls = AutoModelForCausalLM if is_decoder else AutoModelForMaskedLM
    model = cls.from_pretrained(repo, revision=revision, attn_implementation="eager", dtype=torch.float32).eval()
    head = model.lm_head if is_decoder else model.head
    weight = model.decoder.weight.double()
    bias = model.decoder.bias.double() if model.decoder.bias is not None else 0.0
    window = model.config.sliding_window

    rng = np.random.default_rng(0)
    corpus = corpus_ids(tok)
    out: dict[str, np.ndarray] = {}
    fp32_logits: dict[str, list[torch.Tensor]] = {}
    recon_err = 0.0
    for name, lengths, offsets in CASES:
        ids, mask = build_case(name, lengths, offsets, corpus, tok, is_decoder, rng)
        t_ids, t_mask = torch.from_numpy(ids), torch.from_numpy(mask)
        hidden = model.model(input_ids=t_ids, attention_mask=t_mask).last_hidden_state
        head_out = head(hidden)
        logits = model.decoder(head_out)
        full = model(input_ids=t_ids, attention_mask=t_mask).logits
        assert torch.equal(full, logits), "manual head path differs from model.forward"
        recon = head_out.double() @ weight.T + bias
        recon_err = max(recon_err, (recon - logits.double()).abs().max().item())
        top2 = logits.topk(2, dim=-1).values
        pos = sample_positions(mask, rng, window)
        fp32_logits[name] = [logits]
        for seed in range(NOISE_RUNS):
            handles = add_ulp_noise(model, seed)
            fp32_logits[name].append(model(input_ids=t_ids, attention_mask=t_mask).logits)
            for h in handles:
                h.remove()
        model.set_attn_implementation("sdpa")
        fp32_logits[name].append(model(input_ids=t_ids, attention_mask=t_mask).logits)
        model.set_attn_implementation("eager")
        out |= {
            f"{name}/input_ids": ids.astype(np.int32),
            f"{name}/attention_mask": mask.astype(np.int32),
            f"{name}/hidden": hidden.numpy(),
            f"{name}/head": head_out.numpy(),
            f"{name}/top1": logits.argmax(-1).numpy().astype(np.int32),
            f"{name}/margin": (top2[..., 0] - top2[..., 1]).numpy(),
            f"{name}/sample_pos": pos,
            f"{name}/sample_logits": logits[pos[:, 0], pos[:, 1]].numpy(),
        }

    if is_decoder:
        model.generation_config.eos_token_id = None  # always produce GEN_NEW_TOKENS tokens
        prompts = [tok(p)["input_ids"][:-1] for p in GEN_PROMPTS]  # [CLS] text, no trailing [SEP]
        off, n = GEN_LONG_PROMPT
        prompts.append([tok.cls_token_id] + corpus[off : off + n - 1])
        for i, prompt in enumerate(prompts):
            gen = model.generate(
                torch.tensor([prompt]), attention_mask=torch.ones(1, len(prompt), dtype=torch.long),
                max_new_tokens=GEN_NEW_TOKENS, do_sample=False, pad_token_id=tok.pad_token_id,
            )
            out[f"gen/{i}/prompt"] = np.array(prompt, dtype=np.int32)
            out[f"gen/{i}/tokens"] = gen[0, len(prompt) :].numpy().astype(np.int32)
    else:
        ids = tok(FILL_MASK_TEXT.format(mask=tok.mask_token), return_tensors="pt")["input_ids"]
        logits = model(input_ids=ids).logits[0, (ids[0] == tok.mask_token_id).nonzero()[0, 0]]
        out["fill_mask/input_ids"] = ids.numpy().astype(np.int32)
        out["fill_mask/top5"] = logits.topk(5).indices.numpy().astype(np.int32)

    assert recon_err < 1e-4, f"head @ W.T + b does not reproduce HF logits (max abs {recon_err})"

    # Same model in float64: exact values for the fp32 weights. Rows run one at a time
    # without padding: HF's eager path casts the fp64 mask value finfo.min to -inf in its
    # fp32 softmax, so fully masked pad rows become NaN and leak into real tokens.
    del model, head
    model = cls.from_pretrained(repo, revision=revision, attn_implementation="eager", dtype=torch.float64).eval()
    head = model.lm_head if is_decoder else model.head
    fp32_vs_fp64, fp32_spread = {}, {}
    for name, *_ in CASES:
        ids, mask = out[f"{name}/input_ids"], out[f"{name}/attention_mask"]
        head64 = torch.zeros(*ids.shape, model.config.hidden_size, dtype=torch.float64)
        for r, n in enumerate(mask.sum(1)):
            row = torch.from_numpy(ids[r : r + 1, :n].astype(np.int64))
            head64[r, :n] = head(model.model(input_ids=row).last_hidden_state)[0]
        valid = torch.from_numpy(mask).bool()
        logits64 = model.decoder(head64)
        errors = [(logits64 - l.double()).abs().amax(-1)[valid].max().item() for l in fp32_logits.pop(name)]
        fp32_vs_fp64[name], fp32_spread[name] = errors[0], max(errors)
        out[f"{name}/head64"] = head64.numpy()

    meta = {
        "model": short_name, "repo": repo, "revision": revision, "is_decoder": is_decoder,
        "transformers": transformers.__version__, "torch": torch.__version__,
        "attn_implementation": model.config._attn_implementation, "dtype": "float32", "device": "cpu",
        "reconstruction_max_abs": recon_err, "fp32_vs_fp64": fp32_vs_fp64, "fp32_spread": fp32_spread,
    }
    out["meta"] = np.array(json.dumps(meta))
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{short_name}.npz"
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, **out)
    tmp.replace(path)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, choices=sorted(PINNED_REVISIONS))
    parser.add_argument("--out-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args()
    torch.set_num_threads(max(1, torch.get_num_threads()))
    for name in args.models:
        path = make_reference(name, args.out_dir)
        meta = json.loads(str(np.load(path)["meta"]))
        floors = ", ".join(f"{k} {v:.1e}/{meta['fp32_spread'][k]:.1e}" for k, v in meta["fp32_vs_fp64"].items())
        print(f"{name}: wrote {path} ({path.stat().st_size / 1e6:.1f} MB), reconstruction max abs "
              f"{meta['reconstruction_max_abs']:.1e}, HF fp32 vs fp64 (eager/worst variant): {floors}")


if __name__ == "__main__":
    main()
