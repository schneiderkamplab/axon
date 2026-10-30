"""Command line: ``python -m ettin_mlx fill-mask ...`` and ``python -m ettin_mlx generate ...``."""

from __future__ import annotations

import argparse
import sys
import time

import mlx.core as mx

from .generate import generate_step
from .loading import load


def fill_mask(args) -> None:
    model, tok = load(args.model, revision=args.revision, dtype=args.dtype)
    if model.config.is_decoder:
        sys.exit(f"{args.model} is a decoder; fill-mask needs an encoder")
    ids = tok.encode(args.text)
    positions = [i for i, t in enumerate(ids) if t == tok.mask_token_id]
    if not positions:
        sys.exit(f"no {tok.mask_token} token in the input")
    hidden = model.encode(mx.array([ids]))[0, mx.array(positions)]
    probs = mx.softmax(model.logits(hidden).astype(mx.float32), axis=-1)
    for pos, p in zip(positions, probs):
        top = mx.argsort(-p)[: args.top_k].tolist()
        print(f"{tok.mask_token} at token {pos}:")
        for t in top:
            print(f"  {p[t].item():.4f}  {tok.decode([t])!r}")


def generate(args) -> None:
    model, tok = load(args.model, revision=args.revision, dtype=args.dtype)
    if not model.config.is_decoder:
        sys.exit(f"{args.model} is an encoder; generate needs a decoder")
    prompt = tok.encode(args.prompt) if args.keep_eos else tok.encode_prompt(args.prompt)
    eos = tok.eos_token_id
    start = time.perf_counter()
    first = None
    out: list[int] = []
    printed = 0
    sys.stdout.write(args.prompt)
    for token in generate_step(
        model, mx.array([prompt]), args.max_new_tokens,
        temperature=args.temperature, top_p=args.top_p, seed=args.seed,
    ):
        t = token.item()
        first = first or time.perf_counter()
        if t == eos and not args.ignore_eos:
            break
        out.append(t)
        text = tok.decode(out)
        sys.stdout.write(text[printed:])
        sys.stdout.flush()
        printed = len(text)
    end = time.perf_counter()
    print(flush=True)
    decode_tps = (len(out) - 1) / (end - first) if len(out) > 1 else float("nan")
    print(f"[{len(prompt)} prompt tokens, {len(out)} new tokens; prefill+first token "
          f"{(first - start) * 1e3:.0f} ms, decode {decode_tps:.1f} tok/s]", file=sys.stderr)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m ettin_mlx", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument("--model", required=True, help="short name (ettin-encoder-150m), repo id or local dir")
        p.add_argument("--revision", help="override the pinned revision")
        p.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])

    p = sub.add_parser("fill-mask", help="top-k predictions for each mask token")
    common(p)
    p.add_argument("text", help='text containing the mask token, e.g. "The capital of France is [MASK]."')
    p.add_argument("--top-k", type=int, default=5)
    p.set_defaults(func=fill_mask)

    p = sub.add_parser("generate", help="continue a prompt with a decoder")
    common(p)
    p.add_argument("--prompt", required=True)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.0, help="0 = greedy")
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--seed", type=int)
    p.add_argument("--keep-eos", action="store_true",
                   help="keep the [SEP] that the tokenizer appends to the prompt (transformers 5 behaviour)")
    p.add_argument("--ignore-eos", action="store_true", help="do not stop at the EOS token")
    p.set_defaults(func=generate)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
