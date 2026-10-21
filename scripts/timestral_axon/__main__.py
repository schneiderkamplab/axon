"""Export a runtime or forecast NumPy arrays: python -m scripts.timestral_axon."""

import argparse
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser(
        "export", help="generate standalone MLX package (no Apple GPU needed)"
    )
    export.add_argument("--output", type=Path, required=True)
    export.add_argument(
        "--optimize", action="store_true", help="opt in to Axon SDPA and RMSNorm-scale graph fusion"
    )
    forecast = commands.add_parser("forecast", help="run on an MLX-capable machine")
    forecast.add_argument("checkpoint")
    forecast.add_argument(
        "--context", type=Path, required=True, help=".npy array: (T,), (B,T), or (B,V,T)"
    )
    forecast.add_argument("--horizon", type=int, required=True)
    forecast.add_argument("--output", type=Path, required=True, help="new .npz file")
    forecast.add_argument("--revision")
    forecast.add_argument("--local-files-only", action="store_true")
    forecast.add_argument("--max-context", type=int)
    forecast.add_argument("--max-batch-rows", type=int, default=128)
    forecast.add_argument("--max-batch-tokens", type=int, default=4096)
    forecast.add_argument(
        "--prune-final-block", action="store_true",
        help="opt in to final-block pruning, which may change numerical results",
    )
    for name in (
        "observation-mask",
        "past-covariates",
        "known-covariates",
        "categorical-past-covariates",
        "categorical-known-covariates",
    ):
        forecast.add_argument(f"--{name}", type=Path)
    forecast.add_argument("--no-symmetric-averaging", action="store_true")
    forecast.add_argument("--no-clip-nonnegative", action="store_true")
    args = parser.parse_args()
    if args.command == "export":
        from .build import export_package

        print(export_package(args.output, optimize=args.optimize))
        return
    if args.output.suffix != ".npz" or args.output.exists():
        parser.error("--output must be a new .npz file")
    from .forecaster import TimestralForecaster

    model = TimestralForecaster.from_pretrained(
        args.checkpoint,
        revision=args.revision,
        local_files_only=args.local_files_only,
        max_context=args.max_context,
        max_batch_rows=args.max_batch_rows,
        max_batch_tokens=args.max_batch_tokens,
        prune_final_block=args.prune_final_block,
    )
    kwargs = {}
    for name in (
        "observation_mask",
        "past_covariates",
        "known_covariates",
        "categorical_past_covariates",
        "categorical_known_covariates",
    ):
        path = getattr(args, name)
        if path is not None:
            kwargs[name] = np.load(path, allow_pickle=False)
    result = model.forecast(
        np.load(args.context, allow_pickle=False),
        args.horizon,
        symmetric_averaging=not args.no_symmetric_averaging,
        clip_nonnegative=not args.no_clip_nonnegative,
        **kwargs,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        quantiles=result.quantiles,
        median=result.median,
        quantile_levels=result.quantile_levels,
    )
    print(f"Wrote {result.quantiles.shape} quantiles to {args.output}")


if __name__ == "__main__":
    main()
