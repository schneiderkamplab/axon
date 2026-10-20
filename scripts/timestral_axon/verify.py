"""Create upstream forecast fixtures, or verify them with Axon Torch/MLX.

All fixture inputs are synthetic. Reference generation needs the pinned upstream
source and PyTorch. MLX checking uses only the exported inference dependencies.
"""

import argparse
import hashlib
import importlib.metadata
import json
import shutil
import sys
from pathlib import Path

import numpy as np

UPSTREAM_REVISION = "a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff"
UPSTREAM_HASHES = {
    "__init__.py": "46777bcb15b903f7721a3d6d91a6cb93463fc454816da238799eb8d5360e5ab5",
    "config.py": "d46e37020875c536e7a75e1b6ace3f67419ba2f2c9d9e8465f4b07f837e92ed6",
    "covariate_encoding.py": "1c7f5c320fa4b7622d3705f765f53ab9676b829cb1d4fd68f839ec593b8404c0",
    "forecaster.py": "12a712f9be777d1e160513b7f8c81d545b991fcfe6cd4a46004a03533b7cb8fd",
    "layers.py": "480ba93e7e3681dea185e5fa08a96dcce327cf938931c715ae54e6b02934527d",
    "loading.py": "179257fa74e798c207887e2fed162e9a2fef9c37cadadb9437fabfeae68b3d6f",
    "masks.py": "1792898d809b99034bb70e2cc1ab74f897ce8478886f83140e705fc67e6762b3",
    "model.py": "988121c6a089e843c4941732cbc5c64b10cb0eccd9310e228abe38ecf2dfb668",
    "scaler.py": "838da836f70b1640bcaed7701a70a1a358e30442693e63522a93f878e157a8ca",
}


def cases(covariates=True, long_horizon=70):
    rng = np.random.default_rng(71)
    specs = [
        ("univariate", 1, 1, 39, 12, {}),
        ("multivariate_missing", 2, 3, 37, 70, {}),
        ("positive", 1, 1, 41, 65, {"positive": True}),
        ("all_missing", 1, 1, 31, 20, {"blank": True}),
        ("constant", 2, 1, 31, 20, {"constant": True}),
        ("large_offset", 1, 1, 41, 40, {"offset": True}),
        ("single_step", 1, 1, 1, 1, {}),
        ("multi_block", 1, 1, 71, long_horizon, {"raw": True}),
        ("context_cap_mask", 2, 1, 137, 40, {"mask": True, "max_context": 32}),
    ]
    if covariates:
        specs += [
            ("past_known", 2, 1, 41, 40, {"cov": True}),
            ("past_only", 2, 1, 41, 40, {"past": True}),
            ("categorical", 2, 1, 41, 40, {"cat": True}),
            ("raw_categories", 1, 2, 41, 40, {"cat": True}),
            ("covariate_multi_block", 1, 1, 71, long_horizon, {"cov": True, "raw": True}),
        ]
    for name, batch, variates, length, horizon, options in specs:
        x = rng.standard_normal((batch, variates, length)).astype(np.float32).cumsum(-1)
        if options.get("positive"):
            x = np.abs(x) + 1
        if options.get("constant"):
            x.fill(2)
        if options.get("offset"):
            x += 1e5
        if length > 1:
            x[..., :3] = np.nan
        if length > 12:
            x[..., 7:11] = np.nan
        if options.get("blank"):
            x.fill(np.nan)
        arrays = {"context": x}
        if options.get("cov") or options.get("past"):
            arrays["past_covariates"] = rng.standard_normal((batch, 2, length)).astype(np.float32)
            arrays["past_covariates"][..., 1:5] = np.nan
        if options.get("cov"):
            known = rng.standard_normal((batch, 2, length + horizon)).astype(np.float32)
            known[:, 1] = rng.integers(0, 2, size=(batch, length + horizon))
            known[..., 6:9] = np.nan
            arrays["known_covariates"] = known
        if options.get("cat"):
            arrays["categorical_past_covariates"] = (
                rng.integers(0, 4, size=(batch, 1, length)).astype(np.float64) + 2**25
            )
            codes = rng.integers(0, 6, size=(batch, 1, length + horizon)).astype(np.float64)
            codes[..., 3:7] = np.nan
            arrays["categorical_known_covariates"] = codes
        if options.get("mask"):
            arrays["observation_mask"] = rng.random(x.shape) > 0.3
        kwargs = {"prediction_length": horizon}
        if options.get("raw"):
            kwargs.update(symmetric_averaging=False, clip_nonnegative=False)
        constructor = {"max_context": options["max_context"]} if "max_context" in options else {}
        yield name, arrays, kwargs, constructor


def write_reference(output, upstream, checkpoint=None):
    import torch
    from safetensors.torch import save_file

    upstream = Path(upstream).resolve()
    for name, expected in UPSTREAM_HASHES.items():
        actual = hashlib.sha256((upstream / "timestral" / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"{name} does not match upstream {UPSTREAM_REVISION}")
    sys.path.insert(0, str(upstream))
    from timestral import TimestralConfig, TimestralForecaster, TimestralModel
    from timestral.loading import load_model

    if output.exists():
        raise FileExistsError(f"choose a new fixture directory: {output}")
    output.mkdir(parents=True)
    torch.set_num_threads(2)
    base = dict(
        patch_size=8,
        d_model=32,
        num_heads=4,
        mlp_hidden_dim=64,
        block_schedule=("time", "time_variate", "variate", "variate_time"),
        quantile_knots=(0.01, 0.1, 0.5, 0.9, 0.99),
        block_size=4,
        context_length=64,
    )
    models = []
    for name, changes in [
        ("tiny", {}),
        (
            "tiny_bias_no_gates",
            dict(
                attn_bias=True,
                mlp_bias=True,
                time_attn_gate="none",
                variate_attn_gate="none",
                use_asinh=False,
                attend_missing_patches=False,
                split_context_horizon_attn=False,
            ),
        ),
        ("tiny_no_covariates", dict(covariate_enabled=False)),
    ]:
        torch.manual_seed(17)
        model = TimestralModel(TimestralConfig(**(base | changes))).eval()
        for parameter in model.parameters():
            torch.nn.init.normal_(parameter, std=0.05)
        models.append((name, model))
    if checkpoint is not None:
        models.append(("checkpoint", load_model(checkpoint)))
    manifest = {
        "upstream_revision": UPSTREAM_REVISION,
        "torch": torch.__version__,
        "source_hashes": UPSTREAM_HASHES,
        "cases": [],
    }
    with torch.inference_mode():
        for name, model in models:
            directory = output / name
            directory.mkdir()
            model.config.save(directory / "config.json")
            if name == "checkpoint":
                shutil.copy2(
                    Path(checkpoint) / "model.safetensors", directory / "model.safetensors"
                )
            else:
                save_file(model.state_dict(), directory / "model.safetensors")
            long_horizon = model.config.block_size * model.config.patch_size + 17
            for case, arrays, kwargs, constructor in cases(
                model.config.covariate_enabled, long_horizon
            ):
                forecaster = TimestralForecaster(model, **constructor)
                result = forecaster.forecast(**arrays, **kwargs).quantiles.numpy()
                filename = f"{name}/{case}.npz"
                np.savez(output / filename, expected=result, **arrays)
                manifest["cases"].append(
                    {
                        "model": name,
                        "case": case,
                        "file": filename,
                        "kwargs": kwargs,
                        "constructor": constructor,
                    }
                )
                print(f"reference {name}/{case}", flush=True)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def check(fixtures, backend, output, device=None, optimize=False):
    from .config import TimestralConfig
    from .forecaster import TimestralForecaster
    from .model import TimestralModel

    manifest = json.loads((fixtures / "manifest.json").read_text())
    model_cache = {}
    namespace = {}
    if backend == "torch":
        import torch
        from safetensors.torch import load_file

        from .build import generate_core

        torch.set_num_threads(2)
        exec(
            compile(
                generate_core(backend="torch", optimize=optimize), "<axon-timestral-torch>", "exec"
            ),
            namespace,
        )
    else:
        import mlx.core as mx

        if device is not None:
            mx.set_default_device(mx.cpu if device == "cpu" else mx.gpu)
    records = []
    for entry in manifest["cases"]:
        name = entry["model"]
        if name not in model_cache:
            path = fixtures / name
            if backend == "mlx":
                from .loading import load_model

                model_cache[name] = load_model(path)
            else:
                config = TimestralConfig.load(path / "config.json")
                generated = namespace["TimestralCore"].from_state_dict(
                    load_file(path / "model.safetensors"), model_config=config.to_dict()
                )

                def core(_generated=generated, **inputs):
                    with torch.inference_mode():
                        result = _generated.forward(
                            **{k: torch.from_numpy(v.copy()) for k, v in inputs.items()}
                        )
                    return result.numpy()

                model_cache[name] = TimestralModel(config, core)
        with np.load(fixtures / entry["file"], allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files if key != "expected"}
            expected = data["expected"]
        forecaster = TimestralForecaster(model_cache[name], **entry["constructor"])
        actual = forecaster.forecast(**arrays, **entry["kwargs"]).quantiles
        difference = np.abs(actual - expected)
        tolerance = 2e-4 + 2e-4 * np.abs(expected)
        passed = (
            actual.shape == expected.shape
            and np.isfinite(actual).all()
            and (np.diff(actual, axis=-1) >= 0).all()
            and np.all(difference <= tolerance)
        )
        record = dict(
            model=name,
            case=entry["case"],
            passed=bool(passed),
            max_abs=float(difference.max()),
            mean_abs=float(difference.mean()),
            max_tolerance_fraction=float((difference / tolerance).max()),
        )
        records.append(record)
        print(json.dumps(record), flush=True)
    build_info = Path(__file__).with_name("build-info.json")
    report = dict(
        backend=backend,
        device=device,
        optimize=optimize,
        upstream_revision=manifest["upstream_revision"],
        numpy=np.__version__,
        runtime=importlib.metadata.version(backend),
        cases=records,
        torch_imported="torch" in sys.modules,
        exported_build=json.loads(build_info.read_text()) if build_info.exists() else None,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    if not all(record["passed"] for record in records):
        raise SystemExit("forecast parity failed; see report")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    reference = commands.add_parser("reference")
    reference.add_argument("--upstream-src", type=Path, required=True)
    reference.add_argument("--checkpoint", type=Path)
    reference.add_argument("--output", type=Path, required=True)
    verify = commands.add_parser("check")
    verify.add_argument("--fixtures", type=Path, required=True)
    verify.add_argument("--backend", choices=("torch", "mlx"), default="mlx")
    verify.add_argument("--device", choices=("cpu", "gpu"))
    verify.add_argument(
        "--optimize", action="store_true", help="Torch check: use existing SDPA graph fusion"
    )
    verify.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "reference":
        write_reference(args.output, args.upstream_src, args.checkpoint)
    else:
        if args.optimize and args.backend != "torch":
            parser.error("for MLX, select optimization when exporting the package")
        check(args.fixtures, args.backend, args.output, args.device, args.optimize)


if __name__ == "__main__":
    main()
