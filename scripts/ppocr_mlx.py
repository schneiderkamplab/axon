"""PP-OCR through Axon on MLX: build, check against ONNX Runtime, time, and run OCR.

Experiment harness for ``synapse/models/ppocr``. Run from the repository root.

    python scripts/ppocr_mlx.py build  --run log/ppocr-mlx [--onnx-dir DIR | --repo ID]
    python scripts/ppocr_mlx.py parity --run log/ppocr-mlx
    python scripts/ppocr_mlx.py bench  --run log/ppocr-mlx [--providers CPUExecutionProvider,...]
    python scripts/ppocr_mlx.py ocr    --run log/ppocr-mlx --image page.png [--backend mlx|onnx]

    python scripts/ppocr_mlx.py tune   --run log/ppocr-mlx

``parity`` and ``bench`` cover the MLX variants listed in ``MlxBackend``. ``tune`` times
them on this machine and saves the fastest per network; ``ocr`` then uses that choice.

``build`` needs ``onnx``; ``parity`` and the ONNX rows of ``bench`` need ``onnxruntime``;
``ocr`` needs ``opencv-python-headless``. Image pre- and post-processing follow PaddleOCR's
defaults and stay in NumPy/OpenCV; only the two networks run on MLX.

``bench`` times what the pipeline consumes: the detector's probability map as a NumPy
array, and the recognizer's best class and probability per column (taken on the device
for MLX, from the full NumPy output for ONNX Runtime).
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = REPO_ROOT / "synapse" / "models" / "ppocr"
ONNX_FILES = {"det": "pp-ocrv6_small_det.onnx", "rec": "pp-ocrv6_small_rec.onnx"}
DICT_FILE = "ppocrv6_dict.txt"
DEFAULT_REPO = "blackpen-sk/ppocrv6-small"

DET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
DET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
REC_HEIGHT = 48
REC_BASE_WIDTH = 320


# --------------------------------------------------------------------------- build


def _fetch_onnx(repo: str, target: Path) -> None:
    from huggingface_hub import hf_hub_download

    target.mkdir(parents=True, exist_ok=True)
    for name in (*ONNX_FILES.values(), DICT_FILE):
        shutil.copyfile(hf_hub_download(repo, name), target / name)


def _emit_mlx(axon_path: Path, class_name: str) -> str:
    from synapse.axon import (
        GraphOptimizeConfig,
        elaborate_closed_axon_file,
        flatten_closed_axon_file,
        lower_axon_program_to_graph_ir,
        normalize_closed_axon_file,
        optimize_graph_program,
        resolve_axon_program_from_path,
        typecheck2_flat_axon_file,
    )
    from synapse.axon.codegen2_mlx import emit_model_code_from_graph_ir

    program = resolve_axon_program_from_path(axon_path, strict=False).ast
    program = normalize_closed_axon_file(program)
    program = elaborate_closed_axon_file(program)
    program = flatten_closed_axon_file(program)
    program = typecheck2_flat_axon_file(program)
    graph = lower_axon_program_to_graph_ir(program)
    graph = optimize_graph_program(
        graph, config=GraphOptimizeConfig(backend_intrinsics="codegen2-mlx")
    )
    return emit_model_code_from_graph_ir(graph, class_name=class_name)


def cmd_build(args: argparse.Namespace) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from convert_ppocr_onnx import convert_det, convert_rec, write_checkpoint

    run = args.run
    onnx_dir = args.onnx_dir
    if onnx_dir is None:
        onnx_dir = run / "onnx"
        _fetch_onnx(args.repo, onnx_dir)
    write_checkpoint(run / "det", *convert_det(onnx_dir / ONNX_FILES["det"]))
    write_checkpoint(run / "rec", *convert_rec(onnx_dir / ONNX_FILES["rec"]))
    for kind in ("det", "rec"):
        code = _emit_mlx(MODEL_DIR / f"generic-ppocr-{kind}.axon", f"PpocrMlx{kind.title()}")
        (run / f"ppocr_{kind}_mlx.py").write_text(code)
        print(f"{run / f'ppocr_{kind}_mlx.py'}: {len(code.splitlines())} lines")
    (run / "build.json").write_text(json.dumps({"onnx_dir": str(onnx_dir.resolve())}, indent=2))


# ------------------------------------------------------------------------ backends


def _onnx_dir(run: Path) -> Path:
    """ONNX files next to the run (a published kit) or where ``build`` read them."""
    if (run / "onnx" / ONNX_FILES["det"]).exists():
        return run / "onnx"
    return Path(json.loads((run / "build.json").read_text())["onnx_dir"])


def _dictionary(run: Path) -> Path:
    local = run / DICT_FILE
    return local if local.exists() else _onnx_dir(run) / DICT_FILE


def _use_vendored_common(run: Path) -> None:
    """Let the generated modules import their helper module without the synapse package."""
    import importlib.util
    import types

    name = "synapse.axon.codegen2_common"
    vendored = run / "axon_codegen2_common.py"
    if name in sys.modules or not vendored.exists():
        return
    try:
        if importlib.util.find_spec(name) is not None:
            return
    except (ImportError, ValueError):
        pass
    for package in ("synapse", "synapse.axon"):
        sys.modules.setdefault(package, types.ModuleType(package)).__path__ = []
    spec = importlib.util.spec_from_file_location(name, vendored)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)


VARIANTS = ("base", "compiled", "shift", "shift+compiled", "shift49+compiled")
TUNED_FILE = "tuned.json"


class MlxBackend:
    """The Axon-generated MLX modules. Inputs and outputs are channels-last NumPy arrays.

    Variants, combined with ``+``:
      compiled  trace each network once per input shape with ``mx.compile`` (normalization
                and the recognizer's argmax included), instead of rebuilding the graph in
                Python on every call
      shift     compute 3x3 depthwise convolutions as a sum of nine shifted, per-channel
                scaled copies of the input instead of a grouped convolution
      shift49   the same for every depthwise convolution up to 7x7 (the detector neck)

    ``variant`` is one name for both networks, ``auto`` (the choice ``tune`` saved, else
    ``compiled``), or a mapping from ``det``/``rec`` to a name.
    """

    def __init__(self, run: Path, dtype: str = "float32", variant: Any = "base") -> None:
        import mlx.core as mx

        _use_vendored_common(run)
        if variant == "auto":
            tuned = run / TUNED_FILE
            variant = json.loads(tuned.read_text())["variants"] if tuned.exists() else "compiled"
        variants = variant if isinstance(variant, dict) else {"det": variant, "rec": variant}
        self.mx = mx
        self.name = "axon-mlx:" + (variant if isinstance(variant, str) else "/".join(variants.values()))
        self.dtype = getattr(mx, dtype)
        models = {}
        compiled = {}
        for kind in ("det", "rec"):
            options = set(variants[kind].split("+")) - {"base"}
            if options - {"compiled", "shift", "shift49"}:
                raise SystemExit(f"unknown MLX variant {variants[kind]!r}; choose from {', '.join(VARIANTS)}")
            compiled[kind] = "compiled" in options
            namespace: dict[str, Any] = {}
            exec((run / f"ppocr_{kind}_mlx.py").read_text(), namespace)
            if options & {"shift", "shift49"}:
                namespace["_SPATIAL_DEPTHWISE_SHIFT_TAPS"] = 49 if "shift49" in options else 9
            weights = mx.load(str(run / kind / "model.safetensors"))
            weights = {key: value.astype(self.dtype) for key, value in weights.items()}
            mx.eval(*weights.values())
            config = json.loads((run / kind / "config.json").read_text())
            models[kind] = namespace[f"PpocrMlx{kind.title()}"].from_state_dict(
                weights, model_config=config
            )
        scale = mx.array(1.0 / (255.0 * DET_STD)).astype(self.dtype)
        offset = mx.array(-DET_MEAN / DET_STD).astype(self.dtype)

        def forward(kind: str) -> Any:
            return lambda x: models[kind].forward(pixel_values=x.astype(self.dtype)).astype(mx.float32)

        def det_pixels(pixels: Any) -> Any:
            return forward("det")(pixels.astype(self.dtype) * scale + offset)

        def rec_best(x: Any) -> tuple[Any, Any]:
            probs = forward("rec")(x)
            return mx.argmax(probs, axis=-1), mx.max(probs, axis=-1)

        self.fns = {"det": forward("det"), "rec": forward("rec"), "det_pixels": det_pixels, "rec_best": rec_best}
        for name, fn in list(self.fns.items()):
            if compiled[name[:3]]:
                self.fns[name] = mx.compile(fn)

    def _call(self, name: str, x: np.ndarray) -> Any:
        out = self.fns[name](self.mx.array(x))
        self.mx.eval(out)
        return out

    def run(self, kind: str, x: np.ndarray) -> np.ndarray:
        return np.array(self._call(kind, x))

    def det_pixels(self, pixels: np.ndarray) -> np.ndarray:
        """Probability map for a uint8 image batch; normalization runs on the device."""
        return np.array(self._call("det_pixels", pixels))

    def rec_submit(self, x: np.ndarray) -> Any:
        """Queue a recognizer batch on the device without waiting for it."""
        out = self.fns["rec_best"](self.mx.array(x))
        self.mx.async_eval(*out)
        return out

    def rec_collect(self, handle: Any) -> tuple[np.ndarray, np.ndarray]:
        """Best class and its probability per column; only these leave the device."""
        best, score = handle
        return np.array(best), np.array(score)

    def rec_best(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.rec_collect(self.rec_submit(x))


class OnnxBackend:
    """The published ONNX graphs under ONNX Runtime, wrapped to the channels-last interface."""

    def __init__(self, run: Path, providers: list[str] | None = None) -> None:
        import onnxruntime as ort

        providers = providers or ["CPUExecutionProvider"]
        self.name = "onnxruntime:" + providers[0].removesuffix("ExecutionProvider")
        self.sessions = {
            kind: ort.InferenceSession(str(_onnx_dir(run) / name), providers=providers)
            for kind, name in ONNX_FILES.items()
        }

    def run(self, kind: str, x: np.ndarray) -> np.ndarray:
        out = self.sessions[kind].run(None, {"x": np.ascontiguousarray(x.transpose(0, 3, 1, 2))})[0]
        return out.transpose(0, 2, 3, 1) if kind == "det" else out

    def det_pixels(self, pixels: np.ndarray) -> np.ndarray:
        return self.run("det", (pixels.astype(np.float32) / 255.0 - DET_MEAN) / DET_STD)

    def rec_best(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        probs = self.run("rec", x)
        return probs.argmax(-1), probs.max(-1)

    def rec_submit(self, x: np.ndarray) -> Any:
        return self.rec_best(x)

    def rec_collect(self, handle: Any) -> tuple[np.ndarray, np.ndarray]:
        return handle


# ---------------------------------------------------------------- parity and bench


def _random_inputs() -> dict[str, list[np.ndarray]]:
    rng = np.random.default_rng(0)
    det = [(1, 64, 96), (2, 160, 224), (1, 736, 960)]
    rec = [(1, 160), (6, 320), (2, 48), (1, 1000)]
    return {
        "det": [rng.standard_normal((b, h, w, 3)).astype(np.float32) for b, h, w in det],
        "rec": [rng.standard_normal((b, REC_HEIGHT, w, 3)).astype(np.float32) for b, w in rec],
    }


def _sample_pixels(run: Path) -> np.ndarray:
    """A real page when the run has one (and OpenCV is installed), else random pixels.

    Random inputs give near-empty detector maps and small activations; a real page is what
    exposes range problems such as half-precision overflow.
    """
    sample = run / "sample.png"
    if sample.exists():
        try:
            import cv2

            return det_preprocess(cv2.imread(str(sample)), 960)
        except ImportError:
            pass
    return np.random.default_rng(1).integers(0, 256, (1, 96, 128, 3), dtype=np.uint8)


def cmd_parity(args: argparse.Namespace) -> None:
    onnx = OnnxBackend(args.run)
    inputs = _random_inputs()
    reference = {kind: [onnx.run(kind, x) for x in xs] for kind, xs in inputs.items()}
    pixels = _sample_pixels(args.run)
    pixels_reference = onnx.det_pixels(pixels)
    worst = 0.0
    for variant in args.variants.split(","):
        mlx = MlxBackend(args.run, args.dtype, variant)
        for kind, xs in inputs.items():
            for x, want in zip(xs, reference[kind], strict=True):
                got = mlx.run(kind, x)
                diff = float(np.abs(got - want).max())
                worst = max(worst, diff)
                line = f"{mlx.name:26s} {kind} {tuple(x.shape)}  max|d|={diff:.2e}"
                if kind == "rec":
                    best, _ = mlx.rec_best(x)
                    line += f"  argmax agreement={(best == want.argmax(-1)).mean():.4f}"
                print(line)
        diff = float(np.abs(mlx.det_pixels(pixels) - pixels_reference).max())
        worst = max(worst, diff)
        print(f"{mlx.name:26s} det from uint8 pixels {tuple(pixels.shape)}  max|d|={diff:.2e}")
    print(f"worst max|d| = {worst:.2e}")
    if worst > args.tolerance:
        raise SystemExit(f"parity failed: {worst:.2e} > {args.tolerance:.0e}")


def _time(fn: Any, warmup: int, samples: int) -> tuple[float, float]:
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(samples):
        start = time.perf_counter()
        fn()
        times.append((time.perf_counter() - start) * 1e3)
    return statistics.median(times), min(times)


def cmd_bench(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(0)
    workloads = [
        ("det 1x736x960", "det", rng.standard_normal((1, 736, 960, 3)).astype(np.float32)),
        ("det 1x960x960", "det", rng.standard_normal((1, 960, 960, 3)).astype(np.float32)),
        ("rec 1x48x320", "rec", rng.standard_normal((1, REC_HEIGHT, 320, 3)).astype(np.float32)),
        ("rec 6x48x320", "rec", rng.standard_normal((6, REC_HEIGHT, 320, 3)).astype(np.float32)),
        ("rec 16x48x480", "rec", rng.standard_normal((16, REC_HEIGHT, 480, 3)).astype(np.float32)),
    ]
    backends: list[Any] = [MlxBackend(args.run, args.dtype, v) for v in args.variants.split(",")]
    try:
        import onnxruntime as ort

        available = ort.get_available_providers()
        wanted = args.providers.split(",") if args.providers else ["CPUExecutionProvider"]
        for provider in wanted:
            if provider in available:
                backends.append(OnnxBackend(args.run, list(dict.fromkeys([provider, "CPUExecutionProvider"]))))
            else:
                print(f"skipping {provider}: not in {available}")
    except ImportError:
        print("onnxruntime not installed: MLX timings only")
    results = []
    print("median ms / min ms per call")
    print(f"{'workload':15s} " + " ".join(f"{b.name:>24s}" for b in backends))
    for label, kind, x in workloads:
        row = {"workload": label}
        cells = []
        for backend in backends:
            call = backend.rec_best if kind == "rec" else (lambda x, b=backend: b.run("det", x))
            median, best = _time(lambda call=call: call(x), args.warmup, args.samples)
            row[backend.name] = {"median_ms": median, "min_ms": best}
            cells.append(f"{median:13.2f} / {best:8.2f}")
        results.append(row)
        print(f"{label:15s} " + " ".join(cells))
    out = args.run / "bench.json"
    out.write_text(json.dumps({"dtype": args.dtype, "samples": args.samples, "results": results}, indent=2))
    print(f"wrote {out}")


def cmd_tune(args: argparse.Namespace) -> None:
    """Time every variant on this machine and save the fastest one per network.

    Variants are timed in interleaved rounds and compared by their fastest call, which is
    the statistic least disturbed by other load. A variant replaces ``compiled`` only when
    it is at least 5% faster, so that noise does not pick an exotic lowering.
    """
    rng = np.random.default_rng(0)
    inputs = {
        "det": rng.integers(0, 256, (1, 736, 960, 3), dtype=np.uint8),
        "rec": rng.standard_normal((6, REC_HEIGHT, 640, 3)).astype(np.float32),
    }
    backends = {
        (dtype, variant): MlxBackend(args.run, dtype, variant)
        for dtype in ("float32", "float16")
        for variant in VARIANTS
    }
    best_ms: dict[str, dict[str, float]] = {"det": {}, "rec": {}}
    for round_index in range(args.rounds):
        for (dtype, variant), backend in backends.items():
            for kind, call in (("det", backend.det_pixels), ("rec", backend.rec_best)):
                warmup = args.warmup if round_index == 0 else 1
                _, fastest = _time(lambda call=call, kind=kind: call(inputs[kind]), warmup, args.samples)
                key = f"{dtype}:{variant}"
                best_ms[kind][key] = min(fastest, best_ms[kind].get(key, math.inf))
    print(f"fastest call in ms ({args.rounds} rounds of {args.samples})")
    print(f"{'variant':20s} {'det f32':>9s} {'det f16':>9s} {'rec f32':>9s} {'rec f16':>9s}")
    for variant in VARIANTS:
        cells = [best_ms[kind][f"{dtype}:{variant}"] for kind in ("det", "rec") for dtype in ("float32", "float16")]
        print(f"{variant:20s} " + " ".join(f"{cell:9.2f}" for cell in cells))
    chosen = {}
    for kind, table in best_ms.items():
        fastest = min(VARIANTS, key=lambda v, table=table: table[f"float32:{v}"])
        keep_default = table[f"float32:{fastest}"] > 0.95 * table["float32:compiled"]
        chosen[kind] = "compiled" if keep_default else fastest
    (args.run / TUNED_FILE).write_text(json.dumps({"variants": chosen, "fastest_ms": best_ms}, indent=2))
    print(f"float32 choice: {chosen}; saved to {args.run / TUNED_FILE}")


# ----------------------------------------------------------------------------- OCR


def det_preprocess(image: np.ndarray, limit: int) -> np.ndarray:
    """The image resized to multiples of 32, as a uint8 batch; backends normalize it."""
    import cv2

    height, width = image.shape[:2]
    ratio = min(1.0, limit / max(height, width))
    new_h = max(32, int(round(height * ratio / 32)) * 32)
    new_w = max(32, int(round(width * ratio / 32)) * 32)
    return cv2.resize(image, (new_w, new_h))[None]


def _order_points(points: np.ndarray) -> np.ndarray:
    """Top-left, top-right, bottom-right, bottom-left."""
    points = points[np.argsort(points[:, 0])]
    left = points[:2][np.argsort(points[:2, 1])]
    right = points[2:][np.argsort(points[2:, 1])]
    return np.array([left[0], right[0], right[1], left[1]], dtype=np.float32)


def db_boxes(
    prob: np.ndarray,
    src_h: int,
    src_w: int,
    *,
    thresh: float = 0.3,
    box_thresh: float = 0.6,
    unclip_ratio: float = 1.5,
    max_candidates: int = 1000,
    min_size: float = 3.0,
) -> list[np.ndarray]:
    """Rotated text boxes from a probability map, as 4x2 arrays in source-image pixels."""
    import cv2

    height, width = prob.shape
    contours, _ = cv2.findContours(
        (prob > thresh).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
    boxes = []
    for contour in contours[:max_candidates]:
        (cx, cy), (w, h), angle = cv2.minAreaRect(contour)
        if min(w, h) < min_size:
            continue
        points = cv2.boxPoints(((cx, cy), (w, h), angle))
        x0, x1 = int(max(0, math.floor(points[:, 0].min()))), int(min(width - 1, math.ceil(points[:, 0].max())))
        y0, y1 = int(max(0, math.floor(points[:, 1].min()))), int(min(height - 1, math.ceil(points[:, 1].max())))
        mask = np.zeros((y1 - y0 + 1, x1 - x0 + 1), dtype=np.uint8)
        cv2.fillPoly(mask, [(points - [x0, y0]).astype(np.int32)], 1)
        if cv2.mean(prob[y0 : y1 + 1, x0 : x1 + 1], mask)[0] < box_thresh:
            continue
        # Offsetting a rectangle by area * ratio / perimeter grows each side by twice that.
        grow = w * h * unclip_ratio / (2 * (w + h))
        if min(w, h) + 2 * grow < min_size + 2:
            continue
        points = cv2.boxPoints(((cx, cy), (w + 2 * grow, h + 2 * grow), angle))
        points[:, 0] = np.clip(points[:, 0] / width * src_w, 0, src_w)
        points[:, 1] = np.clip(points[:, 1] / height * src_h, 0, src_h)
        boxes.append(_order_points(points))
    return sorted(boxes, key=lambda box: (round(float(box[:, 1].min()) / 10), float(box[:, 0].min())))


def crop_box(image: np.ndarray, box: np.ndarray) -> np.ndarray:
    import cv2

    width = int(max(np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[3] - box[2])))
    height = int(max(np.linalg.norm(box[0] - box[3]), np.linalg.norm(box[1] - box[2])))
    width, height = max(width, 1), max(height, 1)
    target = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
    crop = cv2.warpPerspective(
        image,
        cv2.getPerspectiveTransform(box, target),
        (width, height),
        borderMode=cv2.BORDER_REPLICATE,
        flags=cv2.INTER_CUBIC,
    )
    return np.rot90(crop) if height / width >= 1.5 else crop


def rec_batches(
    crops: list[np.ndarray], batch_size: int, width_bucket: int
) -> Any:
    """Yield batches of similar aspect ratio, each zero-padded to its widest crop.

    The batch width is rounded up to a multiple of ``width_bucket`` so that compiled
    networks see few distinct shapes.
    """
    import cv2

    order = sorted(range(len(crops)), key=lambda i: crops[i].shape[1] / crops[i].shape[0])
    for start in range(0, len(order), batch_size):
        indexes = order[start : start + batch_size]
        ratios = [crops[i].shape[1] / crops[i].shape[0] for i in indexes]
        width = int(REC_HEIGHT * max(REC_BASE_WIDTH / REC_HEIGHT, max(ratios)))
        width = -(-width // width_bucket) * width_bucket
        batch = np.zeros((len(indexes), REC_HEIGHT, width, 3), dtype=np.float32)
        for row, (i, ratio) in enumerate(zip(indexes, ratios, strict=True)):
            resized_w = min(width, int(math.ceil(REC_HEIGHT * ratio)))
            resized = cv2.resize(crops[i], (resized_w, REC_HEIGHT)).astype(np.float32)
            batch[row, :, :resized_w] = (resized / 255.0 - 0.5) / 0.5
        yield indexes, batch


def load_characters(path: Path) -> list[str]:
    """Index 0 is the CTC blank; the dictionary is followed by the space character."""
    return ["", *path.read_text(encoding="utf-8").splitlines(), " "]


def ctc_decode(indexes: np.ndarray, scores: np.ndarray, characters: list[str]) -> tuple[str, float]:
    """Collapse repeats and drop blanks; the score is the mean probability of kept columns."""
    keep = (indexes != 0) & (indexes != np.concatenate([[-1], indexes[:-1]]))
    if not keep.any():
        return "", 0.0
    return "".join(characters[i] for i in indexes[keep]), float(scores[keep].mean())


def run_ocr(
    backend: Any,
    image: np.ndarray,
    characters: list[str],
    *,
    limit: int,
    batch_size: int,
    width_bucket: int,
) -> dict[str, Any]:
    timings = {}
    start = time.perf_counter()
    det_input = det_preprocess(image, limit)
    timings["det_pre_ms"] = (time.perf_counter() - start) * 1e3
    start = time.perf_counter()
    prob = backend.det_pixels(det_input)[0, :, :, 0]
    timings["det_net_ms"] = (time.perf_counter() - start) * 1e3
    start = time.perf_counter()
    boxes = db_boxes(prob, image.shape[0], image.shape[1])
    crops = [crop_box(image, box) for box in boxes]
    timings["boxes_crops_ms"] = (time.perf_counter() - start) * 1e3
    # Each batch is queued as soon as it is built, so the device works on it while the
    # host resizes the next one; results are read back once at the end.
    start = time.perf_counter()
    pending = [
        (indexes, backend.rec_submit(batch))
        for indexes, batch in rec_batches(crops, batch_size, width_bucket)
    ]
    texts: list[tuple[str, float]] = [("", 0.0)] * len(crops)
    for indexes, handle in pending:
        best, scores = backend.rec_collect(handle)
        for row, i in enumerate(indexes):
            texts[i] = ctc_decode(best[row], scores[row], characters)
    timings["rec_ms"] = (time.perf_counter() - start) * 1e3
    lines = [
        {"text": text, "score": score, "box": box.round(1).tolist()}
        for box, (text, score) in zip(boxes, texts, strict=True)
    ]
    return {"lines": lines, "timings": timings, "det_shape": list(det_input.shape)}


def cmd_ocr(args: argparse.Namespace) -> None:
    import cv2

    image = cv2.imread(str(args.image))
    if image is None:
        raise SystemExit(f"cannot read image: {args.image}")
    characters = load_characters(_dictionary(args.run))
    if args.backend == "mlx":
        backend = MlxBackend(args.run, args.dtype, args.variant)
    else:
        backend = OnnxBackend(args.run)
    for _ in range(args.repeat):
        result = run_ocr(
            backend,
            image,
            characters,
            limit=args.limit,
            batch_size=args.batch_size,
            width_bucket=args.width_bucket,
        )
    for line in result["lines"]:
        print(f"{line['score']:.3f}  {line['text']}")
    timings = result["timings"]
    print(
        f"[{backend.name}] {len(result['lines'])} lines, det input {result['det_shape']}; "
        + ", ".join(f"{key}={value:.1f}" for key, value in timings.items())
        + f", total={sum(timings.values()):.1f} ms (last of {args.repeat} runs)"
    )
    if args.json is not None:
        args.json.write_text(json.dumps(result, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)

    def add(name: str, handler: Any) -> argparse.ArgumentParser:
        sub = commands.add_parser(name)
        here = Path(__file__).resolve().parent
        kit = here if (here / "ppocr_det_mlx.py").exists() else None
        sub.add_argument(
            "--run",
            type=Path,
            default=kit,
            required=kit is None,
            help="run directory, e.g. log/ppocr-mlx (default: this directory when it is a kit)",
        )
        sub.set_defaults(handler=handler)
        if name != "build":
            sub.add_argument("--dtype", default="float32", choices=["float32", "float16"])
        return sub

    build = add("build", cmd_build)
    build.add_argument("--onnx-dir", type=Path, help="directory holding the ONNX files and dictionary")
    build.add_argument("--repo", default=DEFAULT_REPO, help="Hugging Face repo used without --onnx-dir")
    all_variants = ",".join(VARIANTS)
    parity = add("parity", cmd_parity)
    parity.add_argument("--variants", default=all_variants, help="comma-separated MLX variants")
    parity.add_argument("--tolerance", type=float, default=1e-3)
    bench = add("bench", cmd_bench)
    bench.add_argument("--variants", default=all_variants, help="comma-separated MLX variants")
    bench.add_argument("--providers", help="comma-separated ONNX Runtime providers to time")
    bench.add_argument("--warmup", type=int, default=3)
    bench.add_argument("--samples", type=int, default=15)
    tune = add("tune", cmd_tune)
    tune.add_argument("--warmup", type=int, default=3)
    tune.add_argument("--samples", type=int, default=10)
    tune.add_argument("--rounds", type=int, default=3)
    ocr = add("ocr", cmd_ocr)
    ocr.add_argument("--image", type=Path, required=True)
    ocr.add_argument("--backend", default="mlx", choices=["mlx", "onnx"])
    ocr.add_argument("--limit", type=int, default=960, help="longest detector input side")
    ocr.add_argument("--variant", default="auto", help="auto, or one of: " + ", ".join(VARIANTS))
    ocr.add_argument("--batch-size", type=int, default=6)
    ocr.add_argument("--width-bucket", type=int, default=32, help="round recognizer batch widths up to this")
    ocr.add_argument("--repeat", type=int, default=1, help="run this many times and report the last")
    ocr.add_argument("--json", type=Path, help="write lines, boxes and timings here")
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
