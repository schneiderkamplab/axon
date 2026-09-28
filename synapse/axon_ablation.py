from __future__ import annotations

import csv
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

from .axon import (
    GraphOptimizeConfig,
    elaborate_closed_axon_file,
    flatten_closed_axon_file,
    lower_axon_program_to_graph_ir,
    normalize_closed_axon_file,
    optimize_graph_program,
    optimize_safe_flat_typed_axon_file,
    resolve_axon_program_from_path,
    typecheck2_flat_axon_file,
)
from .axon.graph_ir import GraphProgram

# ---------------------------------------------------------------------------
# Backend intrinsic sets (mirrors graph_ir/optimize.py for matrix generation)
# ---------------------------------------------------------------------------

_TORCH_INTRINSICS = frozenset({
    "__torch_add_rmsnorm_noscale",
    "__torch_expert_packed_swiglu_ffn",
    "__torch_expert_swiglu_ffn",
    "__torch_rope_apply_factors",
    "__torch_rope_pair_apply_factors",
    "__torch_rmsnorm_noscale",
    "__torch_rmsnorm_scaled",
    "__torch_sdpa",
    "__torch_selected_expert_clamped_packed_swiglu_ffn",
    "__torch_selected_expert_packed_gegelu_ffn",
    "__torch_selected_expert_packed_swiglu_ffn",
    "__torch_selected_expert_relu2_ffn",
    "__torch_selected_expert_swiglu_ffn",
    "__torch_sigmoid_gated_merge",
    "__torch_sigmoid_gated_mul",
    "__torch_split_swiglu",
    "__torch_swiglu_ffn",
    "__torch_gelu_ffn",
    "__torch_topk_normalize",
    "__torch_weighted_topk_sum",
})

_TRITON_INTRINSICS = frozenset({
    "__triton_add_rmsnorm_noscale",
    "__triton_rmsnorm_noscale",
    "__triton_rmsnorm_scaled",
    "__triton_rmsnorm_unit_offset_scaled",
    "__triton_layernorm",
    "__triton_geglu_tanh_activation",
    "__triton_sdpa",
    "__triton_selected_expert_packed_swiglu_ffn",
    "__triton_sigmoid_gated_merge",
    "__triton_sigmoid_gated_mul",
    "__triton_split_swiglu",
    "__triton_swiglu_activation",
    "__torch_gelu_ffn",
    "__torch_rope_apply_factors",
    "__torch_rope_pair_apply_factors",
})

_BACKEND_INTRINSICS: dict[str, frozenset[str]] = {
    "codegen2-torch": _TORCH_INTRINSICS,
    "codegen2-triton": _TRITON_INTRINSICS,
}

_BACKEND_PREFIX = {
    "codegen2-torch": "__torch_",
    "codegen2-triton": "__triton_",
}

# Pass flags that can be individually toggled for T2 ablation
_PASS_FLAGS = (
    "prune_to_main",
    "atomic_alias_cleanup",
    "dead_temp_elimination",
    "constant_folding",
    "constant_dim_substitution",
    "common_subexpression_elimination",
    "inline_safe",
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AblationConfig:
    name: str
    tier: str  # "t1-stages" | "t2-per-pass" | "t3-per-intrinsic"
    description: str
    optimize_ast: bool = False
    graph_config: GraphOptimizeConfig | None = None


@dataclass
class IRMetrics:
    graph_modules: int = 0
    graph_nodes: int = 0
    op_distribution: Counter = field(default_factory=Counter)
    packed_params: int = 0


@dataclass
class StageTimings:
    parse_s: float = 0.0
    resolve_s: float = 0.0
    normalize_s: float = 0.0
    typecheck_s: float = 0.0
    lower_s: float = 0.0
    optimize_graph_s: float = 0.0
    codegen_lines: int = 0


@dataclass
class RuntimeMetrics:
    kernel_count: int | None = None
    peak_memory_bytes: int | None = None
    memory_delta_bytes: int | None = None
    op_call_count: int | None = None
    cpu_op_count: int | None = None
    error: str = ""


@dataclass
class AblationResult:
    config: AblationConfig
    ir_metrics: IRMetrics
    timings: StageTimings
    model_name: str = ""
    runtime: dict[str, Any] = field(default_factory=dict)
    runtime_metrics: RuntimeMetrics = field(default_factory=RuntimeMetrics)

    @property
    def config_name(self) -> str:
        return self.config.name

    @property
    def tier(self) -> str:
        return self.config.tier


# ---------------------------------------------------------------------------
# IR metrics helpers
# ---------------------------------------------------------------------------

def _graph_node_count(graph: GraphProgram) -> int:
    return sum(len(module.nodes) for module in graph.modules)


def _collect_ir_metrics(graph: GraphProgram) -> IRMetrics:
    ops: Counter[str] = Counter()
    for module in graph.modules:
        for node in module.nodes:
            ops[node.op.name] += 1
    return IRMetrics(
        graph_modules=len(graph.modules),
        graph_nodes=_graph_node_count(graph),
        op_distribution=ops,
        packed_params=len(graph.packed_parameters),
    )


def _run_pipeline_to_graph(
    axon_file: Path,
    *,
    optimize_ast: bool = False,
    builtins_overlays: Sequence[str] | None = None,
) -> tuple[GraphProgram, StageTimings]:
    timings = StageTimings()

    t0 = time.perf_counter()
    resolved = resolve_axon_program_from_path(
        axon_file, builtins_overlays=builtins_overlays
    ).ast
    timings.resolve_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    normalized = normalize_closed_axon_file(resolved)
    timings.normalize_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    elaborated = elaborate_closed_axon_file(normalized)
    timings.typecheck_s = 0.0  # typecheck below
    flat = flatten_closed_axon_file(elaborated)
    timings.lower_s = 0.0  # lower below

    t0 = time.perf_counter()
    typed = typecheck2_flat_axon_file(flat)
    timings.typecheck_s = time.perf_counter() - t0

    if optimize_ast:
        typed = optimize_safe_flat_typed_axon_file(typed)

    t0 = time.perf_counter()
    graph = lower_axon_program_to_graph_ir(typed)
    timings.lower_s = time.perf_counter() - t0

    return graph, timings


def _short_intrinsic_name(full: str, backend: str) -> str:
    prefix = _BACKEND_PREFIX.get(backend, "__")
    if full.startswith(prefix):
        return full[len(prefix):]
    return full.removeprefix("_")


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def _build_csv_fieldnames(results: Sequence[AblationResult]) -> list[str]:
    base_fields = [
        "model", "config_name", "tier",
        "optimize_ast", "optimize_graph", "backend_intrinsics",
        "graph_modules", "graph_nodes", "packed_params",
        "resolve_s", "normalize_s", "typecheck_s", "lower_s", "optimize_graph_s",
        "codegen_lines",
        "forward_time_s", "hf_time_s", "speed_ratio",
        "masked_top1_eq", "masked_max_abs_diff", "masked_max_rel_diff",
        "runtime_error",
        "kernel_count", "peak_memory_bytes", "memory_delta_bytes", "op_call_count", "cpu_op_count",
    ]
    all_ops: set[str] = set()
    for r in results:
        all_ops.update(r.ir_metrics.op_distribution.keys())
    op_fields = sorted(all_ops)
    return base_fields + [f"op_{op}" for op in op_fields]


def export_ablation_csv(
    results: Sequence[AblationResult],
    csv_path: Path,
) -> None:
    fieldnames = _build_csv_fieldnames(results)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row: dict[str, Any] = {
                "model": r.model_name,
                "config_name": r.config.name,
                "tier": r.config.tier,
                "optimize_ast": r.config.optimize_ast,
                "optimize_graph": r.config.graph_config is not None,
                "backend_intrinsics": (
                    r.config.graph_config.backend_intrinsics
                    if r.config.graph_config
                    else ""
                ),
                "graph_modules": r.ir_metrics.graph_modules,
                "graph_nodes": r.ir_metrics.graph_nodes,
                "packed_params": r.ir_metrics.packed_params,
                "resolve_s": r.timings.resolve_s,
                "normalize_s": r.timings.normalize_s,
                "typecheck_s": r.timings.typecheck_s,
                "lower_s": r.timings.lower_s,
                "optimize_graph_s": r.timings.optimize_graph_s,
                "codegen_lines": r.timings.codegen_lines,
                "forward_time_s": r.runtime.get("axon_time"),
                "hf_time_s": r.runtime.get("hf_time"),
                "speed_ratio": r.runtime.get("speed_ratio_axon_over_hf"),
                "masked_top1_eq": r.runtime.get("masked_top1_eq"),
                "masked_max_abs_diff": r.runtime.get("masked_max_abs_diff"),
                "masked_max_rel_diff": r.runtime.get("masked_max_rel_diff"),
                "runtime_error": r.runtime.get("error", ""),
                "kernel_count": r.runtime_metrics.kernel_count,
                "peak_memory_bytes": r.runtime_metrics.peak_memory_bytes,
                "memory_delta_bytes": r.runtime_metrics.memory_delta_bytes,
                "op_call_count": r.runtime_metrics.op_call_count,
                "cpu_op_count": r.runtime_metrics.cpu_op_count,
            }
            for op, count in r.ir_metrics.op_distribution.items():
                row[f"op_{op}"] = count
            writer.writerow(row)


# ---------------------------------------------------------------------------
# SVG visualizations
# ---------------------------------------------------------------------------

_TIER_COLORS = {
    "t1-stages": "#2563eb",
    "t2-per-pass": "#dc2626",
    "t3-per-intrinsic": "#16a34a",
}


def _fmt_ms(seconds: float | None) -> str:
    if seconds is None:
        return "N/A"
    return f"{seconds * 1000:.1f}ms"


def plot_ablation_waterfall(
    results: Sequence[AblationResult],
    svg_path: Path,
    *,
    title: str = "Compiler Ablation — IR Node Count",
) -> None:
    if not results:
        return
    configs = [r.config.name for r in results]
    node_counts = [r.ir_metrics.graph_nodes for r in results]
    n = len(configs)
    bar_h = 32
    gap = 8
    label_w = 260
    plot_w = 600
    width = label_w + plot_w + 60
    height = 70 + n * (bar_h + gap) + 40
    max_count = max(node_counts) if node_counts else 1

    out: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"'
        f' viewBox="0 0 {width} {height}">',
        "<style>",
        "text{font-family:Arial,Helvetica,sans-serif;fill:#111827}",
        ".small{font-size:11px}.axis{font-size:13px}.title{font-size:18px;font-weight:700}",
        "</style>",
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text class="title" x="{label_w}" y="32">{escape(title)}</text>',
    ]

    for i, (name, count) in enumerate(zip(configs, node_counts)):
        y = 50 + i * (bar_h + gap)
        color = _TIER_COLORS.get(results[i].tier, "#6b7280")
        bar_w = int((count / max_count) * plot_w) if max_count else 0
        out.append(
            f'<text class="small" x="{label_w - 10}" y="{y + bar_h // 2 + 4}"'
            f' text-anchor="end">{escape(name)}</text>'
        )
        out.append(
            f'<rect x="{label_w}" y="{y}" width="{bar_w}" height="{bar_h}"'
            f' fill="{color}" opacity="0.85" rx="3"/>'
        )
        out.append(
            f'<text class="small" x="{label_w + bar_w + 6}" y="{y + bar_h // 2 + 4}">{count}</text>'
        )

    out.append("</svg>")
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    svg_path.write_text("\n".join(out), encoding="utf-8")


def plot_ablation_runtime(
    results: Sequence[AblationResult],
    svg_path: Path,
    *,
    title: str = "Compiler Ablation — Runtime per Config",
) -> None:
    timed = [r for r in results if r.runtime.get("axon_time") is not None]
    if not timed:
        return
    configs = [r.config.name for r in timed]
    axon_times = [r.runtime["axon_time"] * 1000 for r in timed]
    hf_times = [
        r.runtime["hf_time"] * 1000 if r.runtime.get("hf_time") is not None else 0.0
        for r in timed
    ]
    n = len(timed)
    bar_w = 28
    gap = 12
    label_h = 140
    plot_h = 400
    height = 60 + label_h + plot_h + 50
    width = 80 + n * (bar_w * 2 + gap) + 40
    max_time = max(axon_times + [t for t in hf_times if t > 0]) if axon_times else 1.0

    out: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"'
        f' viewBox="0 0 {width} {height}">',
        "<style>",
        "text{font-family:Arial,Helvetica,sans-serif;fill:#111827}",
        ".small{font-size:10px}.axis{font-size:13px}.title{font-size:18px;font-weight:700}",
        "</style>",
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text class="title" x="80" y="32">{escape(title)}</text>',
    ]

    base_y = 60 + label_h
    for i, (name, ax_t, hf_t) in enumerate(zip(configs, axon_times, hf_times)):
        x = 80 + i * (bar_w * 2 + gap)
        ax_h = int((ax_t / max_time) * plot_h) if max_time else 0
        hf_h = int((hf_t / max_time) * plot_h) if max_time and hf_t > 0 else 0
        out.append(
            f'<rect x="{x}" y="{base_y + plot_h - ax_h}" width="{bar_w}"'
            f' height="{ax_h}" fill="#2563eb" opacity="0.85" rx="2"/>'
        )
        if hf_t > 0:
            out.append(
                f'<rect x="{x + bar_w}" y="{base_y + plot_h - hf_h}" width="{bar_w}"'
                f' height="{hf_h}" fill="#dc2626" opacity="0.85" rx="2"/>'
            )
        out.append(
            f'<text class="small" x="{x + bar_w}" y="{base_y + plot_h + 14}"'
            f' text-anchor="middle" transform="rotate(-45 {x + bar_w} {base_y + plot_h + 14})">'
            f'{escape(name)}</text>'
        )

    out.append(
        f'<line x1="80" y1="{base_y + plot_h}" x2="{width - 40}"'
        f' y2="{base_y + plot_h}" stroke="#111827" stroke-width="1"/>'
    )
    out.append(
        f'<rect x="80" y="{base_y - 25}" width="12" height="12" fill="#2563eb" opacity="0.85"/>'
        f'<text class="small" x="98" y="{base_y - 16}">Axon</text>'
        f'<rect x="140" y="{base_y - 25}" width="12" height="12" fill="#dc2626" opacity="0.85"/>'
        f'<text class="small" x="158" y="{base_y - 16}">HF baseline</text>'
    )
    out.append("</svg>")
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    svg_path.write_text("\n".join(out), encoding="utf-8")


def plot_ablation_kernels(
    results: Sequence[AblationResult],
    svg_path: Path,
    *,
    title: str = "Compiler Ablation — CUDA Kernel Count per Config",
) -> None:
    profiled = [r for r in results if r.runtime_metrics.kernel_count is not None]
    if not profiled:
        return
    configs = [r.config.name for r in profiled]
    counts = [r.runtime_metrics.kernel_count for r in profiled]
    n = len(profiled)
    bar_w = 36
    gap = 16
    label_h = 140
    plot_h = 400
    height = 60 + label_h + plot_h + 50
    width = 80 + n * (bar_w + gap) + 40
    max_count = max(counts) if counts else 1

    out: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"'
        f' viewBox="0 0 {width} {height}">',
        "<style>",
        "text{font-family:Arial,Helvetica,sans-serif;fill:#111827}",
        ".small{font-size:10px}.axis{font-size:13px}.title{font-size:18px;font-weight:700}",
        "</style>",
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text class="title" x="80" y="32">{escape(title)}</text>',
    ]

    base_y = 60 + label_h
    for i, (name, cnt) in enumerate(zip(configs, counts)):
        x = 80 + i * (bar_w + gap)
        h = int((cnt / max_count) * plot_h) if max_count else 0
        out.append(
            f'<rect x="{x}" y="{base_y + plot_h - h}" width="{bar_w}"'
            f' height="{h}" fill="#7c3aed" opacity="0.85" rx="2"/>'
        )
        out.append(
            f'<text class="small" x="{x + bar_w // 2}" y="{base_y + plot_h - h - 4}"'
            f' text-anchor="middle">{cnt}</text>'
        )
        out.append(
            f'<text class="small" x="{x + bar_w // 2}" y="{base_y + plot_h + 14}"'
            f' text-anchor="middle" transform="rotate(-45 {x + bar_w // 2} {base_y + plot_h + 14})">'
            f'{escape(name)}</text>'
        )

    out.append(
        f'<line x1="80" y1="{base_y + plot_h}" x2="{width - 40}"'
        f' y2="{base_y + plot_h}" stroke="#111827" stroke-width="1"/>'
    )
    out.append("</svg>")
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    svg_path.write_text("\n".join(out), encoding="utf-8")


def plot_ablation_memory(
    results: Sequence[AblationResult],
    svg_path: Path,
    *,
    title: str = "Compiler Ablation — Peak GPU Memory per Config",
) -> None:
    profiled = [r for r in results if r.runtime_metrics.peak_memory_bytes is not None]
    if not profiled:
        return
    configs = [r.config.name for r in profiled]
    mem_mb = [r.runtime_metrics.peak_memory_bytes / (1024 * 1024) for r in profiled]
    n = len(profiled)
    bar_w = 36
    gap = 16
    label_h = 140
    plot_h = 400
    height = 60 + label_h + plot_h + 50
    width = 80 + n * (bar_w + gap) + 40
    max_mb = max(mem_mb) if mem_mb else 1.0

    out: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"'
        f' viewBox="0 0 {width} {height}">',
        "<style>",
        "text{font-family:Arial,Helvetica,sans-serif;fill:#111827}",
        ".small{font-size:10px}.axis{font-size:13px}.title{font-size:18px;font-weight:700}",
        "</style>",
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text class="title" x="80" y="32">{escape(title)}</text>',
    ]

    base_y = 60 + label_h
    for i, (name, mb) in enumerate(zip(configs, mem_mb)):
        x = 80 + i * (bar_w + gap)
        h = int((mb / max_mb) * plot_h) if max_mb else 0
        out.append(
            f'<rect x="{x}" y="{base_y + plot_h - h}" width="{bar_w}"'
            f' height="{h}" fill="#059669" opacity="0.85" rx="2"/>'
        )
        out.append(
            f'<text class="small" x="{x + bar_w // 2}" y="{base_y + plot_h - h - 4}"'
            f' text-anchor="middle">{mb:.1f}MB</text>'
        )
        out.append(
            f'<text class="small" x="{x + bar_w // 2}" y="{base_y + plot_h + 14}"'
            f' text-anchor="middle" transform="rotate(-45 {x + bar_w // 2} {base_y + plot_h + 14})">'
            f'{escape(name)}</text>'
        )

    out.append(
        f'<line x1="80" y1="{base_y + plot_h}" x2="{width - 40}"'
        f' y2="{base_y + plot_h}" stroke="#111827" stroke-width="1"/>'
    )
    out.append("</svg>")
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    svg_path.write_text("\n".join(out), encoding="utf-8")


def plot_ablation_ops_heatmap(
    results: Sequence[AblationResult],
    svg_path: Path,
    *,
    title: str = "Compiler Ablation — Op Distribution",
) -> None:
    if not results:
        return
    all_ops: set[str] = set()
    for r in results:
        all_ops.update(r.ir_metrics.op_distribution.keys())
    ops = sorted(all_ops)
    configs = [r.config.name for r in results]
    if not ops:
        return

    n_configs = len(configs)
    n_ops = len(ops)
    label_w = 220
    label_h = 140
    cell_w = 40
    cell_h = 22
    width = label_w + n_ops * cell_w + 40
    height = 60 + label_h + n_configs * cell_h + 20
    max_count = 1
    for r in results:
        for op in ops:
            c = r.ir_metrics.op_distribution.get(op, 0)
            if c > max_count:
                max_count = c

    out: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"'
        f' viewBox="0 0 {width} {height}">',
        "<style>",
        "text{font-family:Arial,Helvetica,sans-serif;fill:#111827}",
        ".small{font-size:10px}.axis{font-size:13px}.title{font-size:18px;font-weight:700}",
        "</style>",
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text class="title" x="{label_w}" y="32">{escape(title)}</text>',
    ]

    for j, op in enumerate(ops):
        x = label_w + j * cell_w
        out.append(
            f'<text class="small" x="{x + cell_w // 2}" y="55"'
            f' text-anchor="middle" transform="rotate(-60 {x + cell_w // 2} 55)">{escape(op)}</text>'
        )

    for i, r in enumerate(configs):
        y = 60 + label_h + i * cell_h
        out.append(
            f'<text class="small" x="{label_w - 8}" y="{y + cell_h // 2 + 3}"'
            f' text-anchor="end">{escape(r)}</text>'
        )
        for j, op in enumerate(ops):
            count = results[i].ir_metrics.op_distribution.get(op, 0)
            intensity = count / max_count if max_count else 0
            if count == 0:
                bg = "#f9fafb"
            else:
                level = int(intensity * 255)
                bg = f"rgb({255 - level}, {255 - int(level * 0.3)}, {255 - level})"
            x = label_w + j * cell_w
            out.append(
                f'<rect x="{x}" y="{y}" width="{cell_w}" height="{cell_h}"'
                f' fill="{bg}" stroke="#e5e7eb" stroke-width="0.5"/>'
            )
            if count > 0:
                out.append(
                    f'<text class="small" x="{x + cell_w // 2}" y="{y + cell_h // 2 + 3}"'
                    f' text-anchor="middle" fill="{("#111827" if intensity < 0.5 else "#ffffff")}">{count}</text>'
                )

    out.append("</svg>")
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    svg_path.write_text("\n".join(out), encoding="utf-8")


# ---------------------------------------------------------------------------
# CompilerAblation — main class
# ---------------------------------------------------------------------------

class CompilerAblation:
    def __init__(
        self,
        axon_file: Path,
        weights: Path,
        *,
        backend: str = "codegen2-torch",
        device: str = "cpu",
        dtype: str = "float32",
        forward_warmup: int = 2,
        forward_repeat: int = 10,
        text: str | Sequence[str] = ("The future of AI is", "Hello World"),
        max_len: int = 32,
        builtins_overlays: Sequence[str] | None = None,
        skip_hf: bool = False,
        compile_axon: bool = False,
        compile_hf: bool = False,
        compile_mode: str | None = None,
    ) -> None:
        self.axon_file = Path(axon_file)
        self.weights = Path(weights)
        self.backend = backend
        self.device = device
        self.dtype = dtype
        self.forward_warmup = forward_warmup
        self.forward_repeat = forward_repeat
        self.text = text
        self.max_len = max_len
        self.builtins_overlays = builtins_overlays
        self.skip_hf = skip_hf
        self.compile_axon = compile_axon
        self.compile_hf = compile_hf
        self.compile_mode = compile_mode
        self._base_graph: GraphProgram | None = None
        self._base_timings: StageTimings | None = None

    def _ensure_base_graph(self) -> tuple[GraphProgram, StageTimings]:
        if self._base_graph is None:
            self._base_graph, self._base_timings = _run_pipeline_to_graph(
                self.axon_file,
                optimize_ast=False,
                builtins_overlays=self.builtins_overlays,
            )
        return self._base_graph, self._base_timings

    def build_ablation_matrix(self) -> list[AblationConfig]:
        configs: list[AblationConfig] = []

        # T1: Pipeline stages
        configs.append(AblationConfig(
            name="no-opt",
            tier="t1-stages",
            description="No optimization",
            optimize_ast=False,
            graph_config=None,
        ))
        configs.append(AblationConfig(
            name="ast-only",
            tier="t1-stages",
            description="AST optimization only",
            optimize_ast=True,
            graph_config=None,
        ))
        configs.append(AblationConfig(
            name="graph-no-intrinsics",
            tier="t1-stages",
            description="Graph optimization without backend intrinsics",
            optimize_ast=False,
            graph_config=GraphOptimizeConfig(),
        ))
        configs.append(AblationConfig(
            name="graph-full",
            tier="t1-stages",
            description="Full optimization with backend intrinsics",
            optimize_ast=False,
            graph_config=GraphOptimizeConfig(backend_intrinsics=self.backend),
        ))

        # T2: Per-pass ablation (full config minus one pass)
        for flag_name in _PASS_FLAGS:
            kwargs: dict[str, Any] = {
                "backend_intrinsics": self.backend,
                flag_name: False,
            }
            configs.append(AblationConfig(
                name=f"no-{flag_name.replace('_', '-')}",
                tier="t2-per-pass",
                description=f"Full optimization with {flag_name} disabled",
                optimize_ast=False,
                graph_config=GraphOptimizeConfig(**kwargs),
            ))
        # Also ablate specialize_definitions
        configs.append(AblationConfig(
            name="no-specialize",
            tier="t2-per-pass",
            description="Full optimization with specialization disabled",
            optimize_ast=False,
            graph_config=GraphOptimizeConfig(
                backend_intrinsics=self.backend,
                specialize_definitions="off",
            ),
        ))

        # T3: Per-intrinsic ablation (only one intrinsic at a time)
        intrinsics = _BACKEND_INTRINSICS.get(self.backend)
        if intrinsics:
            for intrinsic in sorted(intrinsics):
                short = _short_intrinsic_name(intrinsic, self.backend)
                display_name = short.replace("_", "-")
                configs.append(AblationConfig(
                    name=f"only-{display_name}",
                    tier="t3-per-intrinsic",
                    description=f"Full optimization with only {display_name} intrinsic",
                    optimize_ast=False,
                    graph_config=GraphOptimizeConfig(
                        backend_intrinsics=f"{self.backend}:{short}",
                    ),
                ))

        return configs

    def build_baseline_configs(self) -> list[AblationConfig]:
        """Build t4-baselines configs for torch.compile comparison."""
        configs: list[AblationConfig] = []
        configs.append(AblationConfig(
            name="hf-baseline",
            tier="t4-baselines",
            description="HF reference model, no compile",
            optimize_ast=False,
            graph_config=None,
        ))
        configs.append(AblationConfig(
            name="hf-compiled",
            tier="t4-baselines",
            description="HF model with torch.compile",
            optimize_ast=False,
            graph_config=None,
        ))
        configs.append(AblationConfig(
            name="axon-no-opt",
            tier="t4-baselines",
            description="Axon without graph optimization",
            optimize_ast=False,
            graph_config=None,
        ))
        configs.append(AblationConfig(
            name="axon-graph-full",
            tier="t4-baselines",
            description="Axon with full graph optimization + intrinsics",
            optimize_ast=False,
            graph_config=GraphOptimizeConfig(backend_intrinsics=self.backend),
        ))
        configs.append(AblationConfig(
            name="axon-graph-full-compiled",
            tier="t4-baselines",
            description="Axon full optimization + torch.compile",
            optimize_ast=False,
            graph_config=GraphOptimizeConfig(backend_intrinsics=self.backend),
        ))
        return configs

    def run_single(self, config: AblationConfig) -> AblationResult:
        base_graph, base_timings = self._ensure_base_graph()

        # If the config has no graph optimization, use the base graph directly
        if config.graph_config is None:
            ir_metrics = _collect_ir_metrics(base_graph)
            timings = StageTimings(
                resolve_s=base_timings.resolve_s,
                normalize_s=base_timings.normalize_s,
                typecheck_s=base_timings.typecheck_s,
                lower_s=base_timings.lower_s,
            )
        else:
            # Run graph optimization with the specific config
            t0 = time.perf_counter()
            optimized = optimize_graph_program(
                base_graph,
                config=config.graph_config,
            )
            opt_time = time.perf_counter() - t0
            ir_metrics = _collect_ir_metrics(optimized)
            timings = StageTimings(
                resolve_s=base_timings.resolve_s,
                normalize_s=base_timings.normalize_s,
                typecheck_s=base_timings.typecheck_s,
                lower_s=base_timings.lower_s,
                optimize_graph_s=opt_time,
            )

        # Run runtime measurement via run_axon_test
        is_t4 = config.tier == "t4-baselines"
        use_compile_axon = is_t4 and config.name in ("axon-graph-full-compiled",)
        use_compile_hf = is_t4 and config.name in ("hf-compiled",)
        use_skip_hf = is_t4 and config.name.startswith("axon-")
        runtime: dict[str, Any] = {}
        try:
            from .axon_test import run_axon_test

            result = run_axon_test(
                axon_file=self.axon_file,
                weights=self.weights,
                device=self.device,
                dtype=self.dtype,
                text=self.text,
                max_len=self.max_len,
                axon_backend=self.backend,
                optimize_ast=config.optimize_ast,
                optimize_graph=config.graph_config is not None,
                graph_optimize_config=config.graph_config,
                builtins_overlays=self.builtins_overlays,
                forward_warmup=self.forward_warmup,
                forward_repeat=self.forward_repeat,
                benchmark_mode="forward",
                track_runtime=True,
                skip_hf=use_skip_hf or self.skip_hf,
                compile_axon=use_compile_axon,
                compile_hf=use_compile_hf or (self.compile_hf and not is_t4),
                compile_mode=self.compile_mode,
            )
            runtime = {
                "axon_time": result.get("axon_time"),
                "hf_time": result.get("hf_time"),
                "speed_ratio_axon_over_hf": result.get("speed_ratio_axon_over_hf"),
                "masked_top1_eq": result.get("masked_top1_eq"),
                "masked_max_abs_diff": result.get("masked_max_abs_diff"),
                "masked_max_rel_diff": result.get("masked_max_rel_diff"),
            }
            rt_metrics_raw = result.get("runtime_metrics", {})
            hf_rt_metrics_raw = result.get("hf_runtime_metrics", {})
            if is_t4 and not use_skip_hf:
                rt_metrics_raw = hf_rt_metrics_raw
                runtime["axon_time"] = result.get("hf_time")
                runtime["hf_time"] = result.get("hf_time")
                runtime["speed_ratio_axon_over_hf"] = 1.0
            rt_metrics = RuntimeMetrics(
                kernel_count=rt_metrics_raw.get("kernel_count"),
                peak_memory_bytes=rt_metrics_raw.get("peak_memory_bytes"),
                memory_delta_bytes=rt_metrics_raw.get("memory_delta_bytes"),
                op_call_count=rt_metrics_raw.get("op_call_count"),
                cpu_op_count=rt_metrics_raw.get("cpu_op_count"),
                error=rt_metrics_raw.get("error", ""),
            )
            # Extract codegen line count from the generated code
            source = result.get("axon_source_code")
            if source:
                timings.codegen_lines = source.count("\n") + 1
        except Exception as exc:
            print(f"[ablation]   runtime error: {exc}")
            runtime = {
                "error": str(exc),
                "axon_time": None,
                "hf_time": None,
                "speed_ratio_axon_over_hf": None,
                "masked_top1_eq": None,
                "masked_max_abs_diff": None,
                "masked_max_rel_diff": None,
            }
            rt_metrics = RuntimeMetrics(error=str(exc))

        return AblationResult(
            config=config,
            ir_metrics=ir_metrics,
            timings=timings,
            model_name=self.axon_file.stem,
            runtime=runtime,
            runtime_metrics=rt_metrics,
        )

    def run(
        self,
        configs: Sequence[AblationConfig] | None = None,
    ) -> list[AblationResult]:
        if configs is None:
            configs = self.build_ablation_matrix()
        results: list[AblationResult] = []
        for i, config in enumerate(configs):
            print(f"[ablation] ({i + 1}/{len(configs)}) {config.name} ({config.tier})")
            result = self.run_single(config)
            nodes = result.ir_metrics.graph_nodes
            fwd = result.runtime.get("axon_time")
            fwd_str = f"{fwd * 1000:.1f}ms" if fwd else "N/A"
            kernels = result.runtime_metrics.kernel_count
            kernels_str = f"{kernels}" if kernels is not None else "N/A"
            print(f"[ablation]   nodes={nodes}  forward={fwd_str}  kernels={kernels_str}")
            results.append(result)
        return results

    def export(
        self,
        results: Sequence[AblationResult],
        output_dir: Path,
    ) -> dict[str, Path]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / "ablation.csv"
        export_ablation_csv(results, csv_path)
        waterfall_path = output_dir / "ablation_waterfall.svg"
        plot_ablation_waterfall(results, waterfall_path)
        runtime_path = output_dir / "ablation_runtime.svg"
        plot_ablation_runtime(results, runtime_path)
        ops_path = output_dir / "ablation_ops.svg"
        plot_ablation_ops_heatmap(results, ops_path)
        kernels_path = output_dir / "ablation_kernels.svg"
        plot_ablation_kernels(results, kernels_path)
        memory_path = output_dir / "ablation_memory.svg"
        plot_ablation_memory(results, memory_path)
        return {
            "csv": csv_path,
            "waterfall": waterfall_path,
            "runtime": runtime_path,
            "ops_heatmap": ops_path,
            "kernels": kernels_path,
            "memory": memory_path,
        }


__all__ = [
    "AblationConfig",
    "AblationResult",
    "IRMetrics",
    "StageTimings",
    "RuntimeMetrics",
    "CompilerAblation",
    "export_ablation_csv",
    "plot_ablation_waterfall",
    "plot_ablation_runtime",
    "plot_ablation_kernels",
    "plot_ablation_memory",
    "plot_ablation_ops_heatmap",
    "plot_cross_model_nodes",
    "plot_cross_model_speedup",
    "plot_cross_model_kernels",
    "plot_cross_model_memory",
    "merge_ablation_csvs",
]


# ---------------------------------------------------------------------------
# Cross-model visualizations (for batch / overnight runs)
# ---------------------------------------------------------------------------

_T1_CONFIG_NAMES = ("no-opt", "ast-only", "graph-no-intrinsics", "graph-full")
_T1_CONFIG_COLORS = {
    "no-opt": "#9ca3af",
    "ast-only": "#2563eb",
    "graph-no-intrinsics": "#16a34a",
    "graph-full": "#dc2626",
}


def merge_ablation_csvs(
    csv_paths: Sequence[Path],
    output_csv: Path,
) -> Path:
    """Merge multiple per-model ablation CSVs into one combined CSV."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, str]] = []
    fieldnames: set[str] = set()
    for csv_path in csv_paths:
        if not csv_path.exists():
            continue
        with csv_path.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                all_rows.append(row)
                fieldnames.update(row.keys())
    fieldnames_sorted = sorted(fieldnames)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames_sorted, extrasaction="ignore")
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)
    return output_csv


def plot_cross_model_nodes(
    csv_path: Path,
    svg_path: Path,
    *,
    title: str = "Cross-Model IR Node Count by T1 Config",
) -> None:
    """Grouped horizontal bars: one group per model, bars colored by T1 config."""
    models: list[str] = []
    model_nodes: dict[str, dict[str, int]] = {}
    with csv_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            tier = row.get("tier", "")
            if tier != "t1-stages":
                continue
            model_key = row.get("model", row.get("backend", "unknown"))
            if model_key not in model_nodes:
                model_nodes[model_key] = {}
                models.append(model_key)
            nodes_str = row.get("graph_nodes", "0")
            model_nodes[model_key][row["config_name"]] = int(nodes_str) if nodes_str else 0

    if not models:
        return

    n_configs = len(_T1_CONFIG_NAMES)
    bar_h = 14
    group_gap = 10
    label_w = 200
    plot_w = 500
    n_models = len(models)
    height = 60 + n_models * (n_configs * bar_h + group_gap) + 20
    width = label_w + plot_w + 80
    max_count = 1
    for m in models:
        for c in _T1_CONFIG_NAMES:
            v = model_nodes[m].get(c, 0)
            if v > max_count:
                max_count = v

    out: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"'
        f' viewBox="0 0 {width} {height}">',
        "<style>",
        "text{font-family:Arial,Helvetica,sans-serif;fill:#111827}",
        ".small{font-size:10px}.axis{font-size:13px}.title{font-size:18px;font-weight:700}",
        "</style>",
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text class="title" x="{label_w}" y="32">{escape(title)}</text>',
    ]

    for i, m in enumerate(models):
        y_base = 50 + i * (n_configs * bar_h + group_gap)
        out.append(
            f'<text class="small" x="{label_w - 10}" y="{y_base + n_configs * bar_h // 2}"'
            f' text-anchor="end">{escape(m)}</text>'
        )
        for j, config_name in enumerate(_T1_CONFIG_NAMES):
            y = y_base + j * bar_h
            count = model_nodes[m].get(config_name, 0)
            bar_w = int((count / max_count) * plot_w) if max_count else 0
            color = _T1_CONFIG_COLORS.get(config_name, "#6b7280")
            out.append(
                f'<rect x="{label_w}" y="{y}" width="{bar_w}" height="{bar_h - 1}"'
                f' fill="{color}" opacity="0.85" rx="2"/>'
            )
            if count > 0:
                out.append(
                    f'<text class="small" x="{label_w + bar_w + 4}" y="{y + bar_h - 2}">{count}</text>'
                )

    for j, config_name in enumerate(_T1_CONFIG_NAMES):
        color = _T1_CONFIG_COLORS.get(config_name, "#6b7280")
        out.append(
            f'<rect x="{label_w + j * 130}" y="{height - 18}" width="12" height="12"'
            f' fill="{color}" opacity="0.85"/>'
        )
        out.append(
            f'<text class="small" x="{label_w + j * 130 + 16}" y="{height - 8}">{escape(config_name)}</text>'
        )

    out.append("</svg>")
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    svg_path.write_text("\n".join(out), encoding="utf-8")


def plot_cross_model_speedup(
    csv_path: Path,
    svg_path: Path,
    *,
    title: str = "Cross-Model Speed Ratio by T1 Config (lower is better)",
) -> None:
    """Grouped vertical bars: one group per model, bars colored by T1 config."""
    models: list[str] = []
    model_ratios: dict[str, dict[str, float | None]] = {}
    with csv_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            tier = row.get("tier", "")
            if tier != "t1-stages":
                continue
            model_key = row.get("model", row.get("backend", "unknown"))
            if model_key not in model_ratios:
                model_ratios[model_key] = {}
                models.append(model_key)
            ratio_str = row.get("speed_ratio", "")
            try:
                ratio = float(ratio_str) if ratio_str else None
            except ValueError:
                ratio = None
            model_ratios[model_key][row["config_name"]] = ratio

    if not models:
        return

    n_configs = len(_T1_CONFIG_NAMES)
    bar_w = 24
    group_gap = 16
    label_h = 140
    plot_h = 350
    n_models = len(models)
    width = 80 + n_models * (n_configs * bar_w + group_gap) + 40
    height = 60 + label_h + plot_h + 50
    max_ratio = 1.0
    for m in models:
        for c in _T1_CONFIG_NAMES:
            v = model_ratios[m].get(c)
            if v is not None and v > max_ratio:
                max_ratio = v

    out: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"'
        f' viewBox="0 0 {width} {height}">',
        "<style>",
        "text{font-family:Arial,Helvetica,sans-serif;fill:#111827}",
        ".small{font-size:10px}.axis{font-size:13px}.title{font-size:18px;font-weight:700}",
        "</style>",
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text class="title" x="80" y="32">{escape(title)}</text>',
    ]

    base_y = 60 + label_h
    for i, m in enumerate(models):
        x_base = 80 + i * (n_configs * bar_w + group_gap)
        for j, config_name in enumerate(_T1_CONFIG_NAMES):
            x = x_base + j * bar_w
            ratio = model_ratios[m].get(config_name)
            if ratio is None:
                continue
            bar_h = int((ratio / max_ratio) * plot_h) if max_ratio else 0
            color = _T1_CONFIG_COLORS.get(config_name, "#6b7280")
            out.append(
                f'<rect x="{x}" y="{base_y + plot_h - bar_h}" width="{bar_w - 2}"'
                f' height="{bar_h}" fill="{color}" opacity="0.85" rx="2"/>'
            )
        out.append(
            f'<text class="small" x="{x_base + n_configs * bar_w // 2}" y="{base_y + plot_h + 14}"'
            f' text-anchor="middle" transform="rotate(-40 {x_base + n_configs * bar_w // 2} {base_y + plot_h + 14})">'
            f'{escape(m)}</text>'
        )

    out.append(
        f'<line x1="80" y1="{base_y + plot_h}" x2="{width - 40}"'
        f' y2="{base_y + plot_h}" stroke="#111827" stroke-width="1"/>'
    )
    out.append(
        f'<line x1="80" y1="{base_y}" x2="80" y2="{base_y + plot_h}"'
        f' stroke="#111827" stroke-width="1"/>'
    )
    for j, config_name in enumerate(_T1_CONFIG_NAMES):
        color = _T1_CONFIG_COLORS.get(config_name, "#6b7280")
        out.append(
            f'<rect x="{80 + j * 130}" y="{base_y - 25}" width="12" height="12" fill="{color}" opacity="0.85"/>'
        )
        out.append(
            f'<text class="small" x="{80 + j * 130 + 16}" y="{base_y - 16}">{escape(config_name)}</text>'
        )

    out.append("</svg>")
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    svg_path.write_text("\n".join(out), encoding="utf-8")


def plot_cross_model_kernels(
    csv_path: Path,
    svg_path: Path,
    *,
    title: str = "Cross-Model CUDA Kernel Count by T1 Config",
) -> None:
    models: list[str] = []
    model_kernels: dict[str, dict[str, int | None]] = {}
    with csv_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("tier") != "t1-stages":
                continue
            model_key = row.get("model", "unknown")
            if model_key not in model_kernels:
                model_kernels[model_key] = {}
                models.append(model_key)
            kc_str = row.get("kernel_count", "")
            try:
                kc = int(kc_str) if kc_str else None
            except ValueError:
                kc = None
            model_kernels[model_key][row["config_name"]] = kc

    if not models:
        return

    n_configs = len(_T1_CONFIG_NAMES)
    bar_w = 24
    group_gap = 16
    label_h = 140
    plot_h = 350
    n_models = len(models)
    width = 80 + n_models * (n_configs * bar_w + group_gap) + 40
    height = 60 + label_h + plot_h + 50
    max_kc = 1
    for m in models:
        for c in _T1_CONFIG_NAMES:
            v = model_kernels[m].get(c)
            if v is not None and v > max_kc:
                max_kc = v

    out: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"'
        f' viewBox="0 0 {width} {height}">',
        "<style>",
        "text{font-family:Arial,Helvetica,sans-serif;fill:#111827}",
        ".small{font-size:10px}.axis{font-size:13px}.title{font-size:18px;font-weight:700}",
        "</style>",
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text class="title" x="80" y="32">{escape(title)}</text>',
    ]

    base_y = 60 + label_h
    for i, m in enumerate(models):
        x_base = 80 + i * (n_configs * bar_w + group_gap)
        for j, config_name in enumerate(_T1_CONFIG_NAMES):
            x = x_base + j * bar_w
            kc = model_kernels[m].get(config_name)
            if kc is None:
                continue
            bar_h = int((kc / max_kc) * plot_h) if max_kc else 0
            color = _T1_CONFIG_COLORS.get(config_name, "#6b7280")
            out.append(
                f'<rect x="{x}" y="{base_y + plot_h - bar_h}" width="{bar_w - 2}"'
                f' height="{bar_h}" fill="{color}" opacity="0.85" rx="2"/>'
            )
        out.append(
            f'<text class="small" x="{x_base + n_configs * bar_w // 2}" y="{base_y + plot_h + 14}"'
            f' text-anchor="middle" transform="rotate(-40 {x_base + n_configs * bar_w // 2} {base_y + plot_h + 14})">'
            f'{escape(m)}</text>'
        )

    out.append(
        f'<line x1="80" y1="{base_y + plot_h}" x2="{width - 40}"'
        f' y2="{base_y + plot_h}" stroke="#111827" stroke-width="1"/>'
    )
    for j, config_name in enumerate(_T1_CONFIG_NAMES):
        color = _T1_CONFIG_COLORS.get(config_name, "#6b7280")
        out.append(
            f'<rect x="{80 + j * 130}" y="{base_y - 25}" width="12" height="12" fill="{color}" opacity="0.85"/>'
        )
        out.append(
            f'<text class="small" x="{80 + j * 130 + 16}" y="{base_y - 16}">{escape(config_name)}</text>'
        )

    out.append("</svg>")
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    svg_path.write_text("\n".join(out), encoding="utf-8")


def plot_cross_model_memory(
    csv_path: Path,
    svg_path: Path,
    *,
    title: str = "Cross-Model Peak GPU Memory by T1 Config",
) -> None:
    models: list[str] = []
    model_mem: dict[str, dict[str, float | None]] = {}
    with csv_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("tier") != "t1-stages":
                continue
            model_key = row.get("model", "unknown")
            if model_key not in model_mem:
                model_mem[model_key] = {}
                models.append(model_key)
            mem_str = row.get("peak_memory_bytes", "")
            try:
                mem = float(mem_str) / (1024 * 1024) if mem_str else None
            except ValueError:
                mem = None
            model_mem[model_key][row["config_name"]] = mem

    if not models:
        return

    n_configs = len(_T1_CONFIG_NAMES)
    bar_w = 24
    group_gap = 16
    label_h = 140
    plot_h = 350
    n_models = len(models)
    width = 80 + n_models * (n_configs * bar_w + group_gap) + 40
    height = 60 + label_h + plot_h + 50
    max_mb = 1.0
    for m in models:
        for c in _T1_CONFIG_NAMES:
            v = model_mem[m].get(c)
            if v is not None and v > max_mb:
                max_mb = v

    out: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"'
        f' viewBox="0 0 {width} {height}">',
        "<style>",
        "text{font-family:Arial,Helvetica,sans-serif;fill:#111827}",
        ".small{font-size:10px}.axis{font-size:13px}.title{font-size:18px;font-weight:700}",
        "</style>",
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text class="title" x="80" y="32">{escape(title)}</text>',
    ]

    base_y = 60 + label_h
    for i, m in enumerate(models):
        x_base = 80 + i * (n_configs * bar_w + group_gap)
        for j, config_name in enumerate(_T1_CONFIG_NAMES):
            x = x_base + j * bar_w
            mem = model_mem[m].get(config_name)
            if mem is None:
                continue
            bar_h = int((mem / max_mb) * plot_h) if max_mb else 0
            color = _T1_CONFIG_COLORS.get(config_name, "#6b7280")
            out.append(
                f'<rect x="{x}" y="{base_y + plot_h - bar_h}" width="{bar_w - 2}"'
                f' height="{bar_h}" fill="{color}" opacity="0.85" rx="2"/>'
            )
        out.append(
            f'<text class="small" x="{x_base + n_configs * bar_w // 2}" y="{base_y + plot_h + 14}"'
            f' text-anchor="middle" transform="rotate(-40 {x_base + n_configs * bar_w // 2} {base_y + plot_h + 14})">'
            f'{escape(m)}</text>'
        )

    out.append(
        f'<line x1="80" y1="{base_y + plot_h}" x2="{width - 40}"'
        f' y2="{base_y + plot_h}" stroke="#111827" stroke-width="1"/>'
    )
    for j, config_name in enumerate(_T1_CONFIG_NAMES):
        color = _T1_CONFIG_COLORS.get(config_name, "#6b7280")
        out.append(
            f'<rect x="{80 + j * 130}" y="{base_y - 25}" width="12" height="12" fill="{color}" opacity="0.85"/>'
        )
        out.append(
            f'<text class="small" x="{80 + j * 130 + 16}" y="{base_y - 16}">{escape(config_name)}</text>'
        )

    out.append("</svg>")
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    svg_path.write_text("\n".join(out), encoding="utf-8")
