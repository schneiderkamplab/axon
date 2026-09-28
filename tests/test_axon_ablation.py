from __future__ import annotations

import csv
from pathlib import Path

import pytest

from synapse.axon import GraphOptimizeConfig
from synapse.axon.elaborate import elaborate_closed_axon_file
from synapse.axon.flatten import flatten_closed_axon_file
from synapse.axon.graph_ir import lower_axon_program_to_graph_ir
from synapse.axon.normalize import normalize_closed_axon_file
from synapse.axon.resolve import resolve_axon_program_from_path
from synapse.axon.typecheck2 import typecheck2_flat_axon_file
from synapse.axon_ablation import (
    AblationConfig,
    AblationResult,
    CompilerAblation,
    IRMetrics,
    StageTimings,
    export_ablation_csv,
    plot_ablation_ops_heatmap,
    plot_ablation_runtime,
    plot_ablation_waterfall,
)
from tests.test_flags import run_long_tests_enabled

LONG_TEST_ENV = "BS_LONG"

_GPT2_AXON = Path("synapse/models/gpt2/generic-gpt2.axon")


def _build_graph(axon_path: Path = _GPT2_AXON) -> object:
    resolved = resolve_axon_program_from_path(axon_path).ast
    normalized = normalize_closed_axon_file(resolved)
    elaborated = elaborate_closed_axon_file(normalized)
    flat = flatten_closed_axon_file(elaborated)
    typed = typecheck2_flat_axon_file(flat)
    return lower_axon_program_to_graph_ir(typed)


# ---------------------------------------------------------------------------
# Ablation matrix generation
# ---------------------------------------------------------------------------

class TestAblationMatrix:
    def test_matrix_has_three_tiers(self) -> None:
        abl = CompilerAblation(_GPT2_AXON, Path("models/gpt2"))
        configs = abl.build_ablation_matrix()
        tiers = {c.tier for c in configs}
        assert "t1-stages" in tiers
        assert "t2-per-pass" in tiers
        assert "t3-per-intrinsic" in tiers

    def test_t1_has_four_configs(self) -> None:
        abl = CompilerAblation(_GPT2_AXON, Path("models/gpt2"))
        configs = abl.build_ablation_matrix()
        t1 = [c for c in configs if c.tier == "t1-stages"]
        assert len(t1) == 4
        names = {c.name for c in t1}
        assert "no-opt" in names
        assert "ast-only" in names
        assert "graph-no-intrinsics" in names
        assert "graph-full" in names

    def test_t2_ablates_each_pass_flag(self) -> None:
        abl = CompilerAblation(_GPT2_AXON, Path("models/gpt2"))
        configs = abl.build_ablation_matrix()
        t2 = [c for c in configs if c.tier == "t2-per-pass"]
        assert len(t2) == 8  # 7 pass flags + specialize
        names = {c.name for c in t2}
        assert "no-constant-folding" in names
        assert "no-common-subexpression-elimination" in names
        assert "no-inline-safe" in names
        assert "no-specialize" in names
        assert "no-prune-to-main" in names

    def test_t3_ablates_each_torch_intrinsic(self) -> None:
        abl = CompilerAblation(_GPT2_AXON, Path("models/gpt2"), backend="codegen2-torch")
        configs = abl.build_ablation_matrix()
        t3 = [c for c in configs if c.tier == "t3-per-intrinsic"]
        # Should have one config per torch intrinsic
        assert len(t3) == 20
        names = {c.name for c in t3}
        assert "only-sdpa" in names
        assert "only-swiglu-ffn" in names
        assert "only-rope-apply-factors" in names
        assert "only-rmsnorm-scaled" in names

    def test_t3_empty_for_unknown_backend(self) -> None:
        abl = CompilerAblation(_GPT2_AXON, Path("models/gpt2"), backend="codegen2-jax")
        configs = abl.build_ablation_matrix()
        t3 = [c for c in configs if c.tier == "t3-per-intrinsic"]
        assert len(t3) == 0  # jax intrinsics not in the ablation tool's hardcoded set


# ---------------------------------------------------------------------------
# IR metrics collection
# ---------------------------------------------------------------------------

class TestIRMetrics:
    def test_collect_ir_metrics_returns_counts(self) -> None:
        from synapse.axon_ablation import _collect_ir_metrics

        graph = _build_graph()
        metrics = _collect_ir_metrics(graph)
        assert isinstance(metrics, IRMetrics)
        assert metrics.graph_modules > 0
        assert metrics.graph_nodes > 0
        assert len(metrics.op_distribution) > 0

    def test_op_distribution_is_counter(self) -> None:
        from collections import Counter

        from synapse.axon_ablation import _collect_ir_metrics

        graph = _build_graph()
        metrics = _collect_ir_metrics(graph)
        assert isinstance(metrics.op_distribution, Counter)


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

class TestCSVExport:
    def test_csv_has_correct_columns(self, tmp_path: Path) -> None:
        results = [
            AblationResult(
                config=AblationConfig(
                    name="test-config",
                    tier="t1-stages",
                    description="test",
                    graph_config=GraphOptimizeConfig(),
                ),
                ir_metrics=IRMetrics(
                    graph_modules=5,
                    graph_nodes=42,
                    op_distribution={"_linear": 10, "__torch_sdpa": 2},
                ),
                timings=StageTimings(resolve_s=0.01, lower_s=0.02),
                runtime={"axon_time": 0.005, "hf_time": 0.007},
            ),
        ]
        csv_path = tmp_path / "ablation.csv"
        export_ablation_csv(results, csv_path)
        assert csv_path.exists()
        with csv_path.open() as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 1
        row = rows[0]
        assert row["config_name"] == "test-config"
        assert row["tier"] == "t1-stages"
        assert row["graph_nodes"] == "42"
        assert "runtime_error" in row

    def test_csv_handles_missing_runtime(self, tmp_path: Path) -> None:
        results = [
            AblationResult(
                config=AblationConfig(
                    name="no-opt",
                    tier="t1-stages",
                    description="no opt",
                ),
                ir_metrics=IRMetrics(graph_nodes=100),
                timings=StageTimings(),
                runtime={},
            ),
        ]
        csv_path = tmp_path / "ablation.csv"
        export_ablation_csv(results, csv_path)
        with csv_path.open() as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert rows[0]["forward_time_s"] == ""


# ---------------------------------------------------------------------------
# SVG visualizations
# ---------------------------------------------------------------------------

class TestSVGPlots:
    def _sample_results(self) -> list[AblationResult]:
        return [
            AblationResult(
                config=AblationConfig(
                    name="no-opt", tier="t1-stages", description="d",
                ),
                ir_metrics=IRMetrics(
                    graph_nodes=100,
                    op_distribution={"_linear": 50, "_layernorm": 10},
                ),
                timings=StageTimings(),
                runtime={"axon_time": 0.01, "hf_time": 0.008},
            ),
            AblationResult(
                config=AblationConfig(
                    name="graph-full", tier="t1-stages", description="d",
                    graph_config=GraphOptimizeConfig(),
                ),
                ir_metrics=IRMetrics(
                    graph_nodes=60,
                    op_distribution={"_linear": 30, "__torch_sdpa": 5},
                ),
                timings=StageTimings(optimize_graph_s=0.1),
                runtime={"axon_time": 0.005, "hf_time": 0.008},
            ),
        ]

    def test_waterfall_svg(self, tmp_path: Path) -> None:
        svg_path = tmp_path / "waterfall.svg"
        plot_ablation_waterfall(self._sample_results(), svg_path)
        assert svg_path.exists()
        content = svg_path.read_text()
        assert "<svg" in content
        assert "</svg>" in content

    def test_runtime_svg(self, tmp_path: Path) -> None:
        svg_path = tmp_path / "runtime.svg"
        plot_ablation_runtime(self._sample_results(), svg_path)
        assert svg_path.exists()
        content = svg_path.read_text()
        assert "<svg" in content

    def test_ops_heatmap_svg(self, tmp_path: Path) -> None:
        svg_path = tmp_path / "ops.svg"
        plot_ablation_ops_heatmap(self._sample_results(), svg_path)
        assert svg_path.exists()
        content = svg_path.read_text()
        assert "<svg" in content

    def test_empty_results_skips_svg(self, tmp_path: Path) -> None:
        svg_path = tmp_path / "empty.svg"
        plot_ablation_waterfall([], svg_path)
        assert not svg_path.exists()


# ---------------------------------------------------------------------------
# Cross-model merge + visualization
# ---------------------------------------------------------------------------

class TestCrossModel:
    def _write_csv(self, path: Path, model: str, nodes: int, ratio: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "model", "config_name", "tier", "graph_nodes", "speed_ratio",
            ])
            writer.writeheader()
            configs = [
                ("no-opt", nodes),
                ("ast-only", int(nodes * 0.95)),
                ("graph-no-intrinsics", int(nodes * 0.8)),
                ("graph-full", int(nodes * 0.6)),
            ]
            for config_name, n in configs:
                writer.writerow({
                    "model": model, "config_name": config_name,
                    "tier": "t1-stages", "graph_nodes": n, "speed_ratio": ratio,
                })

    def test_merge_csvs(self, tmp_path: Path) -> None:
        from synapse.axon_ablation import merge_ablation_csvs

        self._write_csv(tmp_path / "a" / "ablation.csv", "gpt2", 100, "0.8")
        self._write_csv(tmp_path / "b" / "ablation.csv", "bert", 80, "0.9")
        merged = tmp_path / "combined.csv"
        merge_ablation_csvs(
            [tmp_path / "a" / "ablation.csv", tmp_path / "b" / "ablation.csv"],
            merged,
        )
        assert merged.exists()
        with merged.open() as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 8  # 4 configs x 2 models
        models = {row["model"] for row in rows}
        assert "gpt2" in models
        assert "bert" in models

    def test_cross_model_nodes_svg(self, tmp_path: Path) -> None:
        from synapse.axon_ablation import plot_cross_model_nodes

        csv_path = tmp_path / "combined.csv"
        self._write_csv(tmp_path / "a" / "ablation.csv", "gpt2", 100, "0.8")
        self._write_csv(tmp_path / "b" / "ablation.csv", "bert", 80, "0.9")
        from synapse.axon_ablation import merge_ablation_csvs

        merge_ablation_csvs(
            [tmp_path / "a" / "ablation.csv", tmp_path / "b" / "ablation.csv"],
            csv_path,
        )
        svg_path = tmp_path / "cross_nodes.svg"
        plot_cross_model_nodes(csv_path, svg_path)
        assert svg_path.exists()
        content = svg_path.read_text()
        assert "<svg" in content
        assert "gpt2" in content

    def test_cross_model_speedup_svg(self, tmp_path: Path) -> None:
        from synapse.axon_ablation import plot_cross_model_speedup

        csv_path = tmp_path / "combined.csv"
        self._write_csv(tmp_path / "a" / "ablation.csv", "gpt2", 100, "0.8")
        self._write_csv(tmp_path / "b" / "ablation.csv", "bert", 80, "0.9")
        from synapse.axon_ablation import merge_ablation_csvs

        merge_ablation_csvs(
            [tmp_path / "a" / "ablation.csv", tmp_path / "b" / "ablation.csv"],
            csv_path,
        )
        svg_path = tmp_path / "cross_speedup.svg"
        plot_cross_model_speedup(csv_path, svg_path)
        assert svg_path.exists()
        content = svg_path.read_text()
        assert "<svg" in content


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

class TestCLI:
    def test_ablation_help_lists_command(self) -> None:
        from typer.testing import CliRunner

        from synapse.cli.synapse import app

        runner = CliRunner()
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "axon-ablation" in result.output

    def test_ablation_list_configs(self) -> None:
        from typer.testing import CliRunner

        from synapse.cli.synapse import app

        runner = CliRunner()
        result = runner.invoke(app, [
            "axon-ablation",
            str(_GPT2_AXON),
            "models/gpt2",
            "--list-configs",
        ])
        assert result.exit_code == 0
        assert "no-opt" in result.output
        assert "t1-stages" in result.output
        assert "Total:" in result.output


# ---------------------------------------------------------------------------
# Integration test (gated by BS_LONG=1, requires model weights)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not run_long_tests_enabled(),
    reason=f"Set {LONG_TEST_ENV}=1 to run integration tests (requires model weights + GPU)",
)
class TestIntegration:
    def test_run_single_no_opt(self) -> None:
        abl = CompilerAblation(
            _GPT2_AXON,
            Path("models/gpt2"),
            device="cpu",
            forward_warmup=0,
            forward_repeat=1,
        )
        config = AblationConfig(
            name="no-opt",
            tier="t1-stages",
            description="No optimization",
        )
        result = abl.run_single(config)
        assert result.ir_metrics.graph_nodes > 0
        assert result.config_name == "no-opt"

    def test_run_single_graph_full(self) -> None:
        abl = CompilerAblation(
            _GPT2_AXON,
            Path("models/gpt2"),
            device="cpu",
            forward_warmup=0,
            forward_repeat=1,
        )
        config = AblationConfig(
            name="graph-full",
            tier="t1-stages",
            description="Full optimization",
            graph_config=GraphOptimizeConfig(backend_intrinsics="codegen2-torch"),
        )
        result = abl.run_single(config)
        assert result.ir_metrics.graph_nodes > 0
        # Full optimization should reduce nodes vs no-opt
        no_opt = abl.run_single(AblationConfig(
            name="no-opt", tier="t1-stages", description="d",
        ))
        assert result.ir_metrics.graph_nodes <= no_opt.ir_metrics.graph_nodes

    def test_full_export(self, tmp_path: Path) -> None:
        abl = CompilerAblation(
            _GPT2_AXON,
            Path("models/gpt2"),
            device="cpu",
            forward_warmup=0,
            forward_repeat=1,
        )
        # Only run T1 to keep it fast
        configs = [c for c in abl.build_ablation_matrix() if c.tier == "t1-stages"]
        results = abl.run(configs)
        paths = abl.export(results, tmp_path)
        assert paths["csv"].exists()
        assert paths["waterfall"].exists()
        assert paths["runtime"].exists()
        assert paths["ops_heatmap"].exists()
