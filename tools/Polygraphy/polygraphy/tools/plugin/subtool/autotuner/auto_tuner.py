"""
AutoTuner entry that orchestrates plugin pattern matching, performance analysis,
and strategy search.
"""

import os
import shutil
import tempfile
import time
from typing import Optional, Any

from polygraphy import mod, util
from polygraphy.mod.trt_importer import lazy_import_trt
from polygraphy.logger import G_LOGGER

trt = lazy_import_trt()

from .performance_analyzer import PerformanceAnalyzer, PerformanceMetrics
from .optimization_strategy import OptimizationStrategy, OptimizationResult


class AutoTuner:

    def __init__(
        self,
        builder: Optional[Any] = None,
        config: Optional[Any] = None,
        replacement_engine: Optional[Any] = None,
        workspace_size: int = 1 << 30,
        trt_log_level: Optional[int] = None,
        enable_timing_cache: bool = True,
        timing_cache_path: Optional[str] = None,
    ):
        # Lazily create builder/config if not provided
        if builder is None or config is None:
            # Import here to avoid module import-time dependencies
            from polygraphy.backend.trt import get_trt_logger

            if builder is None:
                builder = trt.Builder(get_trt_logger())
            if config is None:
                config = builder.create_builder_config()

        self.builder = builder
        self.config = config
        self.workspace_size = workspace_size
        self.trt_log_level = (
            trt_log_level if trt_log_level is not None else trt.Logger.INFO
        )
        self.enable_timing_cache = enable_timing_cache
        self.timing_cache_path = timing_cache_path
        self._temp_timing_cache_path = None

        self.replacement_engine = replacement_engine
        self.performance_analyzer = PerformanceAnalyzer(
            builder, config, workspace_size, trt_log_level
        )
        self.optimization_strategy = OptimizationStrategy()
        self.optimization_strategy.set_components(replacement_engine, builder)

        if self.enable_timing_cache:
            self._setup_timing_cache()

    def _setup_timing_cache(self):
        try:
            if self.timing_cache_path:
                if os.path.exists(self.timing_cache_path):
                    cache_data = util.load_file(
                        self.timing_cache_path, description="timing cache"
                    )
                    cache = self.config.create_timing_cache(cache_data)
                else:
                    cache = self.config.create_timing_cache(b"")
            else:
                fd, self._temp_timing_cache_path = tempfile.mkstemp(
                    suffix=".trt_cache", prefix="auto_tuner_"
                )
                os.close(fd)
                cache = self.config.create_timing_cache(b"")
            self.config.set_timing_cache(cache, ignore_mismatch=False)
        except Exception as e:
            G_LOGGER.warning(f"Failed to setup timing cache: {e}")
            self.enable_timing_cache = False

    def _save_timing_cache(self):
        if not self.enable_timing_cache:
            return
        try:
            cache = self.config.get_timing_cache()
            if cache:
                data = cache.serialize()
                path = self.timing_cache_path or self._temp_timing_cache_path
                if path:
                    util.save_file(data, path, description="timing cache")
        except Exception as e:
            G_LOGGER.warning(f"Failed to save timing cache: {e}")

    def cleanup_timing_cache(self):
        """Clean up temporary timing cache file if it exists."""
        if self._temp_timing_cache_path and os.path.exists(
            self._temp_timing_cache_path
        ):
            try:
                os.remove(self._temp_timing_cache_path)
            except Exception as e:
                G_LOGGER.warning(f"Failed to cleanup timing cache: {e}")

    def optimize_network(
        self,
        input_source: str,
        use_greedy: bool = True,
        max_combinations: int = 1000,
        performance_threshold: float = 0.1,
        save_config_yaml: Optional[str] = None,
        load_config_yaml: Optional[str] = None,
    ) -> OptimizationResult:
        start_time = time.time()

        result = OptimizationResult()
        result.input_source = input_source
        if not isinstance(input_source, str) or not input_source.endswith(".onnx"):
            G_LOGGER.critical("Input must be an ONNX file")

        # Baseline network
        baseline_network = self.optimization_strategy._create_network_from_onnx(
            input_source
        )

        # Baseline metrics
        baseline_metrics = self.performance_analyzer.analyze_network(baseline_network)
        result.baseline_metrics = baseline_metrics

        # Log baseline performance
        G_LOGGER.info(
            f"Baseline Performance (without plugin replacements): {baseline_metrics}"
        )

        possible_replacements = {}
        if self.replacement_engine is not None:
            already_populated = bool(
                getattr(self.replacement_engine, "plugin_subgraphs", {})
            )
            if not already_populated:
                if load_config_yaml:
                    self.replacement_engine._load_config_from_yaml(load_config_yaml)
                else:
                    self.replacement_engine.match_subgraphs(
                        input_source, save_config_yaml
                    )

            all_subgraphs = self.replacement_engine.get_all_subgraphs()
            for plugin_name, subgraphs in all_subgraphs.items():
                for subgraph in subgraphs:
                    pattern_name = f"{plugin_name}_{subgraph['index']}"
                    replacement = {
                        "pattern_name": pattern_name,
                        "plugin_name": plugin_name,
                        "subgraph_index": subgraph["index"],
                    }
                    replacement.update(subgraph)
                    possible_replacements.setdefault(pattern_name, []).append(
                        replacement
                    )

        self.optimization_strategy.max_combinations = max_combinations
        self.optimization_strategy.performance_threshold = performance_threshold
        self.optimization_strategy.set_input_source(input_source)

        if use_greedy:
            opt_result = self.optimization_strategy.greedy_optimization(
                possible_replacements, baseline_metrics, self.performance_analyzer
            )
        else:
            opt_result = self.optimization_strategy.find_optimal_strategy(
                possible_replacements, baseline_metrics, self.performance_analyzer
            )

        opt_result.baseline_metrics = baseline_metrics
        opt_result.total_time = time.time() - start_time
        if self.enable_timing_cache:
            self._save_timing_cache()
        return opt_result

    def get_optimized_model_path(self, result: OptimizationResult) -> Optional[str]:
        """
        Get the path to the optimized ONNX model (as a temporary file).
        The caller is responsible for loading and saving the model properly,
        and cleaning up the temporary file.

        Args:
            result: OptimizationResult containing the best strategy

        Returns:
            Path to the temporary optimized ONNX model, or None if optimization failed
        """
        if not result.best_strategy:
            G_LOGGER.critical("No optimization strategy found")

        optimized_onnx_path = self.optimization_strategy._apply_onnx_replacements(
            result.best_strategy, return_network=False
        )
        if not optimized_onnx_path or not os.path.exists(optimized_onnx_path):
            G_LOGGER.critical("Failed to generate optimized ONNX model")

        return optimized_onnx_path

    def save_optimization_report(self, result: OptimizationResult, report_path: str):
        """Save optimization report to the given path.

        Args:
            result: OptimizationResult containing performance metrics and strategy
            report_path: Path where the report should be saved
        """
        lines = []
        lines.append("TRT Plugin Operator Auto-tuning Report\n")
        lines.append("=" * 50 + "\n\n")
        if result.total_time:
            lines.append(f"Optimization Time: {result.total_time:.2f}s\n")
        lines.append(f"Total Combinations Tested: {len(result.all_results)}\n")
        lines.append(f"Number of Engine Builds: {result.num_engine_builds}\n")
        if result.total_compile_time:
            lines.append(f"Total Compilation Time: {result.total_compile_time:.2f}s\n")
        if result.baseline_metrics:
            lines.append("\nBaseline Performance:\n")
            lines.append(f"  {result.baseline_metrics}\n")
        if result.best_metrics:
            lines.append("\nBest Performance:\n")
            lines.append(f"  {result.best_metrics}\n")
        if result.best_strategy:
            lines.append(
                f"\nBest Strategy ({len(result.best_strategy)} replacements):\n"
            )
            for i, replacement in enumerate(result.best_strategy):
                plugin_name = replacement.get("plugin_name", "Unknown")
                plugin_op = replacement.get("plugin_op", "Unknown")
                inputs = replacement.get("inputs", [])
                outputs = replacement.get("outputs", [])
                attributes = replacement.get("attributes", {})
                lines.append(f"  {i+1}. {plugin_op} ({plugin_name})\n")
                lines.append(f"     Inputs: {', '.join(inputs) if inputs else 'N/A'}\n")
                lines.append(
                    f"     Outputs: {', '.join(outputs) if outputs else 'N/A'}\n"
                )
                if attributes:
                    lines.append(f"     Attributes: {attributes}\n")

        report_content = "".join(lines)
        util.save_file(
            report_content, report_path, mode="w", description="optimization report"
        )
