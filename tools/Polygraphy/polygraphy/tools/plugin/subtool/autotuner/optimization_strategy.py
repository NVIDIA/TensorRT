"""
Optimization strategy and result types for plugin autotuning.
"""

import os
import tempfile
import time
from typing import Dict, List, Any, Optional, Tuple, Union

from polygraphy import mod, util
from polygraphy.mod.trt_importer import lazy_import_trt
from polygraphy.logger import G_LOGGER

trt = lazy_import_trt()

from .performance_analyzer import PerformanceAnalyzer, PerformanceMetrics


class OptimizationResult:

    def __init__(self):
        self.best_strategy: List[Dict[str, Any]] = []
        self.best_metrics: Optional[PerformanceMetrics] = None
        self.baseline_metrics: Optional[PerformanceMetrics] = None
        self.all_results: List[Tuple[List[Dict[str, Any]], PerformanceMetrics]] = []
        self.optimization_time: float = 0.0
        self.total_time: float = 0.0
        self.total_compile_time: float = 0.0
        self.num_engine_builds: int = 0
        self.num_combinations_tested: int = 0
        self.performance_improvement: float = 0.0
        self.input_source: Optional[str] = None

    def __str__(self):
        return (
            f"Best strategy: {len(self.best_strategy)} replacements, "
            f"Performance: {self.best_metrics}"
        )


class OptimizationStrategy:

    def __init__(
        self, max_combinations: int = 1000, performance_threshold: float = 0.0
    ):
        self.max_combinations = max_combinations
        self.performance_threshold = performance_threshold
        self.replacement_engine = None
        self.builder: Optional[Any] = None
        self.input_source: Optional[str] = None

    def set_components(self, replacement_engine, builder):
        self.replacement_engine = replacement_engine
        self.builder = builder

    def set_input_source(self, input_source: str):
        self.input_source = input_source

    def _update_compile_stats(self, result: OptimizationResult, performance_analyzer):
        """Update compile time statistics in the result."""
        if hasattr(performance_analyzer, "get_compile_time_stats"):
            compile_time, num_builds = performance_analyzer.get_compile_time_stats()
            result.total_compile_time = compile_time
            result.num_engine_builds = num_builds

    def _calculate_performance_improvement(
        self, baseline: PerformanceMetrics, optimized: PerformanceMetrics
    ) -> float:
        """Calculate performance improvement percentage."""
        if baseline.latency_ms > 0 and optimized.latency_ms > 0:
            return (
                (baseline.latency_ms - optimized.latency_ms) / baseline.latency_ms * 100
            )
        return 0.0

    def _create_network_from_onnx(self, onnx_path: str):
        if not onnx_path.endswith(".onnx"):
            G_LOGGER.critical("Input must be an ONNX file")

        network = self.builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt.OnnxParser(network, self.builder.logger)
        model_data = util.load_file(onnx_path, description="ONNX model")
        if not parser.parse(model_data):
            errors = [
                f"ONNX parser error {idx}: {parser.get_error(idx)}"
                for idx in range(parser.num_errors)
            ]
            G_LOGGER.critical("Failed to parse ONNX model:\n" + "\n".join(errors))
        return network

    def _apply_onnx_replacements(
        self, replacements: List[Dict[str, Any]], return_network: bool = False
    ) -> Optional[Union[str, Any]]:
        temp_file = None
        config_temp_file = None
        temp_file_path = None
        try:
            if not (self.input_source and self.input_source.endswith(".onnx")):
                G_LOGGER.critical("ONNX replacements only work with ONNX files")

            temp_file = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False)
            temp_file_path = temp_file.name
            temp_file.close()

            if self.replacement_engine is not None and replacements:
                config_temp_file = tempfile.NamedTemporaryFile(
                    suffix=".yaml", delete=False
                )
                config_temp_file.close()
                selected_subgraphs = [r for r in replacements if r]
                if selected_subgraphs:
                    self.replacement_engine.generate_config_for_subgraphs(
                        selected_subgraphs, config_temp_file.name
                    )
                    self.replacement_engine.replace_subgraphs(
                        self.input_source, config_temp_file.name, temp_file_path
                    )
                    if return_network:
                        network = self._create_network_from_onnx(temp_file_path)
                        return network
                    return temp_file_path

            G_LOGGER.critical("No valid replacements to apply")
        finally:
            # Always clean up config temp file
            if config_temp_file and os.path.exists(config_temp_file.name):
                try:
                    os.remove(config_temp_file.name)
                except Exception as e:
                    G_LOGGER.warning(f"Failed to remove temporary config file: {e}")

            # Clean up temp ONNX file only when returning network object
            # If returning file path, caller is responsible for cleanup
            if return_network and temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                except Exception as e:
                    G_LOGGER.warning(f"Failed to remove temporary ONNX file: {e}")

    def find_optimal_strategy(
        self,
        possible_replacements: Dict[str, List[Dict[str, Any]]],
        baseline_metrics: PerformanceMetrics,
        performance_analyzer,
    ) -> OptimizationResult:
        """Find the optimal combination of replacements."""
        start_time = time.time()

        result = OptimizationResult()
        result.input_source = self.input_source

        if not possible_replacements:
            G_LOGGER.warning("No possible replacements found")
            return result

        combinations = self._generate_combinations(possible_replacements)
        G_LOGGER.info(f"Testing {len(combinations)} replacement combinations")

        for i, combination in enumerate(combinations):
            try:
                network = self._apply_onnx_replacements(
                    combination, return_network=True
                )
                if network is None:
                    continue

                metrics = performance_analyzer.analyze_network(network)
                result.all_results.append((combination, metrics))

                if self._is_better_performance(metrics, result.best_metrics):
                    result.best_strategy = combination
                    result.best_metrics = metrics
                    G_LOGGER.info(
                        f"New best strategy found (combination {i+1}): {metrics}"
                    )

                if (
                    result.best_metrics
                    and baseline_metrics.latency_ms > 0
                    and (baseline_metrics.latency_ms - result.best_metrics.latency_ms)
                    / baseline_metrics.latency_ms
                    > self.performance_threshold
                ):
                    G_LOGGER.verbose("Performance threshold reached, stopping early")
                    break

            except Exception as e:
                G_LOGGER.warning(f"Failed to test combination {i+1}: {e}")
                continue

        result.optimization_time = time.time() - start_time
        result.num_combinations_tested = len(combinations)
        self._update_compile_stats(result, performance_analyzer)

        if result.best_strategy:
            G_LOGGER.info(f"Optimization completed in {result.optimization_time:.2f}s")
            G_LOGGER.info(f"Best strategy: {len(result.best_strategy)} replacements")
            improvement_str = self._calculate_improvement(
                baseline_metrics, result.best_metrics
            )
            G_LOGGER.info(f"Performance improvement: {improvement_str}")
            result.performance_improvement = self._calculate_performance_improvement(
                baseline_metrics, result.best_metrics
            )
        else:
            G_LOGGER.warning("No valid strategy found")

        return result

    def _generate_combinations(
        self, replacements: Dict[str, List[Dict[str, Any]]]
    ) -> List[List[Dict[str, Any]]]:
        """Generate all possible combinations of replacements (2^n combinations for n plugin layers)."""
        combinations = [[]]  # Start with empty combination (baseline)
        subgraph_patterns = list(replacements.keys())

        for i in range(1, 2 ** len(subgraph_patterns)):
            combination = []
            for j, pattern_name in enumerate(subgraph_patterns):
                if i & (1 << j):
                    combination.extend(replacements[pattern_name])

            combinations.append(combination)

            if len(combinations) >= self.max_combinations:
                break

        return combinations

    def _is_better_performance(
        self, current: PerformanceMetrics, best: Optional[PerformanceMetrics]
    ) -> bool:
        """Check if current performance is better than best."""
        if best is None:
            return True

        if current.latency_ms < best.latency_ms:
            return True
        elif current.latency_ms > best.latency_ms:
            return False

        if current.memory_usage_mb < best.memory_usage_mb:
            return True
        elif current.memory_usage_mb > best.memory_usage_mb:
            return False

        return current.throughput_fps > best.throughput_fps

    def _calculate_improvement(
        self, baseline: PerformanceMetrics, optimized: PerformanceMetrics
    ) -> str:
        """Calculate performance improvement as a string."""
        if baseline.latency_ms <= 0:
            return "N/A"

        latency_improvement = (
            (baseline.latency_ms - optimized.latency_ms) / baseline.latency_ms * 100
        )
        memory_improvement = (
            (baseline.memory_usage_mb - optimized.memory_usage_mb)
            / baseline.memory_usage_mb
            * 100
            if baseline.memory_usage_mb > 0
            else 0
        )

        return (
            f"Latency: {latency_improvement:+.1f}%, "
            f"Memory: {memory_improvement:+.1f}%"
        )

    def greedy_optimization(
        self,
        possible_replacements: Dict[str, List[Dict[str, Any]]],
        baseline_metrics: PerformanceMetrics,
        performance_analyzer,
    ) -> OptimizationResult:
        """Greedy optimization: test each plugin layer individually, decide replace or not."""
        start_time = time.time()

        result = OptimizationResult()
        result.input_source = self.input_source
        current_strategy = []
        current_metrics = baseline_metrics

        subgraph_patterns = list(possible_replacements.keys())
        G_LOGGER.info(
            f"Starting greedy optimization with {len(subgraph_patterns)} subgraph patterns"
        )

        for pattern_name in subgraph_patterns:
            G_LOGGER.verbose(f"Testing subgraph pattern: {pattern_name}")

            already_applied = any(
                replacement.get("pattern_name", "") == pattern_name
                for replacement in current_strategy
            )

            if already_applied:
                G_LOGGER.verbose(f"Pattern {pattern_name} already applied, skipping")
                continue

            metrics_no_replace = current_metrics
            test_strategy = current_strategy + possible_replacements[pattern_name]
            try:
                network_with_replace = self._apply_onnx_replacements(
                    test_strategy, return_network=True
                )
                if network_with_replace is not None:
                    metrics_with_replace = performance_analyzer.analyze_network(
                        network_with_replace
                    )
                    G_LOGGER.verbose(
                        f"With replacement performance: {metrics_with_replace}"
                    )
                else:
                    metrics_with_replace = current_metrics
            except Exception as e:
                G_LOGGER.warning(f"Failed to test replacement for {pattern_name}: {e}")
                metrics_with_replace = current_metrics

            if self._is_better_performance(metrics_with_replace, metrics_no_replace):
                current_strategy = test_strategy
                current_metrics = metrics_with_replace
                G_LOGGER.info(
                    f"Decided to replace {pattern_name}: {metrics_with_replace}"
                )
            else:
                G_LOGGER.info(
                    f"Decided not to replace {pattern_name}: {metrics_no_replace}"
                )

        result.best_strategy = current_strategy
        result.best_metrics = current_metrics
        result.optimization_time = time.time() - start_time
        result.all_results = [(current_strategy, current_metrics)]
        result.num_combinations_tested = len(subgraph_patterns)
        self._update_compile_stats(result, performance_analyzer)

        result.performance_improvement = self._calculate_performance_improvement(
            baseline_metrics, result.best_metrics
        )

        G_LOGGER.info(
            f"Greedy optimization completed in {result.optimization_time:.2f}s"
        )
        G_LOGGER.info(f"Final strategy: {len(result.best_strategy)} replacements")
        return result
