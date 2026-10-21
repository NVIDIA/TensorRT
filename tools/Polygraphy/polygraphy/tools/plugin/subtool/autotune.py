# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA
# SPDX-License-Identifier: Apache-2.0

from polygraphy.tools import Tool
from polygraphy import mod
from polygraphy.mod.trt_importer import lazy_import_trt
from polygraphy.logger import G_LOGGER
from polygraphy.tools.args import OnnxSaveArgs
import os

trt = lazy_import_trt()
yaml = mod.lazy_import("yaml", pkg_name="pyyaml")
onnx = mod.lazy_import("onnx")


class Autotune(Tool):
    """
    Autotune ONNX models by replacing matched subgraphs with TensorRT plugins and
    searching for the best-performing combination.
    """

    def __init__(self):
        super().__init__(name="autotune")

    def get_subscriptions_impl(self):
        return [OnnxSaveArgs(output_opt_required=True)]

    def _load_yaml_config(self, replacement_engine, yaml_path):
        """Load plugin configuration from YAML file."""
        from polygraphy import util

        config_content = util.load_file(
            yaml_path, mode="r", description="plugin config"
        )
        plugin_docs = list(yaml.safe_load_all(config_content))
        replacement_engine.plugin_subgraphs = {}
        for plugin_data in plugin_docs:
            plugin_name = plugin_data.get("name", "unknown")
            instances = plugin_data.get("instances", [])
            replacement_engine.plugin_subgraphs[plugin_name] = []
            for i, instance in enumerate(instances):
                replacement_engine.plugin_subgraphs[plugin_name].append(
                    {
                        "index": i,
                        "inputs": instance.get("inputs", []),
                        "outputs": instance.get("outputs", []),
                        "attributes": instance.get("attributes", {}),
                        "plugin_name": plugin_name,
                        "plugin_op": plugin_data.get("op", plugin_name),
                    }
                )

    def add_parser_args_impl(self, parser):
        # Positional arguments
        parser.add_argument("model", help="Input ONNX model file")

        # Plugin configuration
        plugin_group = parser.add_argument_group(
            "Plugin Configuration",
            "Specify plugin patterns and libraries to use for replacement",
        )
        plugin_group.add_argument(
            "--plugin-dir",
            help="Directory containing plugin pattern definitions (pattern.py files)",
            required=True,
        )
        plugin_group.add_argument(
            "--python-plugins",
            nargs="+",
            help="Python file(s) to import that register TensorRT plugins",
        )
        plugin_group.add_argument(
            "--plugin-libraries",
            nargs="+",
            help="Shared library (.so) plugin file(s) to load",
        )
        include_exclude = plugin_group.add_mutually_exclusive_group()
        include_exclude.add_argument(
            "--include", nargs="+", help="Names of plugins to include"
        )
        include_exclude.add_argument(
            "--exclude", nargs="+", help="Names of plugins to exclude"
        )

        # AutoTuner configuration
        tuner_group = parser.add_argument_group(
            "AutoTuner Configuration",
            "Configure optimization search strategy and outputs",
        )
        tuner_group.add_argument("--report", help="Output optimization report file")
        tuner_group.add_argument(
            "--autotune-mode",
            choices=["greedy", "exhaustive"],
            default="greedy",
            help="AutoTuner search strategy (default: greedy)",
        )
        tuner_group.add_argument(
            "--max-combinations",
            type=int,
            default=1000,
            help="Maximum combinations to test (default: 1000)",
        )
        tuner_group.add_argument(
            "--threshold",
            type=float,
            default=0.1,
            help="Performance improvement threshold (default: 0.1)",
        )
        tuner_group.add_argument(
            "--no-timing-cache",
            action="store_true",
            help="Disable TensorRT timing cache",
        )
        tuner_group.add_argument(
            "--timing-cache",
            help="Path to TensorRT timing cache file (created if missing)",
        )
        tuner_group.add_argument(
            "--save-config-yaml",
            nargs="?",
            const="config.yaml",
            default=None,
            help="Save matched plugin replacement config YAML (default: config.yaml if path omitted)",
        )
        tuner_group.add_argument(
            "--load-config-yaml",
            help="Load a previously saved config.yaml (skip pattern matching)",
        )
        tuner_group.add_argument(
            "--skip-autotuning",
            action="store_true",
            help="Skip optimization; only perform matching and optionally save config YAML",
        )

    def run_impl(self, args):
        from polygraphy.tools.plugin.subtool.autotuner.replacement_engine import (
            ReplacementEngine,
        )
        from polygraphy.tools.plugin.subtool.autotuner.auto_tuner import AutoTuner
        import importlib.util
        import ctypes

        # Import Python plugin modules
        if getattr(args, "python_plugins", None):
            G_LOGGER.info("Importing Python plugin modules...")
            for module_path in args.python_plugins:
                try:
                    if not os.path.isfile(module_path):
                        G_LOGGER.warning(f"  Plugin file not found: {module_path}")
                        continue
                    mod_name = f"polygraphy_plugins.{os.path.splitext(os.path.basename(module_path))[0]}"
                    spec = importlib.util.spec_from_file_location(mod_name, module_path)
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                        G_LOGGER.info(f"  Imported: {os.path.basename(module_path)}")
                except Exception as e:
                    G_LOGGER.warning(f"  Failed to import {module_path}: {e}")

        # Load shared library plugins
        if getattr(args, "plugin_libraries", None):
            G_LOGGER.info("Loading plugin libraries...")
            for lib_path in args.plugin_libraries:
                try:
                    if not os.path.isfile(lib_path):
                        G_LOGGER.warning(f"  Library not found: {lib_path}")
                        continue
                    ctypes.CDLL(lib_path)
                    G_LOGGER.info(f"  Loaded: {os.path.basename(lib_path)}")
                except Exception as e:
                    G_LOGGER.warning(f"  Failed to load {lib_path}: {e}")

        repl = ReplacementEngine()
        repl.load_plugins(
            args.plugin_dir,
            getattr(args, "include", None),
            getattr(args, "exclude", None),
        )

        # Matching (or load from YAML)
        if getattr(args, "load_config_yaml", None):
            self._load_yaml_config(repl, args.load_config_yaml)
        else:
            config_yaml_path = getattr(args, "save_config_yaml", None)
            if config_yaml_path:
                config_dir = os.path.dirname(config_yaml_path)
                if config_dir:
                    os.makedirs(config_dir, exist_ok=True)
            repl.match_subgraphs(args.model, config_yaml_path)

        if getattr(args, "skip_autotuning", False):
            return 0

        # Create TRT builder/config
        builder = trt.Builder(trt.Logger(trt.Logger.INFO))
        config = builder.create_builder_config()
        if getattr(args, "timing_cache", None) and not os.path.exists(
            args.timing_cache
        ):
            try:
                cache_dir = os.path.dirname(args.timing_cache)
                if cache_dir:
                    os.makedirs(cache_dir, exist_ok=True)
                with open(args.timing_cache, "wb") as f:
                    f.write(b"")
            except Exception as e:
                G_LOGGER.warning(f"Failed to create timing cache file: {e}")

        autotuner = AutoTuner(
            builder=builder,
            config=config,
            replacement_engine=repl,
            workspace_size=1 << 32,
            trt_log_level=trt.Logger.INFO,
            enable_timing_cache=not getattr(args, "no_timing_cache", False),
            timing_cache_path=getattr(args, "timing_cache", None),
        )

        result = autotuner.optimize_network(
            input_source=args.model,
            use_greedy=(args.autotune_mode == "greedy"),
            max_combinations=args.max_combinations,
            performance_threshold=args.threshold,
        )

        # Get the optimized model as a temporary file
        optimized_model_path = autotuner.get_optimized_model_path(result)

        try:
            # Load the ONNX model
            model = onnx.load(optimized_model_path)

            # Use OnnxSaveArgs to save the model (handles external data, large models, etc.)
            self.arg_groups[OnnxSaveArgs].save_onnx(model)

            G_LOGGER.info(
                f"Optimized model saved to: {self.arg_groups[OnnxSaveArgs].path}"
            )
        finally:
            # Clean up temporary file
            if optimized_model_path and os.path.exists(optimized_model_path):
                try:
                    os.remove(optimized_model_path)
                except Exception as e:
                    G_LOGGER.warning(
                        f"Failed to remove temporary file {optimized_model_path}: {e}"
                    )

        # Clean up timing cache if it was temporary
        if not autotuner.timing_cache_path:
            autotuner.cleanup_timing_cache()

        if getattr(args, "report", None):
            try:
                autotuner.save_optimization_report(result, args.report)
                G_LOGGER.info(f"Optimization report saved to: {args.report}")
            except Exception as e:
                G_LOGGER.warning(f"Failed to save report: {e}")

        G_LOGGER.finish("Optimization completed successfully!")
        return 0
