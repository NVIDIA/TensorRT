#!/usr/bin/env python3
"""
Demo for TensorRT timing cache performance comparison.
Shows how to use the timing cache feature to speed up TensorRT engine building
by caching and reusing kernel timing information.
Updated to use Polygraphy modules directly.
"""

import os
import sys
import argparse
import tempfile
import shutil
import time

from polygraphy import mod

trt = mod.lazy_import("tensorrt")
onnx = mod.lazy_import("onnx")
from polygraphy.logger import G_LOGGER

from polygraphy.tools.plugin.subtool.autotuner.replacement_engine import (
    ReplacementEngine,
)
from polygraphy.tools.plugin.subtool.autotuner.auto_tuner import AutoTuner

# Register the ConvReluPlugin with TensorRT
# Add the parent directory (plugin_autotuner) to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import plugins.trt_conv_relu_plugin as trt_conv_relu_plugin


def get_patterns_directory():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(os.path.dirname(current_dir), "patterns")


def create_large_conv_relu_model(output_path: str, num_groups: int = 20):
    """Create a large ONNX model with multiple Conv+ReLU layers for timing cache demo."""
    try:
        import onnx
        from onnx import helper, numpy_helper, TensorProto
        import numpy as np

        # Model parameters (smaller than conv_relu_demo for faster demo)
        batch_size = 1
        input_channels = 64
        output_channels = 64
        height = 128
        width = 128
        kernel_size = [3, 3]
        stride = [1, 1]
        padding = [1, 1, 1, 1]  # [top, left, bottom, right]
        dilation = [1, 1]
        groups = 1

        # Create input
        input_name = "input"
        input_shape = [batch_size, input_channels, height, width]
        input_tensor = helper.make_tensor_value_info(
            input_name, TensorProto.FLOAT, input_shape
        )

        # Create graph nodes and initializers
        nodes = []
        initializers = []
        current_input = input_name

        for group_idx in range(num_groups):
            # Create weights for this group
            kernel_shape = [output_channels, input_channels] + kernel_size
            kernel_data = np.random.randn(*kernel_shape).astype(np.float32) * 0.1
            kernel_name = f"conv_kernel_{group_idx}"
            kernel_initializer = numpy_helper.from_array(kernel_data, name=kernel_name)
            initializers.append(kernel_initializer)

            bias_shape = [output_channels]
            bias_data = np.random.randn(*bias_shape).astype(np.float32) * 0.1
            bias_name = f"conv_bias_{group_idx}"
            bias_initializer = numpy_helper.from_array(bias_data, name=bias_name)
            initializers.append(bias_initializer)

            # Create Conv node
            conv_output = f"conv_output_{group_idx}"
            conv_node = helper.make_node(
                "Conv",
                inputs=[current_input, kernel_name, bias_name],
                outputs=[conv_output],
                name=f"conv{group_idx}",
                kernel_shape=kernel_size,
                strides=stride,
                pads=padding,
                dilations=dilation,
                group=groups,
            )
            nodes.append(conv_node)

            # Create ReLU node
            relu_output = f"relu_output_{group_idx}"
            relu_node = helper.make_node(
                "Relu",
                inputs=[conv_output],
                outputs=[relu_output],
                name=f"relu{group_idx}",
            )
            nodes.append(relu_node)

            # Add Abs node between groups (except after the last group)
            if group_idx < num_groups - 1:
                abs_output = f"abs_output_{group_idx}"
                abs_node = helper.make_node(
                    "Abs",
                    inputs=[relu_output],
                    outputs=[abs_output],
                    name=f"abs{group_idx}",
                )
                nodes.append(abs_node)
                current_input = abs_output
            else:
                current_input = relu_output

        # Create output tensor (same shape as input)
        output_tensor = helper.make_tensor_value_info(
            current_input, TensorProto.FLOAT, input_shape
        )

        # Create graph
        graph = helper.make_graph(
            nodes,
            "large_conv_relu_model",
            [input_tensor],
            [output_tensor],
            initializers,
        )

        # Create model
        model = helper.make_model(
            graph, producer_name="PluginAutotuner_TimingCache_Demo"
        )
        model.opset_import[0].version = 11

        # Save model
        onnx.save(model, output_path)
        G_LOGGER.info(
            f"Created large Conv+ReLU model with {num_groups} groups: {output_path}"
        )
        return True

    except Exception as e:
        G_LOGGER.error(f"Failed to create model: {e}")
        return False


def run_optimization_with_timing_cache_mode(
    input_model: str,
    cache_mode: str = "none",
    cache_path: str = None,
    verbose: bool = False,
):
    """Run optimization with different timing cache modes."""
    try:
        # Get patterns directory
        patterns_dir = get_patterns_directory()
        G_LOGGER.verbose(f"Using patterns directory: {patterns_dir}")

        # Initialize pattern-based replacement engine
        replacement_engine = ReplacementEngine()
        replacement_engine.load_plugins(patterns_dir)

        # Initialize TensorRT components
        trt_log_level = trt.Logger.VERBOSE if verbose else trt.Logger.INFO
        builder = trt.Builder(trt.Logger(trt_log_level))
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 32)  # 4GB

        # Configure timing cache based on mode
        timing_cache_path = None
        enable_timing_cache = False

        if cache_mode == "none":
            G_LOGGER.info("Running WITHOUT timing cache")
            enable_timing_cache = False
        elif cache_mode == "temp":
            G_LOGGER.info("Running WITH temporary timing cache")
            enable_timing_cache = True
            timing_cache_path = None  # AutoTuner will create temp cache
        elif cache_mode == "given":
            if cache_path and os.path.exists(cache_path):
                G_LOGGER.info(f"Running WITH given timing cache: {cache_path}")
                enable_timing_cache = True
                timing_cache_path = cache_path
            else:
                G_LOGGER.info("Given cache file not found, falling back to temp cache")
                enable_timing_cache = True
                timing_cache_path = None

        # Initialize AutoTuner with timing cache settings
        autotuner = AutoTuner(
            builder=builder,
            config=config,
            replacement_engine=replacement_engine,
            enable_timing_cache=enable_timing_cache,
            timing_cache_path=timing_cache_path,
        )

        # Run optimization
        start_time = time.time()
        result = autotuner.optimize_network(
            input_source=input_model,
            use_greedy=True,
            max_combinations=5,  # Limit for demo
            performance_threshold=0.001,
        )
        end_time = time.time()

        # Get timing cache info if available
        cache_info = None
        if enable_timing_cache:
            # Check if we have a temp cache path
            temp_cache_path = getattr(autotuner, "_temp_timing_cache_path", None)
            if temp_cache_path and os.path.exists(temp_cache_path):
                cache_size = os.path.getsize(temp_cache_path)
                cache_info = {"path": temp_cache_path, "size": cache_size}
            # Also check if we have a given cache path
            elif timing_cache_path and os.path.exists(timing_cache_path):
                cache_size = os.path.getsize(timing_cache_path)
                cache_info = {"path": timing_cache_path, "size": cache_size}

        return {
            "result": result,
            "total_time": end_time - start_time,
            "compile_time": getattr(result, "total_compile_time", 0),
            "num_builds": getattr(result, "num_engine_builds", 0),
            "cache_mode": cache_mode,
            "cache_info": cache_info,
        }

    except Exception as e:
        G_LOGGER.error(
            f"Error in optimization with timing cache mode '{cache_mode}': {e}"
        )
        import traceback

        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(
        description="TensorRT timing cache performance demo"
    )
    parser.add_argument(
        "--input", "-i", help="Input ONNX model path (will create one if not provided)"
    )
    parser.add_argument(
        "--num-groups",
        "-g",
        type=int,
        default=20,
        help="Number of Conv+ReLU groups in the model (default: 20)",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    # Validate num_groups parameter
    if args.num_groups < 1:
        G_LOGGER.error("Number of groups must be at least 1")
        return 1

    # Set Polygraphy logger verbosity
    if args.verbose:
        G_LOGGER.module_severity = G_LOGGER.VERBOSE
    else:
        G_LOGGER.module_severity = G_LOGGER.INFO

    # Create input model if not provided
    if not args.input:
        args.input = "large_conv_relu_model.onnx"
        if not create_large_conv_relu_model(args.input, args.num_groups):
            G_LOGGER.error("Failed to create input model")
            return 1

    G_LOGGER.info("=" * 60)
    G_LOGGER.info("TensorRT Timing Cache Performance Demo")
    G_LOGGER.info(f"Model: {args.num_groups} Conv+ReLU groups")
    G_LOGGER.info("=" * 60)

    # Step 1: Run optimization WITHOUT timing cache
    G_LOGGER.info("1. Running optimization WITHOUT timing cache...")
    result_without_cache = run_optimization_with_timing_cache_mode(
        args.input, cache_mode="none", verbose=args.verbose
    )

    if result_without_cache is None:
        G_LOGGER.error("Failed to run optimization without cache")
        return 1

    # Step 2: Run optimization WITH temporary timing cache
    G_LOGGER.info("2. Running optimization WITH temporary timing cache...")
    result_with_temp_cache = run_optimization_with_timing_cache_mode(
        args.input, cache_mode="temp", verbose=args.verbose
    )

    if result_with_temp_cache is None:
        G_LOGGER.error("Failed to run optimization with temp cache")
        return 1

    # Step 3: Run optimization WITH given timing cache (using temp cache from step 2)
    G_LOGGER.info("3. Running optimization WITH given timing cache...")
    given_cache_path = "persistent_cache.trt"

    # Copy temp cache to persistent location if it exists
    if result_with_temp_cache["cache_info"]:
        temp_cache_path = result_with_temp_cache["cache_info"]["path"]
        if os.path.exists(temp_cache_path):
            shutil.copy2(temp_cache_path, given_cache_path)
            G_LOGGER.info(
                f"Copied temp cache to persistent location: {given_cache_path}"
            )
        else:
            G_LOGGER.warning(f"Temp cache file not found at: {temp_cache_path}")
            with open(given_cache_path, "wb") as f:
                f.write(b"")
            G_LOGGER.info(f"Created empty cache file for third run: {given_cache_path}")
    else:
        G_LOGGER.warning("No cache info available from temp cache run")
        with open(given_cache_path, "wb") as f:
            f.write(b"")
        G_LOGGER.info(f"Created empty cache file for third run: {given_cache_path}")

    result_with_given_cache = run_optimization_with_timing_cache_mode(
        args.input,
        cache_mode="given",
        cache_path=given_cache_path,
        verbose=args.verbose,
    )

    if result_with_given_cache is None:
        G_LOGGER.error("Failed to run optimization with given cache")
        return 1

    # Step 4: Display comparison results
    G_LOGGER.info("4. Timing Cache Performance Comparison:")
    G_LOGGER.info("=" * 60)

    # Collect all results
    results = [
        ("Without Cache", result_without_cache),
        ("Temp Cache", result_with_temp_cache),
        ("Given Cache", result_with_given_cache),
    ]

    # Total time comparison
    G_LOGGER.info("Total Optimization Time:")
    baseline_time = result_without_cache["total_time"]
    for name, result in results:
        time_taken = result["total_time"]
        if name == "Without Cache":
            G_LOGGER.info(f"  {name}: {time_taken:.2f} seconds (baseline)")
        else:
            time_savings = ((baseline_time - time_taken) / baseline_time) * 100
            speedup = baseline_time / time_taken if time_taken > 0 else 0
            G_LOGGER.info(
                f"  {name}: {time_taken:.2f} seconds (savings: {time_savings:+.1f}%, speedup: {speedup:.2f}x)"
            )

    # Compile time comparison (if available)
    if all(r["compile_time"] for _, r in results):
        G_LOGGER.info("")
        G_LOGGER.info("Engine Compilation Time:")
        baseline_compile = result_without_cache["compile_time"]
        for name, result in results:
            compile_time = result["compile_time"]
            if name == "Without Cache":
                G_LOGGER.info(f"  {name}: {compile_time:.2f} seconds (baseline)")
            else:
                compile_savings = (
                    ((baseline_compile - compile_time) / baseline_compile) * 100
                    if baseline_compile > 0
                    else 0
                )
                compile_speedup = (
                    baseline_compile / compile_time if compile_time > 0 else 0
                )
                G_LOGGER.info(
                    f"  {name}: {compile_time:.2f} seconds (savings: {compile_savings:+.1f}%, speedup: {compile_speedup:.2f}x)"
                )

    # Number of builds comparison (if available)
    if all(r["num_builds"] for _, r in results):
        G_LOGGER.info("")
        G_LOGGER.info("Engine Builds:")
        for name, result in results:
            num_builds = result["num_builds"]
            compile_time = result["compile_time"]
            if compile_time and num_builds > 0:
                avg_time = compile_time / num_builds
                G_LOGGER.info(
                    f"  {name}: {num_builds} builds (avg: {avg_time:.2f}s per build)"
                )
            else:
                G_LOGGER.info(f"  {name}: {num_builds} builds")

    # Cache information
    G_LOGGER.info("")
    G_LOGGER.info("Cache Information:")
    for name, result in results:
        if result["cache_info"]:
            cache_size = result["cache_info"]["size"]
            G_LOGGER.info(f"  {name}: {cache_size / 1024:.1f} KB")
        else:
            G_LOGGER.info(f"  {name}: No cache")

    # Persistent cache file information
    if os.path.exists(given_cache_path):
        cache_size = os.path.getsize(given_cache_path)
        G_LOGGER.info("")
        G_LOGGER.info("Persistent Cache File:")
        G_LOGGER.info(f"  Path: {given_cache_path}")
        G_LOGGER.info(f"  Size: {cache_size / 1024:.1f} KB")

    G_LOGGER.info("=" * 60)
    G_LOGGER.info("Timing cache demo completed successfully!")
    G_LOGGER.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
