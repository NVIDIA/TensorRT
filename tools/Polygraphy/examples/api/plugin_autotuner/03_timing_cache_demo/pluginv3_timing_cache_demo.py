#!/usr/bin/env python3
"""
Demo for TensorRT PluginV3 timing cache performance comparison.
Shows how timing cache works with plugins that have multiple tactics.
Updated to use Polygraphy modules directly.
"""

import os
import sys
import argparse
import time
import numpy as np

from polygraphy import mod

trt = mod.lazy_import("tensorrt")
onnx = mod.lazy_import("onnx")
from polygraphy.logger import G_LOGGER

# Register the ConvReluPlugin with TensorRT
# Add the parent directory (plugin_autotuner) to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import plugins.trt_conv_relu_plugin as trt_conv_relu_plugin


def create_plugin_test_model(
    output_path: str, num_groups: int = 10, use_plugins: bool = True
):
    """Create a test model with Conv+ReLU layers, optionally using plugins."""
    try:
        import onnx
        from onnx import helper, numpy_helper, TensorProto

        # Model parameters
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

            if use_plugins:
                # Create ConvReluPlugin node
                plugin_output = f"plugin_output_{group_idx}"
                plugin_node = helper.make_node(
                    "ConvReluPlugin",
                    inputs=[current_input, kernel_name, bias_name],
                    outputs=[plugin_output],
                    name=f"convReluPlugin{group_idx}",
                    domain="nvidia.com",  # Use NVIDIA domain for custom plugins
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    groups=groups,
                )
                nodes.append(plugin_node)
                current_input = plugin_output
            else:
                # Create native Conv node
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

                # Create native ReLU node
                relu_output = f"relu_output_{group_idx}"
                relu_node = helper.make_node(
                    "Relu",
                    inputs=[conv_output],
                    outputs=[relu_output],
                    name=f"relu{group_idx}",
                )
                nodes.append(relu_node)
                current_input = relu_output

            # Add Abs node between groups (except after the last group)
            if group_idx < num_groups - 1:
                abs_output = f"abs_output_{group_idx}"
                abs_node = helper.make_node(
                    "Abs",
                    inputs=[current_input],
                    outputs=[abs_output],
                    name=f"abs{group_idx}",
                )
                nodes.append(abs_node)
                current_input = abs_output

        # Create output tensor
        output_tensor = helper.make_tensor_value_info(
            current_input, TensorProto.FLOAT, input_shape
        )

        # Create graph
        graph = helper.make_graph(
            nodes, "plugin_test_model", [input_tensor], [output_tensor], initializers
        )

        # Create model
        model = helper.make_model(
            graph, producer_name="PluginAutotuner_PluginV3_TimingCache_Demo"
        )
        model.opset_import[0].version = 11

        # Save model
        onnx.save(model, output_path)
        model_type = "plugin" if use_plugins else "native"
        G_LOGGER.info(
            f"Created {model_type} test model with {num_groups} groups: {output_path}"
        )
        return True

    except Exception as e:
        G_LOGGER.error(f"Failed to create model: {e}")
        return False


def build_tensorrt_engine_with_timing_cache(
    model_path: str,
    cache_path: str = None,
    enable_cache: bool = True,
    verbose: bool = False,
):
    """Build TensorRT engine with optional timing cache."""
    try:
        # Initialize TensorRT components
        trt_log_level = trt.Logger.VERBOSE if verbose else trt.Logger.INFO
        logger_obj = trt.Logger(trt_log_level)
        builder = trt.Builder(logger_obj)
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB

        # Configure timing cache
        if enable_cache and cache_path and os.path.exists(cache_path):
            G_LOGGER.verbose(f"Loading timing cache from: {cache_path}")
            with open(cache_path, "rb") as f:
                cache_data = f.read()
            timing_cache = config.create_timing_cache(cache_data)
            config.set_timing_cache(timing_cache, ignore_mismatch=False)
        elif enable_cache:
            G_LOGGER.verbose("Creating new timing cache")
            empty_cache_data = b""
            timing_cache = config.create_timing_cache(empty_cache_data)
            config.set_timing_cache(timing_cache, ignore_mismatch=False)
        else:
            G_LOGGER.verbose("Running without timing cache")

        # Parse ONNX model
        G_LOGGER.verbose(f"Parsing ONNX model: {model_path}")
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt.OnnxParser(network, logger_obj)

        with open(model_path, "rb") as f:
            if not parser.parse(f.read()):
                for error in range(parser.num_errors):
                    G_LOGGER.error(f"ONNX parsing error: {parser.get_error(error)}")
                return None, None

        G_LOGGER.verbose("ONNX model parsed successfully")

        # Build engine
        start_time = time.time()
        try:
            serialized_engine = builder.build_serialized_network(network, config)
            if serialized_engine is None:
                G_LOGGER.error("Failed to build TensorRT engine")
                return None, None
        except Exception as e:
            G_LOGGER.error(f"Failed to build engine: {e}")
            return None, None

        end_time = time.time()

        # Save timing cache if enabled
        saved_cache_path = None
        if enable_cache:
            timing_cache = config.get_timing_cache()
            if timing_cache:
                cache_data = timing_cache.serialize()
                if cache_path:
                    saved_cache_path = cache_path
                else:
                    saved_cache_path = "temp_timing_cache.trt"
                with open(saved_cache_path, "wb") as f:
                    f.write(cache_data)
                G_LOGGER.verbose(f"Saved timing cache to: {saved_cache_path}")

        return serialized_engine, {
            "build_time": end_time - start_time,
            "cache_path": saved_cache_path,
        }

    except Exception as e:
        G_LOGGER.error(f"Error building TensorRT engine: {e}")
        import traceback

        traceback.print_exc()
        return None, None


def main():
    parser = argparse.ArgumentParser(
        description="TensorRT PluginV3 timing cache performance demo"
    )
    parser.add_argument(
        "--num-groups",
        "-g",
        type=int,
        default=10,
        help="Number of Conv+ReLU groups in the model (default: 10)",
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

    G_LOGGER.info("=" * 80)
    G_LOGGER.info("TensorRT PluginV3 Timing Cache Performance Demo")
    G_LOGGER.info(f"Model: {args.num_groups} Conv+ReLU groups")
    G_LOGGER.info("=" * 80)

    # Step 1: Create native model
    G_LOGGER.info("1. Creating native Conv+ReLU model...")
    native_model_path = "native_conv_relu_model.onnx"
    if not create_plugin_test_model(
        native_model_path, args.num_groups, use_plugins=False
    ):
        G_LOGGER.error("Failed to create native model")
        return 1

    # Step 2: Create plugin model
    G_LOGGER.info("2. Creating plugin Conv+ReLU model...")
    plugin_model_path = "plugin_conv_relu_model.onnx"
    if not create_plugin_test_model(
        plugin_model_path, args.num_groups, use_plugins=True
    ):
        G_LOGGER.error("Failed to create plugin model")
        return 1

    # Step 3: Build native model without cache
    G_LOGGER.info("3. Building native model WITHOUT timing cache...")
    native_result = build_tensorrt_engine_with_timing_cache(
        native_model_path, enable_cache=False, verbose=args.verbose
    )

    if native_result[0] is None:
        G_LOGGER.error("Failed to build native model")
        return 1

    native_engine, native_info = native_result
    G_LOGGER.info(f"Native model build time: {native_info['build_time']:.2f} seconds")

    # Step 4: Build plugin model without cache
    G_LOGGER.info("4. Building plugin model WITHOUT timing cache...")
    plugin_result = build_tensorrt_engine_with_timing_cache(
        plugin_model_path, enable_cache=False, verbose=args.verbose
    )

    if plugin_result[0] is None:
        G_LOGGER.error("Failed to build plugin model")
        return 1

    plugin_engine, plugin_info = plugin_result
    G_LOGGER.info(f"Plugin model build time: {plugin_info['build_time']:.2f} seconds")

    # Step 5: Build native model with cache
    G_LOGGER.info("5. Building native model WITH timing cache...")
    native_cache_path = "native_timing_cache.trt"
    native_with_cache_result = build_tensorrt_engine_with_timing_cache(
        native_model_path,
        cache_path=native_cache_path,
        enable_cache=True,
        verbose=args.verbose,
    )

    if native_with_cache_result[0] is None:
        G_LOGGER.error("Failed to build native model with cache")
        return 1

    native_with_cache_engine, native_with_cache_info = native_with_cache_result
    G_LOGGER.info(
        f"Native model with cache build time: {native_with_cache_info['build_time']:.2f} seconds"
    )

    # Step 6: Build plugin model with cache
    G_LOGGER.info("6. Building plugin model WITH timing cache...")
    plugin_cache_path = "plugin_timing_cache.trt"
    plugin_with_cache_result = build_tensorrt_engine_with_timing_cache(
        plugin_model_path,
        cache_path=plugin_cache_path,
        enable_cache=True,
        verbose=args.verbose,
    )

    if plugin_with_cache_result[0] is None:
        G_LOGGER.error("Failed to build plugin model with cache")
        return 1

    plugin_with_cache_engine, plugin_with_cache_info = plugin_with_cache_result
    G_LOGGER.info(
        f"Plugin model with cache build time: {plugin_with_cache_info['build_time']:.2f} seconds"
    )

    # Step 7: Build plugin model using native cache
    G_LOGGER.info("7. Building plugin model using native cache...")
    plugin_using_native_cache_result = build_tensorrt_engine_with_timing_cache(
        plugin_model_path,
        cache_path=native_cache_path,
        enable_cache=True,
        verbose=args.verbose,
    )

    if plugin_using_native_cache_result[0] is None:
        G_LOGGER.error("Failed to build plugin model using native cache")
        return 1

    plugin_using_native_cache_engine, plugin_using_native_cache_info = (
        plugin_using_native_cache_result
    )
    G_LOGGER.info(
        f"Plugin model using native cache build time: {plugin_using_native_cache_info['build_time']:.2f} seconds"
    )

    # Step 8: Display comparison results
    G_LOGGER.info("8. PluginV3 Timing Cache Performance Comparison:")
    G_LOGGER.info("=" * 80)

    # Collect all results
    results = [
        ("Native (No Cache)", native_info),
        ("Plugin (No Cache)", plugin_info),
        ("Native (With Cache)", native_with_cache_info),
        ("Plugin (With Cache)", plugin_with_cache_info),
        ("Plugin (Using Native Cache)", plugin_using_native_cache_info),
    ]

    # Build time comparison
    G_LOGGER.info("Build Time Comparison:")
    baseline_time = native_info["build_time"]
    for name, info in results:
        build_time = info["build_time"]
        if name == "Native (No Cache)":
            G_LOGGER.info(f"  {name}: {build_time:.2f} seconds (baseline)")
        else:
            time_savings = (
                ((baseline_time - build_time) / baseline_time) * 100
                if baseline_time > 0
                else 0
            )
            speedup = baseline_time / build_time if build_time > 0 else 0
            G_LOGGER.info(
                f"  {name}: {build_time:.2f} seconds (savings: {time_savings:+.1f}%, speedup: {speedup:.2f}x)"
            )

    # Cache information
    G_LOGGER.info("")
    G_LOGGER.info("Cache Information:")
    for name, info in results:
        if info.get("cache_path") and os.path.exists(info["cache_path"]):
            cache_size = os.path.getsize(info["cache_path"])
            G_LOGGER.info(f"  {name}: {cache_size / 1024:.1f} KB")
        else:
            G_LOGGER.info(f"  {name}: No cache")

    # Plugin tactic information
    G_LOGGER.info("")
    G_LOGGER.info("Plugin Tactics:")
    G_LOGGER.info("  ConvReluPlugin supports 2 tactics:")
    G_LOGGER.info("    - Tactic 1: Pointer copy (fast)")
    G_LOGGER.info("    - Tactic 2: Content copy (slower)")
    G_LOGGER.info("  Timing cache helps TensorRT choose the best tactic")

    G_LOGGER.info("=" * 80)
    G_LOGGER.info("PluginV3 timing cache demo completed successfully!")
    G_LOGGER.info("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
