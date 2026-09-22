#!/usr/bin/env python3
"""
Pattern-based native-to-plugin replacement demo with AutoTuner.
Updated to use Polygraphy modules directly.
"""

import os
import sys
import argparse

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


def create_native_conv_relu_model(output_path: str, num_groups: int = 2):
    """Create a simple ONNX model with multiple Conv + ReLU layers."""
    try:
        import onnx
        from onnx import helper, numpy_helper, TensorProto
        import numpy as np

        # Model parameters
        batch_size = 1
        input_channels = 128
        output_channels = 128
        height = 256
        width = 256
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

            current_input = relu_output

        # Create output tensor (same shape as input)
        output_tensor = helper.make_tensor_value_info(
            current_input, TensorProto.FLOAT, input_shape
        )

        # Create graph
        graph = helper.make_graph(
            nodes, "conv_relu_model", [input_tensor], [output_tensor], initializers
        )

        # Create model
        model = helper.make_model(graph, producer_name="PluginAutotuner_CLI_Demo")
        model.opset_import[0].version = 11

        # Save model
        onnx.save(model, output_path)
        G_LOGGER.info(
            f"Created native Conv+ReLU model with {num_groups} groups: {output_path}"
        )
        return True

    except Exception as e:
        G_LOGGER.error(f"Failed to create model: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Conv+ReLU demo using plugin autotuner"
    )
    parser.add_argument(
        "--input", "-i", help="Input ONNX model path (will create one if not provided)"
    )
    parser.add_argument("--output", "-o", default="optimized_conv_relu_model.onnx")
    parser.add_argument(
        "--num-groups",
        "-g",
        type=int,
        default=2,
        help="Number of Conv+ReLU groups in the input model (default: 2)",
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
        args.input = "native_conv_relu_model.onnx"
        G_LOGGER.info(f"Creating test model with {args.num_groups} Conv+ReLU groups...")
        if not create_native_conv_relu_model(args.input, args.num_groups):
            G_LOGGER.error("Failed to create test model")
            return 1

    patterns_dir = get_patterns_directory()
    G_LOGGER.info(f"Using patterns directory: {patterns_dir}")

    repl = ReplacementEngine()
    repl.load_plugins(patterns_dir)

    if not os.path.exists(args.input):
        G_LOGGER.error(f"Input model not found: {args.input}")
        return 1

    # Match patterns (will auto-generate config.yaml in model directory)
    repl.match_subgraphs(args.input)

    # Set TensorRT logger verbosity based on --verbose flag
    trt_log_level = trt.Logger.VERBOSE if args.verbose else trt.Logger.INFO
    builder = trt.Builder(trt.Logger(trt_log_level))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

    autotuner = AutoTuner(builder=builder, config=config, replacement_engine=repl)

    G_LOGGER.info("Running optimization...")
    result = autotuner.optimize_network(
        input_source=args.input,
        use_greedy=True,
        max_combinations=16,
        performance_threshold=0.001,
    )

    # Display optimization results
    G_LOGGER.info("=" * 60)
    G_LOGGER.info("Optimization Results:")
    G_LOGGER.info(f"  Total time: {result.total_time:.2f} seconds")
    G_LOGGER.info(f"  Combinations tested: {result.num_combinations_tested}")
    G_LOGGER.info(f"  Engine builds: {result.num_engine_builds}")
    if result.total_compile_time > 0:
        G_LOGGER.info(f"  Total compilation time: {result.total_compile_time:.2f}s")

    if result.baseline_metrics and result.best_metrics:
        baseline_latency = result.baseline_metrics.latency_ms
        best_latency = result.best_metrics.latency_ms

        G_LOGGER.info("")
        G_LOGGER.info("Performance Comparison:")
        G_LOGGER.info(f"  Baseline latency: {baseline_latency:.4f} ms")
        G_LOGGER.info(f"  Optimized latency: {best_latency:.4f} ms")

        if baseline_latency > 0:
            improvement = ((baseline_latency - best_latency) / baseline_latency) * 100
            speedup = baseline_latency / best_latency
            G_LOGGER.info(f"  Improvement: {improvement:+.2f}%")
            G_LOGGER.info(f"  Speedup: {speedup:.2f}x")

    if result.best_strategy:
        G_LOGGER.info("")
        G_LOGGER.info("Best Replacement Strategy:")
        for i, replacement in enumerate(result.best_strategy):
            plugin_name = replacement.get("plugin_name", "Unknown")
            plugin_op = replacement.get("plugin_op", "Unknown")
            inputs = replacement.get("inputs", [])
            outputs = replacement.get("outputs", [])
            G_LOGGER.info(f"  {i+1}. {plugin_op} ({plugin_name})")
            G_LOGGER.info(f"     Inputs: {', '.join(inputs) if inputs else 'N/A'}")
            G_LOGGER.info(f"     Outputs: {', '.join(outputs) if outputs else 'N/A'}")
    G_LOGGER.info("=" * 60)
    G_LOGGER.info("")

    # Save optimized model
    temp_path = None
    try:
        temp_path = autotuner.get_optimized_model_path(result)
        model = onnx.load(temp_path)
        onnx.save(model, args.output)
        G_LOGGER.info(f"Optimized model saved to: {args.output}")
    finally:
        # Cleanup temporary files
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception as e:
                G_LOGGER.warning(f"Failed to remove temporary file {temp_path}: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
