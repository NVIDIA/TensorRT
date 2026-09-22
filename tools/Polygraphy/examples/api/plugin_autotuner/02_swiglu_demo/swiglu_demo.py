#!/usr/bin/env python3
"""
SwiGLU pattern-based replacement demo with AutoTuner.
Demonstrates multi-input/output, nested patterns, and check_func functionality.
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

# Register the SwiGLUPlugin with TensorRT
# Add the current directory to Python path to access local plugins
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import plugins.trt_swiglu_plugin as trt_swiglu_plugin


def get_patterns_directory():
    """Get the local patterns directory."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_dir, "patterns")


def create_native_swiglu_model(output_path: str, use_fp16: bool = True):
    """Create a simple ONNX model with SwiGLU layers (both fp16 and fp32 versions)."""
    try:
        import onnx
        from onnx import helper, numpy_helper, TensorProto
        import numpy as np

        # Model parameters - Use 2D input for Gemm compatibility
        seq_len = 2048
        hidden_size = 2048

        # Choose data type based on use_fp16 flag
        if use_fp16:
            data_type = TensorProto.FLOAT16
            np_dtype = np.float16
            model_suffix = "fp16"
        else:
            data_type = TensorProto.FLOAT
            np_dtype = np.float32
            model_suffix = "fp32"

        # Create input - 2D for Gemm compatibility
        input_name = "input"
        input_shape = [seq_len, hidden_size]  # 2D input
        input_tensor = helper.make_tensor_value_info(input_name, data_type, input_shape)

        # Create graph nodes and initializers
        nodes = []
        initializers = []
        current_input = input_name

        # Create weights for SwiGLU
        weight1_shape = [hidden_size, hidden_size]
        weight1_data = np.random.randn(*weight1_shape).astype(np_dtype) * 0.1
        weight1_name = "weight1"
        weight1_initializer = numpy_helper.from_array(weight1_data, name=weight1_name)
        initializers.append(weight1_initializer)

        weight2_shape = [hidden_size, hidden_size]
        weight2_data = np.random.randn(*weight2_shape).astype(np_dtype) * 0.1
        weight2_name = "weight2"
        weight2_initializer = numpy_helper.from_array(weight2_data, name=weight2_name)
        initializers.append(weight2_initializer)

        # First GEMM: c1 = gemm(input, weight1)
        c1_output = "c1"
        nodes.append(
            helper.make_node(
                "Gemm",
                inputs=[current_input, weight1_name],
                outputs=[c1_output],
                name="gemm1",
                alpha=1.0,
                beta=0.0,
                transA=0,
                transB=0,
            )
        )

        # Second GEMM: temp = gemm(input, weight2)
        temp_output = "temp"
        nodes.append(
            helper.make_node(
                "Gemm",
                inputs=[current_input, weight2_name],
                outputs=[temp_output],
                name="gemm2",
                alpha=1.0,
                beta=0.0,
                transA=0,
                transB=0,
            )
        )

        # Swish activation: c2 = swish(temp) = temp * sigmoid(temp)
        sigmoid_output = "sigmoid"
        nodes.append(
            helper.make_node(
                "Sigmoid",
                inputs=[temp_output],
                outputs=[sigmoid_output],
                name="sigmoid",
            )
        )

        c2_output = "c2"
        nodes.append(
            helper.make_node(
                "Mul",
                inputs=[temp_output, sigmoid_output],
                outputs=[c2_output],
                name="mul_swish",
            )
        )

        # Final multiplication: d = mul(c1, c2)
        d_output = "d"
        nodes.append(
            helper.make_node(
                "Mul",
                inputs=[c1_output, c2_output],
                outputs=[d_output],
                name="mul_final",
            )
        )

        # Create output tensors (both d and c2)
        output_tensor_d = helper.make_tensor_value_info(
            d_output, data_type, input_shape
        )
        output_tensor_c2 = helper.make_tensor_value_info(
            c2_output, data_type, input_shape
        )

        # Create graph
        graph = helper.make_graph(
            nodes,
            f"swiglu_{model_suffix}_model",
            [input_tensor],
            [output_tensor_d, output_tensor_c2],  # Multiple outputs
            initializers,
        )

        # Create model
        model = helper.make_model(graph, producer_name="PluginAutotuner_SwiGLU_Demo")
        model.opset_import[0].version = 11

        # Save model
        onnx.save(model, output_path)
        G_LOGGER.info(f"Created native SwiGLU {model_suffix} model: {output_path}")
        return True

    except Exception as e:
        G_LOGGER.error(f"Failed to create model: {e}")
        return False


def create_combined_swiglu_model(output_path: str):
    """Create a model with both fp16 and fp32 SwiGLU layers to demonstrate check_func."""
    try:
        import onnx
        from onnx import helper, numpy_helper, TensorProto
        import numpy as np

        # Model parameters - Use 2D input for Gemm compatibility
        seq_len = 2048
        hidden_size = 2048

        # Create input - 2D for Gemm compatibility
        input_name = "input"
        input_shape = [seq_len, hidden_size]  # 2D input
        input_tensor = helper.make_tensor_value_info(
            input_name, TensorProto.FLOAT16, input_shape  # Start with fp16
        )

        # Create graph nodes and initializers
        nodes = []
        initializers = []
        current_input = input_name

        # Create weights for fp16 SwiGLU
        weight1_fp16_shape = [hidden_size, hidden_size]
        weight1_fp16_data = (
            np.random.randn(*weight1_fp16_shape).astype(np.float16) * 0.1
        )
        weight1_fp16_name = "weight1_fp16"
        weight1_fp16_initializer = numpy_helper.from_array(
            weight1_fp16_data, name=weight1_fp16_name
        )
        initializers.append(weight1_fp16_initializer)

        weight2_fp16_shape = [hidden_size, hidden_size]
        weight2_fp16_data = (
            np.random.randn(*weight2_fp16_shape).astype(np.float16) * 0.1
        )
        weight2_fp16_name = "weight2_fp16"
        weight2_fp16_initializer = numpy_helper.from_array(
            weight2_fp16_data, name=weight2_fp16_name
        )
        initializers.append(weight2_fp16_initializer)

        # FP16 SwiGLU (should be matched by check_func)
        # First GEMM: c1_fp16 = gemm(input, weight1_fp16)
        c1_fp16_output = "c1_fp16"
        nodes.append(
            helper.make_node(
                "Gemm",
                inputs=[current_input, weight1_fp16_name],
                outputs=[c1_fp16_output],
                name="gemm1_fp16",
                alpha=1.0,
                beta=0.0,
                transA=0,
                transB=0,
            )
        )

        # Second GEMM: temp_fp16 = gemm(input, weight2_fp16)
        temp_fp16_output = "temp_fp16"
        nodes.append(
            helper.make_node(
                "Gemm",
                inputs=[current_input, weight2_fp16_name],
                outputs=[temp_fp16_output],
                name="gemm2_fp16",
                alpha=1.0,
                beta=0.0,
                transA=0,
                transB=0,
            )
        )

        # Swish activation: c2_fp16 = swish(temp_fp16)
        sigmoid_fp16_output = "sigmoid_fp16"
        nodes.append(
            helper.make_node(
                "Sigmoid",
                inputs=[temp_fp16_output],
                outputs=[sigmoid_fp16_output],
                name="sigmoid_fp16",
            )
        )

        c2_fp16_output = "c2_fp16"
        nodes.append(
            helper.make_node(
                "Mul",
                inputs=[temp_fp16_output, sigmoid_fp16_output],
                outputs=[c2_fp16_output],
                name="mul_swish_fp16",
            )
        )

        # Final multiplication: d_fp16 = mul(c1_fp16, c2_fp16)
        d_fp16_output = "d_fp16"
        nodes.append(
            helper.make_node(
                "Mul",
                inputs=[c1_fp16_output, c2_fp16_output],
                outputs=[d_fp16_output],
                name="mul_final_fp16",
            )
        )

        # fp16 subgraph: final_fp16 = mul(d_fp16, c2_fp16)
        final_fp16_output = "final_fp16_output"
        nodes.append(
            helper.make_node(
                "Mul",
                inputs=[d_fp16_output, c2_fp16_output],
                outputs=[final_fp16_output],
                name="mul_final_fp16_subgraph",
            )
        )

        # Create a separate fp32 input for the second SwiGLU (avoid Cast operation)
        input_fp32_name = "input_fp32"
        input_fp32_tensor = helper.make_tensor_value_info(
            input_fp32_name, TensorProto.FLOAT, input_shape
        )

        # Create weights for fp32 SwiGLU
        weight1_fp32_shape = [hidden_size, hidden_size]
        weight1_fp32_data = (
            np.random.randn(*weight1_fp32_shape).astype(np.float32) * 0.1
        )
        weight1_fp32_name = "weight1_fp32"
        weight1_fp32_initializer = numpy_helper.from_array(
            weight1_fp32_data, name=weight1_fp32_name
        )
        initializers.append(weight1_fp32_initializer)

        weight2_fp32_shape = [hidden_size, hidden_size]
        weight2_fp32_data = (
            np.random.randn(*weight2_fp32_shape).astype(np.float32) * 0.1
        )
        weight2_fp32_name = "weight2_fp32"
        weight2_fp32_initializer = numpy_helper.from_array(
            weight2_fp32_data, name=weight2_fp32_name
        )
        initializers.append(weight2_fp32_initializer)

        # FP32 SwiGLU (should NOT be matched by check_func)
        # First GEMM: c1_fp32 = gemm(input_fp32, weight1_fp32)
        c1_fp32_output = "c1_fp32"
        nodes.append(
            helper.make_node(
                "Gemm",
                inputs=[input_fp32_name, weight1_fp32_name],
                outputs=[c1_fp32_output],
                name="gemm1_fp32",
                alpha=1.0,
                beta=0.0,
                transA=0,
                transB=0,
            )
        )

        # Second GEMM: temp_fp32 = gemm(input_fp32, weight2_fp32)
        temp_fp32_output = "temp_fp32"
        nodes.append(
            helper.make_node(
                "Gemm",
                inputs=[input_fp32_name, weight2_fp32_name],
                outputs=[temp_fp32_output],
                name="gemm2_fp32",
                alpha=1.0,
                beta=0.0,
                transA=0,
                transB=0,
            )
        )

        # Swish activation: c2_fp32 = swish(temp_fp32)
        sigmoid_fp32_output = "sigmoid_fp32"
        nodes.append(
            helper.make_node(
                "Sigmoid",
                inputs=[temp_fp32_output],
                outputs=[sigmoid_fp32_output],
                name="sigmoid_fp32",
            )
        )

        c2_fp32_output = "c2_fp32"
        nodes.append(
            helper.make_node(
                "Mul",
                inputs=[temp_fp32_output, sigmoid_fp32_output],
                outputs=[c2_fp32_output],
                name="mul_swish_fp32",
            )
        )

        # Final multiplication: d_fp32 = mul(c1_fp32, c2_fp32)
        d_fp32_output = "d_fp32"
        nodes.append(
            helper.make_node(
                "Mul",
                inputs=[c1_fp32_output, c2_fp32_output],
                outputs=[d_fp32_output],
                name="mul_final_fp32",
            )
        )

        # fp32 subgraph: final_fp32 = mul(d_fp32, c2_fp32)
        final_fp32_output = "final_fp32_output"
        nodes.append(
            helper.make_node(
                "Mul",
                inputs=[d_fp32_output, c2_fp32_output],
                outputs=[final_fp32_output],
                name="mul_final_fp32_subgraph",
            )
        )

        # Create output tensors
        output_tensor1 = helper.make_tensor_value_info(
            final_fp16_output, TensorProto.FLOAT16, input_shape
        )

        output_tensor2 = helper.make_tensor_value_info(
            final_fp32_output, TensorProto.FLOAT, input_shape
        )

        # Create graph
        graph = helper.make_graph(
            nodes,
            "combined_swiglu_model",
            [input_tensor, input_fp32_tensor],
            [output_tensor1, output_tensor2],
            initializers,
        )

        # Create model
        model = helper.make_model(graph, producer_name="PluginAutotuner_SwiGLU_Demo")
        model.opset_import[0].version = 11

        # Save model
        onnx.save(model, output_path)
        G_LOGGER.info(
            f"Created combined SwiGLU model with fp16 and fp32: {output_path}"
        )
        return True

    except Exception as e:
        G_LOGGER.error(f"Failed to create combined model: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="SwiGLU pattern-based replacement demo - demonstrates multi-input/output and check_func"
    )
    parser.add_argument(
        "--input", "-i", help="Input ONNX model path (will create one if not provided)"
    )
    parser.add_argument("--output", "-o", default="optimized_swiglu_model.onnx")
    parser.add_argument(
        "--simple",
        "-s",
        action="store_true",
        help="Use simple fp16 model instead of combined fp16/fp32 model",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    # Set Polygraphy logger verbosity
    if args.verbose:
        G_LOGGER.module_severity = G_LOGGER.VERBOSE
    else:
        G_LOGGER.module_severity = G_LOGGER.INFO

    G_LOGGER.info("=" * 60)
    G_LOGGER.info("SwiGLU Pattern-Based Replacement Demo")
    G_LOGGER.info("Features: Multi-input/output, Nested patterns, Check function")
    G_LOGGER.info("=" * 60)

    # Create input model if not provided
    if not args.input:
        if args.simple:
            args.input = "native_swiglu_model_fp16.onnx"
            G_LOGGER.info("Creating simple fp16 SwiGLU model...")
            if not create_native_swiglu_model(args.input, use_fp16=True):
                G_LOGGER.error("Failed to create fp16 test model")
                return 1
        else:
            args.input = "combined_swiglu_model.onnx"
            G_LOGGER.info("Creating combined fp16/fp32 SwiGLU model...")
            if not create_combined_swiglu_model(args.input):
                G_LOGGER.error("Failed to create combined test model")
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

    # Get subgraph information
    all_subgraphs = repl.get_all_subgraphs()
    total_matches = sum(len(subgraphs) for subgraphs in all_subgraphs.values())
    G_LOGGER.info(f"Found {total_matches} total matches")
    for plugin, subgraphs in all_subgraphs.items():
        if subgraphs:
            G_LOGGER.info(f"  {plugin}: {len(subgraphs)} instances")

    # Set TensorRT logger verbosity based on --verbose flag
    trt_log_level = trt.Logger.VERBOSE if args.verbose else trt.Logger.INFO
    builder = trt.Builder(trt.Logger(trt_log_level))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 32)  # 4GB

    autotuner = AutoTuner(builder=builder, config=config, replacement_engine=repl)

    G_LOGGER.info("Running optimization...")
    result = autotuner.optimize_network(
        input_source=args.input,
        use_greedy=True,
        max_combinations=10,
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
    G_LOGGER.info("=" * 60)
    G_LOGGER.info("SwiGLU demo completed successfully!")
    G_LOGGER.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
