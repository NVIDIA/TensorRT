#!/usr/bin/env python3
"""
Demo for pattern-based replacement that injects constant-only input transforms
on the Conv weights via Transpose + Reshape inside replace_with_plugin.

This demonstrates how plugin pattern code can add pre-processing nodes that
TensorRT can constant-fold away when weights are constants.
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


def create_native_conv_relu_model(output_path: str, num_groups: int = 1):
    """Create a simple ONNX model with multiple Conv + ReLU layers."""
    try:
        import onnx
        from onnx import helper, numpy_helper, TensorProto
        import numpy as np

        batch_size = 1
        input_channels = 64
        output_channels = 64
        height = 64
        width = 64
        kernel_size = [3, 3]
        stride = [1, 1]
        padding = [1, 1, 1, 1]
        dilation = [1, 1]
        groups = 1

        input_name = "input"
        input_shape = [batch_size, input_channels, height, width]
        input_tensor = helper.make_tensor_value_info(
            input_name, TensorProto.FLOAT, input_shape
        )

        nodes = []
        initializers = []
        current_input = input_name

        for group_idx in range(num_groups):
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

            relu_output = f"relu_output_{group_idx}"
            relu_node = helper.make_node(
                "Relu",
                inputs=[conv_output],
                outputs=[relu_output],
                name=f"relu{group_idx}",
            )
            nodes.append(relu_node)

            current_input = relu_output

        output_tensor = helper.make_tensor_value_info(
            current_input, TensorProto.FLOAT, input_shape
        )

        graph = helper.make_graph(
            nodes, "conv_relu_model", [input_tensor], [output_tensor], initializers
        )

        model = helper.make_model(
            graph, producer_name="PluginAutotuner_WeightsTransformation_Demo"
        )
        model.opset_import[0].version = 11
        onnx.save(model, output_path)
        G_LOGGER.info(
            f"Created native Conv+ReLU model with {num_groups} groups: {output_path}"
        )
        return True

    except Exception as e:
        G_LOGGER.error(f"Failed to create model: {e}")
        return False


def get_patterns_directory():
    """Get the path to the local patterns directory for this demo."""
    return os.path.join(current_dir, "patterns")


def print_onnx_model_structure(model_path: str, title: str = "ONNX Model Structure"):
    """Print the structure of an ONNX model for visualization."""
    try:
        import onnx

        model = onnx.load(model_path)
        graph = model.graph

        G_LOGGER.info("=" * 80)
        G_LOGGER.info(f"{title}: {os.path.basename(model_path)}")
        G_LOGGER.info("=" * 80)

        # Print inputs
        G_LOGGER.info("Graph Inputs:")
        for inp in graph.input:
            shape = [
                dim.dim_value if dim.dim_value > 0 else "?"
                for dim in inp.type.tensor_type.shape.dim
            ]
            dtype = onnx.TensorProto.DataType.Name(inp.type.tensor_type.elem_type)
            G_LOGGER.info(f"  {inp.name}: {dtype} {shape}")

        # Print initializers (constants/weights)
        if graph.initializer:
            G_LOGGER.info("")
            G_LOGGER.info(f"Initializers (Constants/Weights): {len(graph.initializer)}")
            for init in graph.initializer:
                G_LOGGER.info(f"  {init.name}: {list(init.dims)}")

        # Print nodes (operations)
        G_LOGGER.info("")
        G_LOGGER.info(f"Graph Nodes: {len(graph.node)}")
        for i, node in enumerate(graph.node):
            inputs_str = ", ".join(node.input)
            outputs_str = ", ".join(node.output)

            # Highlight transformation nodes (Transpose, Reshape)
            if node.op_type in ["Transpose", "Reshape"]:
                marker = " ⚠️ [Injected Transform]"
            elif "Plugin" in node.op_type:
                marker = " 🔌 [Plugin]"
            else:
                marker = ""

            G_LOGGER.info(f"  [{i}] {node.op_type}{marker}")
            G_LOGGER.info(f"      Name: {node.name if node.name else 'N/A'}")
            G_LOGGER.info(f"      Inputs: {inputs_str}")
            G_LOGGER.info(f"      Outputs: {outputs_str}")

            # Print attributes for key operations
            if (
                node.op_type in ["Transpose", "Reshape", "Conv"]
                or "Plugin" in node.op_type
            ):
                if node.attribute:
                    attrs = {}
                    for attr in node.attribute:
                        if attr.type == onnx.AttributeProto.INTS:
                            attrs[attr.name] = list(attr.ints)
                        elif attr.type == onnx.AttributeProto.INT:
                            attrs[attr.name] = attr.i
                        elif attr.type == onnx.AttributeProto.FLOATS:
                            attrs[attr.name] = list(attr.floats)
                        elif attr.type == onnx.AttributeProto.FLOAT:
                            attrs[attr.name] = attr.f
                        elif attr.type == onnx.AttributeProto.STRING:
                            attrs[attr.name] = (
                                attr.s.decode("utf-8")
                                if isinstance(attr.s, bytes)
                                else attr.s
                            )
                    if attrs:
                        G_LOGGER.info(f"      Attributes: {attrs}")

        # Print outputs
        G_LOGGER.info("")
        G_LOGGER.info("Graph Outputs:")
        for out in graph.output:
            shape = [
                dim.dim_value if dim.dim_value > 0 else "?"
                for dim in out.type.tensor_type.shape.dim
            ]
            dtype = onnx.TensorProto.DataType.Name(out.type.tensor_type.elem_type)
            G_LOGGER.info(f"  {out.name}: {dtype} {shape}")

        G_LOGGER.info("=" * 80)
        G_LOGGER.info("")

        return True
    except Exception as e:
        G_LOGGER.warning(f"Could not print model structure: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Weights transformation demo with Transpose + Reshape in pattern replacement"
    )
    parser.add_argument(
        "--input", "-i", help="Input ONNX model path (will create one if not provided)"
    )
    parser.add_argument(
        "--output",
        "-o",
        default="optimized_conv_relu_model.onnx",
        help="Output ONNX model path",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument(
        "--num-groups",
        "-g",
        type=int,
        default=1,
        help="Number of Conv+ReLU groups in the input model (default: 1)",
    )
    parser.add_argument(
        "--show-structure",
        "-s",
        action="store_true",
        help="Show detailed ONNX model structure before and after transformation",
    )
    args = parser.parse_args()

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
        if not create_native_conv_relu_model(args.input, args.num_groups):
            G_LOGGER.error("Failed to create input model")
            return 1

    G_LOGGER.info("=" * 60)
    G_LOGGER.info("Weights Transformation Demo")
    G_LOGGER.info("Demonstrates: Transpose + Reshape injection on constant weights")
    G_LOGGER.info("=" * 60)
    G_LOGGER.info("")

    # Show original model structure if requested
    if args.show_structure:
        print_onnx_model_structure(args.input, "Original Model Structure")

    patterns_dir = get_patterns_directory()
    G_LOGGER.info(f"Using patterns directory: {patterns_dir}")

    G_LOGGER.info("1. Initializing pattern-based replacement engine...")
    replacement_engine = ReplacementEngine()
    replacement_engine.load_plugins(patterns_dir)

    G_LOGGER.info("2. Finding matching subgraphs...")
    replacement_engine.match_subgraphs(args.input)
    if not replacement_engine.plugin_subgraphs:
        G_LOGGER.error("Failed to find matching subgraphs")
        return 1

    # Get and display match information
    all_subgraphs = replacement_engine.get_all_subgraphs()
    total_matches = sum(len(subgraphs) for subgraphs in all_subgraphs.values())
    G_LOGGER.info(f"Found {total_matches} matching Conv+ReLU patterns")

    G_LOGGER.info("3. Initializing TensorRT components...")
    # Set TensorRT logger verbosity based on --verbose flag
    trt_log_level = trt.Logger.VERBOSE if args.verbose else trt.Logger.INFO
    builder = trt.Builder(trt.Logger(trt_log_level))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 32)

    G_LOGGER.info("4. Initializing AutoTuner...")
    autotuner = AutoTuner(
        builder=builder, config=config, replacement_engine=replacement_engine
    )

    G_LOGGER.info("5. Running AutoTuner optimization...")
    G_LOGGER.info("   (This will inject Transpose + Reshape on constant weights)")
    opt_result = autotuner.optimize_network(
        input_source=args.input,
        use_greedy=True,
        max_combinations=10,
        performance_threshold=0.001,
    )

    G_LOGGER.info("6. Saving optimized model...")
    try:
        temp_path = autotuner.get_optimized_model_path(opt_result)
        model = onnx.load(temp_path)
        onnx.save(model, args.output)
        G_LOGGER.info(f"Optimized model saved to: {args.output}")
    except Exception as e:
        G_LOGGER.error(f"Failed to save optimized model: {e}")
        return 1
    G_LOGGER.info("")

    # Always show the optimized model structure to visualize injected transforms
    G_LOGGER.info("📊 Optimized Model Structure (showing injected transformations):")
    G_LOGGER.info("")
    print_onnx_model_structure(
        args.output, "Optimized Model with Weight Transformations"
    )

    # Print summary
    G_LOGGER.info("=" * 60)
    G_LOGGER.info("Summary:")
    G_LOGGER.info(f"  Original model: {args.input}")
    G_LOGGER.info(f"  Optimized model: {args.output}")
    G_LOGGER.info(f"  Patterns matched: {total_matches}")
    if opt_result.best_strategy:
        G_LOGGER.info(f"  Plugins inserted: {len(opt_result.best_strategy)}")
    G_LOGGER.info("")
    G_LOGGER.info("Key Features Demonstrated:")
    G_LOGGER.info("  ⚠️  Transpose + Reshape nodes injected on constant weights")
    G_LOGGER.info("  🔌 Conv+ReLU replaced with ConvReluPlugin")
    G_LOGGER.info("  ⚡ TensorRT will constant-fold the transforms at compile time")
    G_LOGGER.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
