#!/usr/bin/env python3
"""
Test model generator for plugin autotuner examples.

Creates ONNX models with configurable numbers of Conv+ReLU layers
to demonstrate the plugin autotuner optimization workflow.
"""

import argparse
import logging
import sys
import traceback

# Use Polygraphy's lazy import for optional dependencies
from polygraphy import mod

np = mod.lazy_import("numpy")
onnx = mod.lazy_import("onnx")

logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# Import onnx modules after lazy import (they'll be available when onnx is imported)
def get_onnx_modules():
    """Get ONNX helper modules (ensures onnx is imported first)."""
    import onnx.helper as helper
    import onnx.numpy_helper as numpy_helper
    from onnx import TensorProto

    return helper, numpy_helper, TensorProto


def create_conv_relu_model(
    num_groups: int, output_name: str = "conv_relu_test_model.onnx"
) -> str:
    """
    Create an ONNX model with multiple Conv+ReLU groups.

    Args:
        num_groups: Number of Conv+ReLU groups to create
        output_name: Output ONNX model filename

    Returns:
        Path to the created ONNX model
    """
    logger.info(f"Creating ONNX model with {num_groups} Conv+ReLU groups...")

    # Get ONNX modules (this triggers the actual import)
    helper, numpy_helper, TensorProto = get_onnx_modules()

    # Model parameters
    batch_size = 1
    input_channels = 64
    output_channels = 64
    height = 224
    width = 224
    kernel_size = [3, 3]
    stride = [1, 1]
    padding = [1, 1, 1, 1]
    dilation = [1, 1]
    groups = 1

    # Create input tensor
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
        # Create weights
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
            "Relu", inputs=[conv_output], outputs=[relu_output], name=f"relu{group_idx}"
        )
        nodes.append(relu_node)

        current_input = relu_output
        input_channels = output_channels

    # Create output tensor
    output_tensor = helper.make_tensor_value_info(
        current_input, TensorProto.FLOAT, [batch_size, output_channels, height, width]
    )

    # Create the model
    graph = helper.make_graph(
        nodes,
        f"conv_relu_{num_groups}_groups",
        [input_tensor],
        [output_tensor],
        initializers,
    )
    model = helper.make_model(graph, producer_name="polygraphy_plugin_autotuner")
    model.opset_import[0].version = 11

    # Save the model
    onnx.save(model, output_name)
    logger.info(f"Model saved to: {output_name}")

    # Log statistics
    total_params = sum(np.prod(init.dims) for init in initializers)
    logger.info(f"Model statistics:")
    logger.info(f"  Total layers: {len(nodes)}")
    logger.info(f"  Conv+ReLU groups: {num_groups}")
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Input shape: {input_shape}")
    logger.info(f"  Output shape: [1, {output_channels}, {height}, {width}]")

    return output_name


def main():
    parser = argparse.ArgumentParser(
        description="Create test ONNX models for plugin autotuner examples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 create_test_model.py --groups 5
  python3 create_test_model.py --groups 3 --output custom_model.onnx
        """,
    )

    parser.add_argument(
        "--groups",
        "-g",
        type=int,
        default=5,
        help="Number of Conv+ReLU groups (default: 5)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="conv_relu_test_model.onnx",
        help="Output ONNX model filename",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.groups <= 0:
        logger.error("Number of groups must be positive")
        return 1

    try:
        model_path = create_conv_relu_model(
            num_groups=args.groups, output_name=args.output
        )
        logger.info("=" * 60)
        logger.info("Model creation completed successfully!")
        logger.info(f"Model file: {model_path}")
        logger.info("=" * 60)
        return 0
    except Exception as e:
        logger.error(f"Failed to create model: {e}")
        if args.verbose:
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
