#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""
Real TensorRT PluginV3 implementation for SwiGLU operation.
This plugin demonstrates multi-input, multi-output and nested pattern support.
"""

import numpy as np
from typing import List, Optional, Tuple

# Import TensorRT and CUDA
try:
    import tensorrt as trt
    import cuda
    import cuda.bindings.driver as cuda_driver
except ImportError as e:
    raise ImportError(f"Required dependencies not available: {e}")

# Add utils directory to Python path for direct execution
import sys
import os

utils_path = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, utils_path)

# Import logging utilities
from polygraphy.logger import G_LOGGER as logger


class SwiGLUPlugin(
    trt.IPluginV3,
    trt.IPluginV3OneCore,
    trt.IPluginV3OneBuild,
    trt.IPluginV3OneRuntime,
):
    """
    A TensorRT plugin that implements SwiGLU operation with multi-output support.
    This plugin can be replaced with native SwiGLU layers.

    Inputs:
        - input: Input tensor [batch, seq_len, hidden_size]
        - weight1: First weight tensor [hidden_size, hidden_size]
        - weight2: Second weight tensor [hidden_size, hidden_size]

    Outputs:
        - output1: Main output [batch, seq_len, hidden_size]
        - output2: Secondary output (c2) [batch, seq_len, hidden_size]
    """

    def __init__(self, fc=None, phase=None):
        """Initialize SwiGLUPlugin according to official TensorRT documentation."""
        # Initialize all parent classes
        trt.IPluginV3.__init__(self)
        trt.IPluginV3OneCore.__init__(self)
        trt.IPluginV3OneBuild.__init__(self)
        trt.IPluginV3OneRuntime.__init__(self)

        self.num_outputs = 2  # Two outputs: d and c2
        self.plugin_namespace = ""
        self.plugin_version = "1"
        self.plugin_name = "SwiGLUPlugin"
        self.name = "SwiGLUPlugin"

        # Parse plugin fields if provided (for deserialization)
        if fc is not None:
            try:
                for i in range(len(fc)):
                    field = fc[i]
                    field_name = field.name
                    field_type = field.type
                    field_data = field.data

                    # Add any plugin-specific fields here if needed
                    pass
            except Exception as e:
                logger.warning(f"Error parsing plugin fields: {e}")

        # Store phase for capability interface selection
        self.phase = phase

    def get_capability_interface(self, type):
        """Get capability interface according to official TensorRT documentation."""
        return self

    def get_output_data_types(self, input_types):
        """Get output data types."""
        # Both outputs have the same type as the first input
        return [input_types[0], input_types[0]]

    def get_output_shapes(self, inputs, shape_inputs, exprBuilder):
        """Calculate output shapes using expression builder."""
        # inputs[0]: input tensor [seq_len, hidden_size] (2D for Gemm compatibility)
        # inputs[1]: weight1 tensor [hidden_size, hidden_size]
        # inputs[2]: weight2 tensor [hidden_size, hidden_size]

        input_dims = inputs[0]
        weight1_dims = inputs[1]
        weight2_dims = inputs[2]

        # Validate input dimensions
        if len(input_dims) != 2:
            raise ValueError(f"Expected 2D input tensor, got {len(input_dims)}D")
        if len(weight1_dims) != 2:
            raise ValueError(f"Expected 2D weight1 tensor, got {len(weight1_dims)}D")
        if len(weight2_dims) != 2:
            raise ValueError(f"Expected 2D weight2 tensor, got {len(weight2_dims)}D")

        seq_len = input_dims[0]
        hidden_size = input_dims[1]

        # Both outputs have the same shape as input
        output_shape = [seq_len, hidden_size]

        return [
            trt.DimsExprs(output_shape),  # output1 (d)
            trt.DimsExprs(output_shape),  # output2 (c2)
        ]

    def get_fields_to_serialize(self):
        """Get fields to serialize."""
        fields = [
            # Add any plugin-specific fields here if needed
        ]

        return trt.PluginFieldCollection(fields)

    def configure_plugin(self, inp, out):
        """Configure the plugin."""
        # This plugin doesn't need special configuration
        pass

    def on_shape_change(self, inp, out):
        """Handle shape changes."""
        # This plugin doesn't need special shape change handling
        pass

    def supports_format_combination(self, pos, in_out, num_inputs):
        """Check if format combination is supported."""
        # Support common formats
        return True

    def enqueue(self, input_desc, output_desc, inputs, outputs, workspace, stream):
        """Execute the plugin."""
        # Simple pass-through for demo purposes
        # In real implementation, this would perform the actual SwiGLU computation
        outputs[0] = inputs[0]  # output1 = input
        outputs[1] = inputs[0]  # output2 = input
        return 0

    def attach_to_context(self, context):
        """Attach to TensorRT context."""
        return self.clone()

    def set_tactic(self, tactic):
        pass

    def clone(self):
        """Clone the plugin."""
        plugin = SwiGLUPlugin()
        return plugin

    def get_workspace_size(self, input_desc, output_desc):
        """Get workspace size required."""
        return 0

    def destroy(self):
        """Destroy the plugin."""
        pass


class SwiGLUPluginCreator(trt.IPluginCreatorV3One):
    """Creator for SwiGLUPlugin."""

    def __init__(self):
        trt.IPluginCreatorV3One.__init__(self)

        self.num_outputs = 2
        self.name = "SwiGLUPlugin"
        self.plugin_name = "SwiGLUPlugin"
        self.plugin_version = "1"
        self.plugin_namespace = ""
        self.field_names = trt.PluginFieldCollection([])

    def create_plugin(self, name, fc, phase):
        """Create plugin instance according to official TensorRT documentation."""
        return SwiGLUPlugin(fc, phase)


def register_swiglu_plugin():
    """Register the SwiGLUPlugin with TensorRT according to official documentation."""
    try:
        # Create plugin creator
        creator = SwiGLUPluginCreator()

        # Get TensorRT plugin registry
        registry = trt.get_plugin_registry()

        # Register the plugin creator
        success = registry.register_creator(creator, creator.plugin_namespace)

        if success:
            logger.verbose(
                f"PluginV3 SwiGLUPlugin successfully registered with TensorRT"
            )
            return creator
        else:
            logger.warning("Failed to register PluginV3 SwiGLUPlugin with TensorRT")
            return None

    except Exception as e:
        logger.error(f"Error registering PluginV3 SwiGLUPlugin: {e}")
        return None


def create_swiglu_plugin_weights(hidden_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create weights for SwiGLUPlugin.

    Args:
        hidden_size: Hidden size dimension

    Returns:
        Tuple of (weight1, weight2) - these should be used as input tensors
    """
    # Create random weights
    weight1 = np.random.randn(hidden_size, hidden_size).astype(np.float32) * 0.1
    weight2 = np.random.randn(hidden_size, hidden_size).astype(np.float32) * 0.1

    return weight1, weight2


# Auto-register the plugin when this module is imported
_creator = register_swiglu_plugin()

if __name__ == "__main__":
    # Register the plugin (also runs on import)
    if _creator:
        logger.verbose("SwiGLUPlugin is ready for use!")
    else:
        logger.error("SwiGLUPlugin registration failed!")
