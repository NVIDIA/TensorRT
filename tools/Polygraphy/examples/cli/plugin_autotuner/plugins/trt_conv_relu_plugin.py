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
Real TensorRT PluginV3 implementation for Conv+ReLU operation.
This plugin demonstrates how to create a proper TensorRT plugin that can be replaced with native layers.
Now supports multiple tactics for timing cache testing.
"""

import hashlib
from typing import List, Optional, Tuple

# Import Polygraphy modules and lazy imports
from polygraphy import mod
from polygraphy.mod.trt_importer import lazy_import_trt
from polygraphy.logger import G_LOGGER as logger

# Lazy import dependencies
np = mod.lazy_import("numpy")
trt = lazy_import_trt()


class ConvReluPlugin(
    trt.IPluginV3,
    trt.IPluginV3OneCore,
    trt.IPluginV3OneBuild,
    trt.IPluginV3OneRuntime,
):
    """
    A TensorRT plugin that implements Conv + ReLU operation.
    This plugin can be replaced with native Conv + ReLU layers.
    Using IPluginV3 for compatibility with current TensorRT versions.

    Inputs:
        - input: Input tensor [batch, in_channels, height, width]
        - weights: Kernel weights [out_channels, in_channels, kernel_h, kernel_w]
        - bias: Bias weights [out_channels] (optional)

    Tactics:
        - Tactic 1: Pointer copy (fast, minimal work)
        - Tactic 2: Content copy (slower, actual data copying)
    """

    def __init__(self, fc=None, phase=None):
        """Initialize ConvReluPlugin according to official TensorRT documentation."""
        # Initialize all parent classes
        trt.IPluginV3.__init__(self)
        trt.IPluginV3OneCore.__init__(self)
        trt.IPluginV3OneBuild.__init__(self)
        trt.IPluginV3OneRuntime.__init__(self)

        self.num_outputs = 1
        self.plugin_namespace = ""
        self.plugin_version = "1"
        self.plugin_name = "ConvReluPlugin"
        self.name = "ConvReluPlugin"

        # Default parameters
        self.kernel_size = [3, 3]
        self.stride = [1, 1]
        self.padding = [1, 1, 1, 1]  # [top, left, bottom, right] format to match demo
        self.dilation = [1, 1]
        self.groups = 1
        self.bias_term = True

        # Tactic selection (1 or 2)
        self.selected_tactic = 1

        # Timing cache ID - this is crucial for enabling timing cache
        # The ID should reflect the plugin's creation state and not evolve after creation
        # We'll use a hash of the plugin's key attributes to create a unique ID
        attr_str = f"{self.kernel_size}_{self.stride}_{self.padding}_{self.dilation}_{self.groups}_{self.bias_term}"
        self.timing_cache_id = hashlib.md5(attr_str.encode()).hexdigest()[:16]

        # Parse plugin fields if provided (for deserialization)
        if fc is not None:
            try:
                for i in range(len(fc)):
                    field = fc[i]
                    field_name = field.name
                    field_type = field.type
                    field_data = field.data

                    if (
                        field_name == "kernel_size"
                        and field_type == trt.PluginFieldType.INT32
                    ):
                        self.kernel_size = list(field_data)
                    elif (
                        field_name == "stride"
                        and field_type == trt.PluginFieldType.INT32
                    ):
                        self.stride = list(field_data)
                    elif (
                        field_name == "padding"
                        and field_type == trt.PluginFieldType.INT32
                    ):
                        self.padding = list(field_data)
                    elif (
                        field_name == "dilation"
                        and field_type == trt.PluginFieldType.INT32
                    ):
                        self.dilation = list(field_data)
                    elif (
                        field_name == "groups"
                        and field_type == trt.PluginFieldType.INT32
                    ):
                        self.groups = int(field_data[0])
                    elif (
                        field_name == "bias_term"
                        and field_type == trt.PluginFieldType.INT32
                    ):
                        self.bias_term = bool(field_data[0])

                # Recalculate timing cache ID after parsing fields
                attr_str = f"{self.kernel_size}_{self.stride}_{self.padding}_{self.dilation}_{self.groups}_{self.bias_term}"
                self.timing_cache_id = hashlib.md5(attr_str.encode()).hexdigest()[:16]
            except Exception as e:
                logger.warning(f"Error parsing plugin fields: {e}")

        # Store phase for capability interface selection
        self.phase = phase

    def get_capability_interface(self, type):
        """Get capability interface according to official TensorRT documentation."""
        # Return self like the official example
        return self

    def get_output_data_types(self, input_types):
        """Get output data types."""
        return [input_types[0]]

    def get_output_shapes(self, inputs, shape_inputs, exprBuilder):
        """Calculate output shapes using expression builder."""
        # inputs[0]: input tensor [batch, in_channels, height, width]
        # inputs[1]: weights tensor [out_channels, in_channels, kernel_h, kernel_w]
        # inputs[2]: bias tensor [out_channels] (optional)

        input_dims = inputs[0]
        weights_dims = inputs[1]

        batch_size = input_dims[0]

        # Get output channels from weights tensor shape
        # weights shape: [out_channels, in_channels, kernel_h, kernel_w]
        out_channels = weights_dims[0]

        # Handle padding - support both [top, left, bottom, right] and [h, w] formats
        if len(self.padding) == 4:
            # Format: [top, left, bottom, right]
            padding_h = exprBuilder.constant(self.padding[0])  # top
            padding_w = exprBuilder.constant(self.padding[1])  # left
        elif len(self.padding) == 2:
            # Format: [h, w] - use same padding for top/bottom and left/right
            padding_h = exprBuilder.constant(self.padding[0])
            padding_w = exprBuilder.constant(self.padding[1])
        else:
            # Fallback to zero padding
            padding_h = exprBuilder.constant(0)
            padding_w = exprBuilder.constant(0)

        # Calculate output height
        height = input_dims[2]
        kernel_h = exprBuilder.constant(self.kernel_size[0])
        stride_h = exprBuilder.constant(self.stride[0])
        dilation_h = exprBuilder.constant(self.dilation[0])

        # Standard convolution output formula: (H + 2*P - D*(K-1) - 1) / S + 1
        effective_kernel_h = exprBuilder.operation(
            trt.DimensionOperation.SUM,
            exprBuilder.constant(1),
            exprBuilder.operation(
                trt.DimensionOperation.PROD,
                exprBuilder.operation(
                    trt.DimensionOperation.SUM, kernel_h, exprBuilder.constant(-1)
                ),
                dilation_h,
            ),
        )

        height = exprBuilder.operation(
            trt.DimensionOperation.SUM,
            exprBuilder.constant(1),
            exprBuilder.operation(
                trt.DimensionOperation.FLOOR_DIV,
                exprBuilder.operation(
                    trt.DimensionOperation.SUM,
                    exprBuilder.operation(
                        trt.DimensionOperation.SUM,
                        height,
                        exprBuilder.operation(
                            trt.DimensionOperation.PROD,
                            padding_h,
                            exprBuilder.constant(2),
                        ),
                    ),
                    exprBuilder.operation(
                        trt.DimensionOperation.SUB,
                        exprBuilder.constant(0),
                        effective_kernel_h,
                    ),
                ),
                stride_h,
            ),
        )

        # Calculate output width
        width = input_dims[3]
        kernel_w = exprBuilder.constant(self.kernel_size[1])
        stride_w = exprBuilder.constant(self.stride[1])
        dilation_w = exprBuilder.constant(self.dilation[1])

        # Standard convolution output formula: (W + 2*P - D*(K-1) - 1) / S + 1
        effective_kernel_w = exprBuilder.operation(
            trt.DimensionOperation.SUM,
            exprBuilder.constant(1),
            exprBuilder.operation(
                trt.DimensionOperation.PROD,
                exprBuilder.operation(
                    trt.DimensionOperation.SUM, kernel_w, exprBuilder.constant(-1)
                ),
                dilation_w,
            ),
        )

        width = exprBuilder.operation(
            trt.DimensionOperation.SUM,
            exprBuilder.constant(1),
            exprBuilder.operation(
                trt.DimensionOperation.FLOOR_DIV,
                exprBuilder.operation(
                    trt.DimensionOperation.SUM,
                    exprBuilder.operation(
                        trt.DimensionOperation.SUM,
                        width,
                        exprBuilder.operation(
                            trt.DimensionOperation.PROD,
                            padding_w,
                            exprBuilder.constant(2),
                        ),
                    ),
                    exprBuilder.operation(
                        trt.DimensionOperation.SUB,
                        exprBuilder.constant(0),
                        effective_kernel_w,
                    ),
                ),
                stride_w,
            ),
        )

        return [trt.DimsExprs([batch_size, out_channels, height, width])]

    def get_fields_to_serialize(self):
        """Get fields to serialize."""
        fields = [
            trt.PluginField(
                "kernel_size",
                np.array(self.kernel_size, dtype=np.int32),
                trt.PluginFieldType.INT32,
            ),
            trt.PluginField(
                "stride",
                np.array(self.stride, dtype=np.int32),
                trt.PluginFieldType.INT32,
            ),
            trt.PluginField(
                "padding",
                np.array(self.padding, dtype=np.int32),
                trt.PluginFieldType.INT32,
            ),
            trt.PluginField(
                "dilation",
                np.array(self.dilation, dtype=np.int32),
                trt.PluginFieldType.INT32,
            ),
            trt.PluginField(
                "groups",
                np.array([self.groups], dtype=np.int32),
                trt.PluginFieldType.INT32,
            ),
            trt.PluginField(
                "bias_term",
                np.array([1 if self.bias_term else 0], dtype=np.int32),
                trt.PluginFieldType.INT32,
            ),
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

    def get_valid_tactics(self):
        """Return valid tactics for this plugin."""
        logger.verbose(
            f"ConvReluPlugin get_valid_tactics called, returning [1, 2] for timing_cache_id: {self.timing_cache_id}"
        )
        return [1, 2]

    def set_tactic(self, tactic):
        """Set the selected tactic for this plugin."""
        if tactic in [1, 2]:
            self.selected_tactic = tactic
            logger.verbose(
                f"ConvReluPlugin tactic set to: {tactic} for timing_cache_id: {self.timing_cache_id}"
            )
        else:
            logger.warning(f"Invalid tactic {tactic}, using default tactic 1")
            self.selected_tactic = 1

    def enqueue(self, input_desc, output_desc, inputs, outputs, workspace, stream):
        """Execute the plugin with different tactics."""
        if self.selected_tactic == 1:
            # Tactic 1: Pointer copy (fast, minimal work)
            outputs[0] = inputs[0]
            logger.verbose("ConvReluPlugin executing tactic 1: pointer copy")
        elif self.selected_tactic == 2:
            # Tactic 2: Content copy (slower, actual data copying)
            try:
                input_tensor = inputs[0]

                # Copy data from input to output
                # This is a simplified copy - in a real implementation, you'd use CUDA kernels
                # For demo purposes, we'll just do a basic copy operation
                if hasattr(input_tensor, "copy_to"):
                    input_tensor.copy_to(outputs[0])
                else:
                    # Fallback: just assign the pointer (same as tactic 1)
                    outputs[0] = inputs[0]

                logger.verbose("ConvReluPlugin executing tactic 2: content copy")
            except Exception as e:
                logger.warning(f"Error in tactic 2, falling back to tactic 1: {e}")
                outputs[0] = inputs[0]
        else:
            # Fallback to tactic 1
            outputs[0] = inputs[0]
            logger.warning(f"Unknown tactic {self.selected_tactic}, using tactic 1")

        return 0

    def attach_to_context(self, context):
        """Attach to TensorRT context."""
        return self.clone()

    def clone(self):
        """Clone the plugin."""
        plugin = ConvReluPlugin()
        plugin.kernel_size = self.kernel_size.copy()
        plugin.stride = self.stride.copy()
        plugin.padding = self.padding.copy()
        plugin.dilation = self.dilation.copy()
        plugin.groups = self.groups
        plugin.bias_term = self.bias_term
        plugin.selected_tactic = self.selected_tactic
        plugin.timing_cache_id = self.timing_cache_id
        return plugin

    def get_workspace_size(self, input_desc, output_desc):
        """Get workspace size required."""
        return 0

    def destroy(self):
        """Destroy the plugin."""
        pass


class ConvReluPluginCreator(trt.IPluginCreatorV3One):
    """Creator for ConvReluPlugin."""

    def __init__(self):
        trt.IPluginCreatorV3One.__init__(self)

        self.num_outputs = 1
        self.name = "ConvReluPlugin"
        self.plugin_name = "ConvReluPlugin"
        self.plugin_version = "1"
        self.plugin_namespace = ""
        self.field_names = trt.PluginFieldCollection([])

    def create_plugin(self, name, fc, phase):
        """Create plugin instance according to official TensorRT documentation."""
        return ConvReluPlugin(fc, phase)


def register_conv_relu_plugin():
    """Register the ConvReluPlugin with TensorRT according to official documentation."""
    try:
        # Create plugin creator
        creator = ConvReluPluginCreator()

        # Get TensorRT plugin registry
        registry = trt.get_plugin_registry()

        # Register the plugin creator
        success = registry.register_creator(creator, creator.plugin_namespace)

        if success:
            logger.verbose(
                "PluginV3 ConvReluPlugin successfully registered with TensorRT"
            )
            return creator
        else:
            logger.warning("Failed to register PluginV3 ConvReluPlugin with TensorRT")
            return None

    except Exception as e:
        logger.error(f"Error registering PluginV3 ConvReluPlugin: {e}")
        return None


def create_conv_relu_plugin_weights(
    num_output_maps: int,
    input_channels: int,
    kernel_size: List[int],
    bias_term: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Create weights for ConvReluPlugin.
    Note: These weights should be passed as input tensors to the plugin,
    not stored within the plugin itself.

    Args:
        num_output_maps: Number of output channels
        input_channels: Number of input channels
        kernel_size: Kernel size [height, width]
        bias_term: Whether to include bias

    Returns:
        Tuple of (kernel_weights, bias_weights) - these should be used as input tensors
    """
    # Create random kernel weights
    kernel_shape = (num_output_maps, input_channels, kernel_size[0], kernel_size[1])
    kernel_weights = np.random.randn(*kernel_shape).astype(np.float32) * 0.1

    # Create bias weights if needed
    bias_weights = None
    if bias_term:
        bias_weights = np.random.randn(num_output_maps).astype(np.float32) * 0.1

    return kernel_weights, bias_weights


# Auto-register the plugin when this module is imported
_creator = register_conv_relu_plugin()

if __name__ == "__main__":
    # Register the plugin (also runs on import)
    if _creator:
        logger.verbose("ConvReluPlugin is ready for use!")
    else:
        logger.error("ConvReluPlugin registration failed!")
