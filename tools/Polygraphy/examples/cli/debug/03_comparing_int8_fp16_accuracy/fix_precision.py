#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""
Network postprocessing script to set specific layers to FP32 precision.

This script is used with Polygraphy's --trt-network-postprocess-script option
to constrain certain layers to run in FP32 precision when building an INT8 engine.

Usage:
    polygraphy convert model.onnx \
        --int8 \
        --calibration-cache calibration.cache \
        --trt-network-postprocess-script fix_precision.py \
        --precision-constraints obey \
        -o model_int8_fixed.engine
"""

import tensorrt as trt


def postprocess(network):
    """
    Set specific layers to FP32 precision.
    
    This function is called by Polygraphy after parsing the network.
    It iterates through all layers and sets precision constraints for
    layers that need higher precision to maintain accuracy.
    
    Args:
        network (trt.INetworkDefinition): The TensorRT network to modify
    """
    # List of layer names that should run in FP32
    # Replace these with actual layer names identified from error analysis
    fp32_layers = [
        # Example layer names - replace with your actual problematic layers
        # "Conv_0",
        # "Conv_5",
        # "Add_10",
        # "MatMul_15",
    ]
    
    # Alternative: Set layers by type
    # Uncomment to force all layers of certain types to FP32
    fp32_layer_types = [
        # trt.LayerType.CONVOLUTION,
        # trt.LayerType.FULLY_CONNECTED,
        # trt.LayerType.MATRIX_MULTIPLY,
    ]
    
    print(f"Postprocessing network with {network.num_layers} layers")
    
    layers_modified = 0
    
    for layer_idx in range(network.num_layers):
        layer = network.get_layer(layer_idx)
        
        # Method 1: Set precision by layer name
        if layer.name in fp32_layers:
            print(f"Setting layer '{layer.name}' (type: {layer.type}) to FP32")
            layer.precision = trt.float32
            
            # Also set output type to FP32 to prevent FP16 storage
            # This is important to avoid reformatting overhead
            for output_idx in range(layer.num_outputs):
                layer.set_output_type(output_idx, trt.float32)
            
            layers_modified += 1
        
        # Method 2: Set precision by layer type
        elif layer.type in fp32_layer_types:
            print(f"Setting layer '{layer.name}' (type: {layer.type}) to FP32 by type")
            layer.precision = trt.float32
            
            for output_idx in range(layer.num_outputs):
                layer.set_output_type(output_idx, trt.float32)
            
            layers_modified += 1
    
    print(f"Modified {layers_modified} layers to FP32 precision")


def postprocess_by_pattern(network):
    """
    Alternative postprocessing function that sets layers to FP32 based on name patterns.
    
    This is useful when you want to set all layers matching a certain pattern
    (e.g., all "Conv" layers, all layers in a specific block, etc.)
    
    To use this function instead of postprocess(), specify:
        --trt-network-postprocess-script fix_precision.py:postprocess_by_pattern
    """
    # Patterns to match in layer names
    fp32_patterns = [
        "Conv",      # All convolution layers
        "block_3",   # All layers in block 3
        "head",      # All layers in the head
    ]
    
    print(f"Postprocessing network with {network.num_layers} layers")
    
    layers_modified = 0
    
    for layer_idx in range(network.num_layers):
        layer = network.get_layer(layer_idx)
        
        # Check if layer name matches any pattern
        if any(pattern in layer.name for pattern in fp32_patterns):
            print(f"Setting layer '{layer.name}' to FP32 (matched pattern)")
            layer.precision = trt.float32
            
            for output_idx in range(layer.num_outputs):
                layer.set_output_type(output_idx, trt.float32)
            
            layers_modified += 1
    
    print(f"Modified {layers_modified} layers to FP32 precision")


def postprocess_by_index(network):
    """
    Alternative postprocessing function that sets layers to FP32 based on layer indices.
    
    This is useful when you know the specific layer indices that need FP32.
    
    To use this function instead of postprocess(), specify:
        --trt-network-postprocess-script fix_precision.py:postprocess_by_index
    """
    # Layer indices that should run in FP32
    fp32_layer_indices = [
        0, 1, 2,     # First few layers
        10, 15, 20,  # Specific middle layers
        # Add more indices as needed
    ]
    
    print(f"Postprocessing network with {network.num_layers} layers")
    
    layers_modified = 0
    
    for layer_idx in fp32_layer_indices:
        if layer_idx < network.num_layers:
            layer = network.get_layer(layer_idx)
            print(f"Setting layer {layer_idx} '{layer.name}' to FP32")
            layer.precision = trt.float32
            
            for output_idx in range(layer.num_outputs):
                layer.set_output_type(output_idx, trt.float32)
            
            layers_modified += 1
    
    print(f"Modified {layers_modified} layers to FP32 precision")


def postprocess_first_n_layers(network, n=10):
    """
    Set the first N layers to FP32 precision.
    
    This is useful for debugging or when early layers are problematic.
    """
    print(f"Setting first {n} layers to FP32 precision")
    
    for layer_idx in range(min(n, network.num_layers)):
        layer = network.get_layer(layer_idx)
        print(f"Setting layer {layer_idx} '{layer.name}' to FP32")
        layer.precision = trt.float32
        
        for output_idx in range(layer.num_outputs):
            layer.set_output_type(output_idx, trt.float32)


def postprocess_last_n_layers(network, n=10):
    """
    Set the last N layers to FP32 precision.
    
    This is useful when output layers are problematic.
    """
    print(f"Setting last {n} layers to FP32 precision")
    
    start_idx = max(0, network.num_layers - n)
    
    for layer_idx in range(start_idx, network.num_layers):
        layer = network.get_layer(layer_idx)
        print(f"Setting layer {layer_idx} '{layer.name}' to FP32")
        layer.precision = trt.float32
        
        for output_idx in range(layer.num_outputs):
            layer.set_output_type(output_idx, trt.float32)


def postprocess_with_threshold(network):
    """
    Advanced: Set layers to FP32 based on dynamic criteria.
    
    This example shows how you might implement more sophisticated logic
    for determining which layers need FP32 precision.
    """
    print(f"Postprocessing network with {network.num_layers} layers")
    
    layers_modified = 0
    
    for layer_idx in range(network.num_layers):
        layer = network.get_layer(layer_idx)
        
        # Example criteria for setting FP32:
        # 1. Layers with many outputs (might be critical)
        # 2. Certain layer types
        # 3. Layers with specific characteristics
        
        should_be_fp32 = False
        
        # Criterion 1: Layers with multiple outputs
        if layer.num_outputs > 1:
            should_be_fp32 = True
            reason = "multiple outputs"
        
        # Criterion 2: Specific layer types
        elif layer.type in [trt.LayerType.MATRIX_MULTIPLY, trt.LayerType.FULLY_CONNECTED]:
            should_be_fp32 = True
            reason = "critical layer type"
        
        # Criterion 3: Layers near the output
        elif layer_idx >= network.num_layers - 5:
            should_be_fp32 = True
            reason = "near output"
        
        if should_be_fp32:
            print(f"Setting layer '{layer.name}' to FP32 ({reason})")
            layer.precision = trt.float32
            
            for output_idx in range(layer.num_outputs):
                layer.set_output_type(output_idx, trt.float32)
            
            layers_modified += 1
    
    print(f"Modified {layers_modified} layers to FP32 precision")


# Example usage in comments:
"""
# Basic usage with layer names:
polygraphy convert model.onnx \
    --int8 \
    --calibration-cache calibration.cache \
    --trt-network-postprocess-script fix_precision.py \
    --precision-constraints obey \
    -o model_int8_fixed.engine

# Using alternative function:
polygraphy convert model.onnx \
    --int8 \
    --calibration-cache calibration.cache \
    --trt-network-postprocess-script fix_precision.py:postprocess_by_pattern \
    --precision-constraints obey \
    -o model_int8_fixed.engine

# Using prefer instead of obey (allows TensorRT to override if necessary):
polygraphy convert model.onnx \
    --int8 \
    --calibration-cache calibration.cache \
    --trt-network-postprocess-script fix_precision.py \
    --precision-constraints prefer \
    -o model_int8_fixed.engine
"""
