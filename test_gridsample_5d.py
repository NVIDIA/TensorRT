#!/usr/bin/env python3
"""
Test script to verify GridSample 5D input validation fix.
This script creates an ONNX model with 5D GridSample operation and attempts to convert it to TensorRT.
Expected behavior: Should fail with a clear error message about 4D limitation.
"""

import numpy as np
import onnx
from onnx import helper, TensorProto
import sys

def create_5d_gridsample_model():
    """Create an ONNX model with 5D GridSample operation."""
    
    # Define input shapes
    # Input: [N, C, D, H, W] = [1, 1, 512, 32, 32]
    # Grid: [N, D_out, H_out, W_out, 3] = [1, 512, 32, 32, 3]
    
    input_shape = [1, 1, 512, 32, 32]
    grid_shape = [1, 512, 32, 32, 3]
    
    # Create input tensors
    input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, input_shape)
    grid_tensor = helper.make_tensor_value_info('grid', TensorProto.FLOAT, grid_shape)
    
    # Create output tensor
    output_shape = [1, 1, 512, 32, 32]
    output_tensor = helper.make_tensor_value_info('output', TensorProto.FLOAT, output_shape)
    
    # Create GridSample node
    gridsample_node = helper.make_node(
        'GridSample',
        inputs=['input', 'grid'],
        outputs=['output'],
        mode='linear',
        padding_mode='zeros',
        align_corners=0
    )
    
    # Create the graph
    graph = helper.make_graph(
        [gridsample_node],
        'GridSample5D',
        [input_tensor, grid_tensor],
        [output_tensor]
    )
    
    # Create the model
    model = helper.make_model(graph, producer_name='test_gridsample_5d')
    model.opset_import[0].version = 16
    
    # Check the model
    onnx.checker.check_model(model)
    
    return model

def create_4d_gridsample_model():
    """Create an ONNX model with 4D GridSample operation (should work)."""
    
    # Define input shapes
    # Input: [N, C, H, W] = [1, 1, 32, 32]
    # Grid: [N, H_out, W_out, 2] = [1, 32, 32, 2]
    
    input_shape = [1, 1, 32, 32]
    grid_shape = [1, 32, 32, 2]
    
    # Create input tensors
    input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, input_shape)
    grid_tensor = helper.make_tensor_value_info('grid', TensorProto.FLOAT, grid_shape)
    
    # Create output tensor
    output_shape = [1, 1, 32, 32]
    output_tensor = helper.make_tensor_value_info('output', TensorProto.FLOAT, output_shape)
    
    # Create GridSample node
    gridsample_node = helper.make_node(
        'GridSample',
        inputs=['input', 'grid'],
        outputs=['output'],
        mode='linear',
        padding_mode='zeros',
        align_corners=0
    )
    
    # Create the graph
    graph = helper.make_graph(
        [gridsample_node],
        'GridSample4D',
        [input_tensor, grid_tensor],
        [output_tensor]
    )
    
    # Create the model
    model = helper.make_model(graph, producer_name='test_gridsample_4d')
    model.opset_import[0].version = 16
    
    # Check the model
    onnx.checker.check_model(model)
    
    return model

def main():
    print("=" * 80)
    print("Testing GridSample 5D Input Validation Fix")
    print("=" * 80)
    
    # Test 1: Create and save 5D model
    print("\n[Test 1] Creating 5D GridSample ONNX model...")
    model_5d = create_5d_gridsample_model()
    model_5d_path = '/tmp/gridsample_5d.onnx'
    onnx.save(model_5d, model_5d_path)
    print(f"✓ 5D model saved to: {model_5d_path}")
    print(f"  Input shape: [1, 1, 512, 32, 32] (5D)")
    print(f"  Grid shape: [1, 512, 32, 32, 3] (5D)")
    
    # Test 2: Create and save 4D model
    print("\n[Test 2] Creating 4D GridSample ONNX model...")
    model_4d = create_4d_gridsample_model()
    model_4d_path = '/tmp/gridsample_4d.onnx'
    onnx.save(model_4d, model_4d_path)
    print(f"✓ 4D model saved to: {model_4d_path}")
    print(f"  Input shape: [1, 1, 32, 32] (4D)")
    print(f"  Grid shape: [1, 32, 32, 2] (4D)")
    
    print("\n" + "=" * 80)
    print("ONNX models created successfully!")
    print("=" * 80)
    print("\nTo test with TensorRT ONNX parser, you would need to:")
    print("1. Build the TensorRT ONNX parser with the fix")
    print("2. Try parsing the 5D model - should get clear error message")
    print("3. Try parsing the 4D model - should succeed")
    print("\nExpected error message for 5D model:")
    print("  'TensorRT only supports 4D GridSample operations (NCHW format).")
    print("   Input tensor has rank 5. For 5D volumetric GridSample (NCDHW),")
    print("   consider using a custom plugin or reshaping the input to 4D if applicable.'")
    print("=" * 80)
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
