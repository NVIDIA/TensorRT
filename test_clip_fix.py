#!/usr/bin/env python3
"""
Test script to reproduce GitHub issue #4606:
Clip layer upper bound not respected by TRT 10.x in MatMul->Add->Clip chains

This script creates a minimal ONNX model with MatMul->Add->Clip(min=0, max=6) pattern
and saves it for testing with TensorRT.
"""

import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

def create_matmul_add_clip_model():
    """
    Create an ONNX model with the pattern: MatMul -> Add -> Clip(min=0, max=6)
    This reproduces the issue where TensorRT incorrectly ignores the upper bound.
    """
    
    # Define input shapes
    matmul_input_shape = [2, 3]
    matmul_weight_shape = [3, 4]
    add_bias_shape = [4]
    
    # Create input tensor
    matmul_input = helper.make_tensor_value_info('matmul_input', TensorProto.FLOAT, matmul_input_shape)
    
    # Create weight initializer for MatMul
    matmul_weight_data = np.random.randn(*matmul_weight_shape).astype(np.float32)
    matmul_weight = numpy_helper.from_array(matmul_weight_data, name='matmul_weight')
    
    # Create bias initializer for Add
    add_bias_data = np.random.randn(*add_bias_shape).astype(np.float32)
    add_bias = numpy_helper.from_array(add_bias_data, name='add_bias')
    
    # Create min and max initializers for Clip (opset 11+)
    clip_min = numpy_helper.from_array(np.array(0.0, dtype=np.float32), name='clip_min')
    clip_max = numpy_helper.from_array(np.array(6.0, dtype=np.float32), name='clip_max')
    
    # Create MatMul node
    matmul_node = helper.make_node(
        'MatMul',
        inputs=['matmul_input', 'matmul_weight'],
        outputs=['matmul_output']
    )
    
    # Create Add node
    add_node = helper.make_node(
        'Add',
        inputs=['matmul_output', 'add_bias'],
        outputs=['add_output']
    )
    
    # Create Clip node (opset 11+ uses inputs instead of attributes)
    clip_node = helper.make_node(
        'Clip',
        inputs=['add_output', 'clip_min', 'clip_max'],
        outputs=['clip_output']
    )
    
    # Create output tensor
    clip_output = helper.make_tensor_value_info('clip_output', TensorProto.FLOAT, [2, 4])
    
    # Create the graph
    graph = helper.make_graph(
        nodes=[matmul_node, add_node, clip_node],
        name='MatMulAddClipGraph',
        inputs=[matmul_input],
        outputs=[clip_output],
        initializer=[matmul_weight, add_bias, clip_min, clip_max]
    )
    
    # Create the model
    model = helper.make_model(graph, producer_name='test_clip_fix')
    model.opset_import[0].version = 13  # Use opset 13
    
    # Check the model
    onnx.checker.check_model(model)
    
    return model

def create_simple_clip_model():
    """
    Create a simpler ONNX model with just Clip(min=0, max=6) for basic testing.
    """
    
    # Create input tensor
    input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [2, 4])
    
    # Create min and max initializers for Clip
    clip_min = numpy_helper.from_array(np.array(0.0, dtype=np.float32), name='clip_min')
    clip_max = numpy_helper.from_array(np.array(6.0, dtype=np.float32), name='clip_max')
    
    # Create Clip node
    clip_node = helper.make_node(
        'Clip',
        inputs=['input', 'clip_min', 'clip_max'],
        outputs=['output']
    )
    
    # Create output tensor
    output_tensor = helper.make_tensor_value_info('output', TensorProto.FLOAT, [2, 4])
    
    # Create the graph
    graph = helper.make_graph(
        nodes=[clip_node],
        name='SimpleClipGraph',
        inputs=[input_tensor],
        outputs=[output_tensor],
        initializer=[clip_min, clip_max]
    )
    
    # Create the model
    model = helper.make_model(graph, producer_name='test_clip_fix')
    model.opset_import[0].version = 13
    
    # Check the model
    onnx.checker.check_model(model)
    
    return model

if __name__ == '__main__':
    # Create and save the MatMul->Add->Clip model
    print("Creating MatMul->Add->Clip model...")
    model = create_matmul_add_clip_model()
    output_path = '/vercel/sandbox/matmul_add_clip_test.onnx'
    onnx.save(model, output_path)
    print(f"Model saved to: {output_path}")
    
    # Create and save the simple Clip model
    print("\nCreating simple Clip model...")
    simple_model = create_simple_clip_model()
    simple_output_path = '/vercel/sandbox/simple_clip_test.onnx'
    onnx.save(simple_model, simple_output_path)
    print(f"Simple model saved to: {simple_output_path}")
    
    print("\nModels created successfully!")
    print("\nTo test with polygraphy, run:")
    print(f"  polygraphy run --trt --onnxrt {output_path} --val-range [-10,10] --iterations 10")
