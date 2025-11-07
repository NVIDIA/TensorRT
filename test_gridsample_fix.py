#!/usr/bin/env python3
"""
Test script to demonstrate the GridSample 5D input validation fix.

This script creates an ONNX model with 5D GridSample inputs to verify
that the fix provides a clear error message instead of a cryptic API error.

Note: This test requires the TensorRT ONNX parser to be built with the fix.
"""

import torch
import torch.nn as nn
import numpy as np

try:
    import onnx
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("Warning: onnx and onnxruntime not available. Install with: pip install onnx onnxruntime")


class GridSample5DModel(nn.Module):
    """Model that uses 5D grid_sample operation."""
    
    def forward(self, img, grid):
        # This will create a 5D GridSample operation in ONNX
        return torch.nn.functional.grid_sample(img, grid, align_corners=False)


def create_5d_gridsample_onnx():
    """Create an ONNX model with 5D GridSample inputs."""
    
    if not ONNX_AVAILABLE:
        print("Cannot create ONNX model without onnx package installed.")
        return None
    
    print("Creating ONNX model with 5D GridSample inputs...")
    
    # Create model
    model = GridSample5DModel()
    model.eval()
    
    # Create 5D dummy inputs matching the issue description
    # Input: (1, 1, 512, 32, 32) - 5D tensor
    # Grid: (1, 512, 32, 32, 3) - 5D tensor
    dummy_img = torch.ones((1, 1, 512, 32, 32), dtype=torch.float32)
    dummy_grid = torch.ones((1, 512, 32, 32, 3), dtype=torch.float32)
    
    # Export to ONNX
    onnx_path = "/tmp/gridsample_5d_test.onnx"
    
    try:
        torch.onnx.export(
            model,
            (dummy_img, dummy_grid),
            onnx_path,
            input_names=['input', 'grid'],
            output_names=['output'],
            opset_version=16,
            do_constant_folding=False
        )
        print(f"✓ ONNX model created successfully: {onnx_path}")
        
        # Verify the model
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("✓ ONNX model is valid")
        
        # Print model info
        print("\nModel Information:")
        for input_tensor in onnx_model.graph.input:
            print(f"  Input: {input_tensor.name}, Shape: {[d.dim_value for d in input_tensor.type.tensor_type.shape.dim]}")
        
        return onnx_path
        
    except Exception as e:
        print(f"✗ Error creating ONNX model: {e}")
        return None


def test_4d_gridsample_onnx():
    """Create a valid 4D GridSample ONNX model for comparison."""
    
    if not ONNX_AVAILABLE:
        print("Cannot create ONNX model without onnx package installed.")
        return None
    
    print("\n" + "="*60)
    print("Creating ONNX model with 4D GridSample inputs (valid)...")
    
    # Create model
    model = GridSample5DModel()
    model.eval()
    
    # Create 4D dummy inputs (valid for TensorRT)
    # Input: (1, 1, 32, 32) - 4D tensor
    # Grid: (1, 32, 32, 2) - 4D tensor
    dummy_img = torch.ones((1, 1, 32, 32), dtype=torch.float32)
    dummy_grid = torch.ones((1, 32, 32, 2), dtype=torch.float32)
    
    # Export to ONNX
    onnx_path = "/tmp/gridsample_4d_test.onnx"
    
    try:
        torch.onnx.export(
            model,
            (dummy_img, dummy_grid),
            onnx_path,
            input_names=['input', 'grid'],
            output_names=['output'],
            opset_version=16,
            do_constant_folding=False
        )
        print(f"✓ ONNX model created successfully: {onnx_path}")
        
        # Verify the model
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("✓ ONNX model is valid")
        
        # Print model info
        print("\nModel Information:")
        for input_tensor in onnx_model.graph.input:
            print(f"  Input: {input_tensor.name}, Shape: {[d.dim_value for d in input_tensor.type.tensor_type.shape.dim]}")
        
        return onnx_path
        
    except Exception as e:
        print(f"✗ Error creating ONNX model: {e}")
        return None


def main():
    """Main test function."""
    
    print("="*60)
    print("GridSample 5D Input Validation Test")
    print("="*60)
    print()
    print("This test demonstrates the issue described in GitHub Issue #4619")
    print("where 5D GridSample inputs cause a cryptic TensorRT API error.")
    print()
    print("With the fix applied, TensorRT will provide a clear error message:")
    print("  'TensorRT only supports 4D input tensors for GridSample.'")
    print("="*60)
    print()
    
    # Test 5D GridSample (should fail with clear error after fix)
    onnx_5d_path = create_5d_gridsample_onnx()
    
    # Test 4D GridSample (should work)
    onnx_4d_path = test_4d_gridsample_onnx()
    
    print("\n" + "="*60)
    print("Test Summary:")
    print("="*60)
    if onnx_5d_path:
        print(f"✓ 5D GridSample ONNX model created: {onnx_5d_path}")
        print("  → This model will fail TensorRT conversion with a clear error message")
        print("     after the fix is applied.")
    else:
        print("✗ Failed to create 5D GridSample ONNX model")
    
    if onnx_4d_path:
        print(f"✓ 4D GridSample ONNX model created: {onnx_4d_path}")
        print("  → This model should convert to TensorRT successfully.")
    else:
        print("✗ Failed to create 4D GridSample ONNX model")
    
    print("\n" + "="*60)
    print("Next Steps:")
    print("="*60)
    print("1. Build TensorRT with the modified ONNX parser")
    print("2. Try converting the 5D model to TensorRT:")
    print(f"   trtexec --onnx={onnx_5d_path or '/tmp/gridsample_5d_test.onnx'}")
    print("3. Verify the new error message appears:")
    print("   'TensorRT only supports 4D input tensors for GridSample.'")
    print("4. Try converting the 4D model to verify it still works:")
    print(f"   trtexec --onnx={onnx_4d_path or '/tmp/gridsample_4d_test.onnx'}")
    print("="*60)


if __name__ == "__main__":
    main()
