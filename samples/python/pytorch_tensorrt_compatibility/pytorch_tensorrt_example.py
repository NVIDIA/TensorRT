#!/usr/bin/env python3
"""
SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Example: Using PyTorch and TensorRT Together Without Context Conflicts

This example demonstrates how to use PyTorch and TensorRT in the same Python process
without encountering CUDA context conflicts. It uses cuda-python instead of PyCUDA
to avoid the hanging issue described in GitHub Issue #4608.

The key insight is that cuda-python (NVIDIA's official CUDA Python bindings) properly
manages CUDA contexts and doesn't conflict with PyTorch's CUDA context management.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

# Import PyTorch first - this is now safe with cuda-python
try:
    import torch
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    print("Warning: PyTorch not available. PyTorch features will be disabled.")

# Import TensorRT and cuda-python
try:
    import tensorrt as trt
    from cuda import cudart
    TENSORRT_AVAILABLE = True
except ImportError as e:
    TENSORRT_AVAILABLE = False
    print(f"Error: TensorRT or cuda-python not available: {e}")
    print("Please install: pip install tensorrt cuda-python")
    sys.exit(1)


def check_cuda_error(error):
    """
    Helper function to check CUDA errors from cuda-python calls.
    
    Args:
        error: CUDA error code or tuple containing error code
        
    Raises:
        RuntimeError: If CUDA error occurred
    """
    if isinstance(error, tuple):
        error = error[0]
    if error != cudart.cudaError_t.cudaSuccess:
        error_name = cudart.cudaGetErrorName(error)[1]
        error_string = cudart.cudaGetErrorString(error)[1]
        raise RuntimeError(f"CUDA Error: {error_name} ({error_string})")


class TensorRTInference:
    """
    TensorRT inference wrapper using cuda-python for CUDA operations.
    
    This class demonstrates the recommended approach for using TensorRT with PyTorch
    in the same process. By using cuda-python instead of PyCUDA, we avoid CUDA
    context conflicts that cause hangs in get_binding_shape() and other operations.
    """
    
    def __init__(self, engine_path: str, verbose: bool = False):
        """
        Initialize TensorRT inference engine.
        
        Args:
            engine_path: Path to serialized TensorRT engine (.trt or .plan file)
            verbose: Enable verbose logging
        """
        self.verbose = verbose
        
        # Initialize TensorRT logger
        log_level = trt.Logger.INFO if verbose else trt.Logger.WARNING
        self.logger = trt.Logger(log_level)
        
        # Load the TensorRT engine
        if self.verbose:
            print(f"Loading TensorRT engine from: {engine_path}")
        
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
        if self.engine is None:
            raise RuntimeError(f"Failed to load TensorRT engine from {engine_path}")
        
        # Create execution context
        self.context = self.engine.create_execution_context()
        
        # Get binding information
        # NOTE: This is where the hang would occur with PyCUDA + PyTorch
        # With cuda-python, this works without issues!
        self.bindings = []
        self.allocations = []
        
        if self.verbose:
            print(f"\nEngine has {self.engine.num_io_tensors} I/O tensors:")
        
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            dtype = self.engine.get_tensor_dtype(name)
            shape = self.engine.get_tensor_shape(name)
            is_input = self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
            
            # Calculate size and allocate GPU memory
            size = np.dtype(trt.nptype(dtype)).itemsize
            for s in shape:
                size *= s
            
            # Allocate GPU memory using cuda-python
            err, allocation = cudart.cudaMalloc(size)
            check_cuda_error(err)
            
            binding = {
                "name": name,
                "dtype": np.dtype(trt.nptype(dtype)),
                "shape": list(shape),
                "allocation": allocation,
                "size": size,
                "is_input": is_input
            }
            
            self.bindings.append(binding)
            self.allocations.append(allocation)
            
            if self.verbose:
                io_type = "Input" if is_input else "Output"
                print(f"  [{i}] {io_type}: {name}, shape: {shape}, dtype: {dtype}")
        
        # Create CUDA stream for asynchronous execution
        err, self.stream = cudart.cudaStreamCreate()
        check_cuda_error(err)
        
        if self.verbose:
            print("\nTensorRT engine initialized successfully!")
    
    def infer(self, input_data: np.ndarray) -> np.ndarray:
        """
        Run inference on input data.
        
        Args:
            input_data: Input numpy array
            
        Returns:
            Output numpy array
        """
        # Ensure input is contiguous in memory
        input_data = np.ascontiguousarray(input_data)
        
        # Get input binding
        input_bindings = [b for b in self.bindings if b["is_input"]]
        if not input_bindings:
            raise RuntimeError("No input bindings found")
        input_binding = input_bindings[0]
        
        # Validate input shape
        if list(input_data.shape) != input_binding["shape"]:
            raise ValueError(
                f"Input shape mismatch. Expected {input_binding['shape']}, "
                f"got {list(input_data.shape)}"
            )
        
        # Copy input to GPU
        err = cudart.cudaMemcpy(
            input_binding["allocation"],
            input_data.ctypes.data,
            input_binding["size"],
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
        )
        check_cuda_error(err)
        
        # Set tensor addresses for all I/O tensors
        for i, binding in enumerate(self.bindings):
            self.context.set_tensor_address(binding["name"], self.allocations[i])
        
        # Execute inference asynchronously
        self.context.execute_async_v3(stream_handle=self.stream)
        
        # Wait for completion
        err = cudart.cudaStreamSynchronize(self.stream)
        check_cuda_error(err)
        
        # Get output binding
        output_bindings = [b for b in self.bindings if not b["is_input"]]
        if not output_bindings:
            raise RuntimeError("No output bindings found")
        output_binding = output_bindings[0]
        
        # Allocate output array
        output = np.empty(output_binding["shape"], dtype=output_binding["dtype"])
        
        # Copy output from GPU
        err = cudart.cudaMemcpy(
            output.ctypes.data,
            output_binding["allocation"],
            output_binding["size"],
            cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost
        )
        check_cuda_error(err)
        
        return output
    
    def __del__(self):
        """Cleanup GPU resources."""
        # Free GPU memory
        for allocation in self.allocations:
            cudart.cudaFree(allocation)
        
        # Destroy stream
        if hasattr(self, 'stream'):
            cudart.cudaStreamDestroy(self.stream)


def demonstrate_pytorch_tensorrt_compatibility(
    engine_path: str,
    use_pytorch: bool = True,
    verbose: bool = False
) -> None:
    """
    Demonstrate that PyTorch and TensorRT can work together without conflicts.
    
    Args:
        engine_path: Path to TensorRT engine file
        use_pytorch: Whether to demonstrate PyTorch operations
        verbose: Enable verbose output
    """
    print("=" * 80)
    print("PyTorch + TensorRT Compatibility Demonstration")
    print("=" * 80)
    print()
    
    # Step 1: Use PyTorch (if available)
    if use_pytorch and PYTORCH_AVAILABLE:
        print("Step 1: Creating PyTorch tensors and running operations...")
        print("-" * 80)
        
        # Create PyTorch tensors
        torch_tensor = torch.randn(1, 3, 224, 224)
        print(f"Created PyTorch tensor with shape: {torch_tensor.shape}")
        
        # Move to GPU if available
        if torch.cuda.is_available():
            torch_tensor = torch_tensor.cuda()
            print(f"Moved tensor to GPU: {torch_tensor.device}")
            
            # Perform some operations
            result = torch.nn.functional.relu(torch_tensor)
            print(f"Applied ReLU activation, output shape: {result.shape}")
        else:
            print("CUDA not available for PyTorch, using CPU")
        
        print()
    
    # Step 2: Initialize TensorRT
    print("Step 2: Initializing TensorRT engine...")
    print("-" * 80)
    
    try:
        trt_inference = TensorRTInference(engine_path, verbose=verbose)
        print("✓ TensorRT engine initialized successfully (no hanging!)")
        print()
    except FileNotFoundError:
        print(f"Error: Engine file not found: {engine_path}")
        print("Please provide a valid TensorRT engine file.")
        return
    except Exception as e:
        print(f"Error initializing TensorRT: {e}")
        return
    
    # Step 3: Run TensorRT inference
    print("Step 3: Running TensorRT inference...")
    print("-" * 80)
    
    # Get input shape from engine
    input_binding = [b for b in trt_inference.bindings if b["is_input"]][0]
    input_shape = input_binding["shape"]
    input_dtype = input_binding["dtype"]
    
    print(f"Creating random input with shape: {input_shape}, dtype: {input_dtype}")
    input_data = np.random.randn(*input_shape).astype(input_dtype)
    
    # Run inference
    output = trt_inference.infer(input_data)
    print(f"✓ Inference completed successfully!")
    print(f"Output shape: {output.shape}, dtype: {output.dtype}")
    print()
    
    # Step 4: Demonstrate interoperability
    if use_pytorch and PYTORCH_AVAILABLE and torch.cuda.is_available():
        print("Step 4: Demonstrating PyTorch ↔ TensorRT interoperability...")
        print("-" * 80)
        
        # Convert TensorRT output to PyTorch tensor
        torch_output = torch.from_numpy(output).cuda()
        print(f"Converted TensorRT output to PyTorch tensor: {torch_output.shape}")
        
        # Perform PyTorch operations on TensorRT output
        processed = torch.nn.functional.softmax(torch_output, dim=-1)
        print(f"Applied softmax using PyTorch: {processed.shape}")
        print()
    
    print("=" * 80)
    print("SUCCESS! PyTorch and TensorRT work together without conflicts!")
    print("=" * 80)
    print()
    print("Key takeaways:")
    print("  1. Using cuda-python instead of PyCUDA avoids CUDA context conflicts")
    print("  2. PyTorch can be imported before or after TensorRT initialization")
    print("  3. Both frameworks can be used in the same Python process")
    print("  4. Data can be easily transferred between PyTorch and TensorRT")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Demonstrate PyTorch and TensorRT compatibility using cuda-python",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with a TensorRT engine file
  python pytorch_tensorrt_example.py --engine model.trt
  
  # Run with verbose output
  python pytorch_tensorrt_example.py --engine model.trt --verbose
  
  # Run without PyTorch operations
  python pytorch_tensorrt_example.py --engine model.trt --no-pytorch

Note: This example requires a pre-built TensorRT engine file.
You can create one using trtexec or the TensorRT Python API.
        """
    )
    
    parser.add_argument(
        "--engine",
        type=str,
        required=True,
        help="Path to TensorRT engine file (.trt or .plan)"
    )
    
    parser.add_argument(
        "--no-pytorch",
        action="store_true",
        help="Disable PyTorch operations (only test TensorRT)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Check if engine file exists
    if not Path(args.engine).exists():
        print(f"Error: Engine file not found: {args.engine}")
        print("\nTo create a TensorRT engine, you can use trtexec:")
        print("  trtexec --onnx=model.onnx --saveEngine=model.trt")
        sys.exit(1)
    
    # Run demonstration
    demonstrate_pytorch_tensorrt_compatibility(
        engine_path=args.engine,
        use_pytorch=not args.no_pytorch,
        verbose=args.verbose
    )


if __name__ == "__main__":
    main()
