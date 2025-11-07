# Solution for GitHub Issue #4608: TensorRT Engine Hangs with PyTorch and PyCUDA

## Problem Summary

When importing PyTorch before initializing a TensorRT engine with PyCUDA, the program hangs at `engine.get_binding_shape()`. This is caused by a CUDA context conflict between PyTorch and PyCUDA/TensorRT.

## Root Cause

The issue occurs because:

1. **PyTorch creates its own CUDA context** when imported, which becomes the active context
2. **PyCUDA expects to manage its own CUDA context** and may conflict with PyTorch's context
3. **TensorRT operations** (like `get_binding_shape()`) require a valid CUDA context, and when there's a conflict between PyTorch's and PyCUDA's contexts, the operation hangs

## Solutions

### Solution 1: Use `cuda-python` Instead of PyCUDA (Recommended)

The TensorRT team has migrated samples from PyCUDA to `cuda-python` to avoid these conflicts and support newer GPUs. This is the **recommended approach**.

**Benefits:**
- No context conflicts with PyTorch
- Better support for modern GPUs
- Official NVIDIA CUDA Python bindings
- More maintainable and future-proof

**Implementation:**

```python
import torch  # Can import PyTorch without issues
import tensorrt as trt
from cuda import cudart  # Use cuda-python instead of pycuda
import numpy as np

def check_cuda_error(error):
    """Helper function to check CUDA errors"""
    if isinstance(error, tuple):
        error = error[0]
    if error != cudart.cudaError_t.cudaSuccess:
        error_name = cudart.cudaGetErrorName(error)[1]
        error_string = cudart.cudaGetErrorString(error)[1]
        raise RuntimeError(f"CUDA Error: {error_name} ({error_string})")

class TRTInference:
    def __init__(self, engine_path: str):
        # Initialize TensorRT logger and runtime
        self.logger = trt.Logger(trt.Logger.ERROR)
        
        # Load the TensorRT engine
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
        if self.engine is None:
            raise RuntimeError("Failed to load TensorRT engine")
        
        # Create execution context
        self.context = self.engine.create_execution_context()
        
        # Get binding information - this now works without hanging
        self.bindings = []
        self.allocations = []
        
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            dtype = self.engine.get_tensor_dtype(name)
            shape = self.engine.get_tensor_shape(name)
            is_input = self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
            
            # Calculate size and allocate GPU memory
            size = np.dtype(trt.nptype(dtype)).itemsize
            for s in shape:
                size *= s
            
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
        
        # Create CUDA stream
        err, self.stream = cudart.cudaStreamCreate()
        check_cuda_error(err)
    
    def infer(self, input_data: np.ndarray) -> np.ndarray:
        """Run inference on input data"""
        # Ensure input is contiguous
        input_data = np.ascontiguousarray(input_data)
        
        # Copy input to GPU
        input_binding = [b for b in self.bindings if b["is_input"]][0]
        err = cudart.cudaMemcpy(
            input_binding["allocation"],
            input_data.ctypes.data,
            input_binding["size"],
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
        )
        check_cuda_error(err)
        
        # Set tensor addresses
        for i, binding in enumerate(self.bindings):
            self.context.set_tensor_address(binding["name"], self.allocations[i])
        
        # Execute inference
        self.context.execute_async_v3(stream_handle=self.stream)
        err = cudart.cudaStreamSynchronize(self.stream)
        check_cuda_error(err)
        
        # Copy output from GPU
        output_binding = [b for b in self.bindings if not b["is_input"]][0]
        output = np.empty(output_binding["shape"], dtype=output_binding["dtype"])
        err = cudart.cudaMemcpy(
            output.ctypes.data,
            output_binding["allocation"],
            output_binding["size"],
            cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost
        )
        check_cuda_error(err)
        
        return output
    
    def __del__(self):
        """Cleanup GPU resources"""
        # Free GPU memory
        for allocation in self.allocations:
            cudart.cudaFree(allocation)
        
        # Destroy stream
        if hasattr(self, 'stream'):
            cudart.cudaStreamDestroy(self.stream)

# Example usage
if __name__ == "__main__":
    # PyTorch can be imported and used without conflicts
    import torch
    
    # Create some PyTorch tensors (optional)
    torch_tensor = torch.randn(1, 3, 224, 224).cuda()
    print(f"PyTorch tensor shape: {torch_tensor.shape}")
    
    # Initialize TensorRT inference - no hanging!
    trt_inference = TRTInference("model.trt")
    
    # Run inference
    input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
    output = trt_inference.infer(input_data)
    print(f"TensorRT output shape: {output.shape}")
```

**Installation:**

```bash
pip install cuda-python tensorrt
```

### Solution 2: Proper PyCUDA Context Management (Alternative)

If you must use PyCUDA, you need to properly manage CUDA contexts to avoid conflicts with PyTorch.

**Implementation:**

```python
import tensorrt as trt
import numpy as np

# IMPORTANT: Import torch AFTER pycuda initialization
import pycuda.driver as cuda
import pycuda.autoinit  # This initializes CUDA context

# NOW import torch
import torch

class TRTInference:
    def __init__(self, engine_path: str):
        # Make sure PyCUDA context is active
        cuda.init()
        
        # Get the current context (created by pycuda.autoinit)
        self.cuda_ctx = cuda.Device(0).retain_primary_context()
        self.cuda_ctx.push()
        
        # Initialize TensorRT
        self.logger = trt.Logger(trt.Logger.ERROR)
        
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
        if self.engine is None:
            raise RuntimeError("Failed to load TensorRT engine")
        
        # Now get_binding_shape should work
        self.bindings = []
        for i in range(self.engine.num_bindings):
            shape = tuple(self.engine.get_binding_shape(i))
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            name = self.engine.get_binding_name(i)
            is_input = self.engine.binding_is_input(i)
            
            self.bindings.append({
                "name": name,
                "shape": shape,
                "dtype": dtype,
                "is_input": is_input
            })
        
        self.context = self.engine.create_execution_context()
    
    def infer(self, input_data: np.ndarray) -> np.ndarray:
        """Run inference"""
        # Ensure context is active
        self.cuda_ctx.push()
        
        try:
            # Allocate device memory
            d_input = cuda.mem_alloc(input_data.nbytes)
            
            # Get output shape and allocate
            output_shape = [b["shape"] for b in self.bindings if not b["is_input"]][0]
            output_dtype = [b["dtype"] for b in self.bindings if not b["is_input"]][0]
            output = np.empty(output_shape, dtype=output_dtype)
            d_output = cuda.mem_alloc(output.nbytes)
            
            # Copy input to device
            cuda.memcpy_htod(d_input, input_data)
            
            # Create bindings list
            bindings = [int(d_input), int(d_output)]
            
            # Execute
            self.context.execute_v2(bindings=bindings)
            
            # Copy output to host
            cuda.memcpy_dtoh(output, d_output)
            
            return output
        finally:
            self.cuda_ctx.pop()
    
    def __del__(self):
        """Cleanup"""
        if hasattr(self, 'cuda_ctx'):
            self.cuda_ctx.pop()

# Example usage
if __name__ == "__main__":
    # Initialize TensorRT first
    trt_inference = TRTInference("model.trt")
    
    # Now you can use PyTorch
    torch_tensor = torch.randn(1, 3, 224, 224).cuda()
    
    # Run TensorRT inference
    input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
    output = trt_inference.infer(input_data)
```

### Solution 3: Separate Processes (For Complex Scenarios)

If you need to use both PyTorch and TensorRT extensively, consider running them in separate processes:

```python
import multiprocessing as mp
import numpy as np

def pytorch_process(input_queue, output_queue):
    """Process that handles PyTorch operations"""
    import torch
    
    while True:
        data = input_queue.get()
        if data is None:
            break
        
        # PyTorch operations
        tensor = torch.from_numpy(data).cuda()
        result = tensor.cpu().numpy()
        output_queue.put(result)

def tensorrt_process(input_queue, output_queue):
    """Process that handles TensorRT operations"""
    import tensorrt as trt
    from cuda import cudart
    
    # Initialize TensorRT (no PyTorch imported here)
    # ... TensorRT inference code ...
    
    while True:
        data = input_queue.get()
        if data is None:
            break
        
        # TensorRT inference
        result = run_trt_inference(data)
        output_queue.put(result)

# Main process coordinates between PyTorch and TensorRT
if __name__ == "__main__":
    pytorch_in = mp.Queue()
    pytorch_out = mp.Queue()
    tensorrt_in = mp.Queue()
    tensorrt_out = mp.Queue()
    
    # Start processes
    p1 = mp.Process(target=pytorch_process, args=(pytorch_in, pytorch_out))
    p2 = mp.Process(target=tensorrt_process, args=(tensorrt_in, tensorrt_out))
    
    p1.start()
    p2.start()
    
    # Use both without conflicts
    # ...
```

## Recommended Migration Path

1. **Install cuda-python**: `pip install cuda-python`
2. **Replace PyCUDA imports** with cuda-python equivalents:
   - `import pycuda.driver as cuda` → `from cuda import cudart`
   - `cuda.mem_alloc()` → `cudart.cudaMalloc()`
   - `cuda.memcpy_htod()` → `cudart.cudaMemcpy(..., cudaMemcpyHostToDevice)`
   - `cuda.memcpy_dtoh()` → `cudart.cudaMemcpy(..., cudaMemcpyDeviceToHost)`
3. **Update TensorRT API calls** to use modern APIs:
   - Use `engine.num_io_tensors` instead of `engine.num_bindings`
   - Use `engine.get_tensor_name()` instead of `engine.get_binding_name()`
   - Use `context.execute_async_v3()` instead of `context.execute_v2()`

## References

- [TensorRT Refactored Samples](samples/python/refactored/) - Examples using cuda-python
- [TensorRT Changelog](CHANGELOG.md) - See 10.13.0 GA release notes about cuda-python migration
- [CUDA Python Documentation](https://nvidia.github.io/cuda-python/)
- [TensorRT Python API](https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/)

## Additional Notes

- The TensorRT team officially migrated from PyCUDA to cuda-python in version 10.13.0
- cuda-python provides better support for modern GPUs and CUDA versions
- The quickstart guide explicitly warns: "TensorRT and PyTorch can not be loaded into your Python processes at the same time" when using PyCUDA
- Using cuda-python resolves this limitation

## Testing the Solution

To verify the fix works:

```python
# This should NOT hang anymore
import torch  # Import PyTorch first
import tensorrt as trt
from cuda import cudart

# Initialize CUDA
err = cudart.cudaSetDevice(0)
assert err[0] == cudart.cudaError_t.cudaSuccess

# Load TensorRT engine
logger = trt.Logger(trt.Logger.ERROR)
with open("model.trt", "rb") as f:
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(f.read())

# This should work without hanging
for i in range(engine.num_io_tensors):
    name = engine.get_tensor_name(i)
    shape = engine.get_tensor_shape(name)
    print(f"Tensor {i}: {name}, shape: {shape}")

print("Success! No hanging occurred.")
```
