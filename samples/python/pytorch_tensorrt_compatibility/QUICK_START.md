# Quick Start Guide: PyTorch + TensorRT Without Conflicts

## The Problem

```python
import torch  # ← Importing PyTorch first
import pycuda.driver as cuda
import tensorrt as trt

# This hangs! 😱
shape = engine.get_binding_shape(i)
```

## The Solution

```python
import torch  # ← Now safe to import PyTorch first! ✅
import tensorrt as trt
from cuda import cudart  # ← Use cuda-python instead of PyCUDA

# This works! 🎉
shape = engine.get_tensor_shape(name)
```

## Installation

```bash
pip install tensorrt cuda-python torch numpy
```

## Minimal Working Example

```python
#!/usr/bin/env python3
import torch
import tensorrt as trt
from cuda import cudart
import numpy as np

def check_cuda_error(error):
    if isinstance(error, tuple):
        error = error[0]
    if error != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"CUDA Error: {cudart.cudaGetErrorString(error)[1]}")

# 1. Use PyTorch (no conflicts!)
torch_tensor = torch.randn(1, 3, 224, 224).cuda()
print(f"PyTorch tensor: {torch_tensor.shape}")

# 2. Load TensorRT engine
logger = trt.Logger(trt.Logger.WARNING)
with open("model.trt", "rb") as f:
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(f.read())

# 3. Get tensor info (no hanging!)
for i in range(engine.num_io_tensors):
    name = engine.get_tensor_name(i)
    shape = engine.get_tensor_shape(name)
    print(f"Tensor: {name}, shape: {shape}")

# 4. Run inference
context = engine.create_execution_context()

# Allocate GPU memory
input_shape = [1, 3, 224, 224]
input_size = np.prod(input_shape) * np.float32().itemsize
err, d_input = cudart.cudaMalloc(input_size)
check_cuda_error(err)

output_shape = [1, 1000]
output_size = np.prod(output_shape) * np.float32().itemsize
err, d_output = cudart.cudaMalloc(output_size)
check_cuda_error(err)

# Prepare input
input_data = np.random.randn(*input_shape).astype(np.float32)
input_data = np.ascontiguousarray(input_data)

# Copy to GPU
err = cudart.cudaMemcpy(
    d_input,
    input_data.ctypes.data,
    input_size,
    cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
)
check_cuda_error(err)

# Set tensor addresses
context.set_tensor_address("input", d_input)
context.set_tensor_address("output", d_output)

# Execute
err, stream = cudart.cudaStreamCreate()
check_cuda_error(err)

context.execute_async_v3(stream_handle=stream)
err = cudart.cudaStreamSynchronize(stream)
check_cuda_error(err)

# Copy output back
output = np.empty(output_shape, dtype=np.float32)
err = cudart.cudaMemcpy(
    output.ctypes.data,
    d_output,
    output_size,
    cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost
)
check_cuda_error(err)

print(f"Output shape: {output.shape}")

# Cleanup
cudart.cudaFree(d_input)
cudart.cudaFree(d_output)
cudart.cudaStreamDestroy(stream)

print("Success! PyTorch and TensorRT work together! 🎉")
```

## Key Differences: PyCUDA vs cuda-python

| Operation | PyCUDA (Old) | cuda-python (New) |
|-----------|--------------|-------------------|
| Import | `import pycuda.driver as cuda` | `from cuda import cudart` |
| Allocate | `d = cuda.mem_alloc(size)` | `err, d = cudart.cudaMalloc(size)` |
| Copy H→D | `cuda.memcpy_htod(d, h)` | `cudart.cudaMemcpy(d, h.ctypes.data, size, cudaMemcpyHostToDevice)` |
| Copy D→H | `cuda.memcpy_dtoh(h, d)` | `cudart.cudaMemcpy(h.ctypes.data, d, size, cudaMemcpyDeviceToHost)` |
| Free | `d.free()` | `cudart.cudaFree(d)` |
| Stream | `s = cuda.Stream()` | `err, s = cudart.cudaStreamCreate()` |

## Common Errors & Fixes

### Error: "Module 'cuda' has no attribute 'cudart'"

**Fix:** Install cuda-python
```bash
pip install cuda-python
```

### Error: "CUDA Error: invalid device context"

**Fix:** Make sure you're using cuda-python, not PyCUDA
```python
# Wrong
import pycuda.driver as cuda

# Correct
from cuda import cudart
```

### Error: "Engine file not found"

**Fix:** Create a TensorRT engine first
```bash
trtexec --onnx=model.onnx --saveEngine=model.trt
```

## Full Example

See [pytorch_tensorrt_example.py](pytorch_tensorrt_example.py) for a complete, production-ready implementation.

## Run the Example

```bash
# Create a TensorRT engine (if you don't have one)
trtexec --onnx=model.onnx --saveEngine=model.trt

# Run the example
python pytorch_tensorrt_example.py --engine model.trt --verbose
```

## Learn More

- [Full Solution Document](../../../ISSUE_4608_SOLUTION.md)
- [Detailed README](README.md)
- [TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/)
- [cuda-python Documentation](https://nvidia.github.io/cuda-python/)

## Why This Works

**PyCUDA approach:**
- PyTorch creates CUDA context A
- PyCUDA tries to create CUDA context B
- Conflict! → Hang 😱

**cuda-python approach:**
- PyTorch creates CUDA context
- cuda-python uses the existing context
- No conflict! → Works ✅

## Summary

1. ❌ **Don't use:** PyCUDA with PyTorch
2. ✅ **Do use:** cuda-python with PyTorch
3. 🎯 **Result:** No more hanging, seamless integration!
