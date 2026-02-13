# PyTorch and TensorRT Compatibility Example

This example demonstrates how to use PyTorch and TensorRT together in the same Python process without encountering CUDA context conflicts.

## Problem Statement

When using PyCUDA with TensorRT, importing PyTorch before initializing a TensorRT engine causes the program to hang at operations like `get_binding_shape()`. This is due to CUDA context conflicts between PyTorch and PyCUDA.

**Related Issue:** [GitHub Issue #4608](https://github.com/NVIDIA/TensorRT/issues/4608)

## Solution

Use **cuda-python** (NVIDIA's official CUDA Python bindings) instead of PyCUDA. This avoids CUDA context conflicts and allows PyTorch and TensorRT to coexist peacefully in the same process.

## Key Benefits

1. ✅ **No CUDA context conflicts** - PyTorch and TensorRT work together seamlessly
2. ✅ **Import order doesn't matter** - Import PyTorch before or after TensorRT
3. ✅ **Better GPU support** - cuda-python supports newer GPUs and CUDA versions
4. ✅ **Official NVIDIA support** - cuda-python is the recommended approach by the TensorRT team
5. ✅ **Easy data transfer** - Seamlessly move data between PyTorch and TensorRT

## Requirements

```bash
pip install tensorrt cuda-python torch numpy
```

**Minimum versions:**
- Python 3.10+
- TensorRT 8.0+
- CUDA 11.0+
- PyTorch 1.10+ (optional, for PyTorch features)

## Usage

### Basic Usage

```bash
python pytorch_tensorrt_example.py --engine model.trt
```

### With Verbose Output

```bash
python pytorch_tensorrt_example.py --engine model.trt --verbose
```

### Without PyTorch Operations

```bash
python pytorch_tensorrt_example.py --engine model.trt --no-pytorch
```

## Creating a TensorRT Engine

If you don't have a TensorRT engine file, you can create one using `trtexec`:

```bash
# From ONNX model
trtexec --onnx=model.onnx --saveEngine=model.trt

# With FP16 precision
trtexec --onnx=model.onnx --saveEngine=model.trt --fp16

# With specific batch size
trtexec --onnx=model.onnx --saveEngine=model.trt --explicitBatch --minShapes=input:1x3x224x224 --optShapes=input:1x3x224x224 --maxShapes=input:1x3x224x224
```

## Code Example

```python
import torch
import tensorrt as trt
from cuda import cudart
import numpy as np

# Import PyTorch first - no problem!
torch_tensor = torch.randn(1, 3, 224, 224).cuda()

# Initialize TensorRT - no hanging!
logger = trt.Logger(trt.Logger.WARNING)
with open("model.trt", "rb") as f:
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(f.read())

# Get binding information - works perfectly!
for i in range(engine.num_io_tensors):
    name = engine.get_tensor_name(i)
    shape = engine.get_tensor_shape(name)
    print(f"Tensor {i}: {name}, shape: {shape}")

# Run inference
context = engine.create_execution_context()
# ... inference code ...
```

## Migration from PyCUDA

If you're migrating from PyCUDA to cuda-python, here are the key changes:

### Import Changes

```python
# Old (PyCUDA)
import pycuda.driver as cuda
import pycuda.autoinit

# New (cuda-python)
from cuda import cudart
```

### Memory Allocation

```python
# Old (PyCUDA)
d_input = cuda.mem_alloc(size)

# New (cuda-python)
err, d_input = cudart.cudaMalloc(size)
check_cuda_error(err)
```

### Memory Copy

```python
# Old (PyCUDA)
cuda.memcpy_htod(d_input, h_input)
cuda.memcpy_dtoh(h_output, d_output)

# New (cuda-python)
cudart.cudaMemcpy(d_input, h_input.ctypes.data, size, 
                  cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
cudart.cudaMemcpy(h_output.ctypes.data, d_output, size,
                  cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
```

### Stream Creation

```python
# Old (PyCUDA)
stream = cuda.Stream()

# New (cuda-python)
err, stream = cudart.cudaStreamCreate()
check_cuda_error(err)
```

## Architecture

The example demonstrates a complete workflow:

1. **PyTorch Operations** - Create and manipulate PyTorch tensors on GPU
2. **TensorRT Initialization** - Load and initialize TensorRT engine (no hanging!)
3. **TensorRT Inference** - Run inference using cuda-python for CUDA operations
4. **Interoperability** - Convert between PyTorch tensors and NumPy arrays seamlessly

## Common Issues and Solutions

### Issue: "CUDA Error: invalid device context"

**Solution:** Make sure you're using cuda-python, not PyCUDA. Check your imports.

### Issue: "Engine file not found"

**Solution:** Provide a valid path to a TensorRT engine file, or create one using trtexec.

### Issue: "Input shape mismatch"

**Solution:** Ensure your input data matches the shape expected by the engine. Check the engine's input shape with `--verbose` flag.

## Performance Considerations

- **CUDA Streams**: The example uses CUDA streams for asynchronous execution
- **Memory Management**: GPU memory is properly allocated and freed
- **Contiguous Arrays**: Input arrays are made contiguous for efficient GPU transfer
- **Zero-Copy**: Where possible, data is transferred without unnecessary copies

## Additional Resources

- [TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/)
- [cuda-python Documentation](https://nvidia.github.io/cuda-python/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [TensorRT Python API](https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/)
- [Issue #4608 Solution Document](../../../ISSUE_4608_SOLUTION.md)

## Related Samples

- [1_run_onnx_with_tensorrt](../refactored/1_run_onnx_with_tensorrt/) - Basic ONNX to TensorRT conversion
- [2_construct_network_with_layer_apis](../refactored/2_construct_network_with_layer_apis/) - Building networks with TensorRT APIs

## License

This sample is licensed under the Apache License 2.0. See the LICENSE file for details.

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](../../../CONTRIBUTING.md) for guidelines.
