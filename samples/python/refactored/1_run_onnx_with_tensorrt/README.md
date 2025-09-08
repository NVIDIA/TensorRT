# Run ONNX with TensorRT
This sample demonstrates:

- Converting a pre-trained [EfficientNet](https://arxiv.org/abs/1905.11946)-B0 ONNX model to a `TensorRT` engine
- Performing inference with `TensorRT` using Python APIs
- Comparing inference performance between `ONNX Runtime` and `TensorRT`
- Proper memory management and resource cleanup in both Python implementations

## Key features demonstrated:

- `TensorRT`'s ONNX parser + ONNX model -> `TensorRT` engine
- Engine building and serialization
- Input/output tensor handling
- Performance profiling
- Editable timing cache for deterministic engine builds
- Memory pool optimization with workspace configuration

## Implementation Details

### Memory Management
- Configures workspace memory pool for running under limited hardware

### Engine Building
- Supports editable timing cache for deterministic builds
- Serialization and deserialization of TensorRT engines

### Inference Pipeline
- Efficient image preprocessing with `PIL` and `NumPy`
- Supports batch inference
- Implements proper error handling and resource cleanup
- Provides performance comparison between `ONNX Runtime` and `TensorRT`
- Performs inference on a real-world image

## CLI Tools
Users can run their onnx model and generate the engine with similar functionality using `trtexec`:

```bash
# Basic conversion with performance profiling
trtexec --onnx=efficientnet-b0.onnx \
        --saveEngine=efficientnet-b0_trtexec.plan \
        --dumpProfile \
        --iterations=100 \
        --avgRuns=100 \
        --workspace=1024 \
        --batch=1
```

Key options explained:
- `--onnx`: Input ONNX model
- `--saveEngine`: Output TensorRT engine
- `--dumpProfile`: Performance profiling
- `--iterations`: Number of inference iterations
- `--avgRuns`: Number of runs to average for timing
- `--workspace`: Workspace size in MB (1024MB = 1GB)
- `--batch`: Batch size for inference

## Additional Resources

- [TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/latest/index.html)
- [ONNX Documentation](https://onnx.ai/)
- [EfficientNet Paper](https://arxiv.org/abs/1905.11946)

# Changelog

August 2025
Removed support for Python versions < 3.10.
