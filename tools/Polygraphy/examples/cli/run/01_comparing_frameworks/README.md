# Comparing Frameworks

You can use the `run` subtool to compare a model between different frameworks.
In the simplest case, you can supply a model, and one or more framework flags.

For example, to compare an ONNX model between TensorRT and ONNX Runtime:

```bash
polygraphy run ../../../models/identity.onnx --trt --onnxrt
```
