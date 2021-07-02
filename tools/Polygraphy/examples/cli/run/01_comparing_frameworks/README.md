# Comparing Frameworks

You can use the `run` subtool to compare a model between different frameworks.
In the simplest case, you can supply a model, and one or more framework flags.

For example, to compare an ONNX model between TensorRT and ONNX Runtime:

```bash
polygraphy run dynamic_identity.onnx --trt --onnxrt
```

If our model uses dynamic input shapes, we can specify the shapes to use at
runtime with the `--input-shapes` option:

```bash
polygraphy run dynamic_identity.onnx --trt --onnxrt \
    --input-shapes X:[1,2,4,4]
```

For more details on working with dynamic shapes in TensorRT, refer to
[`convert` example 03](../../convert/03_dynamic_shapes_in_tensorrt/) or
[API example 07](../../../api/07_tensorrt_and_dynamic_shapes/).
