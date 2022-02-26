# Comparing Frameworks


## Introduction

You can use the `run` subtool to compare a model between different frameworks.
In the simplest case, you can supply a model, and one or more framework flags.
By default, `run` will generate synthetic input data, run inference using the specified frameworks,
and finally compare outputs.

*NOTE: The default tolerances used by `run` are usually appropriate for FP32 precision.*
    *When using other precisions, you may need to relax the tolerances, which you can do*
    *(optionally on a per-output basis) using `--atol` for absolute tolerance and*
    *`--rtol` for relative tolerance.*

## Running The Example

1. Compare an ONNX model between TensorRT and ONNX Runtime:

    ```bash
    polygraphy run dynamic_identity.onnx --trt --onnxrt
    ```

    If our model uses dynamic input shapes, we can specify the shapes to use at
    runtime with the `--input-shapes` option:

    ```bash
    polygraphy run dynamic_identity.onnx --trt --onnxrt \
        --input-shapes X:[1,2,4,4]
    ```

2. [Optional] Compare per-layer outputs between TensorRT and ONNX-Runtime:

    When network outputs do not match, it can be useful to compare per-layer
    outputs to see where the error is introduced. To do so, you can use the
    `--trt-outputs` and `--onnx-outputs` options respectively. These options
    accept one or more output names as their arguments. The special value `mark all`
    indicates that all tensors in the model should be compared:

    ```bash
    polygraphy run dynamic_identity.onnx --trt --onnxrt \
        --trt-outputs mark all \
        --onnx-outputs mark all
    ```

    *TIP: To find the first mismatched output more easily, you can use the `--fail-fast` option*
        *which will cause the tool to exit after the first mismatch between outputs.*

## Further Reading

For more details on working with dynamic shapes in TensorRT, refer to
[`convert` example 03](../../convert/03_dynamic_shapes_in_tensorrt/) or
[API example 07](../../../api/07_tensorrt_and_dynamic_shapes/).

For details on how to supply real input data, see [`run` example 05](../05_comparing_with_custom_data/).
