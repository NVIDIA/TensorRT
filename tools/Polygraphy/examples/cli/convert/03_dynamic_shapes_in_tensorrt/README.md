# Working With Models With Dynamic Shapes In TensorRT

## Introduction

In order to use dynamic input shapes with TensorRT, we have to specify a range
(or multiple ranges) of possible shapes when we build the engine.
For details on how this works, refer to
[API example 07](../../../api/07_tensorrt_and_dynamic_shapes/).

When using the CLI, we can specify the per-input minimum, optimum, and maximum
shapes one or more times. If shapes are specified more than
once per input, multiple optimization profiles are created.

## Running The Example

1. Build an engine with 3 separate profiles:

    ```bash
    polygraphy convert dynamic_identity.onnx -o dynamic_identity.engine \
        --trt-min-shapes X:[1,3,28,28] --trt-opt-shapes X:[1,3,28,28] --trt-max-shapes X:[1,3,28,28] \
        --trt-min-shapes X:[1,3,28,28] --trt-opt-shapes X:[4,3,28,28] --trt-max-shapes X:[32,3,28,28] \
        --trt-min-shapes X:[128,3,28,28] --trt-opt-shapes X:[128,3,28,28] --trt-max-shapes X:[128,3,28,28]
    ```

    For models with multiple inputs, simply provide multiple arguments to each `--trt-*-shapes` parameter.
    For example: `--trt-min-shapes input0:[10,10] input1:[10,10] input2:[10,10] ...`

    *TIP: If we want to use only a single profile where min == opt == max, we can leverage the runtime input*
        *shapes option: `--input-shapes` as a conveneint shorthand instead of setting min/opt/max separately.*


2. **[Optional]** Inspect the resulting engine:

    ```bash
    polygraphy inspect model dynamic_identity.engine
    ```


## Further Reading

For more information on using dynamic shapes with TensorRT, see the
[developer guide](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#work_dynamic_shapes)
