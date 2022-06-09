# Adding Precision Constraints

## Introduction

When a model trained in FP32 is used to build a TensorRT engine that leverages
[reduced precision optimizations](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#reduced-precision),
certain layers in the model may need to be constrained to run in FP32 to
preserve acceptable accuracy.

The following example demonstrates how to use a custom Polygraphy network script
to selectively constrain precisions of specified layers in the network. The
provided ONNX model:

1. Flips its input horizontally by right-multiplying by a 90 degree rotated identity matrix,
2. Adds `FP16_MAX` to the flipped input, then subtracts `FP16_MAX` from the result,
3. Flips the output of the subtraction horizontally by right-multiplying by the rotated identity.

If `x` is positive, step (2) in this procedure needs to be done in FP32 in
order to achieve acceptable accuracy since values will exceed FP16 representable
range (by design).  However, when FP16 optimizations are enabled without
constraints, TensorRT, having no knowledge of what range of values will be used
for `x`, will usually choose to run all steps in this process in FP16:

* The GEMM operations in steps (1) and (3) will run faster in FP16 than in FP32
  (for large enough problem sizes)
* The pointwise operations in step (2) will run faster in FP16, and leaving the
  data in FP16 eliminates the need for additional reformats to/from FP32.

Hence, you need to constrain the allowed precisions in the TensorRT network in
order for TensorRT to make appropriate choices when assigning layer precisions
in the engine.

## Running The Example

**Warning:** _This example requires TensorRT 8.4 or later._

### Running Without Constraints

Run the following command to compare running the model with TensorRT using FP16
optimizations against ONNX-Runtime in FP32:

<!-- Polygraphy Test: Ignore Start -->
```bash
polygraphy run needs_constraints.onnx \
    --trt --fp16 --onnxrt --val-range x:[1,2] \
    --check-error-stat median
```
<!-- Polygraphy Test: Ignore End -->

If FP16 tactics are selected for the `Add` or `Sub` layers, this should fail for the reasons described above.

### Using A Network Script To Constrain Precisions

The below section assumes you have read through the example on
[Defining a TensorRT Network or Config Manually](/examples/cli/run/04_defining_a_tensorrt_network_or_config_manually)
and have a basic understanding of how to use the [TensorRT Python API](https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/).

First, run ONNX-Runtime on the model to generate reference inputs and golden outputs:

```bash
polygraphy run needs_constraints.onnx --onnxrt --val-range x:[1,2] \
    --save-inputs inputs.json --save-outputs golden_outputs.json
```

Next, run the provided network script
[constrained_network.py](./constrained_network.py) which constrains precisions
in the model, forcing TensorRT to obey constraints, using the saved input and comparing against the saved golden output:

```bash
polygraphy run constrained_network.py --precision-constraints obey \
    --trt --fp16 --load-inputs inputs.json --load-outputs golden_outputs.json \
    --check-error-stat median
```

Note that TensorRT may choose to run other layers in the network in FP32 besides
the explicitly constrained layers if doing so would result in higher overall
engine performance.

**[Optional]**: Run the network script but allow TensorRT to ignore precision
constraints if necessary. This may be required to run the network if TensorRT
has no layer implementation that satisfies the requested precision constraints:

```
polygraphy run constrained_network.py --precision-constraints prefer \
    --trt --fp16 --load-inputs inputs.json --load-outputs golden_outputs.json \
    --check-error-stat median
```


## See Also

* [Working with Reduced Precision](/how-to/work_with_reduced_precision.md) for a more general guide on how to debug
  reduced precision optimizations using Polygraphy.
* [Defining a TensorRT Network or Config Manually](/examples/cli/run/04_defining_a_tensorrt_network_or_config_manually) for
  instructions on how to create network script templates.
* [TensorRT Python API Reference](https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/)
