# Comparing Frameworks

## Introduction

You can use the `run` subtool to compare a model between different frameworks.
In the simplest case, you can supply a model, and one or more framework flags.
By default, it will generate synthetic input data, run inference using the
specified frameworks, then compare outputs of the specified frameworks.

## Running The Example

In this example, we'll outline various common use-cases for the `run` subtool:

- [Comparing TensorRT And ONNX-Runtime Outputs](#comparing-tensorrt-and-onnx-runtime-outputs)
- [Comparing TensorRT Precisions](#comparing-tensorrt-precisions)
- [Changing Tolerances](#changing-tolerances)
- [Changing Comparison Metrics](#changing-comparison-metrics)
- [Comparing Per-Layer Outputs Between ONNX-Runtime And TensorRT](#comparing-per-layer-outputs-between-onnx-runtime-and-tensorrt)

### Comparing TensorRT And ONNX-Runtime Outputs

To run the model in Polygraphy with both frameworks and perform an output
comparison:

```bash
polygraphy run dynamic_identity.onnx --trt --onnxrt
```

The `dynamic_identity.onnx` model has dynamic input shapes. By default,
Polygraphy will override any dynamic input dimensions in the model to
`constants.DEFAULT_SHAPE_VALUE` (defined as `1`) and warn you:

<!-- Polygraphy Test: Ignore Start -->
```
[W]     Input tensor: X (dtype=DataType.FLOAT, shape=(1, 2, -1, -1)) | No shapes provided; Will use shape: [1, 2, 1, 1] for min/opt/max in profile.
[W]     This will cause the tensor to have a static shape. If this is incorrect, please set the range of shapes for this input tensor.
```
<!-- Polygraphy Test: Ignore End -->

In order to suppress this message and explicitly provide input shapes to
Polygraphy, use the `--input-shapes` option:

```
polygraphy run dynamic_identity.onnx --trt --onnxrt \
    --input-shapes X:[1,2,4,4]
```

### Comparing TensorRT Precisions

To build a TensorRT engine with reduced precision layers for comparison against
ONNXRT, use one of the supported precision flags (`--tf32`, `--fp16`, or
`--int8`). For example:

```bash
polygraphy run dynamic_identity.onnx --trt --fp16 --onnxrt \
    --input-shapes X:[1,2,4,4]
```

> :warning: Getting acceptable accuracy with INT8 precision typically requires an additional calibration step:
  see the [developer guide](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#working-with-int8)
  and instructions on [how to do calibration](../../../../examples/cli/convert/01_int8_calibration_in_tensorrt)
  with Polygraphy on the command line.

### Changing Tolerances

The default tolerances used by `run` are usually appropriate for FP32 precision
but may not be appropriate for reduced precisions. In order to relax tolerances,
you can use the `--atol` and `--rtol` options to set absolute and relative
tolerance respectively.

### Changing Comparison Metrics

You can use the `--check-error-stat` option to change the metric used for
comparison. By default, Polygraphy uses an "elementwise" metric
(`--check-error-stat elemwise`).

Other possible metrics for `--check-error-stat` are `mean`, `median`, and `max`, which
compares the mean, median, and maximum absolute/relative error across the tensor, respectively.

To better understand this, suppose we are
comparing two outputs `out0` and `out1`. Polygraphy takes
the elementwise absolute and relative difference of these tensors:

<!-- Polygraphy Test: Ignore Start -->
```
absdiff = out0 - out1
reldiff = absdiff / abs(out1)
```
<!-- Polygraphy Test: Ignore End -->

Then, for each index `i` in the output, Polygraphy checks whether
`absdiff[i] > atol and reldiff[i] > rtol`. If any index  satisfies this,
then the comparison will fail.  This is less stringent than comparing the maximum
absolute and relative error across the entire tensor (`--check-error-stat max`) since if
*different* indices `i` and `j` satisfy `absdiff[i] > atol` and `reldiff[j] > rtol`,
then the `max` comparison will fail but the `elemwise` comparison may
pass.

Putting it all together, the below example runs a `median` comparison between
TensorRT using FP16 and ONNX-Runtime, using absolute and relative tolerances of `0.001`:

```bash
polygraphy run dynamic_identity.onnx --trt --fp16 --onnxrt \
    --input-shapes X:[1,2,4,4] \
    --atol 0.001 --rtol 0.001 --check-error-stat median
```

> You can also specify per-output values for `--atol`/`--rtol`/`--check-error-stat`.
  See the help output of the `run` subtool for more information.

### Comparing Per-Layer Outputs Between ONNX-Runtime And TensorRT

When network outputs do not match, it can be useful to compare per-layer outputs
to see where the error is introduced. To do so, you can use the `--trt-outputs`
and `--onnx-outputs` options respectively. These options accept one or more
output names as their arguments. The special value `mark all` indicates that all
tensors in the model should be compared:

```bash
 polygraphy run dynamic_identity.onnx --trt --onnxrt \
     --trt-outputs mark all \
     --onnx-outputs mark all
```

To find the first mismatched output more easily, you can use the `--fail-fast`
option which will cause the tool to exit after the first mismatch between
outputs.

Note that use of `--trt-outputs mark all` can sometimes perturb the generated
engine due to differences in timing, layer fusion choices, and format
constraints, which can hide the failure.  In that case, you may have to use a
more sophisticated approach to bisect the failing model and generate a reduced
test case that reproduces the error. See [Reducing Failing ONNX
Models](../../../../examples/cli/debug/02_reducing_failing_onnx_models) for a tutorial on
how to do this with Polygraphy.

## Further Reading

* In some cases you may need to do comparisons across multiple Polygraphy runs
  (for example, when comparing the output of a pre-built TensorRT engine or
  [Polygraphy network script](../../../../examples/cli/run/04_defining_a_tensorrt_network_or_config_manually)
  against ONNX-Runtime). See [Comparing Across Runs](../../../../examples/cli/run/02_comparing_across_runs) for a tutorial on how to
  accomplish this.

* For more details on working with dynamic shapes in TensorRT:
  * See [Dynamic Shapes in TensorRT](../../../../examples/cli/convert/03_dynamic_shapes_in_tensorrt/) for how to specify
    optimization profiles for use with the engine using the Polygraphy CLI
  * See [TensorRT and Dynamic Shapes](../../../../examples/api/07_tensorrt_and_dynamic_shapes/) for details on
    how to do this with the Polygraphy API

* For details on how to supply real input data, see [Comparing with Custom Input Data](../05_comparing_with_custom_input_data/).

* See [Debugging TensorRT Accuracy Issues](../../../../how-to/debug_accuracy.md) for a broader tutorial on how to debug accuracy failures using Polygraphy.
