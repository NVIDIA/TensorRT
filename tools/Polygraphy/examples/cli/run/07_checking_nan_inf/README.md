# Checking for Intermediate NaN or Infinities

## Introduction

When debugging model accuracy issues in Polygraphy, it can be helpful to check layerwise outputs for potential problems. Polygraphy's `run` subtool provides a helpful flag `--validate` which can quickly diagnose problematic intermediate outputs.

This example demonstrates use of this flag with a model which intentionally generates an
infinite output by adding infinity to the input tensor.

## Running The Example

 <!-- Polygraphy Test: XFAIL Start -->
```bash
polygraphy run add_infinity.onnx --onnx-outputs '*' --onnxrt --validate
```
 <!-- Polygraphy Test: XFAIL End -->

 <!-- Polygraphy Test: Ignore Start -->
You should see output like:
```
[I] onnxrt-runner-N0-05/13/22-22:35:48  | Iteration 0 | Validating output: B (check_inf=True, check_nan=True)
[I]     mean=inf, std-dev=nan, var=nan, median=inf, min=inf at (0,), max=inf at (0,), avg-magnitude=inf, p90=inf, p95=inf, p99=inf
[E]     Inf Detected | One or more non-finite values were encountered in this output
[I]     Note: Use -vv or set logging verbosity to EXTRA_VERBOSE to display non-finite values
[E]     FAILED | Errors detected in output: B
[!] Output validation failed: iteration 0 contains invalid values (NaNs/Infs). See the errors above.
```
 <!-- Polygraphy Test: Ignore End -->

## See Also

* [Debugging TensorRT Accuracy Issues](../../../../how-to/debug_accuracy.md)
