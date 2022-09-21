# Converting ONNX Models To FP16

## Introduction

When debugging accuracy issues with using TensorRT reduced precision
optimizations (`--fp16` and `--tf32` flags) on an ONNX model trained in FP32,
it can be helpful to convert the model to FP16 and run it under ONNX-Runtime
to check if there are might be problems inherent to running the model
with reduced precision.

## Running The Example

1. Convert the model to FP16:

   ```bash
   polygraphy convert --fp-to-fp16 -o identity_fp16.onnx identity.onnx
   ```
2. **[Optional]** Inspect the resulting model:

   ```bash
   polygraphy inspect model identity_fp16.onnx
   ```
3. **[Optional]** Run the FP32 and FP16 model under ONNX-Runtime, then compare the results:

   ```bash
   polygraphy run --onnxrt identity.onnx \
      --save-inputs inputs.json --save-outputs outputs_fp32.json
   ```

   ```bash
   polygraphy run --onnxrt identity_fp16.onnx \
      --load-inputs inputs.json --load-outputs outputs_fp32.json \
      --atol 0.001 --rtol 0.001
   ```
4. **[Optional]** Check if any intermediate outputs of the FP16 model
   contain NaN or infinity (see [Checking for Intermediate NaN or Infinities](../../../../examples/cli/run/07_checking_nan_inf)):
   ```bash
   polygraphy run --onnxrt identity_fp16.onnx --validate
   ```

## See Also

* [Comparing Across Runs](../../../../examples/cli/run/02_comparing_across_runs)
* [Checking for Intermediate NaN or Infinities](../../../../examples/cli/run/07_checking_nan_inf)
* [Debugging TensorRT Accuracy Issues](../../../../how-to/debug_accuracy.md)
