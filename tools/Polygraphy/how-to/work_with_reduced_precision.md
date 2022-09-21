# Working With Reduced Precision

## Can the model run in reduced precision?

TensorRT can often be used to run models trained in FP32 using lower-precision
implementations (in particular TF32 and FP16) with little to no additional
effort needed.  Note this generally _isn't_ the case when using INT8, which
requires extra steps as described [here](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#working-with-int8)
to achieve acceptable accuracy.

You can easily check whether the generated engine satisfies accuracy
requirements by comparing the resulting engine against ONNX-Runtime using
Polygraphy.  See [Comparing Frameworks](../examples/cli/run/01_comparing_frameworks) for more detailed
instructions on how to do this.

### Sanity-checking for FP16 limitations

If you are using `--trt --fp16` and accuracy is not acceptable, you can
sanity-check whether this might be a limitation with using the model in reduced
precision (and not a TensorRT-specific issue) by running the same model in FP16
using ONNX-Runtime.  Refer to [Converting ONNX Models to FP16](../examples/cli/convert/04_converting_models_to_fp16)
for instructions on generating a model for comparison and validating its
outputs. If running the model in FP16 fails in ONNX-Runtime, then you will likely need to
adjust the model or add precision constraints as described below in order to
achieve acceptable accuracy.

## Debugging accuracy failures

If the output comparison fails, the next step is generally to isolate the
problematic layer(s) in the model contributing to the accuracy failure. See
[How to Debug Accuracy](../how-to/debug_accuracy.md) for techniques on how to do this.

## Overriding precision constraints

Once you've identified the problematic layer(s), the next step is to override
precision constraints for those layers to FP32 to see if accuracy recovers. See
the example on [Overriding Precision Constraints](../examples/cli/run/08_adding_precision_constraints) for details on how to use
Polygraphy to experiment with different precision constraints.

## Other options

If falling back layers to FP32 isn't sufficient to recover accuracy or leads to
unwanted performance degradation, you will generally need to modify and retrain
the model to help keep the dynamic range of intermediate activations within
expressible bounds. Some techniques that can help achieve this include:

* Normalizing input values (for example, scaling RGB input data to `[0, 1]`) when training the model
  and using the trained model for inference.
* Using batch normalization and other regularization techniques when training the model.
* When using INT8, consider using [Quantization Aware Training (QAT)](https://developer.nvidia.com/blog/achieving-fp32-accuracy-for-int8-inference-using-quantization-aware-training-with-tensorrt/) to help improve accuracy.
