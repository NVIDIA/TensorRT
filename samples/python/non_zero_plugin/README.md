# Python-based NonZero Plugin for TensorRT using IPluginV3

## Description

This sample, `non_zero_plugin`, implements a Python-based plugin for the NonZero operation, configurable to use a `CUDA Python` or `PyTorch` backend.

NonZero is an operation where the non-zero indices of the input tensor is found.

## How does this sample work?

This sample creates and runs a TensorRT engine built from a network containing a single NonZeroPlugin node. It demonstrates how
custom layers with data-dependent output shapes can be implemented and added to a TensorRT network using Python.

### Implementing a NonZero plugin using IPluginV3 interface

Until `IPluginV3` (and associated interfaces), TensorRT plugins could not have outputs whose shapes depended on the input values (they could only depend
on input shapes). `IPluginV3OneBuild` which exposes a build capability for `IPluginV3`, provides support for such data-dependent output shapes.

`NonZeroPlugin` in this sample is written to handle 2-D input tensors of shape $R \times C$. Assume that the tensor contains $K$ non-zero elements and that the
non-zero indices are required in a row ordering (each set of indices in its own row). Then the output shape would be $K \times 2$.

The output shapes are expressed to the TensorRT builder through the `IPluginV3OneBuild.get_output_shapes()` API. Expressing the second dimension of the output is
straightforward:
```
# output_dims[0] = trt.DimsExprs(2)
output_dims[0][1] = exprBuilder.constant(2)
```

The extent of each data-dependent dimension in the plugin must be expressed in terms of a *_size tensor_*. A size tensor is a scalar output of type
`trt.int32` or `trt.int64` that must be added as one of the plugin outputs. In this case, it is sufficient to declare one size tensor to denote the extent of the
first dimension of the non-zero indices output. To declare a size tensor, one must provide an upper-bound and optimum value for its extent as `IDimensionExpr`s. These can be formed through the `IExprBuilder` argument passed to the `IPluginV3OneBuild.get_output_shapes()` method.
 - For unknown inputs, the upper-bound is the total number of elements in the input
	```
	upper_bound = exprBuilder.operation(trt.DimensionOperation.PROD, inputs[0][0], inputs[0][1])
	```
 - A good estimate for the optimum is that half of the elements are non-zero
	```
	opt_value = exprBuilder.operation(trt.DimensionOperation.FLOOR_DIV, upper_bound, exprBuilder.constant(2))
	```

Now we can declare the size tensor using the `IExprBuilder.declare_size_tensor()` method, which also requires the specification of the output index at which the size tensor would reside. Let us place it after the non-zero indices output:
```
num_non_zero_size_tensor = exprBuilder.declare_size_tensor(1, opt_value, upper_bound)
```

Now we are ready to specify the extent of the first dimension of the non-zero indices output:
```
# output_dims[0] = trt.DimsExprs(0)
output_dims[0][0] = num_non_zero_size_tensor
```
Note that the size tensor is declared to be a scalar (0-D):

### Creating network and building the engine

To add the plugin to the network, the `INetworkDefinition::add_plugin_v3()` method must be used.

Similar to `IPluginCreator` used for V2 plugins, V3 plugins must be accompanied by the registration of a plugin creator implementing the `IPluginCreatorV3One` interface.

## Running the sample

1.  Run the sample to create a TensorRT inference engine and run inference:
    `python3 non_zero_plugin.py [-h] [--precision {fp32,fp16}] [--backend {cuda_python,torch}] [--net_type {onnx,inetdef}]`

2.  Verify that the sample ran successfully. If the sample runs successfully you should see the following message:
     ```
    Inference result correct!
    ```

### Sample `--help` options

To see the full list of available options and their descriptions, use the `-h` or `--help` command line option.


# Additional resources

The following resources provide a deeper understanding about the V3 TensorRT plugins and the NonZero operation:

**NonZero**
- [ONNX: NonZero](https://onnx.ai/onnx/operators/onnx__NonZero.html)

**C++-based NonZero Plugin sample**
- [NonZero C++ Plugin](../../sampleNonZeroPlugin/)

**TensorRT plugins**
- [Extending TensorRT with Custom Layers](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#extending)
- [TensorRT Python-based Plugins](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/#add_custom_layer_python)

**Other documentation**
- [Introduction To NVIDIA’s TensorRT Samples](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sample-support-guide/index.html#samples)
- [Working With TensorRT Using The Python API](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/#python_topics)
- [NVIDIA’s TensorRT Documentation Library](https://docs.nvidia.com/deeplearning/sdk/tensorrt-archived/index.html)

# License

For terms and conditions for use, reproduction, and distribution, see the [TensorRT Software License Agreement](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html) documentation.

# Changelog

October 2025
Migrate to strongly typed APIs.

August 2025
Removed support for Python versions < 3.10.

April 2024
This is the first version of this `README.md` file.

# Known issues

There are no known issues in this sample.
