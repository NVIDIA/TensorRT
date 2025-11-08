# NonZero Plugin for TensorRT using IPluginV3

**Table Of Contents**
- [Description](#description)
- [How does this sample work?](#how-does-this-sample-work)
	* [Implementing a NonZero plugin using IPluginV3 interface](#implementing-a-nonzero-plugin-using-ipluginv3-interface)
	* [Creating network and building the engine](#creating-network-and-building-the-engine)
	* [Running inference](#running-inference)
- [Running the sample](#running-the-sample)
	* [Sample `--help` options](#sample---help-options)
- [Additional resources](#additional-resources)
- [License](#license)
- [Changelog](#changelog)
- [Known issues](#known-issues)

## Description

This sample, sampleNonZeroPlugin, implements a plugin for the NonZero operation, customizable to output the non-zero indices in
either a row order (each set of indices in the same row) or column order format (each set of indices in the same column).

NonZero is an operation where the non-zero indices of the input tensor is found. 

## How does this sample work?

This sample creates and runs a TensorRT engine built from a network containing a single NonZeroPlugin node. It demonstrates how
custom layers with data-dependent output shapes can be implemented and added to a TensorRT network.

Specifically, this sample:
- [Implements a TensorRT plugin for the NonZero operation](#implementing-a-nonzero-plugin-using-ipluginv3-interface)
- [Creates a network and builds an engine](#creating-network-and-building-the-engine)
- [Runs inference using the generated TensorRT network](#running-inference)

### Implementing a NonZero plugin using IPluginV3 interface

Until `IPluginV3` (and associated interfaces), TensorRT plugins could not have outputs whose shapes depended on the input values (they could only depend
on input shapes). `IPluginV3OneBuild` which exposes a build capability for `IPluginV3`, provides support for such data-dependent output shapes.

`NonZeroPlugin` in this sample is written to handle 2-D input tensors of shape $R \times C$. Assume that the tensor contains $K$ non-zero elements and that the
non-zero indices are required in a row ordering (each set of indices in its own row). Then the output shape would be $K \times 2$.

The output shapes are expressed to the TensorRT builder through the `IPluginV3OneBuild::getOutputShapes()` API. Expressing the second dimension of the output is
straightforward:
```
outputs[0].d[1] = exprBuilder.constant(2);
```

The extent of each data-dependent dimension in the plugin must be expressed in terms of a *_size tensor_*. A size tensor is a scalar output of 
`DataType::kINT32` or `DataType::kINT64` that must be added as one of the plugin outputs. In this case, it is sufficient to declare one size tensor to denote the extent of the
first dimension of the non-zero indices output. To declare a size tensor, one must provide an upper-bound and optimum value for its extent as `IDimensionExpr`s. These can be formed through the `IExprBuilder` argument passed to the `IPluginV3OneBuild::getOutputShapes()` method.
 - For unknown inputs, the upper-bound is the total number of elements in the input
	```
	auto upperBound = exprBuilder.operation(DimensionOperation::kPROD, *inputs[0].d[0], *inputs[0].d[1]);
	```
 - A good estimate for the optimum is that half of the elements are non-zero
	```
	auto optValue = exprBuilder.operation(DimensionOperation::kFLOOR_DIV, *upperBound, *exprBuilder.constant(2));
	```

Now we can declare the size tensor using the `IExprBuilder::declareSizeTensor()` method, which also requires the specification of the output index at which the size tensor would reside. Let us place it after the non-zero indices output:
```
auto numNonZeroSizeTensor = exprBuilder.declareSizeTensor(1, *optValue, *upperBound);
```

Now we are ready to specify the extent of the first dimension of the non-zero indices output:
```
outputs[0].d[0] = numNonZeroSizeTensor;
```
and let's not forget to declare that the size tensor is a scalar (0-D):
```
outputs[1].nbDims = 0;
```

The `NonZeroPlugin` can also be configured to emit the non-zero indices in a column-order fashion through the `rowOrder` plugin attribute, by setting it to `0`.
In this case, the first output of the plugin will have shape $2 \times K$, and the output shape specification must be adjusted accordingly.

### Creating network and building the engine

To add the plugin to the network, the `INetworkDefinition::addPluginV3()` method must be used. 

Similar to `IPluginCreator` used for V2 plugins, V3 plugins must be accompanied by the registration of a plugin creator implementing the `IPluginCreatorV3One`
interface.

### Running inference

As sample inputs, random images from MNIST dataset are selected and scaled to between `[0,1]`. The network will output both the non-zero indices,
as well as the non-zero count.

## Prerequisites
1. Preparing sample data

See [Preparing sample data](../README.md#preparing-sample-data) in the main samples README.

## Running the sample

1. Compile the sample by following build instructions in [TensorRT README](https://github.com/NVIDIA/TensorRT/).

2.  Run the sample to build and run the MNIST engine from the ONNX model.
	```
	./sample_non_zero_plugin [-h or --help] [-d or --datadir=<path to data directory>] [--columnOrder] [--fp16]
	```

3.  Verify that the sample ran successfully. If the sample runs successfully you should see output similar to the following:
	```
	&&&& RUNNING TensorRT.sample_non_zero_plugin # ./sample_non_zero_plugin
	...
	[I] Input:
	[I] 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
	[I] 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
	[I] 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.854902, 0
	[I] 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.858824, 0, 0, 0.0745098, 0, 0.564706, 0
	[I] 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.317647, 0, 0, 0.47451, 0, 0, 0
	[I] 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0431373, 0, 0, 0
	[I] 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
	[I] 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
	[I] 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.854902, 0, 0, 0.145098
	[I] 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.564706, 0, 0, 0.996078
	[I] 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.282353
	[I] 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
	[I] 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
	[I] 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.854902
	[I] 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.854902, 0, 0, 0.145098, 0, 0.564706
	[I] 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.564706, 0, 0, 0.996078, 0, 0
	[I] 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.282353, 0, 0
	[I]
	[I] Output:
	[I] 2 14
	[I] 3 9
	[I] 3 12
	[I] 3 14
	[I] 4 9
	[I] 4 12
	[I] 5 12
	[I] 8 12
	[I] 8 15
	[I] 9 12
	[I] 9 15
	[I] 10 15
	[I] 13 15
	[I] 14 10
	[I] 14 13
	[I] 14 15
	[I] 15 10
	[I] 15 13
	[I] 16 13
	&&&& PASSED TensorRT.sample_non_zero_plugin # ./sample_non_zero_plugin
	```

### Sample `--help` options

To see the full list of available options and their descriptions, use the `-h` or `--help` command line option.


# Additional resources

The following resources provide a deeper understanding about the V3 TensorRT plugins and the NonZero operation:

**NonZero**
- [ONNX: NonZero](https://onnx.ai/onnx/operators/onnx__NonZero.html)

**TensorRT plugins**
- [Extending TensorRT with Custom Layers](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#extending)

**Other documentation**
- [Introduction To NVIDIA’s TensorRT Samples](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sample-support-guide/index.html#samples)
- [Working With TensorRT Using The C++ API](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#c_topics)
- [NVIDIA’s TensorRT Documentation Library](https://docs.nvidia.com/deeplearning/sdk/tensorrt-archived/index.html)

# License

For terms and conditions for use, reproduction, and distribution, see the [TensorRT Software License Agreement](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html) documentation.


# Changelog

March 2024
This is the first version of this `README.md` file.


# Known issues

Windows users building this sample with Visual Studio with a CUDA version different from the TensorRT package will need to retarget the project to build against the installed CUDA version through the `Build Dependencies -> Build Customization` menu.
