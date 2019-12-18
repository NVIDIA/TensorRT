# Building a Simple MNIST Network Layer by Layer


**Table Of Contents**
- [Description](#description)
- [How does this sample work?](#how-does-this-sample-work)
	* [TensorRT API layers](#tensorrt-api-layers)
- [Prerequisites](#prerequisites)
- [Running the sample](#running-the-sample)
	* [Sample `--help` options](#sample---help-options)
- [Additional resources](#additional-resources)
- [License](#license)
- [Changelog](#changelog)
- [Known issues](#known-issues)

## Description

This sample, sampleMNISTAPI, uses the TensorRT API to build an engine for a model trained on the [MNIST dataset](https://github.com/NVIDIA/DIGITS/blob/master/docs/GettingStarted.md). It creates the network layer by layer, sets up weights and inputs/outputs, and then performs inference. This sample is similar to sampleMNIST. Both of these samples use the same model weights, handle the same input, and expect similar output.

## How does this sample work?

This sample uses a Caffe model that was trained on the [MNIST dataset](https://github.com/NVIDIA/DIGITS/blob/master/docs/GettingStarted.md).

In contrast to sampleMNIST, which uses the Caffe parser to import the MNIST model, this sample uses the C++ API, individually creating every layer and loading weights from a trained weights file. For a detailed description of how to create layers using the C++ API, see [Creating A Network Definition In C++](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#network_c).

### TensorRT API layers

In this sample, the following layers are used. For more information about these layers, see the [TensorRT Developer Guide: Layers](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#layers) documentation.

[Activation layer](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#activation-layer)
The Activation layer implements element-wise activation functions. Specifically, this sample uses the Activation layer with the type `kRELU`.

[Convolution layer](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#convolution-layer)
The Convolution layer computes a CHW 2D convolution, with or without bias.

[FullyConnected layer](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#fullyconnected-layer)
The FullyConnected layer implements a matrix-vector product, with or without bias.

[Pooling layer](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#pooling-layer)
The Pooling layer implements pooling within a channel. Supported pooling types are `maximum`, `average` and `maximum-average blend`.

[Scale layer](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#scale-layer)
The Scale layer implements a per-tensor, per-channel, or per-element affine transformation and/or exponentiation by constant values.

[SoftMax layer](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#softmax-layer)
The SoftMax layer applies the SoftMax function on the input tensor along an input dimension specified by the user.

## Prerequisites

When you build a network by individually creating every layer, ensure you provide the per-layer weights to TensorRT in host memory.

1.  Extract the weights from their pre-trained model or deep learning framework.  In this sample, the `mnistapi.wts` weights file stores the weights in a simple space delimited format, as described below:
	```
	<number of weight sets>
	[weights_name] [size] <data x size in hex>
	[weights_name] [size] <data x size in hex>
	[weights_name] [size] <data x size in hex>
	```

	In the `loadWeights` function, the sample reads this file and creates a std::map<string, Weights> structure as a mapping from the `weights_name` to Weights.

2.  Load the per-layer weights into host memory to pass to TensorRT during the network creation. For example:
    In this statement, we are loading the filter weights weightsMap["conv1filter"] and bias weightsMap["conv1bias"] to the
    convolution layer.
	```
	IConvolutionLayer* conv1 = network->addConvolutionNd(*scale_1->getOutput(0), 20, Dims{2, {5, 5}, {}}, weightMap["conv1filter"], weightMap["conv1bias"]);
	```

## Running the sample

1.  Compile this sample by running `make` in the `<TensorRT root directory>/samples/sampleMNISTAPI` directory. The binary named `sample_mnist_api` will be created in the `<TensorRT root directory>/bin` directory.
	```
	cd <TensorRT root directory>/samples/sampleMNISTAPI
	make
	```

	Where `<TensorRT root directory>` is where you installed TensorRT.

2.  Run the sample to perform inference on the digit:
	```
	./sample_mnist_api [-h] [--datadir=/path/to/data/dir/] [--useDLACore=N]
	```

3.  Verify that the sample ran successfully. If the sample runs successfully you should see output similar to the following; ASCII rendering of the input image with digit 9:
	```
	&&&& RUNNING TensorRT.sample_mnist_api # ./sample_mnist_api
	[I] Loading weights: ../../../../../../data/samples/mnist/mnistapi.wts
	[I] Input:
	@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	@@@@@@@@@@@@@@%.-@@@@@@@@@@@
	@@@@@@@@@@@*-    %@@@@@@@@@@
	@@@@@@@@@@= .-.  *@@@@@@@@@@
	@@@@@@@@@= +@@@  *@@@@@@@@@@
	@@@@@@@@* =@@@@  %@@@@@@@@@@
	@@@@@@@@..@@@@%  @@@@@@@@@@@
	@@@@@@@# *@@@@-  @@@@@@@@@@@
	@@@@@@@: @@@@%   @@@@@@@@@@@
	@@@@@@@: @@@@-   @@@@@@@@@@@
	@@@@@@@: =+*= +: *@@@@@@@@@@
	@@@@@@@*.    +@: *@@@@@@@@@@
	@@@@@@@@%#**#@@: *@@@@@@@@@@
	@@@@@@@@@@@@@@@: -@@@@@@@@@@
	@@@@@@@@@@@@@@@+ :@@@@@@@@@@
	@@@@@@@@@@@@@@@*  @@@@@@@@@@
	@@@@@@@@@@@@@@@@  %@@@@@@@@@
	@@@@@@@@@@@@@@@@  #@@@@@@@@@
	@@@@@@@@@@@@@@@@: +@@@@@@@@@
	@@@@@@@@@@@@@@@@- +@@@@@@@@@
	@@@@@@@@@@@@@@@@*:%@@@@@@@@@
	@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	@@@@@@@@@@@@@@@@@@@@@@@@@@@@

	[I] Output:
	0:
	1:
	2:
	3:
	4:
	5:
	6:
	7:
	8:
	9: **********

	&&&& PASSED TensorRT.sample_mnist_api # ./sample_mnist_api
	```

	This output shows that the sample ran successfully; PASSED.

### Sample --help options

To see the full list of available options and their descriptions, use the `-h` or `--help` command line option. For example:
```
Usage: ./sample_mnist_api [-h or --help] [-d or --datadir=<path to data directory>] [--useDLACore=<int>]
-h or --help Display help information
--datadir Specify path to a data directory, overriding the default. This option can be used multiple times to add multiple directories. If no data directories are given, the default is to use (data/samples/mnist/, data/mnist/)
--useDLACore=N Specify a DLA engine for layers that support DLA. Value can range from 0 to n-1, where n is the number of DLA engines on the platform.
--int8 Run in Int8 mode.
--fp16 Run in FP16 mode.
```

# Additional resources

The following resources provide a deeper understanding about MNIST:

**MNIST:**
- [MNIST dataset](https://github.com/NVIDIA/DIGITS/blob/master/docs/GettingStarted.md)

**Documentation**
- [Introduction To NVIDIA’s TensorRT Samples](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sample-support-guide/index.html#samples)
- [Working With TensorRT Using The C++ API](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#c_topics)
- [NVIDIA’s TensorRT Documentation Library](https://docs.nvidia.com/deeplearning/sdk/tensorrt-archived/index.html)

# License

For terms and conditions for use, reproduction, and distribution, see the [TensorRT Software License Agreement](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html) documentation.


# Changelog

- March 2019: This is the first release of this `README.md` file.


# Known issues

There are no known issues in this sample.
