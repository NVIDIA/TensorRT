# “Hello World” For Multilayer Perceptron (MLP)


**Table Of Contents**
- [Description](#description)
- [How does this sample work?](#how-does-this-sample-work)
    * [Defining the network](#defining-the-network)
    * [TensorRT API layers and ops](#tensorrt-api-layers-and-ops)
- [Training an MLP network](#training-an-mlp-network)
- [Running the sample](#running-the-sample)
    * [Sample `--help` options](#sample-help-options)
- [Additional resources](#additional-resources)
- [License](#license)
- [Changelog](#changelog)
- [Known issues](#known-issues)

## Description

This sample, sampleMLP, is a simple hello world example that shows how to create a network that triggers the multilayer perceptron ([MLP](https://en.wikipedia.org/wiki/Multilayer_perceptron)) optimizer. The generated MLP optimizer can then accelerate TensorRT.

## How does this sample work?

This sample uses a publicly accessible [TensorFlow tutorial](https://github.com/aymericdamien/TensorFlow-Examples) to train a [MLP network](https://en.wikipedia.org/wiki/Multilayer_perceptron) based on the [MNIST data set](http://yann.lecun.com/exdb/mnist/) and shows how to transform that data into a format that the samples use.

Specifically, this sample [defines the network](#defining-the-network), triggers the MLP optimizer by creating a sequence of networks to increase performance, and creates a sequence of TensorRT layers that represent an MLP layer.

### Defining the network

This sample follows the same flow as [sampleMNISTAPI](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sample-support-guide/index.html#mnistapi_sample) with one exception. The network is defined as a sequence of addMLP calls, which adds FullyConnected and Activation layers to the network.

Generally, an MLP layer is:
-   a FullyConnected operation that is followed by an optional Scale and an optional Activation; or
-   a MatrixMultiplication operation followed by an optional bias and an optional activation.

An MLP network is more than one MLP layer generated sequentially in the TensorRT network. The optimizer will detect this pattern and generate optimized MLP code.

More formally, the following variations of MLP layers will trigger the MLP optimizer:
```
{MatrixMultiplication [-> ElementWiseSum] [-> Activation]}+
{FullyConnected [-> Scale(with empty scale and power arguments)] [-> Activation]}+
```

### TensorRT API layers and ops

In this sample, the following layers are used. For more information about these layers, see the [TensorRT Developer Guide: Layers](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#layers) documentation.

[Activation layer](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#activation-layer)
The Activation layer implements element-wise activation functions. Specifically, this sample uses the Activation layer with the type `kRELU`.

[FullyConnected layer](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#fullyconnected-layer)
The FullyConnected layer implements a matrix-vector product, with or without bias.

[TopK layer](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#topk-layer)
The TopK layer finds the top K maximum (or minimum) elements along a dimension, returning a reduced tensor and a tensor of index positions.

## Training an MLP network

This sample comes with pre-trained weights. However, if you want to train your own MLP network, you first need to generate the weights by training a TensorFlow based neural network using an MLP optimizer, and then verify that the trained weights are converted into a format that sampleMLP can read. If you want to use the weights that are shipped with this sample, see [Running the sample](#running-the-sample).

1.  [Install Python](https://www.tensorflow.org/install/pip#1.-install-the-python-development-environment-on-your-system).
2.  Install [TensorFlow 1.4 or later](https://www.tensorflow.org/install/pip).
3.  Download the [TensorFlow tutorial](https://github.com/aymericdamien/TensorFlow-Examples).
	```
	git clone https://github.com/aymericdamien/TensorFlow-Examples.git
	cd TensorFlow-Examples
	```
4.  Apply the `update_mlp.patch` file to save the final result.
	```
	patch -p1 < <TensorRT Install>/samples/sampleMLP/update_mlp.patch
	```
5.  Train the MNIST MLP network.
	```
	python examples/3_NeuralNetworks/multilayer_perceptron.py
	```
	This step produces the following file:
	```
	/tmp/sampleMLP.ckpt - Trained MLP checkpoint
	```
	The `sampleMLP.ckpt` file contains the checkpoint for the parameters and weights.

 6. Convert the trained model weights to a format sampleMLP understands.
	```
	python <TensorRT Install>/samples/sampleMLP/convert_weights.py -m /tmp/sampleMLP.ckpt -o sampleMLP
	mkdir -p <TensorRT Install>/data/mlp
	cp sampleMLP.wts2 <TensorRT Install>/data/mlp/
	```

## Running the sample

1. Compile this sample by running `make` in the `<TensorRT root directory>/samples/sampleMLP` directory. The binary named `sample_mlp` will be created in the `<TensorRT root directory>/bin` directory.
	```
	cd <TensorRT root directory>/samples/sampleMLP
	make
	```
	Where `<TensorRT root directory>` is where you installed TensorRT.

2. Run the sample to classify the MNIST digit.
	```
	cd <TensorRT Install>/bin
	./sample_mlp
	```

3. Verify that the sample ran successfully. If the sample runs successfully you should see output similar to the following; ASCII rendering of the input image with digit 9:
```
	&&&& RUNNING TensorRT.sample_mlp # build/x86_64-linux/sample_mlp
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

	[I] Algorithm chose 9
	&&&& PASSED TensorRT.sample_mlp # build/x86_64-linux/sample_mlp
```

This output shows that the sample ran successfully; `PASSED`.

### Sample --help options

To see the full list of available options and their descriptions, use the `-h` or `--help` command line option. For example:
```
./sample_mlp --help

Usage: ./sample_mlp [-h or --help] [-d or --datadir=<path to data directory>] [--useDLACore=<int>]
--help Display help information
--datadir Specify path to a data directory, overriding the default. This option can be used multiple times to add multiple directories. If no data directories are given, the default is to use (data/samples/mnist/, data/mnist/, data/samples/mlp/, data/mlp/)
--useDLACore=N Specify a DLA engine for layers that support DLA. Value can range from 0 to n-1, where n is the number of DLA engines on the platform.
--int8 Run in Int8 mode.
--fp16 Run in FP16 mode.
```

# Additional resources

The following resources provide a deeper understanding about MLP:

**MLP**
- [TensorFlow tutorial](https://github.com/aymericdamien/TensorFlow-Examples)
- [MLP network](https://en.wikipedia.org/wiki/Multilayer_perceptron)

**Models**
- [MNIST data set](http://yann.lecun.com/exdb/mnist/)

**Documentation**
- [TensorRT Sample Support Guide: sampleMLP](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sample-support-guide/index.html#mlp_sample)
- [NVIDIA’s TensorRT Documentation Library](https://docs.nvidia.com/deeplearning/sdk/tensorrt-archived/index.html)

# License

For terms and conditions for use, reproduction, and distribution, see the [TensorRT Software License Agreement](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html) documentation.


# Changelog

February 2019
This `README.md` file was recreated, updated and reviewed.


# Known issues

There are no known issues in this sample.
