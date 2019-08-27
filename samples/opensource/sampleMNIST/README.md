# “Hello World” For TensorRT


**Table Of Contents**
- [Description](#description)
- [How does this sample work?](#how-does-this-sample-work)
	* [TensorRT API layers and ops](#tensorrt-api-layers-and-ops)
- [Running the sample](#running-the-sample)
	* [Sample `--help` options](#sample---help-options)
- [Additional resources](#additional-resources)
- [License](#license)
- [Changelog](#changelog)
- [Known issues](#known-issues)

## Description

This sample, sampleMNIST, is a simple hello world example that performs the basic setup and initialization of TensorRT using the Caffe parser.

## How does this sample work?

This sample uses a Caffe model that was trained on the [MNIST dataset](https://github.com/NVIDIA/DIGITS/blob/master/docs/GettingStarted.md).

Specifically, this sample:
- Performs the basic setup and initialization of TensorRT using the Caffe parser    
- [Imports a trained Caffe model using Caffe parser](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#import_caffe_c)    
- Preprocesses the input and stores the result in a managed buffer
- [Builds an engine](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#build_engine_c)
- [Serializes and deserializes the engine](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#serial_model_c)
- [Uses the engine to perform inference on an input image](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#perform_inference_c)

To verify whether the engine is operating correctly, this sample picks a 28x28 image of a digit at random and runs inference on it using the engine it created. The output of the network is a probability distribution on the digit, showing which digit is likely that in the image.

### TensorRT API layers and ops

In this sample, the following layers are used.  For more information about these layers, see the [TensorRT Developer Guide: Layers](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#layers) documentation.

[Activation layer](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#activation-layer)
The Activation layer implements element-wise activation functions. Specifically, this sample uses the Activation layer with the type `kRELU`. 

[Convolution layer](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#convolution-layer)
The Convolution layer computes a 2D (channel, height, and width) convolution, with or without bias.

[FullyConnected layer](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#fullyconnected-layer)
The FullyConnected layer implements a matrix-vector product, with or without bias.

[Pooling layer](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#pooling-layer)
The Pooling layer implements pooling within a channel. Supported pooling types are `maximum`, `average` and `maximum-average blend`.

[Scale layer](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#scale-layer)
The Scale layer implements a per-tensor, per-channel, or per-element affine transformation and/or exponentiation by constant values.

[SoftMax layer](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#softmax-layer)
The SoftMax layer applies the SoftMax function on the input tensor along an input dimension specified by the user.

	
## Running the sample

1. Compile this sample by running `make` in the `<TensorRT root directory>/samples/sampleMNIST` directory. The binary named `sample_mnist` will be created in the `<TensorRT root directory>/bin` directory.
	```
	cd <TensorRT root directory>/samples/sampleMNIST
	make
	```
	Where `<TensorRT root directory>` is where you installed TensorRT.
	
2. Run the sample to perform inference on the digit:
    ```
	./sample_mnist [-h] [--datadir=/path/to/data/dir/] [--useDLA=N] [--fp16 or --int8]
	```
	This sample reads three Caffe files to build the network:
	-   `mnist.prototxt` 
	The prototxt file that contains the network design.

	-   `mnist.caffemodel`
	The model file which contains the trained weights for the network.

	-   `mnist_mean.binaryproto`
	The binaryproto file which contains the means.

	This sample can be run in FP16 and INT8 modes as well.

	**Note:** By default, the sample expects these files to be in either the `data/samples/mnist/` or `data/mnist/` directories. The list of default directories can be changed by adding one or more paths with `--datadir=/new/path/` as a command line argument.

3.  Verify that the sample ran successfully. If the sample runs successfully you should see output similar to the following; ASCII rendering of the input image with digit 3:
    ```
	&&&& RUNNING TensorRT.sample_mnist # ./sample_mnist
	[I] Building and running a GPU inference engine for MNIST
	[I] Input:
	@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	@@@@@@@@#-:.-=@@@@@@@@@@@@@@
	@@@@@%=     . *@@@@@@@@@@@@@
	@@@@% .:+%%%  *@@@@@@@@@@@@@
	@@@@+=#@@@@@# @@@@@@@@@@@@@@
	@@@@@@@@@@@%  @@@@@@@@@@@@@@
	@@@@@@@@@@@: *@@@@@@@@@@@@@@
	@@@@@@@@@@- .@@@@@@@@@@@@@@@
	@@@@@@@@@:  #@@@@@@@@@@@@@@@
	@@@@@@@@:   +*%#@@@@@@@@@@@@
	@@@@@@@%         :+*@@@@@@@@
	@@@@@@@@#*+--.::     +@@@@@@
	@@@@@@@@@@@@@@@@#=:.  +@@@@@
	@@@@@@@@@@@@@@@@@@@@  .@@@@@
	@@@@@@@@@@@@@@@@@@@@#. #@@@@
	@@@@@@@@@@@@@@@@@@@@#  @@@@@
	@@@@@@@@@%@@@@@@@@@@- +@@@@@
	@@@@@@@@#-@@@@@@@@*. =@@@@@@
	@@@@@@@@ .+%%%%+=.  =@@@@@@@
	@@@@@@@@           =@@@@@@@@
	@@@@@@@@*=:   :--*@@@@@@@@@@
	@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	@@@@@@@@@@@@@@@@@@@@@@@@@@@@

	[I] Output:
	0:
	1:
	2:
	3: **********
	4:
	5:
	6:
	7:
	8:
	9:

	&&&& PASSED TensorRT.sample_mnist # ./sample_mnist
	```

	This output shows that the sample ran successfully; `PASSED`.
 

### Sample --help options

To see the full list of available options and their descriptions, use the `-h` or `--help` command line option. For example:
```
Usage: ./sample_mnist [-h or --help] [-d or --datadir=<path to data directory>] [--useDLACore=<int>]
--help Display help information
--datadir Specify path to a data directory, overriding the default. This option can be used multiple times to add multiple directories. If no data directories are given, the default is to use (data/samples/mnist/, data/mnist/)
--useDLACore=N Specify a DLA engine for layers that support DLA. Value can range from 0 to n-1, where n is the number of DLA engines on the platform.
--int8 Run in Int8 mode.
--fp16 Run in FP16 mode.
```

# Additional resources

The following resources provide a deeper understanding about sampleMNIST:

**MNIST**
- [MNIST dataset](https://github.com/NVIDIA/DIGITS/blob/master/docs/GettingStarted.md)

**Documentation**
- [Introduction To NVIDIA’s TensorRT Samples](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sample-support-guide/index.html#samples)
- [Working With TensorRT Using The C++ API](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#c_topics)
- [NVIDIA’s TensorRT Documentation Library](https://docs.nvidia.com/deeplearning/sdk/tensorrt-archived/index.html)

# License

For terms and conditions for use, reproduction, and distribution, see the [TensorRT Software License Agreement](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html) documentation.


# Changelog

February 2019
This `README.md` file was recreated, updated and reviewed.


# Known issues

There are no known issues in this sample.