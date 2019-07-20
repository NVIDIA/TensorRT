# Import A TensorFlow Model And Run Inference

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

This sample, sampleUffMNIST, imports a TensorFlow model trained on the MNIST dataset.

The MNIST TensorFlow model has been converted to UFF (Universal Framework Format) using the explanation described in [Working With TensorFlow](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#working_tf).

The UFF is designed to store neural networks as a graph. The NvUffParser that we use in this sample parses the UFF file in order to create an inference engine based on that neural network.

With TensorRT, you can take a TensorFlow trained model, export it into a UFF protobuf file (`.uff`) using the [UFF converter](https://docs.nvidia.com/deeplearning/sdk/tensorrt-api/python_api/uff/uff.html#), and import it using the UFF parser.

## How does this sample work?

This sample loads the `.uff` file created from the TensorFlow MNIST model, parses it to create a TensorRT engine and performs inference using the created engine.

Specifically, this sample:
- Loads a trained TensorFlow model that has been pre-converted to the UFF file format
- Creates the UFF Parser (see [Importing From TensorFlow Using Python](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#import_tf_python))
- Uses the UFF Parser, registers inputs and outputs, and provides the dimensions and the order of the input tensor
- Builds an engine (see [Building An Engine In C++](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#build_engine_c))
- Uses the engine to perform inference 10 times and reports average inference time (see [Performing Inference in C++](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#perform_inference_c))

### TensorRT API layers and ops

In this sample, the following layers are used. For more information about these layers, see the [TensorRT Developer Guide: Layers](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#layers) documentation.

[Activation layer](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#activation-layer)
The Activation layer implements element-wise activation functions. 

[Convolution layer](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#convolution-layer)
The Convolution layer computes a 2D (channel, height, and width) convolution, with or without bias.

[FullyConnected layer](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#fullyconnected-layer)
The FullyConnected layer implements a matrix-vector product, with or without bias.

[Pooling layer](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#pooling-layer)
The Pooling layer implements pooling within a channel. Supported pooling types are `maximum`, `average` and `maximum-average blend`.

[Scale layer](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#scale-layer)
The Scale layer implements a per-tensor, per-channel, or per-element affine transformation and/or exponentiation by constant values.

[Shuffle layer](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#shuffle-layer)
The Shuffle layer implements a reshape and transpose operator for tensors.


## Running the sample

1.  Compile this sample by running `make` in the `<TensorRT root directory>/samples/sampleUffMNIST` directory. The binary named `sample_uff_mnist` will be created in the `<TensorRT root directory>/bin` directory.
	```
	cd <TensorRT root directory>/samples/sampleUffMNIST
	make
	```

	Where `<TensorRT root directory>` is where you installed TensorRT.

2.  Run the sample to create an MNIST engine from a UFF model and perform inference using it.
	```
	./sample_uff_mnist [-h or --help] [-d or --datadir=<path to data directory>] [--useDLACore=<int>] [--int8] [--fp16]
	```

3.  Verify that the sample ran successfully. If the sample runs successfully you should see output similar to the following:
	```
	&&&& RUNNING TensorRT.sample_uff_mnist # ./sample_uff_mnist
	[I] ../../../../../../data/samples/mnist/lenet5.uff
	[I] Input:
	@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	@@@@@@@@@@@@@@@@+  :@@@@@@@@
	@@@@@@@@@@@@@@%= :. --%@@@@@
	@@@@@@@@@@@@@%. -@= - :@@@@@
	@@@@@@@@@@@@@: -@@#%@@ #@@@@
	@@@@@@@@@@@@: #@@@@@@@-#@@@@
	@@@@@@@@@@@= #@@@@@@@@=%@@@@
	@@@@@@@@@@= #@@@@@@@@@:@@@@@
	@@@@@@@@@+ -@@@@@@@@@%.@@@@@
	@@@@@@@@@::@@@@@@@@@@+-@@@@@
	@@@@@@@@-.%@@@@@@@@@@.*@@@@@
	@@@@@@@@ *@@@@@@@@@@@ *@@@@@
	@@@@@@@% %@@@@@@@@@%.-@@@@@@
	@@@@@@@:*@@@@@@@@@+. %@@@@@@
	@@@@@@# @@@@@@@@@# .*@@@@@@@
	@@@@@@# @@@@@@@@=  +@@@@@@@@
	@@@@@@# @@@@@@%. .+@@@@@@@@@
	@@@@@@# @@@@@*. -%@@@@@@@@@@
	@@@@@@# ---    =@@@@@@@@@@@@
	@@@@@@#      *%@@@@@@@@@@@@@
	@@@@@@@%: -=%@@@@@@@@@@@@@@@
	@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	@@@@@@@@@@@@@@@@@@@@@@@@@@@@

	[I] Output:
	0 => 14.255573  : ***
	1 => -4.830786  :
	2 =>  1.091855  :
	3 => -6.290083  :
	4 => -0.835606  :
	5 => -6.920589  :
	6 =>  2.403986  :
	7 => -6.011705  :
	8 =>  0.730784  :
	9 =>  1.500333  :

	… (repeated 10 times)

	[I] Average over 10 runs is 0.0643946 ms.
	&&&& PASSED TensorRT.sample_uff_mnist # ./sample_uff_mnist
	```
  
	This output shows that the sample ran successfully; PASSED.

### Sample --help options

To see the full list of available options and their descriptions, use the `-h` or `--help` command line option. For example:
```
Usage: ./sample_uff_mnist [-h or --help] [-d or --datadir=<path to data directory>] [--useDLACore=<int>]
--help Display help information
--datadir Specify path to a data directory, overriding the default. This option can be used multiple times to add multiple directories. If no data directories are given, the default is to use (data/samples/mnist/, data/mnist/)
--useDLACore=N Specify a DLA engine for layers that support DLA. Value can range from 0 to n-1, where n is the number of DLA engines on the platform.
--int8 Run in Int8 mode.
--fp16 Run in FP16 mode.
```

# Additional resources

The following resources provide a deeper understanding about the MNIST model from TensorFlow and using it in TensorRT:

**Models**
- [MNIST](https://keras.io/datasets/#mnist-database-of-handwritten-digits)

**Documentation**
- [Introduction To NVIDIA’s TensorRT Samples](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sample-support-guide/index.html#samples)
- [Working With TensorRT Using The C++ API](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#c_topics)
- [NVIDIA’s TensorRT Documentation Library](https://docs.nvidia.com/deeplearning/sdk/tensorrt-archived/index.html)

# License

For terms and conditions for use, reproduction, and distribution, see the [TensorRT Software License Agreement](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html) documentation.


# Changelog

March 2019
This is the first release of this `README.md` file.


# Known issues

There are no known issues in this sample.
