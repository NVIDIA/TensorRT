# “Hello World” For TensorRT From ONNX


**Table Of Contents**
- [Description](#description)
- [How does this sample work?](#how-does-this-sample-work)
	* [Converting the ONNX model to a TensorRT network](#converting-the-onnx-model-to-a-tensorrt-network)
	* [Building the engine](#building-the-engine)
	* [Running inference](#running-inference)
	* [TensorRT API layers and ops](#tensorrt-api-layers-and-ops)
- [Running the sample](#running-the-sample)
	* [Sample `--help` options](#sample-help-options)
- [Additional resources](#additional-resources)
- [License](#license)
- [Changelog](#changelog)
- [Known issues](#known-issues)

## Description

This sample, sampleOnnxMNIST, converts a model trained on the [MNIST dataset](https://github.com/onnx/models/tree/master/vision/classification/mnist) in Open Neural Network Exchange (ONNX) format to a TensorRT network and runs inference on the network.

ONNX is a standard for representing deep learning models that enables models to be transferred between frameworks.

## How does this sample work?

This sample creates and runs the TensorRT engine from an ONNX model of the MNIST network. It demonstrates how TensorRT can consume an ONNX model as input to create a network.

Specifically, this sample:
- [Converts the ONNX model to a TensorRT network](#converting-the-onnx-model-to-a-tensorrt-network)
- [Builds an engine](#building-an-engine)
- [Runs inference using the generated TensorRT network](#running-inference)

### Converting the ONNX model to a TensorRT network

The model file can be converted to a TensorRT network using the ONNX parser. The parser can be initialized with the
network definition that the parser will write to and the logger object.

`auto parser = nvonnxparser::createParser(*network, sample::gLogger.getTRTLogger());`

The ONNX model file is then passed onto the parser along with the logging level

```
if (!parser->parseFromFile(model_file, static_cast<int>(sample::gLogger.getReportableSeverity())))
{
	  string msg("failed to parse onnx file");
	  sample::gLogger->log(nvinfer1::ILogger::Severity::kERROR, msg.c_str());
	  exit(EXIT_FAILURE);
}
```

After the TensorRT network is constructed by parsing the model, the TensorRT engine can be built to run inference.

### Building the engine

To build the engine, create the builder and pass a logger created for TensorRT which is used for reporting errors, warnings and informational messages in the network:
`IBuilder* builder = createInferBuilder(sample::gLogger);`

To build the engine from the generated TensorRT network, issue the following call:
`nvinfer1::ICudaEngine* engine = builder->buildCudaEngine(*network);`

After you build the engine, verify that the engine is running properly by confirming the output is what you expected. The output format of this sample should be the same as the output of sampleMNIST.

### Running inference

To run inference using the created engine, see [Performing Inference In C++](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#perform_inference_c).

**Note:** It’s important to preprocess the data and convert it to the format accepted by the network. In this example, the sample input is in PGM (portable graymap) format. The model expects an input of image `1x28x28` scaled to between `[0,1]`.

### TensorRT API layers and ops

In this sample, the following layers are used. For more information about these layers, see the [TensorRT Developer Guide: Layers](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#layers) documentation.

[Activation layer](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#activation-layer)
The Activation layer implements element-wise activation functions. Specifically, this sample uses the Activation layer with the type `kRELU`.

[Convolution layer](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#convolution-layer)
The Convolution layer computes a 2D (channel, height, and width) convolution, with or without bias.

[MatrixMultiplyLayer](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#matrixmultiply-layer)
The MatrixMultiply layer implements a matrix multiplication operation.
(The [FullyConnected layer](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#fullyconnected-layer) is deprecated since 8.4.
The bias of a FullyConnected layer can be added with an
[ElementwiseLayer](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#elementwise-layer) of `SUM` operation.)

[Pooling layer](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#pooling-layer)
The Pooling layer implements pooling within a channel. Supported pooling types are `maximum`, `average` and `maximum-average blend`.

[Scale layer](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#scale-layer)
The Scale layer implements a per-tensor, per-channel, or per-element affine transformation and/or exponentiation by constant values.

[Shuffle layer](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#shuffle-layer)
The Shuffle layer implements a reshape and transpose operator for tensors.


## Running the sample

1. Compile the sample by following build instructions in [TensorRT README](https://github.com/NVIDIA/TensorRT/).

2.  Run the sample to build and run the MNIST engine from the ONNX model.
	```
	./sample_onnx_mnist [-h or --help] [-d or --datadir=<path to data directory>] [--useDLACore=<int>] [--int8 or --fp16]
	```

3.  Verify that the sample ran successfully. If the sample runs successfully you should see output similar to the following:
	```
	&&&& RUNNING TensorRT.sample_onnx_mnist # ./sample_onnx_mnist
	----------------------------------------------------------------
	Input filename: ../../../../../../data/samples/mnist/mnist.onnx
	ONNX IR version: 0.0.3
	Opset version: 1
	Producer name: CNTK
	Producer version: 2.4
	Domain:
	Model version: 1
	Doc string:
	----------------------------------------------------------------
	[I] Input:
	@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	@@@@@@@@@@@*.  .*@@@@@@@@@@@
	@@@@@@@@@@*.     +@@@@@@@@@@
	@@@@@@@@@@. :#+   %@@@@@@@@@
	@@@@@@@@@@.:@@@+  +@@@@@@@@@
	@@@@@@@@@@.:@@@@:  +@@@@@@@@
	@@@@@@@@@@=%@@@@:  +@@@@@@@@
	@@@@@@@@@@@@@@@@#  +@@@@@@@@
	@@@@@@@@@@@@@@@@*  +@@@@@@@@
	@@@@@@@@@@@@@@@@:  +@@@@@@@@
	@@@@@@@@@@@@@@@@:  +@@@@@@@@
	@@@@@@@@@@@@@@@*  .@@@@@@@@@
	@@@@@@@@@@%**%@.  *@@@@@@@@@
	@@@@@@@@%+.  .:  .@@@@@@@@@@
	@@@@@@@@=  ..    :@@@@@@@@@@
	@@@@@@@@:  *@@:  :@@@@@@@@@@
	@@@@@@@%   %@*    *@@@@@@@@@
	@@@@@@@%   ++ ++  .%@@@@@@@@
	@@@@@@@@-    +@@-  +@@@@@@@@
	@@@@@@@@=  :*@@@#  .%@@@@@@@
	@@@@@@@@@+*@@@@@%.   %@@@@@@
	@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	@@@@@@@@@@@@@@@@@@@@@@@@@@@@

	[I] Output:
	Prob 0 0.0000 Class 0:
	Prob 1 0.0000 Class 1:
	Prob 2 1.0000 Class 2: **********
	Prob 3 0.0000 Class 3:
	Prob 4 0.0000 Class 4:
	Prob 5 0.0000 Class 5:
	Prob 6 0.0000 Class 6:
	Prob 7 0.0000 Class 7:
	Prob 8 0.0000 Class 8:
	Prob 9 0.0000 Class 9:

	&&&& PASSED TensorRT.sample_onnx_mnist # ./sample_onnx_mnist
	```

	This output shows that the sample ran successfully; PASSED.


### Sample `--help` options

To see the full list of available options and their descriptions, use the `-h` or `--help` command line option.


# Additional resources

The following resources provide a deeper understanding about the ONNX project and MNIST model:

**ONNX**
- [GitHub: ONNX](https://github.com/onnx/onnx)
- [Github: ONNX-TensorRT Open source parser](https://github.com/onnx/onnx-tensorrt)

**Models**
- [MNIST - Handwritten Digit Recognition](https://github.com/onnx/models/tree/master/mnist)
- [GitHub: ONNX Models](https://github.com/onnx/models)

**Documentation**
- [Introduction To NVIDIA’s TensorRT Samples](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sample-support-guide/index.html#samples)
- [Working With TensorRT Using The C++ API](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#c_topics)
- [NVIDIA’s TensorRT Documentation Library](https://docs.nvidia.com/deeplearning/sdk/tensorrt-archived/index.html)

# License

For terms and conditions for use, reproduction, and distribution, see the [TensorRT Software License Agreement](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html) documentation.


# Changelog

March 2019
This `README.md` file was recreated, updated and reviewed.


# Known issues

There are no known issues in this sample.
