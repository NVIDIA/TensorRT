# Implementing CoordConv in TensorRT with a custom plugin


**Table Of Contents**
- [Implementing CoordConv in TensorRT with a custom plugin](#implementing-coordconv-in-tensorrt-with-a-custom-plugin)
	- [Description](#description)
	- [How does this sample work?](#how-does-this-sample-work)
		- [Converting the ONNX model to a TensorRT network](#converting-the-onnx-model-to-a-tensorrt-network)
		- [Building the engine](#building-the-engine)
		- [Running inference](#running-inference)
		- [TensorRT API layers and ops](#tensorrt-api-layers-and-ops)
	- [Running the sample](#running-the-sample)
		- [Sample --help options](#sample---help-options)
- [Additional resources](#additional-resources)
- [License](#license)
- [Changelog](#changelog)
- [Known issues](#known-issues)

## Description

This sample, sampleOnnxMnistCoordConvAC, converts a model trained on the `MNIST dataset` in Open Neural Network Exchange (ONNX) format to a TensorRT network and runs inference on the network.
This model was trained in PyTorch and it contains custom CoordConv layers instead of Conv layers.<br/>
Model with CoordConvAC layers training script and code of CoordConv layers in PyTorch: [link](https://github.com/denti/mnist-coordconv-pytorch/blob/master/main_coord_conv.py)<br/>
Original model with usual Conv layers: [link](https://github.com/pytorch/examples/tree/master/mnist)

[CoordConv](https://arxiv.org/abs/1807.03247)  layer is a layer proposed by Uber AI Labs at 2018.
It improves the quality of regular Conv layers by adding additional channels with relative coordinates to the input data.
This layer is used in classification, detection, segmentation and other NN architectures.
The CoordConv layer maps to the `CoordConvAC_TRT` custom plugin implemented in TensorRT for fast inference.
This plugin can be found at `TensorRT/plugin/coordConvACPlugin`. Additional information about the layer and plugin implementation can be found at `TensorRT/plugin/coordConvACPlugin/README.md`

ONNX is a standard for representing deep learning models that enables models to be transferred between frameworks.

## How does this sample work?

This sample creates and runs a TensorRT engine on an ONNX model of MNIST trained with CoordConv layers. It demonstrates how TensorRT can parse and import ONNX models, as well as use plugins to run custom layers in neural networks.

Specifically, this sample:
- [Converts the ONNX model with custom layer to a TensorRT network](#converting-the-onnx-model-to-a-tensorrt-network)
- [Builds an engine with custom layer](#building-an-engine)
- [Runs inference using the generated TensorRT network](#running-inference)

### Converting the ONNX model to a TensorRT network

The model file can be converted to a TensorRT network using the ONNX parser. The parser can be initialized with the
network definition that the parser will write to and the logger object.

`auto parser = nvonnxparser::createParser(*network, sample::gLogger.getTRTLogger());`

Plugins library needs to be added to the code to parse custom layers implemented as Plugins

`initLibNvInferPlugins(&sample::gLogger, "ONNXTRT_NAMESPACE");`

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

**Note2:** Additional preprocessing needs to be applied to the data before putting it to the NN input due to the same normalization preprocessing were used when model was trained [transforms.Normalize((0.1307,), (0.3081,))](https://github.com/pytorch/examples/tree/master/mnist):

```
const float PYTORCH_NORMALIZE_MEAN = 0.1307;
const float PYTORCH_NORMALIZE_STD = 0.3081;
hostDataBuffer[i] = ((1.0 - float(fileData[i] / 255.0)) - PYTORCH_NORMALIZE_MEAN) / PYTORCH_NORMALIZE_STD;
```

### TensorRT API layers and ops

In this sample, the following layers and plugins are used. For more information about these layers, see the [TensorRT Developer Guide: Layers](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#layers) documentation.

[CoordConvAC layer](https://github.com/NVIDIA/TensorRT/tree/main/plugin/coordConvACPlugin)
Custom layer implemented with CUDA API that implements operation AddChannels. This layer expands the input data by adding additional channels with relative coordinates.

[Activation layer](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#activation-layer)
The Activation layer implements element-wise activation functions. Specifically, this sample uses the Activation layer with the type `kRELU`.

[Convolution layer](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#convolution-layer)
The Convolution layer computes a 2D (channel, height, and width) convolution, with or without bias.

[MatrixMultiplyLayer](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#matrixmultiply-layer)
The MatrixMultiply layer implements a matrix multiplication.
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

1. The sample gets compiled when building the TensorRT OSS following the [instructions](https://github.com/NVIDIA/TensorRT). The binary named sample_onnx_mnist_coord_conv_ac will be created in the output directory.

2. (Optional) If the ONNX model on MNIST dataset is not available, you can generate an ONNX model for running this sample using the following commands:
    ```
    python3 mnist_coord_conv_train.py --save-onnx
    python3 modify_onnx_ac.py
    ``` 
    The first line trains a model for the MNIST dataset and saves it as an ONNX model. The second line modifies the ONNX model structure to make it work with TensorRT for building the MNIST engine. Please use torch 1.10.2 to run these scripts.

3.  Run the sample to build and run the MNIST engine from the ONNX model.
    ```
    ./sample_onnx_mnist_coord_conv_ac [-h or --help] [-d or --datadir=<path to data directory>] [--useDLACore=<int>] [--int8 or --fp16]
    ```

4. Verify that the sample ran successfully. If the sample runs successfully you should see output similar to the following:
    ```
    &&&& RUNNING TensorRT.sample_coord_conv_ac_onnx_mnist # ./sample_onnx_mnist_coord_conv_ac
    ----------------------------------------------------------------
    Input filename:   data/mnist/mnist_with_coordconv.onnx
    ONNX IR version:  0.0.6
    Opset version:    11
    Producer name:
    Producer version:
    Domain:
    Model version:    0
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
    Prob 0  0.0001 Class 0:
    Prob 1  0.0003 Class 1:
    Prob 2  0.9975 Class 2: **********
    Prob 3  0.0009 Class 3:
    Prob 4  0.0000 Class 4:
    Prob 5  0.0001 Class 5:
    Prob 6  0.0001 Class 6:
    Prob 7  0.0000 Class 7:
    Prob 8  0.0009 Class 8:
    Prob 9  0.0000 Class 9:

    &&&& PASSED TensorRT.sample_coord_conv_ac_onnx_mnist # ./sample_onnx_mnist_coord_conv_ac
	```

	This output shows that the sample ran successfully; PASSED.

### Sample --help options

To see the full list of available options and their descriptions, use the `-h` or `--help` command line option. For example:
```
Usage: ./sample_onnx_mnist_coord_conv_ac [-h or --help] [-d or --datadir=<path to data directory>] [--useDLACore=<int>]
--help Display help information
--datadir Specify path to a data directory, overriding the default. This option can be used multiple times to add multiple directories. If no data directories are given, the default is to use (data/samples/mnist/, data/mnist/)
--useDLACore=N Specify a DLA engine for layers that support DLA. Value can range from 0 to n-1, where n is the number of DLA engines on the platform.
--int8 Run in Int8 mode.
--fp16 Run in FP16 mode.
```

# Additional resources

The following resources provide a deeper understanding about the ONNX project and MNIST model:

**CoordConv Layer**
- [Arxiv paper by Uber AI Labs](https://arxiv.org/abs/1807.03247)
- [Blog post about the CoordConv layer](https://eng.uber.com/coordconv/)
- [Path to the layer's plugin in repository](https://github.com/NVIDIA/TensorRT/tree/main/plugin/coordConvACPlugin)

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

April 2020
This `README.md` file was recreated, updated and reviewed.


# Known issues

There are no known issues in this sample.
