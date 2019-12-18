# Digit Recognition With Dynamic Shapes In TensorRT


**Table Of Contents**
- [Description](#description)
- [How does this sample work?](#how-does-this-sample-work)
    * [Creating the preprocessing network](#creating-the-preprocessing-network)
    * [Parsing the ONNX MNIST model](#parsing-the-onnx-mnist-model)
    * [Building engines](#building-engines)
    * [Running inference](#running-inference)
	* [TensorRT API layers and ops](#tensorrt-api-layers-and-ops)
- [Running the sample](#running-the-sample)
	* [Sample `--help` options](#sample---help-options)
- [Additional resources](#additional-resources)
- [License](#license)
- [Changelog](#changelog)
- [Known issues](#known-issues)

## Description

This sample, sampleDynamicReshape, demonstrates how to use dynamic input dimensions in TensorRT. It creates an engine that takes a dynamically shaped input and resizes it to be consumed by an ONNX MNIST model that expects a fixed size input. For more information, see [Working With Dynamic Shapes](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#work_dynamic_shapes) in the TensorRT Developer Guide.

## How does this sample work?

This sample creates an engine for resizing an input with dynamic dimensions to a size that an ONNX MNIST model can consume.

Specifically, this sample:
-   Creates a network with dynamic input dimensions to act as a preprocessor for the model
-   Parses an ONNX MNIST model to create a second network
-   Builds engines for both networks
-   Runs inference using both engines

### Creating the preprocessing network

First, create a network with full dims support:
`auto preprocessorNetwork = makeUnique(builder->createNetworkV2(1U << static_cast<int32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH)));`

Next, add an input layer that accepts an input with a dynamic shape, followed by a resize layer that will reshape the input to the shape the model expects:
```
auto input = preprocessorNetwork->addInput("input", nvinfer1::DataType::kFLOAT, Dims4{1, 1, -1, -1});
auto resizeLayer = preprocessorNetwork->addResize(*input);
resizeLayer->setOutputDimensions(mPredictionInputDims);
preprocessorNetwork->markOutput(*resizeLayer->getOutput(0));
```

The -1 dimensions denote dimensions that will be supplied at runtime.

### Parsing the ONNX MNIST model

First, create an empty full-dims network, and parser:
```
const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
auto network = makeUnique(builder->createNetworkV2(explicitBatch));
auto parser = nvonnxparser::createParser(*network, gLogger.getTRTLogger());
```

Next, parse the model file to populate the network:
```
parser->parseFromFile(locateFile(mParams.onnxFileName, mParams.dataDirs).c_str(), static_cast<int>(gLogger.getReportableSeverity()));
```

### Building engines

When building the preprocessor engine, also provide an optimization profile so that TensorRT knows which input shapes to optimize for:
```
auto preprocessorConfig = makeUnique(builder->createNetworkConfig());
auto profile = builder->createOptimizationProfile();
```

`OptProfileSelector::kOPT` specifies the dimensions that the profile will be optimized for, whereas `OptProfileSelector::kMIN` and `OptProfileSelector::kMAX` specify the minimum and maximum dimensions for which the profile will be valid:
```
profile->setDimensions(input->getName(), OptProfileSelector::kMIN, Dims4{1, 1, 1, 1});
profile->setDimensions(input->getName(), OptProfileSelector::kOPT, Dims4{1, 1, 28, 28});
profile->setDimensions(input->getName(), OptProfileSelector::kMAX, Dims4{1, 1, 56, 56});
preprocessorConfig->addOptimizationProfile(profile);
mPreprocessorEngine = makeUnique(builder->buildEngineWithConfig(*preprocessorNetwork, *preprocessorConfig));
```

For the MNIST model, attach a Softmax layer to the end of the network and replace the existing network output with the Softmax:
```
auto softmax = network->addSoftMax(*network->getOutput(0));
network->unmarkOutput(*network->getOutput(0));
network->markOutput(*softmax->getOutput(0));
```

Finally, build as normal:
`mPredictionEngine = makeUnique(builder->buildEngineWithConfig(*network, *config));`

### Running inference

During inference, first copy the input buffer to the device:
```
CHECK(cudaMemcpy(mInput.deviceBuffer.data(), mInput.hostBuffer.data(), mInput.hostBuffer.nbBytes(), cudaMemcpyHostToDevice));
```

Since the preprocessor engine accepts dynamic shapes, specify the actual shape of the current input to the execution context:
`mPreprocessorContext->setBindingDimensions(0, inputDims);`

Next, run the preprocessor using the `executeV2` function. The example writes the output of the preprocessor engine directly to the input device buffer of the MNIST engine:
```
std::vector<void*> preprocessorBindings = {mInput.deviceBuffer.data(), mPredictionInput.data()};
bool status = mPreprocessorContext->executeV2(preprocessorBindings.data());
```

Then, run the MNIST engine:
```
std::vector<void*> predicitonBindings = {mPredictionInput.data(), mOutput.deviceBuffer.data()};
status = mPredictionContext->executeV2(predicitonBindings.data());
```

Finally, copy the output back to the host:
```
CHECK(cudaMemcpy(mOutput.hostBuffer.data(), mOutput.deviceBuffer.data(), mOutput.deviceBuffer.nbBytes(), cudaMemcpyDeviceToHost));
```

### TensorRT API layers and ops

In this sample, the following layers are used. For more information about these layers, see the [TensorRT Developer Guide: Layers](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#layers) documentation.

[Resize layer](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#resize-layer)
The IResizeLayer implements the resize operation on an input tensor.

## Running the sample

1.  Compile this sample by running `make` in the `<TensorRT root directory>/samples/sampleDynamicReshape` directory. The binary named `sample_dynamic_reshape` will be created in the `<TensorRT root directory>/bin` directory.
	```
	cd <TensorRT root directory>/samples/sampleDynamicReshape
	make
	```

	Where `<TensorRT root directory>` is where you installed TensorRT.

2.  Run the sample.
	```
	./sample_dynamic_reshape [-h or --help] [-d or --datadir=<path to data directory>] [--useDLACore=<int>] [--int8 or --fp16]
	```

3. Verify that the sample ran successfully. If the sample runs successfully you should see output similar to the following:
	```
	&&&& RUNNING TensorRT.sample_dynamic_reshape # ./sample_dynamic_reshape
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
	@@@@@@@@@@@*. .*@@@@@@@@@@@
	@@@@@@@@@@*. +@@@@@@@@@@
	@@@@@@@@@@. :#+ %@@@@@@@@@
	@@@@@@@@@@.:@@@+ +@@@@@@@@@
	@@@@@@@@@@.:@@@@: +@@@@@@@@
	@@@@@@@@@@=%@@@@: +@@@@@@@@
	@@@@@@@@@@@@@@@@# +@@@@@@@@
	@@@@@@@@@@@@@@@@* +@@@@@@@@
	@@@@@@@@@@@@@@@@: +@@@@@@@@
	@@@@@@@@@@@@@@@@: +@@@@@@@@
	@@@@@@@@@@@@@@@* .@@@@@@@@@
	@@@@@@@@@@%**%@. *@@@@@@@@@
	@@@@@@@@%+. .: .@@@@@@@@@@
	@@@@@@@@= .. :@@@@@@@@@@
	@@@@@@@@: *@@: :@@@@@@@@@@
	@@@@@@@% %@* *@@@@@@@@@
	@@@@@@@% ++ ++ .%@@@@@@@@
	@@@@@@@@- +@@- +@@@@@@@@
	@@@@@@@@= :*@@@# .%@@@@@@@
	@@@@@@@@@+*@@@@@%. %@@@@@@
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

	&&&& PASSED TensorRT.sample_dynamic_reshape # ./sample_dynamic_reshape
	```

	This output shows that the sample ran successfully; `PASSED`.


### Sample `--help` options

To see the full list of available options and their descriptions, use the `-h` or `--help` command line option.


# Additional resources

The following resources provide a deeper understanding of dynamic shapes.

**ONNX**
- [GitHub: ONNX](https://github.com/onnx/onnx)
- [GitHub: ONNX-TensorRT open source parser](https://github.com/onnx/onnx-tensorrt)

**Models**
- [MNIST - Handwritten Digit Recognition](https://github.com/onnx/models/tree/master/mnist)
- [GitHub: ONNX Models](https://github.com/onnx/models)

**Documentation**
- [Introduction To NVIDIA’s TensorRT Samples](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sample-support-guide/index.html#samples)
- [Working With TensorRT Using The Python API](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#python_topics)
- [NVIDIA’s TensorRT Documentation Library](https://docs.nvidia.com/deeplearning/sdk/tensorrt-archived/index.html)

# License

For terms and conditions for use, reproduction, and distribution, see the [TensorRT Software License Agreement](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html) documentation.


# Changelog

June 2019
This is the first release of the `README.md` file and sample.


# Known issues

There are no known issues in this sample.
