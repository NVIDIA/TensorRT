# Digit Recognition With Dynamic Shapes In TensorRT


**Table Of Contents**
- [Description](#description)
- [How does this sample work?](#how-does-this-sample-work)
    * [Creating the preprocessing network](#creating-the-preprocessing-network)
    * [Parsing the ONNX MNIST model](#parsing-the-onnx-mnist-model)
    * [Building engines](#building-engines)
    * [Running inference](#running-inference)
	* [TensorRT API layers and ops](#tensorrt-api-layers-and-ops)
- [Preparing sample data](#preparing-sample-data)
- [Running the sample](#running-the-sample)
	* [Sample `--help` options](#sample-help-options)
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
-   Builds engines for both networks and does calibration if running in int8
-   Runs inference using both engines

### Creating the preprocessing network

First, create a network with full dims support:
`auto preprocessorNetwork = makeUnique(builder->createNetworkV2(1U << static_cast<int32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH)));`

Next, add an input layer that accepts an input with a dynamic shape, followed by a resize layer that will reshape the input to the shape the model expects:
```
auto input = preprocessorNetwork->addInput("input", nvinfer1::DataType::kFLOAT, Dims4{-1, 1, -1, -1});
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
auto parser = nvonnxparser::createParser(*network, sample::gLogger.getTRTLogger());
```

Next, parse the model file to populate the network:
```
parser->parseFromFile(locateFile(mParams.onnxFileName, mParams.dataDirs).c_str(), static_cast<int>(sample::gLogger.getReportableSeverity()));
```

### Building engines

When building the preprocessor engine, also provide an optimization profile so that TensorRT knows which input shapes to optimize for:
```
auto preprocessorConfig = makeUnique(builder->createBuilderConfig());
auto profile = builder->createOptimizationProfile();
```

`OptProfileSelector::kOPT` specifies the dimensions that the profile will be optimized for, whereas `OptProfileSelector::kMIN` and `OptProfileSelector::kMAX` specify the minimum and maximum dimensions for which the profile will be valid:
```
profile->setDimensions(input->getName(), OptProfileSelector::kMIN, Dims4{1, 1, 1, 1});
profile->setDimensions(input->getName(), OptProfileSelector::kOPT, Dims4{1, 1, 28, 28});
profile->setDimensions(input->getName(), OptProfileSelector::kMAX, Dims4{1, 1, 56, 56});
preprocessorConfig->addOptimizationProfile(profile);
```

Create an optimization profile for calibration:
```
auto profileCalib = builder->createOptimizationProfile();
const int calibBatchSize{256};
profileCalib->setDimensions(input->getName(), OptProfileSelector::kMIN, Dims4{calibBatchSize, 1, 28, 28});
profileCalib->setDimensions(input->getName(), OptProfileSelector::kOPT, Dims4{calibBatchSize, 1, 28, 28});
profileCalib->setDimensions(input->getName(), OptProfileSelector::kMAX, Dims4{calibBatchSize, 1, 28, 28});
preprocessorConfig->setCalibrationProfile(profileCalib);
```

Prepare and set int8 calibrator if running in int8 mode:
```
std::unique_ptr<IInt8Calibrator> calibrator;
if (mParams.int8)
{
    preprocessorConfig->setFlag(BuilderFlag::kINT8);
    const int nCalibBatches{10};
    MNISTBatchStream calibrationStream(calibBatchSize, nCalibBatches, "train-images-idx3-ubyte",
        "train-labels-idx1-ubyte", mParams.dataDirs);
    calibrator.reset(new Int8EntropyCalibrator2<MNISTBatchStream>(
        calibrationStream, 0, "MNISTPreprocessor", "input"));
    preprocessorConfig->setInt8Calibrator(calibrator.get());
}
```

Run engine build with config:
```
SampleUniquePtr<nvinfer1::IHostMemory> preprocessorPlan = makeUnique(
        builder->buildSerializedNetwork(*preprocessorNetwork, *preprocessorConfig));
if (!preprocessorPlan)
{
    sample::gLogError << "Preprocessor serialized engine build failed." << std::endl;
    return false;
}

mPreprocessorEngine = makeUnique(
    runtime->deserializeCudaEngine(preprocessorPlan->data(), preprocessorPlan->size()));
if (!mPreprocessorEngine)
{
    sample::gLogError << "Preprocessor engine deserialization failed." << std::endl;
    return false;
}
```

For the MNIST model, attach a Softmax layer to the end of the network, set softmax axis to 1 since network output has shape [1, 10] in full dims mode and replace the existing network output with the Softmax:
```
auto softmax = network->addSoftMax(*network->getOutput(0));
softmax->setAxes(1 << 1);
network->unmarkOutput(*network->getOutput(0));
network->markOutput(*softmax->getOutput(0));
```

A calibrator and a calibration profile are set the same way as above for the preprocessor engine config. `calibBatchSize` is set to 1 for the prediction engine as ONNX model has an explicit batch.

Finally, build as normal:
```
SampleUniquePtr<nvinfer1::IHostMemory> predictionPlan = makeUnique(builder->buildSerializedNetwork(*network, *config));
if (!predictionPlan)
{
    sample::gLogError << "Prediction serialized engine build failed." << std::endl;
    return false;
}

mPredictionEngine = makeUnique(
    runtime->deserializeCudaEngine(predictionPlan->data(), predictionPlan->size()));
if (!mPredictionEngine)
{
    sample::gLogError << "Prediction engine deserialization failed." << std::endl;
    return false;
}
```

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

## Preparing sample data

1. Download the sample data from [TensorRT release tarball](https://developer.nvidia.com/nvidia-tensorrt-download#), if not already mounted under `/usr/src/tensorrt/data` (NVIDIA NGC containers) and set it to `$TRT_DATADIR`.
    ```bash
    export TRT_DATADIR=/usr/src/tensorrt/data
    pushd $TRT_DATADIR/mnist
    pip3 install Pillow
    popd
    ```

## Running the sample

1. Compile the sample by following build instructions in [TensorRT README](https://github.com/NVIDIA/TensorRT/).

2.  Run the sample.
    ```bash
    ./sample_dynamic_reshape [-h or --help] [-d or --datadir=<path to data directory>] [--useDLACore=<int>] [--int8 or --fp16]
    ```

    For example:
    ```bash
    ./sample_dynamic_reshape --datadir $TRT_DATADIR/char-rnn --fp16
    ```

3. Verify that the sample ran successfully. If the sample runs successfully you should see output similar to the following:
    ```
  	&&&& RUNNING TensorRT.sample_dynamic_reshape # ./sample_dynamic_reshape
    ----------------------------------------------------------------
    Input filename:   ../../../../../data/samples/mnist/mnist.onnx
    ONNX IR version:  0.0.3
    Opset version:    8
    Producer name:    CNTK
    Producer version: 2.5.1
    Domain:           ai.cntk
    Model version:    1
    Doc string:
    ----------------------------------------------------------------
    [W] [TRT] onnx2trt_utils.cpp:214: Your ONNX model has been generated with INT64 weights, while TensorRT does not natively support INT64. Attempting to cast down to INT32.
    [W] [TRT] onnx2trt_utils.cpp:214: Your ONNX model has been generated with INT64 weights, while TensorRT does not natively support INT64. Attempting to cast down to INT32.
    [I] [TRT] Detected 1 inputs and 1 output network tensors.
    [I] [TRT] Detected 1 inputs and 1 output network tensors.
    [I] Profile dimensions in preprocessor engine:
    [I]     Minimum = (1, 1, 1, 1)
    [I]     Optimum = (1, 1, 28, 28)
    [I]     Maximum = (1, 1, 56, 56)
    [I] Input:
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@
    @@@@@@@@@@@*.  .*@@@@@@@@@@@
    @@@@@@@@@@*.     +@@@@@@@@@@
    @@@@@@@@@@. :#+   %@@@@@@@@@
    @@@@@@@@@@.:@@@+  +@@@@@@@@@
    @@@@@@@@@@.:@@@@: +@@@@@@@@@
    @@@@@@@@@@=%@@@@: +@@@@@@@@@
    @@@@@@@@@@@@@@@@# +@@@@@@@@@
    @@@@@@@@@@@@@@@@* +@@@@@@@@@
    @@@@@@@@@@@@@@@@: +@@@@@@@@@
    @@@@@@@@@@@@@@@@: +@@@@@@@@@
    @@@@@@@@@@@@@@@* .@@@@@@@@@@
    @@@@@@@@@@%**%@. *@@@@@@@@@@
    @@@@@@@@%+.  .: .@@@@@@@@@@@
    @@@@@@@@=  ..   :@@@@@@@@@@@
    @@@@@@@@: *@@:  :@@@@@@@@@@@
    @@@@@@@%  %@*    *@@@@@@@@@@
    @@@@@@@%  ++  ++ .%@@@@@@@@@
    @@@@@@@@-    +@@- +@@@@@@@@@
    @@@@@@@@=  :*@@@# .%@@@@@@@@
    @@@@@@@@@+*@@@@@%.  %@@@@@@@
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@

    [I] Output:
    [I]  Prob 0  0.0000 Class 0:
    [I]  Prob 1  0.0000 Class 1:
    [I]  Prob 2  1.0000 Class 2: **********
    [I]  Prob 3  0.0000 Class 3:
    [I]  Prob 4  0.0000 Class 4:
    [I]  Prob 5  0.0000 Class 5:
    [I]  Prob 6  0.0000 Class 6:
    [I]  Prob 7  0.0000 Class 7:
    [I]  Prob 8  0.0000 Class 8:
    [I]  Prob 9  0.0000 Class 9:
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

February 2020
This is the second release of the `README.md` file and sample.


# Known issues

There are no known issues in this sample.
