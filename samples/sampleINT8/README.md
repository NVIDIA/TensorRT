# Performing Inference In INT8 Using Custom Calibration

**Table Of Contents**
- [Description](#description)
- [How does this sample work?](#how-does-this-sample-work)
    * [Defining the network](#defining-the-network)
    * [Setup the calibrator](#setup-the-calibrator)
        * [Calibration data](#calibration-data)
        * [Calibrator interface](#calibrator-interface)
        * [Calibration file](#calibration-file)
    * [Configuring the builder](#configuring-the-builder)
    * [Building the engine](#building-the-engine)
    * [Running the engine](#running-the-engine)
    * [Verifying the output](#verifying-the-output)
    * [TensorRT API layers and ops](#tensorrt-api-layers-and-ops)
- [Preparing sample data](#preparing-sample-data)
- [Running the sample](#running-the-sample)
    * [Sample `--help` options](#sample-help-options)
- [Additional resources](#additional-resources)
- [License](#license)
- [Changelog](#changelog)
- [Known issues](#known-issues)


## Description

This sample, sampleINT8, performs INT8 calibration and inference.

Specifically, this sample demonstrates how to perform inference in 8-bit integer (INT8). INT8 inference is available only on GPUs with compute capability 6.1 or newer. After the network is calibrated for execution in INT8, output of the calibration is cached to avoid repeating the process. You can then reproduce your own experiments with any deep learning framework in order to validate your results on ImageNet networks.

## How does this sample work?

INT8 engines are build from 32-bit network definitions, similarly to 32-bit and 16-bit engines, but with more configuration steps. In particular, the builder and network must be configured to use INT8, which requires per-tensor dynamic ranges. The INT8 calibrator can determine how best to represent weights and activations as 8-bit integers and sets the per-tensor dynamic ranges accordingly. Alternatively, you can set custom per-tensor dynamic ranges; this is covered in sampleINT8API.

This sample requires the [MNIST training set](https://github.com/BVLC/caffe/blob/master/data/mnist/get_mnist.sh). Download this script into the `TensorRT-x.x.x.x/data/mnist` directory, where `x.x.x.x` is your installed version of TensorRT, and execute the script to download the required data. The packaged MNIST model that is shipped with this sample is based on [lenet.prototxt](https://github.com/BVLC/caffe/edit/master/examples/mnist/lenet.prototxt). For more information, see the [MNIST BVLC Caffe example](https://github.com/BVLC/caffe/tree/master/examples/mnist).

The data set used by this sample is based on the [MNIST data set](https://github.com/BVLC/caffe/tree/master/data/mnist). The batch file generation from the above data set is described in [Batch files for calibration](#batch-files-for-calibration).

Specifically, this sample performs the following steps:

- [Defines the network](#defining-the-network)
- [Sets up the calibrator](#setup-the-calibrator)
- [Configures the builder](#configuring-the-builder)
- [Builds the engine](#building-the-engine)
- [Runs the engine](#running-the-engine)
- [Verifies the output](#verifying-the-output)

### Defining the network

Defining a network for INT8 execution is exactly the same as for any other precision. Weights should be imported as FP32 values, and the builder will calibrate the network to find appropriate quantization factors to reduce the network to INT8 precision. This sample imports the network using the NvCaffeParser:

```cpp
const nvcaffeparser1::IBlobNameToTensor* blobNameToTensor =
parser->parse(locateFile(mParams.prototxtFileName, mParams.dataDirs).c_str(),
locateFile(mParams.weightsFileName, mParams.dataDirs).c_str(),
*network,
dataType == DataType::kINT8 ? DataType::kFLOAT : dataType);
```

### Setup the calibrator

Calibration is an additional step required when building networks for INT8. The application must provide TensorRT with sample input, in other words, calibration data. TensorRT will then perform inference in FP32 and gather statistics about intermediate activation layers that it will use to build the reduced precision INT8 engine.

#### Calibration data

Calibration must be performed using images representative of those which will be used at runtime. Since the sample is based around Caffe, any image preprocessing that caffe would perform prior to running the network (such as scaling, cropping, or mean subtraction) will be done in Caffe and captured as a set of files. The sample uses a utility class (MNISTBatchStream) to read these files and create appropriate input for calibration. Generation of these files is discussed in [Batch files for calibration](#batch-files-for-calibration).

You can create calibration data stream (calibrationStream), for example:
```cpp
MNISTBatchStream calibrationStream(mParams.calBatchSize, mParams.nbCalBatches, "train-images-idx3-ubyte","train-labels-idx1-ubyte", mParams.dataDirs);
```

The MNISTBatchStream class provides helper methods used to retrieve batch data. Batch stream object is used by the calibrator in order to retrieve batch data while calibrating. In general, the BatchStream class should provide implementation for `getBatch()` and `getBatchSize()` which can be invoked by `IInt8Calibrator::getBatch()` and `IInt8Calibrator::getBatchSize()`. Ideally, you can write your own custom BatchStream class to serve calibration data. For more information, see `BatchStream.h`.

**Note:** The calibration data must be representative of the input provided to TensorRT at runtime; for example, for image classification networks, it should not consist of images from just a small subset of categories. For ImageNet networks, around 500 calibration images is adequate.

#### Calibrator interface

The application must implement the `IInt8Calibrator` interface to provide calibration data and helper methods for reading/writing the calibration table file.

We can create calibrator object (`calibrator`), for example:
`std::unique_ptr<IInt8Calibrator> calibrator;`

TensorRT provides 4 implementations for `IInt8Calibrator`:
1.  IInt8EntropyCalibrator
2.  IInt8EntropyCalibrator2
3.  IInt8MinMaxCalibrator
4.  IInt8LegacyCalibrator

See `NvInfer.h` for more information on the `IInt8Calibrator` interface variants.

This sample uses `IInt8EntropyCalibrator2` by default. We can set the calibrator interface to use `IInt8EntropyCalibrator2` as shown:

```cpp
calibrator.reset(new Int8EntropyCalibrator2<MNISTBatchStream>(
calibrationStream, 0, mParams.networkName.c_str(), mParams.inputTensorNames[0].c_str()));
```

where `calibrationStream` is a MNISTBatchStream object. The calibrator object should be configured to use the calibration batch stream.

In order to perform calibration, the interface must provide implementation for `getBatchSize()` and `getBatch()` to retrieve data from the BatchStream object.

The builder calls the `getBatchSize()` method once, at the start of calibration, to obtain the batch size for the calibration set. The method `getBatch()` is then called repeatedly to obtain batches from the application, until the method returns false. Every calibration batch must include exactly the number of images specified as the batch size.

```cpp
bool getBatch(void* bindings[], const char* names[], int nbBindings) override
{
    return mImpl.getBatch(bindings, names, nbBindings);
}
```

For each input tensor, a pointer to input data in GPU memory must be written into the bindings array. The names array contains the names of the input tensors. The position for each tensor in the bindings array matches the position of its name in the names array. Both arrays have size `nbBindings`.

Since the calibration step is time consuming, you can choose to provide the implementation for `writeCalibrationCache()` to write calibration table to the appropriate location to be used for later runs. Then, implement `readCalibrationCache()` method to read calibration table file from desired location.

During calibration, the builder will check if the calibration file exists using `readCalibrationCache()`.  The builder will re-calibrate only if either calibration file does not exist or is incompatible with the current TensorRT version or calibrator variant it was generated with.

For more information on implementing `IInt8Calibrator` interface, see `EntropyCalibrator.h`.

#### Calibration file

A calibration file stores activation scales for each network tensor. Activations scales are calculated using a dynamic range generated from a calibration algorithm, in other words, `abs(max_dynamic_range) / 127.0f`.

The calibration file is called `CalibrationTable<NetworkName>`, where `<NetworkName>` is the name of your network, for example `mnist`. The file is located in the `TensorRT-x.x.x.x/data/mnist` directory, where `x.x.x.x` is your installed version of TensorRT.

If the `CalibrationTable` file is not found, the builder will run the calibration algorithm again to create it. The `CalibrationTable` contents include:

```
TRT-7000-EntropyCalibration2
data: 3c008912
conv1: 3c88edfc
pool1: 3c88edfc
conv2: 3ddc858b
pool2: 3ddc858b
ip1: 3db6bd6e
ip2: 3e691968
prob: 3c010a14
```

Where:
-   `<TRT-xxxx>-<xxxxxxx>` The TensorRT version followed by the calibration algorithm, for example, EntropyCalibration2.
-   `<layer name> :` value corresponds to the floating point activation scales determined during calibration for each tensor in the network.

The `CalibrationTable` file is generated during the build phase while running the calibration algorithm. After the calibration file is created, it can be read for subsequent runs without running the calibration again. You can provide implementation for `readCalibrationCache()` to load calibration file from a desired location. If the read calibration file is compatible with calibrator type (which was used to generate the file) and TensorRT version, builder would skip the calibration step and use per-tensor scales values from the calibration file instead.

### Configuring for the builder

1.  Set minimum and average number of timing iterations.
    `config->setAvgTimingIterations(1);`
    `config->setMinTimingIterations(1);`

2.  Set maximum workspace size.
    `config->setMaxWorkspaceSize(1_GiB);`

3.  Set allowed builder precision to INT8 in addition to FP32. Default builder precision is FP32.
    `config->setFlag(BuilderFlag::kINT8);`

4.  Set maximum batch size.
    `builder->setMaxBatchSize(mParams.batchSize);`

5.  Pass the calibrator object (calibrator) to the builder.
    `config->setInt8Calibrator(calibrator.get());`

### Building the engine

After we configure the builder, we can build the engine similar to any FP32 engine.

```cpp
SampleUniquePtr<IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
SampleUniquePtr<IRuntime> runtime{createInferRuntime(sample::gLogger.getTRTLogger())};

mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
    runtime->deserializeCudaEngine(plan->data(), plan->size()), samplesCommon::InferDeleter());
````

### Running the engine

After the engine has been built, it can be used just like an FP32 engine. For example, inputs and outputs remain in 32-bit floating point.

1.  Allocate memory for input and output buffers.
    ```cpp
    samplesCommon::BufferManager buffers(mEngine, mParams.batchSize);
    ```

2.  Create execution context.
    ```cpp
    auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    ```

3.  Get data dimensions.
    ```cpp
    Dims outputDims = context->getEngine().getBindingDimensions(
    context->getEngine().getBindingIndex(mParams.outputTensorNames[0].c_str()));
    ```

4.  Read the input data into the managed buffers.
    ```cpp
    processInput(buffers, batchStream.getBatch())
    ```

5.  Copy data from host input buffers to device input buffers
    ```cpp
    buffers.copyInputToDevice();
    ```

6.  Run Inference.
    ```cpp
    context->enqueue(mParams.batchSize, buffers.getDeviceBindings().data(), stream, nullptr);
    ```

7.  Copy the CUDA buffer output to CPU output buffers for post processing.
    ```cpp
    buffers.copyOutputToHost();
    ```

### Verifying the output

This sample outputs Top-1 and Top-5 metrics for both FP32 and INT8 precision, as well as for FP16 if it is natively supported by the hardware. These numbers should be within 1%.

### TensorRT API layers and ops

In this sample, the following layers are used. For more information about these layers, see the [TensorRT Developer Guide: Layers](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#layers) documentation.

[Activation layer](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#activation-layer)
The Activation layer implements element-wise activation functions.

[Convolution layer](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#convolution-layer)
The Convolution layer computes a 2D (channel, height, and width) convolution, with or without bias.

[FullyConnected layer](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#fullyconnected-layer)
The FullyConnected layer implements a matrix-vector product, with or without bias.

[SoftMax layer](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#softmax-layer)
The SoftMax layer applies the SoftMax function on the input tensor along an input dimension specified by the user.

## Preparing sample data

1. Download the sample data from [TensorRT release tarball](https://developer.nvidia.com/nvidia-tensorrt-download#), if not already mounted under `/usr/src/tensorrt/data` (NVIDIA NGC containers) and set it to `$TRT_DATADIR`.
    ```bash
    export TRT_DATADIR=/usr/src/tensorrt/data
    ```

2. Download the [MNIST dataset](http://yann.lecun.com/exdb/mnist/)
    - This sample requires the [training set](http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz) and [training labels](http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz)
    ```bash
    pushd $TRT_DATADIR/mnist
    wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
    wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
    gunzip train-images-idx3-ubyte.gz
    gunzip train-labels-idx1-ubyte.gz
    popd
    ```

## Running the sample

1. Compile the sample by following build instructions in [TensorRT README](https://github.com/NVIDIA/TensorRT/).

2. Run the sample to generate characters based on the trained model:
    ```bash
    ./sample_int8 --datadir=<path/to/data> --useDLACore=N batch=N start=N score=N
    ```

    For example:
    ```bash
    ./sample_int8 --datadir $TRT_DATADIR/mnist
    ```

3.  Verify that the sample ran successfully. If the sample runs successfully you should see output similar to the following:

    ```
    &&&& RUNNING TensorRT.sample_int8 # ./sample_int8
    [I] Building and running a GPU inference engine for INT8 sample
    [I] FP32 run:1800 batches of size 32 starting at 16
    [I] [TRT] Detected 1 inputs and 1 output network tensors.
    [I] Processing next set of max 100 batches
    ...
    [I] Processing next set of max 100 batches
    [I] Top1: 0.998646, Top5: 1
    [I] Processing 57600 images averaged 0.00262977 ms/image and 0.0834112 ms/batch.
    [I] FP16 run:1800 batches of size 32 starting at 16
    [I] [TRT] Detected 1 inputs and 1 output network tensors.
    [I] Processing next set of max 100 batches
    ...
    [I] Processing next set of max 100 batches
    [I] Top1: 0.998646, Top5: 1
    [I] Processing 57600 images averaged 0.00298152 ms/image and 0.0945682 ms/batch.
    [I] INT8 run:1800 batches of size 32 starting at 16
    [I] [TRT] Detected 1 inputs and 1 output network tensors.
    [I] [TRT] Starting Calibration with batch size 50.
    [I] [TRT]   Calibrated batch 0 in 0.00211989 seconds.
    ...
    [I] [TRT]   Calibrated batch 9 in 0.00207574 seconds.
    [I] [TRT]   Post Processing Calibration data in 0.180447 seconds.
    [I] [TRT] Calibration completed in 0.317902 seconds.
    [I] [TRT] Writing Calibration Cache for calibrator: TRT-7000-EntropyCalibration2
    [I] [TRT] Detected 1 inputs and 1 output network tensors.
    [I] Processing next set of max 100 batches
    ...
    [I] Processing next set of max 100 batches
    [I] Top1: 0.998576, Top5: 1
    [I] Processing 57600 images averaged 0.00227856 ms/image and 0.0722715 ms/batch.
    &&&& PASSED TensorRT.sample_int8 # ./sample_int8
    ```

	This output shows that the sample ran successfully; `PASSED`.


### Sample `--help` options

To see the full list of available options and their descriptions, use the `-h` or `--help` command line option.


# Additional resources

The following resources provide a deeper understanding how to perform inference in INT8 using custom calibration:

**INT8:**
- [8-bit Inference with TensorRT](http://on-demand.gputechconf.com/gtc/2017/presentation/s7310-8-bit-inference-with-tensorrt.pdf)
- [INT8 Calibration Using C++](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#optimizing_int8_c)

**Models:**
- [MNIST lenet.prototxt](https://github.com/BVLC/caffe/edit/master/examples/mnist/lenet.prototxt)

**Blogs:**
- [Fast INT8 Inference for Autonomous Vehicles with TensorRT 3](https://devblogs.nvidia.com/int8-inference-autonomous-vehicles-tensorrt/)
- [Low Precision Inference with TensorRT](https://towardsdatascience.com/low-precision-inference-with-tensorrt-6eb3cda0730b)
- [8-Bit Quantization and TensorFlow Lite: Speeding up Mobile Inference with Low Precision](https://heartbeat.fritz.ai/8-bit-quantization-and-tensorflow-lite-speeding-up-mobile-inference-with-low-precision-a882dfcafbbd)

**Videos:**
- [Inference and Quantization](https://www.youtube.com/watch?v=VsGX9kFXjbs)
- [8-bit Inference with TensorRT Webinar](http://on-demand.gputechconf.com/gtcdc/2017/video/DC7172)

**Documentation:**
- [Introduction to NVIDIA’s TensorRT Samples](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sample-support-guide/index.html#samples)
- [Working with TensorRT Using the C++ API](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#c_topics)
- [NVIDIA’s TensorRT Documentation Library](https://docs.nvidia.com/deeplearning/sdk/tensorrt-archived/index.html)

# License

For terms and conditions for use, reproduction, and distribution, see the [TensorRT Software License Agreement](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html) documentation.


# Changelog

December 2019
This is the second release of this `README.md` file.


# Known issues

There are no known issues in this sample.
