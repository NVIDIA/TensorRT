# Adding A Custom Layer To Your Network In TensorRT

**Table Of Contents**

- [Description](#description)
- [How does this sample work?](#how-does-this-sample-work)
    * [Defining the network](#defining-the-network)
    * [Enabling custom layers in NvCaffeParser](#enabling-custom-layers-in-nvcaffeparser)
    * [Building the engine](#building-the-engine)
    * [Serializing and deserializing](#serializing-and-deserializing)
    * [Resource management and execution](#resource-management-and-execution)
    * [TensorRT API layers and ops](#tensorrt-api-layers-and-ops)
- [Preparing sample data](#preparing-sample-data)
- [Running the sample](#running-the-sample)
    * [Sample `--help` options](#sample-help-options)
- [Additional resources](#additional-resources)
- [License](#license)
- [Changelog](#changelog)
- [Known issues](#known-issues)

## Description

This sample, samplePlugin, defines a custom layer that supports multiple data formats and demonstrates how to serialize/deserialize plugin layers. This sample also demonstrates how to use a fully connected plugin (`FCPlugin`) as a custom layer and the integration with NvCaffeParser.

## How does this sample work?

This sample implements the MNIST model (`data/samples/mnist/mnist.prototxt`) with the difference that the custom layer implements the Caffe InnerProduct layer using gemm routines (Matrix Multiplication) in cuBLAS and tensor addition in cuDNN (bias offset). Normally, the Caffe InnerProduct layer can be implemented in TensorRT using the IFullyConnected layer. However, in this sample, we use `FCPlugin` for this layer as an example of how to use plugins. The sample demonstrates plugin usage through the `IPluginExt` interface and uses the `nvcaffeparser1::IPluginFactoryExt` to add the plugin object to the network.

Specifically, this sample:
-  [Defines the network](#defining-the-network)
-  [Enables custom layers](#enabling-custom-layers-in-nvcaffeparser)
-  [Builds the engine](#building-the-engine)
-  [Serialize and deserialize](#serializing-and-deserializing)
-  [Manages resources and executes the engine](#resource-management-and-execution)

### Defining the network

The `FCPlugin` redefines the InnerProduct layer, which has a single output. Accordingly, `getNbOutputs` returns `1` and `getOutputDimensions` includes validation checks and returns the dimensions of the output:

```c++
Dims getOutputDimensions(int index, const Dims* inputDims,
                         int nbInputDims) override
{
    assert(index == 0 && nbInputDims == 1 &&
           inputDims[0].nbDims == 3);
    assert(mNbInputChannels == inputDims[0].d[0] *
                               inputDims[0].d[1] *
                               inputDims[0].d[2]);
    return DimsCHW(mNbOutputChannels, 1, 1);
}
```

### Enabling custom layers in NvCaffeParser

The model is imported using the Caffe parser (see [Importing A Caffe Model Using The C++ Parser API](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#import_caffe_c)  and [Using Custom Layers When Importing A Model From a Framework](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#using_custom_layer)). To use the `FCPlugin` implementation for the InnerProduct layer, a plugin factory is defined which recognizes the name of the InnerProduct layer (inner product `ip2`  in Caffe).

```c++
bool isPlugin(const char* name) override
{    return !strcmp(name, "ip2"); }
```

The factory can then instantiate `FCPlugin` objects as directed by the parser. The `createPlugin` method receives the layer name, and a set of weights extracted from the Caffe model file, which are then passed to the plugin constructor. Since the lifetime of the weights and that of the newly created plugin are decoupled, the plugin makes a copy of the weights in the constructor.

```c++
virtual nvinfer1::IPlugin* createPlugin(const char* layerName, const nvinfer1::Weights* weights, int nbWeights) override
{
    ...
    mPlugin =
      std::unique_ptr<FCPlugin>(new FCPlugin(weights,nbWeights));

    return mPlugin.get();
}
```

### Building the engine

`FCPlugin` does not need any scratch space, therefore, for building the engine, the most important methods deal with the formats supported and the configuration. `FCPlugin` supports two formats: NCHW in both single and half precision as defined in the `supportsFormat` method.

```c++
bool supportsFormat(DataType type, PluginFormat format) const override
{
    return (type == DataType::kFLOAT || type == DataType::kHALF) &&
           format == PluginFormat::kNCHW;
}
```

Supported configurations are selected in the building phase. The builder selects a configuration with the networks `configureWithFormat()` method, to give it a chance to select an algorithm based on its inputs. In this example, the inputs are checked to ensure they are in a supported format, and the selected format is recorded in a member variable. No other information needs to be stored in this simple case; in more complex cases, you may need to do so or even choose an ad-hoc algorithm for the given configuration.

```c++
void configureWithFormat(..., DataType type, PluginFormat format, ...) override
{
    assert((type == DataType::kFLOAT || type == DataType::kHALF) &&
            format == PluginFormat::kNCHW);
    mDataType = type;

}
```

The configuration takes place at build time, therefore, any information or state determined here that is required at runtime should be stored as a member variable of the plugin, and serialized and deserialized.

### Serializing and deserializing

Fully compliant plugins support serialization and deserialization, as described in [Serializing A Model In C++](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#serial_model_c). In the example, `FCPlugin` stores the number of channels and weights, the format selected, and the actual weights. The size of these variables makes up for the size of the serialized image; the size is returned by `getSerializationSize`:

```c++
virtual size_t getSerializationSize() override
{
    return sizeof(mNbInputChannels) + sizeof(mNbOutputChannels) +
           sizeof(mBiasWeights.count) + sizeof(mDataType) +
           (mKernelWeights.count + mBiasWeights.count) *
           type2size(mDataType);
}
```

Eventually, when the engine is serialized, these variables are serialized, the weights converted is needed, and written on a buffer:

```c++
virtual void serialize(void* buffer) override
{
    char* d = static_cast<char*>(buffer), *a = d;
    write(d, mNbInputChannels);
    ...
    convertAndCopyToBuffer(d, mKernelWeights);
    convertAndCopyToBuffer(d, mBiasWeights);
    assert(d == a + getSerializationSize());
}
```

Then, when the engine is deployed, it is deserialized. As the runtime scans the serialized image, when a plugin image is encountered, it create a new plugin instance via the factory. The plugin object created during deserialization (shown below using new) is destroyed when the engine is destroyed by calling `FCPlugin::destroy()`.

```c++
IPlugin* createPlugin(...) override
{
    ...
    return new FCPlugin(serialData, serialLength);
}
```

In the same order as in the serialization, the variables are read and their values restored. In addition, at this point the weights have been converted to selected format and can be stored directly on the device.

```c++
FCPlugin(const void* data, size_t length)
{
    const char* d = static_cast<const char*>(data), *a = d;
    read(d, mNbInputChannels);
    ...
    deserializeToDevice(d, mDeviceKernel,
                        mKernelWeights.count*type2size(mDataType));
    deserializeToDevice(d, mDeviceBias,
                        mBiasWeights.count*type2size(mDataType));
    assert(d == a + length);
}
```

### Resource management and execution

Before a custom layer is executed, the plugin is initialized. This is where resources are held for the lifetime of the plugin and can be acquired and initialized. In this example, weights are kept in CPU memory at first, so that during the build phase, for each configuration tested, weights can be converted to the desired format and then copied to the device in the initialization of the plugin. The method `initialize` creates the required cuBLAS and cuDNN handles, sets up tensor descriptors, allocates device memory, and copies the weights to device memory. Conversely, terminate destroys the handles and frees the memory allocated on the device.

```c++
int initialize() override
{
    CHECK(cudnnCreate(&mCudnn));
    CHECK(cublasCreate(&mCublas));
    ...
    if (mKernelWeights.values != nullptr)
        convertAndCopyToDevice(mDeviceKernel, mKernelWeights);
    ...
}
```

The core of the plugin is `enqueue`, which is used to execute the custom layer at runtime. The `call` parameters include the actual batch size, inputs, and outputs. The handles for cuBLAS and cuDNN operations are placed on the given stream; then, according to the data type and format configured, the plugin executes in single or half precision.

**Note:** The two handles are part of the plugin object, therefore, the same engine cannot be executed concurrently on multiple streams. In order to enable multiple streams of execution, plugins must be re-entrant and handle stream-specific data accordingly.

```c++
virtual int enqueue(int batchSize, const void*const * inputs, void** outputs, ...) override
{
    ...
    cublasSetStream(mCublas, stream);
    cudnnSetStream(mCudnn, stream);
    if (mDataType == DataType::kFLOAT)
    {...}
    else
    {
        CHECK(cublasHgemm(mCublas, CUBLAS_OP_T, CUBLAS_OP_N,
                          mNbOutputChannels, batchSize,
                          mNbInputChannels, &oneh,
                          mDeviceKernel), mNbInputChannels,
                          inputs[0], mNbInputChannels, &zeroh,
                          outputs[0], mNbOutputChannels));
    }
    if (mBiasWeights.count)
    {
        cudnnDataType_t cudnnDT = mDataType == DataType::kFLOAT ?
                                  CUDNN_DATA_FLOAT : CUDNN_DATA_HALF;
        ...
    }
    return 0;
}
```

The plugin object created in the sample is cloned by each of the network, builder, and engine by calling the `FCPlugin::clone()` method. The `clone()` method calls the plugin constructor and can also clone plugin parameters, if necessary.

```c++
IPluginExt* clone()
{
        return new FCPlugin(&mKernelWeights, mNbWeights, mNbOutputChannels);
}
```

The cloned plugin objects are deleted when the network, builder, or engine are destroyed. This is done by invoking the `FCPlugin::destroy()` method.
`void destroy() { delete this; }`


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

## Preparing sample data

1. Download the sample data from [TensorRT release tarball](https://developer.nvidia.com/nvidia-tensorrt-download#), if not already mounted under `/usr/src/tensorrt/data` (NVIDIA NGC containers) and set it to `$TRT_DATADIR`.
    ```bash
    export TRT_DATADIR=/usr/src/tensorrt/data
    pushd $TRT_DATADIR/mnist
    pip install Pillow
    python3 download_pgms.py
    popd
    ```

## Running the sample

1. Compile the sample by following build instructions in [TensorRT README](https://github.com/NVIDIA/TensorRT/).

2. Run the sample to perform inference on the digit:
    ```bash
    sample_plugin --datadir=<path/to/data>
    ```

    For example:
    ```bash
    sample_plugin --datadir $TRT_DATADIR/mnist
    ```

3.  Verify that the sample ran successfully. If the sample runs successfully you should see output similar to the following:
    ```
    &&&& RUNNING TensorRT.sample_plugin # ./build/x86_64-linux/sample_plugin
    [I] [TRT] Detected 1 input and 1 output network tensors.
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

    &&&& PASSED TensorRT.sample_plugin # ./build/x86_64-linux/sample_plugin
    ```

    This output shows that the sample ran successfully; `PASSED`.


### Sample `--help` options

To see the full list of available options and their descriptions, use the `-h` or `--help` command line option.


# Additional resources

The following resources provide a deeper understanding about samplePlugin:

**Models**
- [Training LeNet on MNIST with Caffe](http://caffe.berkeleyvision.org/gathered/examples/mnist.html)
- [lenet.prototxt](https://github.com/BVLC/caffe/blob/master/examples/mnist/lenet.prototxt)

**Documentation**
- [Introduction To NVIDIA’s TensorRT Samples](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sample-support-guide/index.html#samples)
- [Working With TensorRT Using The C++ API](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#c_topics)
- [NVIDIA’s TensorRT Documentation Library](https://docs.nvidia.com/deeplearning/sdk/tensorrt-archived/index.html)

# License

For terms and conditions for use, reproduction, and distribution, see the [TensorRT Software License Agreement](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html) documentation.


# Changelog

February 2019
This is the first release of this `README.md` file.


# Known issues

There are no known issues in this sample.
