# Adding A Custom Layer That Supports INT8 I/O To Your Network In TensorRT

**Table of Contents**
-  [Description](#description)
-  [How does this sample work?](#how-does-this-sample-work)
    * [Define layer outputs](#define-layer-outputs)
    * [Restrict supported I/O format and data type](#restrict-supported-io-format-and-data-type)
    * [Store information for layer execution](#store-information-for-layer-execution)
    * [Serialize and deserialize the engine](#serialize-and-deserialize-the-engine)
    * [Implement execution](#implement-execution)
    * [Manage resources](#manage-resources)
-  [Running the sample](#running-the-sample)
    * [Sample `--help` options](#sample---help-options)
-  [Additional resources](#additional-resources)
-  [License](#license)
-  [Changelog](#changelog)
-  [Known issues](#known-issues)

## Description

This sample, `sampleUffPluginV2Ext`, implements the custom pooling layer for the MNIST model
(`data/samples/lenet5_custom_pool.uff`). Since the cuDNN function `cudnnPoolingForward` with float precision is used to
simulate an INT8 kernel, the performance for INT8 precision does not speed up. Nevertheless, the main purpose of this
sample is to demonstrate how to extend INT8 I/O for a plugin that is introduced in TensorRT 6.0. This requires the
interface replacement from `IPlugin/IPluginV2/IPluginV2Ext` to `IPluginV2IOExt` (or `IPluginV2DynamicExt` if dynamic
shape is required).

## How does this sample work?

Specifically, this sample illustrates how to:
- [Define layer outputs](#define-layer-outputs)
- [Restrict supported I/O format and data type](#restrict-supported-io-format-and-data-type)
- [Store information for layer execution](#store-information-for-layer-execution)
- [Serialize and deserialize the engine](#serialize-and-deserialize-the-engine)
- [Implement execution](#implement-execution)
- [Manage resources](#manage-resources)

### Define layer outputs

`UffPoolPluginV2` implements the pooling layer which has a single output. Accordingly, the overridden
`IPluginV2IOExt::getNbOutputs` returns `1` and `IPluginV2IOExt::getOutputDimensions` includes validation checks and
returns the dimensions of the output.

```
    Dims UffPoolPluginV2::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
    {
        assert(index == 0 && nbInputDims == 1 && inputs[0].nbDims == 3);
        int height = (inputs[0].d[1] + mPoolingParams.pH * 2 - mPoolingParams.mR) / mPoolingParams.mU + 1;
        int width = (inputs[0].d[2] + mPoolingParams.pW * 2 - mPoolingParams.mS) / mPoolingParams.mV + 1;
        DimsHW outDims(height, width);
        return Dims3(inputs[0].d[0], outDims.h(), outDims.w());
    }
```

### Restrict supported I/O format and data type

The builder of TensorRT will ask for supported formats by the `IPluginV2IOExt::supportsFormatCombination` method to give
it a chance to select a reasonable algorithm based on its I/O tensor description indexed by `pos`. In this sample, the
supported I/O tensor format is linear CHW while Int32 is excluded, but the I/O tensor must have the same data type. For
a more complex case, refer to [IPluginV2IOExt::supportsFormatCombination()](https://docs.nvidia.com/deeplearning/sdk/tensorrt-api/c_api/classnvinfer1_1_1_i_plugin_v2_i_o_ext.html#a72f5170d7f1043d40e3c8b90b7b2f2f0)
in the API documentation for more details.

```
    bool UffPoolPluginV2::supportsFormatCombination(
        int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) const
    {
        ...
        bool condition = inOut[pos].format == TensorFormat::kLINEAR;
        condition &= inOut[pos].type != DataType::kINT32;
        condition &= inOut[pos].type == inOut[0].type;
        return condition;
    }
```

### Store information for layer execution

TensorRT will invoke `IPluginV2IOExt::configurePlugin` method to pass the information to the plugin through
`PluginTensorDesc`, which are stored as member variables, serialized and deserialized if they are required by the layer
execution.

```
    void UffPoolPluginV2::configurePlugin(
        const PluginTensorDesc* in, int nbInput, const PluginTensorDesc* out, int nbOutput)
    {
        ...
        mDataType = in[0].type;
        mInputDims = in[0].dims;
        mOutputDims = out[0].dims;
        mPoolingParams.mC = mInputDims.d[0];
        mPoolingParams.mH = mInputDims.d[1];
        mPoolingParams.mW = mInputDims.d[2];
        mPoolingParams.mP = mOutputDims.d[1];
        mPoolingParams.mQ = mOutputDims.d[2];
        mInHostScale = in[0].scale >= 0.0f ? in[0].scale : -1.0f;
        mOutHostScale = out[0].scale >= 0.0f ? out[0].scale : -1.0f;
    }
```

### Serialize and deserialize the engine

Fully compliant plugins support serialization and deserialization, as described in
[Serializing A Model In C++](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#serial_model_c).
In this sample, `UffPoolPluginV2` stores `PoolParameters`, I/O tensor dimensions, data types and optional INT8 scales.
The size of these variables is returned by `IPluginV2IOExt::getSerializationSize`.

```
    size_t UffPoolPluginV2::getSerializationSize() const
    {
        size_t serializationSize = 0;
        serializationSize += sizeof(mPoolingParams);
        serializationSize += sizeof(mInputDims.nbDims);
        serializationSize += sizeof(mInputDims.d[0]) * mInputDims.nbDims;
        serializationSize += sizeof(mOutputDims.nbDims);
        serializationSize += sizeof(mOutputDims.d[0]) * mOutputDims.nbDims;
        serializationSize += sizeof(static_cast<int>(mDataType));
        if (mDataType == DataType::kINT8)
        {
            serializationSize += sizeof(float) * 2;
        }
        return serializationSize;
    }
```

Eventually, when the engine is serialized, these variables are written to a buffer:

```
    void UffPoolPluginV2::serialize(void* buffer) const
    {
        char* d = static_cast<char*>(buffer);
        const char* const a = d;
        write(d, mPoolingParams);
        write(d, mInputDims.nbDims);
        ...
    }
```

Then, when the engine is deployed, it is deserialized by `UffPoolPluginV2Creator::deserializePlugin`.

```
    IPluginV2* UffPoolPluginV2Creator::deserializePlugin(
        const char* name, const void* serialData, size_t serialLength)
    {
        auto plugin = new UffPoolPluginV2(serialData, serialLength);
        mPluginName = name;
        return plugin;
    }
```

In the same order as in the serialization, the variables are read and their values are restored.

```
    UffPoolPluginV2::UffPoolPluginV2(const void* data, size_t length)
    {
        const char* const d = static_cast<const char*>(data);
        const char* const a = d;
        mPoolingParams = read<PoolParameters>(d);
        mInputDims.nbDims = read<int>(d);
        ...
    }
```

### Implement execution

TensorRT will invoke `IPluginV2::enqueue` which includes a collection of core algorithms of the plugin to execute the
custom layer at runtime. The execution uses the input parameters including the actual batch size, inputs, outputs, cuDNN
stream and the information configured.

```
    int UffPoolPluginV2::enqueue(
        int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream)
    {
        ...
        CHECK(cudnnPoolingForward(mCudnn, mPoolingDesc, &kONE, mSrcDescriptor, input, &kZERO, mDstDescriptor, output));
        ...
        return 0;
    }
```

### Manage resources

TensorRT will guanturee that `IPluginV2IOExt::initialize` and `IPluginV2IOExt::terminate` are invoked in pairs for
resource allocation and deallocation. In this sample, the overridden method `UffPoolPluginV2::initialize` creates the
required cuDNN handle and sets up tensor descriptors. Conversely, `UffPoolPluginV2::terminate` destroys the handle and
tensor descriptors.

```
    int UffPoolPluginV2::initialize()
    {
        CHECK(cudnnCreate(&mCudnn));
        CHECK(cudnnCreateTensorDescriptor(&mSrcDescriptor));
        CHECK(cudnnCreateTensorDescriptor(&mDstDescriptor));
        CHECK(cudnnCreatePoolingDescriptor(&mPoolingDesc));
        CHECK(cudnnSetPooling2dDescriptor(mPoolingDesc, mMode, CUDNN_NOT_PROPAGATE_NAN, mPoolingParams.mR,
            mPoolingParams.mS, mPoolingParams.pH, mPoolingParams.pW, mPoolingParams.mU, mPoolingParams.mV));
        return 0;
    }
```

```
    void UffPoolPluginV2::terminate()
    {
        CHECK(cudnnDestroyTensorDescriptor(mSrcDescriptor));
        CHECK(cudnnDestroyTensorDescriptor(mDstDescriptor));
        CHECK(cudnnDestroyPoolingDescriptor(mPoolingDesc));
        CHECK(cudnnDestroy(mCudnn));
    }
```

The plugin object created in the sample is cloned by each of the network, builder, and engine by calling the
`IPluginV2IOExt::clone` method which calls the plugin constructor and can also clone plugin parameters, if necessary.

```
    IPluginV2Ext* UffPoolPluginV2::clone() const
    {
        auto* plugin = new UffPoolPluginV2(*this);
        return plugin;
    }
```

The cloned plugin objects are deleted when the network, builder, and engine are destroyed. This is done by invoking the
`IPluginV2IOExt::destroy` method. The plugin object created by `UffPoolPluginV2Creator::createPlugin` is also destroyed
by calling this method when the engine is destroyed.

```
    void destroy() override
    {
        delete this;
    }
```

## Running the sample

1. Compile this sample by running `make` in the `<TensorRT root directory>/samples/sampleUffPluginV2Ext` directory. The
binary named `sample_uff_plugin_v2_ext` will be created in the `<TensorRT root directory>/bin` directory.

```
    cd <TensorRT root directory>/samples/sampleUffPluginV2Ext
    make
```
Where `<TensorRT root directory>` is where you installed TensorRT.

2. Run inference on the digit looping from 0 to 9:

```
    ./sample_uff_plugin_v2_ext
```

3. Verify that all the 10 digits match properly. If the sample runs successfully you should see output similar to the
following.

```
    &&&& RUNNING TensorRT.sample_uff_plugin_v2_ext # ./sample_uff_plugin_v2_ext
    [I] ../../../../../data/samples/mnist/lenet5_custom_pool.uff
    [I] [TRT] Detected 1 input and 1 output network tensors.
    [I] Input:
    ... (omitted messages)
    [I] Average over 10 runs is 0.10516 ms.
    &&&& PASSED TensorRT.sample_uff_plugin_v2_ext # ./sample_uff_plugin_v2_ext
```
This output shows that the sample ran successfully; `PASSED`.

### Sample `--help` options

To see the full list of available options and their descriptions, use the `-h` or `--help` command line option.

```
    ./sample_uff_plugin_v2_ext --help
    Usage: ./sample_uff_plugin_v2_ext [-h or --help] [-d or --datadir=<path to data directory>] [--useDLACore=<int>]
    --help Display help information
    --datadir Specify path to a data directory, overriding the default. This option can be used multiple times to add multiple
    directories. If no data directories are given, the default is to use (data/samples/mnist/, data/mnist/)
    --useDLACore=N Specify a DLA engine for layers that support DLA. Value can range from 0 to n-1, where n is the number of DLA
    engines on the platform.
    --int8 Run in Int8 mode.
    --fp16 Run in FP16 mode.
```

## Additional resources

The following resources provide a deeper understanding of sampleUffPluginV2Ext:

**Models**

-  [Training LeNet on MNIST with Caffe](http://caffe.berkeleyvision.org/gathered/examples/mnist.html)
-  [lenet.prototxt](https://github.com/BVLC/caffe/blob/master/examples/mnist/lenet.prototxt)

**Documentation**

-  [Introduction To NVIDIA’s TensorRT Samples](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sample-support-guide/index.html#samples)
-  [Working With TensorRT Using The C++ API](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#c_topics)
-  [NVIDIA’s TensorRT Documentation Library](https://docs.nvidia.com/deeplearning/sdk/tensorrt-archived/index.html)

## License

For terms and conditions for use, reproduction, and distribution, see the
[TensorRT Software License Agreement](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html) documentation.

## Changelog

June 2019
This is the initial open source release for this sample.

## Known issues

There are no known issues in this sample.
