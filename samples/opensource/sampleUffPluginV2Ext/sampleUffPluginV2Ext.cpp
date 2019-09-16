/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "NvInfer.h"
#include "NvUffParser.h"
#include <cassert>
#include <chrono>
#include <cudnn.h>
#include <iostream>
#include <map>
#include <string.h>
#include <unordered_map>
#include <vector>

#include "NvUtils.h"
#include "argsParser.h"
#include "common.h"
#include "half.h"
#include "logger.h"

using namespace nvuffparser;
using namespace nvinfer1;
using namespace samplesCommon;

const std::string gSampleName = "TensorRT.sample_uff_plugin_v2_ext";
samplesCommon::Args gArgs;

template <DataType in, DataType out>
void transform(const void* src, void* dst, int count)
{
    assert(in == out);
    memcpy(dst, src, count * elementSize(in));
}

template <>
void transform<DataType::kHALF, DataType::kFLOAT>(const void* src, void* dst, int count)
{
    auto srcPtr = static_cast<const half_float::half*>(src);
    auto dstPtr = static_cast<float*>(dst);
    std::transform(srcPtr, srcPtr + count, dstPtr, [](half_float::half in) { return static_cast<float>(in); });
}

template <>
void transform<DataType::kINT8, DataType::kFLOAT>(const void* src, void* dst, int count)
{
    auto srcPtr = static_cast<const int8_t*>(src);
    auto dstPtr = static_cast<float*>(dst);
    std::transform(srcPtr, srcPtr + count, dstPtr, [](int8_t in) { return static_cast<float>(in); });
}

template <>
void transform<DataType::kFLOAT, DataType::kHALF>(const void* src, void* dst, int count)
{
    auto srcPtr = static_cast<const float*>(src);
    auto dstPtr = static_cast<half_float::half*>(dst);
    std::transform(srcPtr, srcPtr + count, dstPtr, [](float in) { return static_cast<half_float::half>(in); });
}

template <>
void transform<DataType::kFLOAT, DataType::kINT8>(const void* src, void* dst, int count)
{
    auto srcPtr = static_cast<const float*>(src);
    auto dstPtr = static_cast<int8_t*>(dst);
    std::transform(srcPtr, srcPtr + count, dstPtr, [](float x) {
        x = std::max(x, float(INT8_MIN));
        x = std::min(x, float(INT8_MAX));
        return static_cast<int8_t>(x);
    });
}

static const int INPUT_H = 28;
static const int INPUT_W = 28;

// simple PGM (portable greyscale map) reader
void readPGMFile(const std::string& filename, uint8_t buffer[INPUT_H * INPUT_W])
{
    readPGMFile(locateFile(filename, gArgs.dataDirs), buffer, INPUT_H, INPUT_W);
}

std::vector<std::pair<size_t, DataType>> calculateBindingBufferSizes(
    const ICudaEngine& engine, int nbBindings, int batchSize)
{
    std::vector<std::pair<size_t, DataType>> sizes;
    for (int i = 0; i < nbBindings; ++i)
    {
        Dims dims = engine.getBindingDimensions(i);
        DataType dtype = engine.getBindingDataType(i);

        size_t eltCount = volume(dims) * batchSize;
        sizes.push_back(std::make_pair(eltCount, dtype));
    }
    return sizes;
}

void* createMnistCudaBuffer(int64_t eltCount, DataType dtype, int num)
{
    // in that specific case, eltCount == INPUT_H * INPUT_W
    assert(eltCount == INPUT_H * INPUT_W);
    assert(elementSize(dtype) == sizeof(float));

    size_t memSize = eltCount * elementSize(dtype);
    std::vector<float> inputs(eltCount);

    // read PGM file
    uint8_t fileData[INPUT_H * INPUT_W];
    readPGMFile(std::to_string(num) + ".pgm", fileData);

    // display the number in an ascii representation
    gLogInfo << "Input:\n";
    for (int i = 0; i < eltCount; i++)
    {
        gLogInfo << (" .:-=+*#%@"[fileData[i] / 26]) << (((i + 1) % INPUT_W) ? "" : "\n");
    }
    gLogInfo << std::endl;

    // initialize the inputs buffer
    for (int i = 0; i < eltCount; i++)
    {
        inputs[i] = 1.0 - float(fileData[i]) / 255.0;
    }

    void* deviceMem = safeCudaMalloc(memSize);
    CHECK(cudaMemcpy(deviceMem, inputs.data(), memSize, cudaMemcpyHostToDevice));

    return deviceMem;
}

bool verifyOutput(int64_t eltCount, DataType dtype, void* buffer, int num)
{
    assert(elementSize(dtype) == sizeof(float));

    bool pass = false;

    size_t memSize = eltCount * elementSize(dtype);
    std::vector<float> outputs(eltCount);
    CHECK(cudaMemcpy(outputs.data(), buffer, memSize, cudaMemcpyDeviceToHost));

    int maxIdx = std::distance(outputs.begin(), std::max_element(outputs.begin(), outputs.end()));

    std::ios::fmtflags prevSettings = gLogInfo.flags();
    gLogInfo.setf(std::ios::fixed, std::ios::floatfield);
    gLogInfo.precision(6);
    gLogInfo << "Output:\n";
    for (int64_t eltIdx = 0; eltIdx < eltCount; ++eltIdx)
    {
        gLogInfo << eltIdx << " => " << std::setw(10) << outputs[eltIdx] << "\t : ";
        if (eltIdx == maxIdx)
        {
            gLogInfo << "***";
            pass = eltIdx == num ? true : false;
        }
        gLogInfo << "\n";
    }
    gLogInfo.flags(prevSettings);
    gLogInfo << std::endl;
    return pass;
}

struct PoolParameters
{
    // Input dimensions
    int mC, mH, mW;
    // Output dimensions
    int mP, mQ;
    // Kernel size
    int mR, mS;
    // Stride
    int mU, mV;
    // Padding
    int pH, pW;
    // Pooling Function
    PoolingType pType;
};

class SampleUffPluginV2Ext
{
public:
    template <typename T>
    using SampleUniquePtr = std::unique_ptr<T, samplesCommon::InferDeleter>;

    explicit SampleUffPluginV2Ext(const UffSampleParams& params)
        : mParams(params)
    {
    }

    //!
    //! \brief Creates the network, configures the builder and creates the network engine
    //!
    bool build()
    {
        SampleUniquePtr<IUffParser> parser{createUffParser()};
        parser->registerInput("in", Dims3(1, 28, 28), UffInputOrder::kNCHW);
        parser->registerOutput("out");

        SampleUniquePtr<IBuilder> builder{createInferBuilder(gLogger.getTRTLogger())};
        if (!builder.get())
        {
            gLogError << "Failed to create infer builder. " << std::endl;
            return false;
        }

        SampleUniquePtr<INetworkDefinition> network{builder->createNetwork()};
        if (!network.get())
        {
            gLogError << "Failed to create network. " << std::endl;
            return false;
        }

        if (!parser->parse(mParams.uffFileName.data(), *network, nvinfer1::DataType::kFLOAT))
        {
            gLogError << "Failure while parsing UFF file" << std::endl;
            return false;
        }

        if (gArgs.runInInt8)
        {
            samplesCommon::setAllTensorScales(network.get(), 5.0f, 5.0f);
        }

        SampleUniquePtr<IBuilderConfig> networkConfig{builder->createBuilderConfig()};
        networkConfig->setMaxWorkspaceSize(1_GiB);
        if (gArgs.runInFp16)
        {
            networkConfig->setFlag(BuilderFlag::kFP16);
        }
        if (gArgs.runInInt8)
        {
            networkConfig->setFlag(BuilderFlag::kINT8);
        }
        networkConfig->setFlag(BuilderFlag::kSTRICT_TYPES);
        if (gArgs.useDLACore >= 0)
        {
            networkConfig->setDLACore(gArgs.useDLACore);
        }

        const int maxBatchSize = 1;
        builder->setMaxBatchSize(maxBatchSize);
        samplesCommon::enableDLA(builder.get(), networkConfig.get(), gArgs.useDLACore);

        mEngine.reset(builder->buildEngineWithConfig(*network, *networkConfig));
        if (!mEngine.get())
        {
            gLogError << "Unable to create engine. " << std::endl;
            return false;
        }
        return true;
    }

    //!
    //! \brief Runs the TensorRT inference engine for this sample
    //!
    bool infer()
    {
        bool pass{true};
        SampleUniquePtr<IExecutionContext> context{mEngine->createExecutionContext()};

        const int batchSize{1};
        const int nbBindings = mEngine->getNbBindings();
        assert(nbBindings == 2);

        std::vector<void*> buffers(nbBindings);
        auto buffersSizes = calculateBindingBufferSizes(*mEngine, nbBindings, batchSize);

        const int bindingIdxInput = mEngine->bindingIsInput(0) ? 0 : 1;
        const int bindingIdxOutput = mEngine->bindingIsInput(0) ? 1 : 0;
        auto bufferSizesOutput = buffersSizes[bindingIdxOutput];
        buffers[bindingIdxOutput] = safeCudaMalloc(bufferSizesOutput.first * elementSize(bufferSizesOutput.second));

        auto bufferSizesInput = buffersSizes[bindingIdxInput];

        const int iterations{1};
        const int numberRun{10};
        for (int i = 0; i < iterations; i++)
        {
            float total{0.0f}, ms{0.0f};
            for (int num = 0; num < numberRun; num++)
            {
                buffers[bindingIdxInput] = createMnistCudaBuffer(bufferSizesInput.first, bufferSizesInput.second, num);
                auto t_start = std::chrono::high_resolution_clock::now();
                context->execute(batchSize, &buffers[0]);
                auto t_end = std::chrono::high_resolution_clock::now();
                ms = std::chrono::duration<float, std::milli>(t_end - t_start).count();
                total += ms;

                for (int bindingIdx = 0; bindingIdx < nbBindings; ++bindingIdx)
                {
                    if (mEngine->bindingIsInput(bindingIdx))
                    {
                        continue;
                    }
                    auto bufferSizesOutput = buffersSizes[bindingIdx];
                    pass &= verifyOutput(bufferSizesOutput.first, bufferSizesOutput.second, buffers[bindingIdx], num);
                }
                CHECK(cudaFree(buffers[bindingIdxInput]));
            }
            total /= numberRun;
            gLogInfo << "Average over " << numberRun << " runs is " << total << " ms." << std::endl;
        }

        for (int bindingIdx = 0; bindingIdx < nbBindings; ++bindingIdx)
        {
            if (!mEngine->bindingIsInput(bindingIdx))
            {
                CHECK(cudaFree(buffers[bindingIdx]));
            }
        }
        return pass;
    }

    //!
    //! \brief Used to clean up any state created in the sample class
    //!
    bool teardown()
    {
        nvuffparser::shutdownProtobufLibrary();
        return true;
    }

private:
    SampleUniquePtr<nvinfer1::ICudaEngine> mEngine;
    samplesCommon::UffSampleParams mParams;
};

class UffPoolPluginV2 : public IPluginV2IOExt
{
public:
    UffPoolPluginV2(const PluginFieldCollection& fc)
    {
        // To do: TRT-TRT-8010 Populate Parameters from fc object w/ hard code
        mPoolingParams.pType = PoolingType::kMAX;
        mPoolingParams.mU = 2;
        mPoolingParams.mV = 2;
        mPoolingParams.mR = 2;
        mPoolingParams.mS = 2;
        mPoolingParams.pH = 0;
        mPoolingParams.pW = 0;
        mMode = CUDNN_POOLING_MAX;
        (void) fc;
    }

    UffPoolPluginV2(const void* data, size_t length)
    {
        const char* d = static_cast<const char*>(data);
        const char* const a = d;
        mPoolingParams = read<PoolParameters>(d);
        mInputDims.nbDims = read<int>(d);
        for (int i = 0; i < mInputDims.nbDims; ++i)
        {
            mInputDims.d[i] = read<int>(d);
        }
        mOutputDims.nbDims = read<int>(d);
        for (int i = 0; i < mOutputDims.nbDims; ++i)
        {
            mOutputDims.d[i] = read<int>(d);
        }
        mDataType = static_cast<DataType>(read<int>(d));
        mMode = mPoolingParams.pType == PoolingType::kMAX ? CUDNN_POOLING_MAX
                                                          : CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
        if (mDataType == DataType::kINT8)
        {
            mInHostScale = read<float>(d);
            mOutHostScale = read<float>(d);
        }
        assert(d == a + length);
    }

    // It makes no sense to construct UffPoolPluginV2 without arguments.
    UffPoolPluginV2() = delete;

    virtual ~UffPoolPluginV2() {}

public:
    int getNbOutputs() const override
    {
        return 1;
    }

    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override
    {
        assert(index == 0 && nbInputDims == 1 && inputs[0].nbDims == 3);
        int height = (inputs[0].d[1] + mPoolingParams.pH * 2 - mPoolingParams.mR) / mPoolingParams.mU + 1;
        int width = (inputs[0].d[2] + mPoolingParams.pW * 2 - mPoolingParams.mS) / mPoolingParams.mV + 1;
        DimsHW outDims(height, width);
        return Dims3(inputs[0].d[0], outDims.h(), outDims.w());
    }

    int initialize() override
    {
        CHECK(cudnnCreate(&mCudnn));
        CHECK(cudnnCreateTensorDescriptor(&mSrcDescriptor));
        CHECK(cudnnCreateTensorDescriptor(&mDstDescriptor));
        CHECK(cudnnCreatePoolingDescriptor(&mPoolingDesc));
        CHECK(cudnnSetPooling2dDescriptor(mPoolingDesc, mMode, CUDNN_NOT_PROPAGATE_NAN, mPoolingParams.mR,
            mPoolingParams.mS, mPoolingParams.pH, mPoolingParams.pW, mPoolingParams.mU, mPoolingParams.mV));
        return 0;
    }

    void terminate() override
    {
        CHECK(cudnnDestroyTensorDescriptor(mSrcDescriptor));
        CHECK(cudnnDestroyTensorDescriptor(mDstDescriptor));
        CHECK(cudnnDestroyPoolingDescriptor(mPoolingDesc));
        CHECK(cudnnDestroy(mCudnn));
    }

    size_t getWorkspaceSize(int maxBatchSize) const override
    {
        return 0;
    }

    int enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream) override
    {
        const float kONE = 1.0f, kZERO = 0.0f;
        cudnnSetStream(mCudnn, stream);

        const int N = 1;
        // Use float to simulate int8 calculation
        std::map<DataType, cudnnDataType_t> typeMap = {{DataType::kFLOAT, CUDNN_DATA_FLOAT},
            {DataType::kHALF, CUDNN_DATA_HALF}, {DataType::kINT8, CUDNN_DATA_FLOAT}};
        assert(mDataType != DataType::kINT32);
        CHECK(cudnnSetTensor4dDescriptor(mSrcDescriptor, CUDNN_TENSOR_NCHW, typeMap[mDataType], N, mPoolingParams.mC,
            mPoolingParams.mH, mPoolingParams.mW));
        CHECK(cudnnSetTensor4dDescriptor(mDstDescriptor, CUDNN_TENSOR_NCHW, typeMap[mDataType], N, mPoolingParams.mC,
            mPoolingParams.mP, mPoolingParams.mQ));
        void* input{nullptr};
        void* output{nullptr};
        if (mDataType == DataType::kINT8)
        {
            copyDeviceInputToFP32(inputs[0], input);
            size_t outCount = getC(mOutputDims) * getH(mOutputDims) * getW(mOutputDims);
            CHECK(cudaMalloc(&output, outCount * elementSize(DataType::kFLOAT)));
        }
        else
        {
            input = const_cast<void*>(inputs[0]);
            output = const_cast<void*>(outputs[0]);
        }
        CHECK(cudnnPoolingForward(mCudnn, mPoolingDesc, &kONE, mSrcDescriptor, input, &kZERO, mDstDescriptor, output));
        if (mDataType == DataType::kINT8)
        {
            copyDeviceToInt8Output(output, outputs[0]);
        }
        return 0;
    }

    size_t getSerializationSize() const override
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

    void serialize(void* buffer) const override
    {
        char* d = static_cast<char*>(buffer);
        const char* const a = d;
        write(d, mPoolingParams);
        write(d, mInputDims.nbDims);
        assert(mInputDims.nbDims <= mInputDims.MAX_DIMS);
        for (int i = 0; i < mInputDims.nbDims; ++i)
        {
            write(d, mInputDims.d[i]);
        }
        write(d, mOutputDims.nbDims);
        assert(mOutputDims.nbDims <= mOutputDims.MAX_DIMS);
        for (int i = 0; i < mOutputDims.nbDims; ++i)
        {
            write(d, mOutputDims.d[i]);
        }
        write(d, static_cast<int>(mDataType));
        if (mDataType == DataType::kINT8)
        {
            write(d, mInHostScale);
            write(d, mOutHostScale);
        }
        assert(d == a + getSerializationSize());
    }

    void configurePlugin(const PluginTensorDesc* in, int nbInput, const PluginTensorDesc* out, int nbOutput) override
    {
        assert(in && nbInput == 1);
        assert(out && nbOutput == 1);
        assert(in[0].type == out[0].type);
        assert(in[0].format == TensorFormat::kLINEAR && out[0].format == TensorFormat::kLINEAR);

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

    //! The combination of kLINEAR + kINT8/kHALF/kFLOAT is supported.
    bool supportsFormatCombination(int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) const override
    {
        assert(nbInputs == 1 && nbOutputs == 1 && pos < nbInputs + nbOutputs);
        bool condition = inOut[pos].format == TensorFormat::kLINEAR;
        condition &= inOut[pos].type != DataType::kINT32;
        condition &= inOut[pos].type == inOut[0].type;
        return condition;
    }

    DataType getOutputDataType(int index, const DataType* inputTypes, int nbInputs) const override
    {
        assert(inputTypes && nbInputs == 1);
        (void) index;
        return inputTypes[0];
    }

    const char* getPluginType() const override
    {
        return "MaxPool";
    }

    const char* getPluginVersion() const override
    {
        return "2";
    }

    void destroy() override
    {
        delete this;
    }

    IPluginV2Ext* clone() const override
    {
        auto* plugin = new UffPoolPluginV2(*this);
        return plugin;
    }

    void setPluginNamespace(const char* libNamespace) override
    {
        mNamespace = libNamespace;
    }

    const char* getPluginNamespace() const override
    {
        return mNamespace.data();
    }

    bool isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const override
    {
        return false;
    }

    bool canBroadcastInputAcrossBatch(int inputIndex) const override
    {
        return false;
    }

private:
    template <typename T>
    void write(char*& buffer, const T& val) const
    {
        *reinterpret_cast<T*>(buffer) = val;
        buffer += sizeof(T);
    }

    template <typename T>
    T read(const char*& buffer) const
    {
        T val = *reinterpret_cast<const T*>(buffer);
        buffer += sizeof(T);
        return val;
    }

    void copyDeviceInputToFP32(const void* src, void*& dst)
    {
        assert(mDataType == DataType::kINT8);
        size_t inCount = getC(mInputDims) * getH(mInputDims) * getW(mInputDims);
        std::unique_ptr<char> inputTmp{new char[inCount * elementSize(mDataType)]};
        CHECK(cudaMemcpy(inputTmp.get(), src, inCount * elementSize(mDataType), cudaMemcpyDeviceToHost));
        std::unique_ptr<float> inputFP32{new float[inCount]};
        transform<DataType::kINT8, DataType::kFLOAT>(inputTmp.get(), inputFP32.get(), inCount);
        // int8 scale
        int hw = mInputDims.d[1] * mInputDims.d[2];
        for (int j = 0; j < mInputDims.d[0]; ++j)
        {
            std::transform(inputFP32.get() + hw * j, inputFP32.get() + hw * (j + 1), inputFP32.get() + hw * j,
                [&](float in) -> float { return in * mInHostScale; });
        }
        CHECK(cudaMalloc(&dst, inCount * elementSize(DataType::kFLOAT)));
        CHECK(cudaMemcpy(dst, inputFP32.get(), inCount * elementSize(DataType::kFLOAT), cudaMemcpyHostToDevice));
    }

    void copyDeviceToInt8Output(const void* src, void* dst)
    {
        size_t outCount = getC(mOutputDims) * getH(mOutputDims) * getW(mOutputDims);
        std::unique_ptr<float> outTmp{new float[outCount]};
        CHECK(cudaMemcpy(outTmp.get(), src, outCount * elementSize(DataType::kFLOAT), cudaMemcpyDeviceToHost));
        std::unique_ptr<char> outInt8{new char[outCount * elementSize(DataType::kINT8)]};
        // int8 + scale
        int hw = mOutputDims.d[1] * mOutputDims.d[2];
        for (int j = 0; j < mInputDims.d[0]; ++j)
        {
            std::transform(outTmp.get() + hw * j, outTmp.get() + hw * (j + 1), outTmp.get() + hw * j,
                [&](float in) -> float { return in / mOutHostScale; });
        }
        transform<DataType::kFLOAT, DataType::kINT8>(outTmp.get(), outInt8.get(), outCount);
        CHECK(cudaMemcpy(dst, outInt8.get(), outCount, cudaMemcpyHostToDevice));
    }

private:
    cudnnHandle_t mCudnn;
    cudnnTensorDescriptor_t mSrcDescriptor, mDstDescriptor;
    cudnnPoolingDescriptor_t mPoolingDesc;
    PoolParameters mPoolingParams;
    cudnnPoolingMode_t mMode;
    DataType mDataType;

    Dims mInputDims;
    Dims mOutputDims;
    float mInHostScale{-1.0f};
    float mOutHostScale{-1.0f};
    std::string mNamespace;
};

class UffPoolPluginV2Creator : public IPluginCreator
{
public:
    const char* getPluginName() const override
    {
        return "MaxPool";
    }

    const char* getPluginVersion() const override
    {
        return "2";
    }

    const PluginFieldCollection* getFieldNames() override
    {
        return &mFieldCollection;
    }

    IPluginV2* createPlugin(const char* name, const PluginFieldCollection* fc) override
    {
        auto plugin = new UffPoolPluginV2(*fc);
        mFieldCollection = *fc;
        mPluginName = name;
        return plugin;
    }

    IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) override
    {
        auto plugin = new UffPoolPluginV2(serialData, serialLength);
        mPluginName = name;
        return plugin;
    }

    void setPluginNamespace(const char* libNamespace) override
    {
        mNamespace = libNamespace;
    }

    const char* getPluginNamespace() const override
    {
        return mNamespace.c_str();
    }

private:
    std::string mNamespace;
    std::string mPluginName;
    PluginFieldCollection mFieldCollection{0, nullptr};
};

REGISTER_TENSORRT_PLUGIN(UffPoolPluginV2Creator);

// This function prints the help information for running this sample
void printHelpInfo()
{
    std::cout << "Usage: ./sample_uff_plugin_v2_ext [-h or --help] [-d or --datadir=<path to data directory>] "
                 "[--useDLACore=<int>]\n";
    std::cout << "--help          Display help information\n";
    std::cout << "--datadir       Specify path to a data directory, overriding the default. This option can be used "
                 "multiple times to add multiple directories. If no data directories are given, the default is to use "
                 "(data/samples/mnist/, data/mnist/)"
              << std::endl;
    std::cout << "--useDLACore=N  Specify a DLA engine for layers that support DLA. Value can range from 0 to n-1, "
                 "where n is the number of DLA engines on the platform."
              << std::endl;
    std::cout << "--int8          Run in Int8 mode.\n";
    std::cout << "--fp16          Run in FP16 mode.\n";
}

int main(int argc, char** argv)
{
    bool argsOK = samplesCommon::parseArgs(gArgs, argc, argv);
    if (gArgs.help)
    {
        printHelpInfo();
        return EXIT_SUCCESS;
    }
    if (!argsOK)
    {
        gLogError << "Invalid arguments" << std::endl;
        printHelpInfo();
        return EXIT_FAILURE;
    }
    if (gArgs.dataDirs.empty())
    {
        gArgs.dataDirs = std::vector<std::string>{"data/samples/mnist/", "data/mnist/"};
    }

    auto sampleTest = gLogger.defineTest(gSampleName, argc, argv);

    gLogger.reportTestStart(sampleTest);

    samplesCommon::UffSampleParams params;
    params.uffFileName = locateFile("lenet5_custom_pool.uff", gArgs.dataDirs);
    gLogInfo << params.uffFileName << std::endl;
    SampleUffPluginV2Ext sample(params);

    if (!sample.build())
    {
        return gLogger.reportFail(sampleTest);
    }

    if (!sample.infer())
    {
        return gLogger.reportFail(sampleTest);
    }

    if (!sample.teardown())
    {
        return gLogger.reportFail(sampleTest);
    }

    return gLogger.reportPass(sampleTest);
}
