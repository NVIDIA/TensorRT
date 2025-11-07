/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//!
//! sampleNonZeroPlugin.cpp
//! This file contains a sample demonstrating a plugin for NonZero.
//! It can be run with the following command line:
//! Command: ./sample_non_zero_plugin [-h or --help] [-d=/path/to/data/dir or --datadir=/path/to/data/dir]
//!

// Define TRT entrypoints used in common code
#define DEFINE_TRT_ENTRYPOINTS 1

#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"
#include "nonZeroKernel.h"
#include "parserOnnxConfig.h"

#include "NvInfer.h"
#include <cuda_runtime_api.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
using namespace nvinfer1;
using samplesCommon::SampleUniquePtr;

std::string const kSAMPLE_NAME = "TensorRT.sample_non_zero_plugin";

using half = __half;

void nonZeroIndicesHelper(nvinfer1::DataType type, void const* X, void* indices, void* count, void const* K, int32_t R,
    int32_t C, bool rowOrder, cudaStream_t stream)
{
    if (type == nvinfer1::DataType::kFLOAT)
    {
        nonZeroIndicesImpl<float>(static_cast<float const*>(X), static_cast<int32_t*>(indices),
            static_cast<int32_t*>(count), static_cast<int32_t const*>(K), R, C, rowOrder, stream);
    }
    else if (type == nvinfer1::DataType::kHALF)
    {
        nonZeroIndicesImpl<half>(static_cast<half const*>(X), static_cast<int32_t*>(indices),
            static_cast<int32_t*>(count), static_cast<int32_t const*>(K), R, C, rowOrder, stream);
    }
    else
    {
        ASSERT(false && "Unsupported data type");
    }
}

class NonZeroPlugin : public IPluginV3, public IPluginV3OneCore, public IPluginV3OneBuild, public IPluginV3OneRuntime
{
public:
    NonZeroPlugin(NonZeroPlugin const& p) = default;

    NonZeroPlugin(bool rowOrder)
        : mRowOrder(rowOrder)
    {
        initFieldsToSerialize();
    }

    void initFieldsToSerialize()
    {
        mDataToSerialize.clear();
        mDataToSerialize.emplace_back(PluginField("rowOrder", &mRowOrder, PluginFieldType::kINT32, 1));
        mFCToSerialize.nbFields = mDataToSerialize.size();
        mFCToSerialize.fields = mDataToSerialize.data();
    }

    // IPluginV3 methods

    IPluginCapability* getCapabilityInterface(PluginCapabilityType type) noexcept override
    {
        try
        {
            if (type == PluginCapabilityType::kBUILD)
            {
                return static_cast<IPluginV3OneBuild*>(this);
            }
            if (type == PluginCapabilityType::kRUNTIME)
            {
                return static_cast<IPluginV3OneRuntime*>(this);
            }
            ASSERT(type == PluginCapabilityType::kCORE);
            return static_cast<IPluginV3OneCore*>(this);
        }
        catch (std::exception const& e)
        {
            sample::gLogError << e.what() << std::endl;
        }
        return nullptr;
    }

    IPluginV3* clone() noexcept override
    {
        auto clone = std::make_unique<NonZeroPlugin>(*this);
        clone->initFieldsToSerialize();
        return clone.release();
    }

    // IPluginV3OneCore methods
    char const* getPluginName() const noexcept override
    {
        return "NonZeroPlugin";
    }

    char const* getPluginVersion() const noexcept override
    {
        return "0";
    }

    char const* getPluginNamespace() const noexcept override
    {
        return "";
    }

    // IPluginV3OneBuild methods
    int32_t getNbOutputs() const noexcept override
    {
        return 2;
    }

    int32_t configurePlugin(DynamicPluginTensorDesc const* in, int32_t nbInputs, DynamicPluginTensorDesc const* out,
        int32_t nbOutputs) noexcept override
    {
        return 0;
    }

    bool supportsFormatCombination(
        int32_t pos, DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept override
    {
        bool typeOk{false};
        if (pos == 0)
        {
            typeOk = inOut[0].desc.type == DataType::kFLOAT || inOut[0].desc.type == DataType::kHALF;
        }
        else if (pos == 1)
        {
            typeOk = inOut[1].desc.type == DataType::kINT32;
        }
        else // pos == 2
        {
            // size tensor outputs must be NCHW INT32
            typeOk = inOut[2].desc.type == DataType::kINT32;
        }

        return inOut[pos].desc.format == PluginFormat::kLINEAR && typeOk;
    }

    int32_t getOutputDataTypes(
        DataType* outputTypes, int32_t nbOutputs, DataType const* inputTypes, int32_t nbInputs) const noexcept override
    {
        outputTypes[0] = DataType::kINT32;
        outputTypes[1] = DataType::kINT32;
        return 0;
    }

    int32_t getOutputShapes(DimsExprs const* inputs, int32_t nbInputs, DimsExprs const* shapeInputs,
        int32_t nbShapeInputs, DimsExprs* outputs, int32_t nbOutputs, IExprBuilder& exprBuilder) noexcept override
    {
        // The input tensor must be 2-D
        if (inputs[0].nbDims != 2)
        {
            return -1;
        }

        outputs[0].nbDims = 2;

        auto upperBound = exprBuilder.operation(DimensionOperation::kPROD, *inputs[0].d[0], *inputs[0].d[1]);

        // On average, we can assume that half of all elements will be non-zero
        auto optValue = exprBuilder.operation(DimensionOperation::kFLOOR_DIV, *upperBound, *exprBuilder.constant(2));
        auto numNonZeroSizeTensor = exprBuilder.declareSizeTensor(1, *optValue, *upperBound);

        if (!mRowOrder)
        {
            outputs[0].d[0] = exprBuilder.constant(2);
            outputs[0].d[1] = numNonZeroSizeTensor;
        }
        else
        {
            outputs[0].d[0] = numNonZeroSizeTensor;
            outputs[0].d[1] = exprBuilder.constant(2);
        }

        // output at index 1 is a size tensor
        outputs[1].nbDims = 0; // size tensors must be declared as 0-D

        return 0;
    }

    // IPluginV3OneRuntime methods
    int32_t enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc, void const* const* inputs,
        void* const* outputs, void* workspace, cudaStream_t stream) noexcept override
    {

        int32_t const R = inputDesc[0].dims.d[0];
        int32_t const C = inputDesc[0].dims.d[1];

        auto type = inputDesc[0].type;

        if (!(type == nvinfer1::DataType::kHALF || type == nvinfer1::DataType::kFLOAT))
        {
            sample::gLogError << "Unsupported: Sample only supports DataType::kHALF and DataType::FLOAT" << std::endl;
            return -1;
        }

        cudaMemsetAsync(outputs[1], 0, sizeof(int32_t), stream);

        if (workspace == nullptr)
        {
            sample::gLogError << "Unsupported: workspace is null" << std::endl;
            return -1;
        }

        if (!mRowOrder)
        {
            // When constructing a column major output, the kernel needs to be aware of the total number of non-zero
            // elements so as to write the non-zero indices at the correct places. Therefore, we will launch the kernel
            // twice: first, only to calculate the total non-zero count, which will be stored in workspace; and
            // then to actually write the non-zero indices to the outputs[0] buffer.
            cudaMemsetAsync(workspace, 0, sizeof(int32_t), stream);
            nonZeroIndicesHelper(type, inputs[0], nullptr, workspace, 0, R, C, mRowOrder, stream);
            nonZeroIndicesHelper(type, inputs[0], outputs[0], outputs[1], workspace, R, C, mRowOrder, stream);
        }
        else
        {
            nonZeroIndicesHelper(type, inputs[0], outputs[0], outputs[1], 0, R, C, mRowOrder, stream);
        }

        return 0;
    }

    int32_t onShapeChange(
        PluginTensorDesc const* in, int32_t nbInputs, PluginTensorDesc const* out, int32_t nbOutputs) noexcept override
    {
        return 0;
    }

    IPluginV3* attachToContext(IPluginResourceContext* context) noexcept override
    {
        return clone();
    }

    PluginFieldCollection const* getFieldsToSerialize() noexcept override
    {
        return &mFCToSerialize;
    }

    size_t getWorkspaceSize(DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
        DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept override
    {
        return sizeof(int32_t);
    }

private:
    bool mRowOrder{true};
    std::vector<nvinfer1::PluginField> mDataToSerialize;
    nvinfer1::PluginFieldCollection mFCToSerialize;
};

class NonZeroPluginCreator : public nvinfer1::IPluginCreatorV3One
{
public:
    NonZeroPluginCreator()
    {
        mPluginAttributes.clear();
        mPluginAttributes.emplace_back(PluginField("rowOrder", nullptr, PluginFieldType::kINT32, 1));
        mFC.nbFields = mPluginAttributes.size();
        mFC.fields = mPluginAttributes.data();
    }

    char const* getPluginName() const noexcept override
    {
        return "NonZeroPlugin";
    }

    char const* getPluginVersion() const noexcept override
    {
        return "0";
    }

    PluginFieldCollection const* getFieldNames() noexcept override
    {
        return &mFC;
    }

    IPluginV3* createPlugin(char const* name, PluginFieldCollection const* fc, TensorRTPhase phase) noexcept override
    {
        try
        {
            bool rowOrder{true};
            for (int32_t i = 0; i < fc->nbFields; ++i)
            {
                auto const fieldName(fc->fields[i].name);
                if (std::strcmp(fieldName, "rowOrder") == 0)
                {
                    rowOrder = *static_cast<bool const*>(fc->fields[i].data);
                }
            }
            return new NonZeroPlugin(rowOrder);
        }
        catch (std::exception const& e)
        {
            sample::gLogError << e.what() << std::endl;
        }
        return nullptr;
    }

    char const* getPluginNamespace() const noexcept override
    {
        return "";
    }

private:
    nvinfer1::PluginFieldCollection mFC;
    std::vector<nvinfer1::PluginField> mPluginAttributes;
};

namespace
{
struct NonZeroParams : public samplesCommon::SampleParams
{
    bool rowOrder{true};
};
} // namespace

//! \brief  The SampleNonZeroPlugin class implements a NonZero plugin
//!
//! \details The plugin is able to output the non-zero indices in row major or column major order
//!
class SampleNonZeroPlugin
{
public:
    SampleNonZeroPlugin(NonZeroParams const& params)
        : mParams(params)
        , mRuntime(nullptr)
        , mEngine(nullptr)
    {
        mSeed = static_cast<uint32_t>(time(nullptr));
    }

    //!
    //! \brief Function builds the network engine
    //!
    bool build();

    //!
    //! \brief Runs the TensorRT inference engine for this sample
    //!
    bool infer();

private:
    NonZeroParams mParams; //!< The parameters for the sample.

    nvinfer1::Dims mInputDims;  //!< The dimensions of the input to the network.
    nvinfer1::Dims mOutputDims; //!< The dimensions of the output to the network.

    std::shared_ptr<nvinfer1::IRuntime> mRuntime;   //!< The TensorRT runtime used to deserialize the engine
    std::shared_ptr<nvinfer1::ICudaEngine> mEngine; //!< The TensorRT engine used to run the network

    uint32_t mSeed{};

    //!
    //! \brief Creates a TensorRT network and inserts a NonZero plugin
    //!
    bool constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
        SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config);

    //!
    //! \brief Reads the input and stores the result in a managed buffer
    //!
    bool processInput(samplesCommon::BufferManager const& buffers);

    //!
    //! \brief Verifies the result
    //!
    bool verifyOutput(samplesCommon::BufferManager const& buffers);
};

//!
//! \brief Creates the network, configures the builder and creates the network engine
//!
//! \details This function creates a network containing a NonZeroPlugin and builds
//!          the engine that will be used to run the plugin (mEngine)
//!
//! \return true if the engine was created successfully and false otherwise
//!
bool SampleNonZeroPlugin::build()
{
    auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));
    if (!builder)
    {
        return false;
    }

    auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(0));
    if (!network)
    {
        return false;
    }

    auto config = SampleUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config)
    {
        return false;
    }

    auto pluginCreator = std::make_unique<NonZeroPluginCreator>();
    getPluginRegistry()->registerCreator(*pluginCreator, "");

    auto constructed = constructNetwork(builder, network, config);
    if (!constructed)
    {
        return false;
    }

    // CUDA stream used for profiling by the builder.
    auto profileStream = samplesCommon::makeCudaStream();
    if (!profileStream)
    {
        return false;
    }
    config->setProfileStream(*profileStream);

    SampleUniquePtr<IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
    if (!plan)
    {
        return false;
    }

    mRuntime = std::shared_ptr<nvinfer1::IRuntime>(createInferRuntime(sample::gLogger.getTRTLogger()));
    if (!mRuntime)
    {
        return false;
    }

    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
        mRuntime->deserializeCudaEngine(plan->data(), plan->size()), samplesCommon::InferDeleter());
    if (!mEngine)
    {
        return false;
    }

    ASSERT(network->getNbInputs() == 1);
    mInputDims = network->getInput(0)->getDimensions();
    ASSERT(mInputDims.nbDims == 2);

    ASSERT(network->getNbOutputs() == 2);
    mOutputDims = network->getOutput(0)->getDimensions();
    ASSERT(mOutputDims.nbDims == 2);

    return true;
}

//!
//! \brief Creates a network with a single custom layer containing the NonZero plugin and marks the
//!        output layers
//!
//! \param network Pointer to the network that will be populated with the NonZero plugin
//!
//! \param builder Pointer to the engine builder
//!
bool SampleNonZeroPlugin::constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
    SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config)
{
    if (mParams.fp16)
    {
        config->setFlag(BuilderFlag::kFP16);
    }

    std::default_random_engine generator(mSeed);
    std::uniform_int_distribution<int32_t> distr(10, 25);

    int32_t const R = distr(generator);
    int32_t const C = distr(generator);
    auto* in = network->addInput("Input", DataType::kFLOAT, {2, {R, C}});
    ASSERT(in != nullptr);

    std::vector<PluginField> const vecPF{{"rowOrder", &mParams.rowOrder, PluginFieldType::kINT32, 1}};
    PluginFieldCollection pfc{static_cast<int32_t>(vecPF.size()), vecPF.data()};

    auto pluginCreator = static_cast<IPluginCreatorV3One*>(getPluginRegistry()->getCreator("NonZeroPlugin", "0", ""));
    auto plugin = std::unique_ptr<IPluginV3>(pluginCreator->createPlugin("NonZeroPlugin", &pfc, TensorRTPhase::kBUILD));

    std::vector<ITensor*> inputsVec{in};
    auto pluginNonZeroLayer = network->addPluginV3(inputsVec.data(), inputsVec.size(), nullptr, 0, *plugin);
    ASSERT(pluginNonZeroLayer != nullptr);
    ASSERT(pluginNonZeroLayer->getOutput(0) != nullptr);
    ASSERT(pluginNonZeroLayer->getOutput(1) != nullptr);

    pluginNonZeroLayer->getOutput(0)->setName("Output0");
    pluginNonZeroLayer->getOutput(1)->setName("Output1");

    network->markOutput(*(pluginNonZeroLayer->getOutput(0)));
    network->markOutput(*(pluginNonZeroLayer->getOutput(1)));

    return true;
}

//!
//! \brief Runs the TensorRT inference engine for this sample
//!
//! \details This function is the main execution function of the sample. It allocates the buffer,
//!          sets inputs and executes the engine.
//!
bool SampleNonZeroPlugin::infer()
{

    // Since the data dependent output size cannot be inferred from the engine denote a sufficient size for the
    // corresponding output buffer (along with the rest of the I/O tensors)
    std::vector<int64_t> ioVolumes = {mInputDims.d[0] * mInputDims.d[1], mInputDims.d[0] * mInputDims.d[1] * 2, 1};

    // Create RAII buffer manager object
    samplesCommon::BufferManager buffers(mEngine, ioVolumes);

    auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    if (!context)
    {
        return false;
    }

    for (int32_t i = 0, e = mEngine->getNbIOTensors(); i < e; ++i)
    {
        auto const name = mEngine->getIOTensorName(i);
        context->setTensorAddress(name, buffers.getDeviceBuffer(name));
    }

    // Read the input data into the managed buffers
    ASSERT(mParams.inputTensorNames.size() == 1);
    if (!processInput(buffers))
    {
        return false;
    }

    // Create CUDA stream for the execution of this inference.
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // Memcpy from host input buffers to device input buffers
    buffers.copyInputToDeviceAsync(stream);

    bool status = context->enqueueV3(stream);
    if (!status)
    {
        return false;
    }

    // Asynchronously copy data from device output buffers to host output buffers.
    buffers.copyOutputToHostAsync(stream);

    // Wait for the work in the stream to complete.
    CHECK(cudaStreamSynchronize(stream));

    // Release stream.
    CHECK(cudaStreamDestroy(stream));

    // Verify results
    if (!verifyOutput(buffers))
    {
        return false;
    }

    return true;
}

//!
//! \brief Reads the input and stores the result in a managed buffer
//!
bool SampleNonZeroPlugin::processInput(samplesCommon::BufferManager const& buffers)
{
    int32_t const inputH = mInputDims.d[0];
    int32_t const inputW = mInputDims.d[1];

    std::vector<uint8_t> fileData(inputH * inputW);

    std::default_random_engine generator(mSeed);
    std::uniform_int_distribution<int32_t> distr(0, 9);
    auto const number = distr(generator);
    samplesCommon::readPGMFile(
        samplesCommon::locateFile(std::to_string(number) + ".pgm", mParams.dataDirs), fileData.data(), inputH, inputW);

    float* hostDataBuffer = static_cast<float*>(buffers.getHostBuffer(mParams.inputTensorNames[0]));
    for (int32_t i = 0; i < inputH * inputW; ++i)
    {
        auto const raw = 1.0 - float(fileData[i] / 255.0);
        hostDataBuffer[i] = raw;
    }

    sample::gLogInfo << "Input:" << std::endl;
    for (int32_t i = 0; i < inputH; ++i)
    {
        for (int32_t j = 0; j < inputW; ++j)
        {
            sample::gLogInfo << hostDataBuffer[i * inputW + j];
            if (j < inputW - 1)
            {
                sample::gLogInfo << ", ";
            }
        }
        sample::gLogInfo << std::endl;
    }
    sample::gLogInfo << std::endl;

    return true;
}

//!
//! \brief Verify result
//!
//! \return whether the output correctly identifies all (and only) non-zero elements
//!
bool SampleNonZeroPlugin::verifyOutput(samplesCommon::BufferManager const& buffers)
{
    float* input = static_cast<float*>(buffers.getHostBuffer(mParams.inputTensorNames[0]));
    int32_t* output = static_cast<int32_t*>(buffers.getHostBuffer(mParams.outputTensorNames[0]));
    int32_t count = *static_cast<int32_t*>(buffers.getHostBuffer(mParams.outputTensorNames[1]));

    std::vector<bool> covered(mInputDims.d[0] * mInputDims.d[1], false);

    sample::gLogInfo << "Output:" << std::endl;
    if (mParams.rowOrder)
    {
        for (int32_t i = 0; i < count; ++i)
        {
            for (int32_t j = 0; j < 2; ++j)
            {
                sample::gLogInfo << output[j + 2 * i] << " ";
            }
            sample::gLogInfo << std::endl;
        }
    }
    else
    {
        for (int32_t i = 0; i < 2; ++i)
        {
            for (int32_t j = 0; j < count; ++j)
            {
                sample::gLogInfo << output[j + count * i] << " ";
            }
            sample::gLogInfo << std::endl;
        }
    }

    if (!mParams.rowOrder)
    {
        for (int32_t i = 0; i < count; ++i)
        {
            auto const idx = output[i] * mInputDims.d[1] + output[i + count];
            covered[idx] = true;
            if (input[idx] == 0.F)
            {
                return false;
            }
        }
    }
    else
    {
        for (int32_t i = 0; i < count; ++i)
        {
            auto const idx = output[2 * i] * mInputDims.d[1] + output[2 * i + 1];
            covered[idx] = true;
            if (input[idx] == 0.F)
            {
                return false;
            }
        }
    }

    for (int32_t i = 0; i < static_cast<int32_t>(covered.size()); ++i)
    {
        if (!covered[i])
        {
            if (input[i] != 0.F)
            {
                return false;
            }
        }
    }

    return true;
}

//!
//! \brief Initializes members of the params struct using the command line args
//!
NonZeroParams initializeSampleParams(samplesCommon::Args const& args)
{
    NonZeroParams params;
    if (args.dataDirs.empty()) // Use default directories if user hasn't provided directory paths
    {
        params.dataDirs.push_back("data/mnist/");
        params.dataDirs.push_back("data/samples/mnist/");
    }
    else // Use the data directory provided by the user
    {
        params.dataDirs = args.dataDirs;
    }

    params.inputTensorNames.push_back("Input");
    params.outputTensorNames.push_back("Output0");
    params.outputTensorNames.push_back("Output1");
    params.fp16 = args.runInFp16;
    params.rowOrder = args.rowOrder;

    return params;
}

//!
//! \brief Prints the help information for running this sample
//!
void printHelpInfo()
{
    std::cout << "Usage: ./sample_non_zero_plugin [-h or --help] [-d or --datadir=<path to data directory>]"
              << std::endl;
    std::cout << "--help          Display help information" << std::endl;
    std::cout << "--datadir       Specify path to a data directory, overriding the default. This option can be used "
                 "multiple times to add multiple directories. If no data directories are given, the default is to use "
                 "(data/samples/mnist/, data/mnist/)"
              << std::endl;
    std::cout << "--fp16          Run in FP16 mode." << std::endl;
    std::cout << "--columnOrder   Run plugin in column major output mode." << std::endl;
}

int main(int argc, char** argv)
{
    samplesCommon::Args args;
    bool argsOK = samplesCommon::parseArgs(args, argc, argv);
    if (!argsOK)
    {
        sample::gLogError << "Invalid arguments" << std::endl;
        printHelpInfo();
        return EXIT_FAILURE;
    }
    if (args.help)
    {
        printHelpInfo();
        return EXIT_SUCCESS;
    }

    auto sampleTest = sample::gLogger.defineTest(kSAMPLE_NAME, argc, argv);

    sample::gLogger.reportTestStart(sampleTest);

    SampleNonZeroPlugin sample(initializeSampleParams(args));

    sample::gLogInfo << "Building and running a GPU inference engine for NonZero plugin" << std::endl;

    if (!sample.build())
    {
        return sample::gLogger.reportFail(sampleTest);
    }
    if (!sample.infer())
    {
        return sample::gLogger.reportFail(sampleTest);
    }

    return sample::gLogger.reportPass(sampleTest);
}
