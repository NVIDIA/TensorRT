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

//! \file SampleIOFormats.cpp
//! \brief This file contains the implementation of the I/O formats sample.
//!
//! It builds a TensorRT engine by from an MNIST network.
//! It uses the engine to identify input images.
//! The goal of this sample is to show how to specify allowed I/O formats.
//! It can be run with the following command line:
//! Command: ./sample_io_formats

// Define TRT entrypoints used in common code
#define DEFINE_TRT_ENTRYPOINTS 1

#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "half.h"
#include "logger.h"
#include "parserOnnxConfig.h"
#include "sampleOptions.h"

#include "NvInfer.h"
#include "NvOnnxParser.h"

#include <algorithm>
#include <cmath>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iostream>
#include <sstream>

#include <array>
#include <cstdlib>
#include <random>
#include <string>
#include <utility>
#include <vector>

using namespace nvinfer1;

std::string const gSampleName = "TensorRT.sample_io_formats";

inline int32_t divUp(int32_t a, int32_t b)
{
    return (a + b - 1) / b;
}

template <typename T>
std::shared_ptr<T> mallocCudaMem(size_t nbElems)
{
    T* ptr = nullptr;
    CHECK(cudaMalloc((void**) &ptr, sizeof(T) * nbElems));
    return std::shared_ptr<T>(ptr, [](T* p) { CHECK(cudaFree(p)); });
}

class BufferDesc
{
public:
    BufferDesc() = default;

    BufferDesc(nvinfer1::Dims dims, int32_t dataWidth, TensorFormat format)
    {
        this->dataWidth = dataWidth;
        if (format == TensorFormat::kLINEAR)
        {
            this->dims[0] = dims.d[0];
            this->dims[1] = dims.d[1];
            this->dims[2] = dims.d[2];
            this->dims[3] = dims.d[3];
            this->dims[4] = 1;
        }
        else if (format == TensorFormat::kCHW32)
        {
            this->dims[0] = dims.d[0];
            this->dims[1] = divUp(dims.d[1], 32);
            this->dims[2] = dims.d[2];
            this->dims[3] = dims.d[3];
            this->dims[4] = 32;
            this->scalarPerVector = 32;
        }
        else if (format == TensorFormat::kHWC)
        {
            this->dims[0] = dims.d[0];
            this->dims[1] = dims.d[2];
            this->dims[2] = dims.d[3];
            this->dims[3] = dims.d[1];
            this->dims[4] = 1;
            this->channelPivot = true;
        }
    }

    // [(C+x-1)/x][H][W][x]
    // or
    // [H][W][(C+x-1)/x*x][1]
    int32_t dims[5] = {1, 1, 1, 1, 1};
    int32_t dataWidth = 1;
    int32_t scalarPerVector = 1;

    bool channelPivot = false;

    int32_t getElememtSize()
    {
        return dims[0] * dims[1] * dims[2] * dims[3] * dims[4];
    }
    int32_t getBufferSize()
    {
        return getElememtSize() * dataWidth;
    }
};

//! Specification for a network I/O tensor.
class TypeSpec
{
public:
    DataType dtype;         //!< datatype
    TensorFormat format;    //!< format
    std::string formatName; //!< name of the format
};

class SampleBuffer
{
public:
    SampleBuffer()
    {
        dims.d[0] = 1;
        dims.d[1] = 1;
        dims.d[2] = 1;
        dims.d[3] = 1;
    }

    SampleBuffer(nvinfer1::Dims dims, int32_t dataWidth, TensorFormat format, bool isInput)
        : dims(dims)
        , dataWidth(dataWidth)
        , format(format)
        , isInput(isInput)
    {

        // Output buffer is unsqueezed to 4D in order to reuse the BufferDesc class
        if (isInput == false)
        {
            dims.d[2] = dims.d[0];
            dims.d[3] = dims.d[1];
            dims.d[0] = 1;
            dims.d[1] = 1;
        }

        desc = BufferDesc(dims, dataWidth, format);

        if (nullptr == buffer)
        {
            buffer = new uint8_t[getBufferSize()]();
        }
    }

    ~SampleBuffer()
    {
        destroy();
    }

    SampleBuffer& operator=(SampleBuffer&& sampleBuffer) noexcept
    {
        destroy();

        this->dims = sampleBuffer.dims;
        this->dataWidth = sampleBuffer.dataWidth;
        this->desc = sampleBuffer.desc;
        this->format = sampleBuffer.format;
        this->isInput = sampleBuffer.isInput;
        this->buffer = sampleBuffer.buffer;
        sampleBuffer.buffer = nullptr;

        return *this;
    }

    void destroy()
    {
        if (buffer != nullptr)
        {
            delete[] buffer;
            buffer = nullptr;
        }
    }

    nvinfer1::Dims dims;

    int32_t dataWidth{1};

    TensorFormat format{TensorFormat::kLINEAR};

    bool isInput{true};

    BufferDesc desc;

    uint8_t* buffer = nullptr;

    int32_t getBufferSize()
    {
        return desc.getBufferSize();
    }
};

//!
//! \brief  The SampleIOFormats class implements the I/O formats sample
//!
//! \details It creates the network using the Onnx parser.
//!
class SampleIOFormats
{
public:
    SampleIOFormats(samplesCommon::OnnxSampleParams const& params)
        : mParams(params)
    {
    }

    //!
    //! \brief Builds the network engine
    //!
    bool build(int32_t dataWidth);

    //!
    //! \brief Verify the built engine I/O types and formats.
    //!
    bool verify(TypeSpec const& spec);

    //!
    //! \brief Runs the TensorRT inference engine for this sample
    //!
    bool infer(SampleBuffer& inputBuf, SampleBuffer& outputBuf);

private:
    //!
    //! \brief Parses an ONNX model for MNIST and creates a TensorRT network
    //!
    bool constructNetwork(std::unique_ptr<nvinfer1::IBuilder>& builder,
        std::unique_ptr<nvinfer1::INetworkDefinition>& network, std::unique_ptr<nvinfer1::IBuilderConfig>& config,
        std::unique_ptr<nvonnxparser::IParser>& parser);

    std::unique_ptr<IRuntime> mRuntime{};                    //!< The TensorRT Runtime used to deserialize the engine.
    std::shared_ptr<nvinfer1::ICudaEngine> mEngine{nullptr}; //!< The TensorRT engine used to run the network

public:
    samplesCommon::OnnxSampleParams mParams;

    nvinfer1::Dims mInputDims; //!< The dimensions of the input to the network.

    nvinfer1::Dims mOutputDims; //!< The dimensions of the output to the network.

    TensorFormat mTensorFormat{TensorFormat::kLINEAR};

    int32_t mDigit;
};

//!
//! \brief Validates engine I/O datatypes and formats against a reference.
//!
//! \details This function queries I/O datatype and format description from the built engine.
//!           Validating them is sufficient to ensure that `ITensor::setType` and `ITensor::setAllowedFormats` API as
//!           expected.
//!
//! \return true if type and format validation succeeds.
//!
bool SampleIOFormats::verify(TypeSpec const& spec)
{
    assert(mEngine->getNbIOTensors() == 2);
    char const* inputName = mEngine->getIOTensorName(0);
    char const* outputName = mEngine->getIOTensorName(1);

    auto verifyType = [](DataType actual, DataType expected) {
        if (actual != expected)
        {
            sample::gLogError << "Expected " << expected << " data type, got " << actual;
            return false;
        }
        return true;
    };

    if (!verifyType(mEngine->getTensorDataType(inputName), spec.dtype))
    {
        return false;
    }

    if (!verifyType(mEngine->getTensorDataType(outputName), spec.dtype))
    {
        return false;
    }

    auto verifyFormat = [](std::string actual, std::string expected) {
        if (expected.find(actual) != std::string::npos)
        {
            sample::gLogError << "Expected " << expected << " format, got " << actual;
            return false;
        }
        return true;
    };

    if (!verifyFormat(std::string(mEngine->getTensorFormatDesc(inputName)), spec.formatName))
    {
        return false;
    }

    if (!verifyFormat(std::string(mEngine->getTensorFormatDesc(inputName)), "kLINEAR"))
    {
        return false;
    }

    return true;
}

//!
//! \brief Creates the network, configures the builder and creates the network engine
//!
//! \details This function creates the single layer network by manual insertion and builds
//!          the engine
//!
//! \return true if the engine was created successfully and false otherwise
//!
bool SampleIOFormats::build(int32_t dataWidth)
{
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));
    if (!builder)
    {
        return false;
    }

    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(
        builder->createNetworkV2(1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kSTRONGLY_TYPED)));
    if (!network)
    {
        return false;
    }

    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config)
    {
        return false;
    }

    auto parser
        = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, sample::gLogger.getTRTLogger()));
    if (!parser)
    {
        return false;
    }

    auto constructed = constructNetwork(builder, network, config, parser);
    if (!constructed)
    {
        return false;
    }

    network->getInput(0)->setAllowedFormats(static_cast<TensorFormats>(1 << static_cast<int32_t>(mTensorFormat)));
    network->getOutput(0)->setAllowedFormats(1U << static_cast<int32_t>(TensorFormat::kLINEAR));

    mEngine.reset();

    config->setFlag(BuilderFlag::kGPU_FALLBACK);

    // CUDA stream used for profiling by the builder.
    auto profileStream = samplesCommon::makeCudaStream();
    if (!profileStream)
    {
        return false;
    }
    config->setProfileStream(*profileStream);

    std::unique_ptr<nvinfer1::ITimingCache> timingCache{};

    // Load timing cache
    if (!mParams.timingCacheFile.empty())
    {
        timingCache
            = samplesCommon::buildTimingCacheFromFile(sample::gLogger.getTRTLogger(), *config, mParams.timingCacheFile);
    }

    std::unique_ptr<IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
    if (!plan)
    {
        return false;
    }

    if (timingCache != nullptr && !mParams.timingCacheFile.empty())
    {
        samplesCommon::updateTimingCacheFile(
            sample::gLogger.getTRTLogger(), mParams.timingCacheFile, timingCache.get(), *builder);
    }

    if (!mRuntime)
    {
        mRuntime = std::unique_ptr<IRuntime>(createInferRuntime(sample::gLogger.getTRTLogger()));
    }

    if (!mRuntime)
    {
        return false;
    }

    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(mRuntime->deserializeCudaEngine(plan->data(), plan->size()));
    if (!mEngine)
    {
        return false;
    }

    ASSERT(network->getNbInputs() == 1);
    mInputDims = network->getInput(0)->getDimensions();
    ASSERT(mInputDims.nbDims == 4);

    ASSERT(network->getNbOutputs() == 1);
    mOutputDims = network->getOutput(0)->getDimensions();
    ASSERT(mOutputDims.nbDims == 2);

    return true;
}

//!
//! \brief Uses a ONNX parser to create the Onnx MNIST Network and marks the
//!        output layers
//!
//! \param network Pointer to the network that will be populated with the Onnx MNIST network
//!
//! \param builder Pointer to the engine builder
//!
bool SampleIOFormats::constructNetwork(std::unique_ptr<nvinfer1::IBuilder>& builder,
    std::unique_ptr<nvinfer1::INetworkDefinition>& network, std::unique_ptr<nvinfer1::IBuilderConfig>& config,
    std::unique_ptr<nvonnxparser::IParser>& parser)
{
    auto parsed = parser->parseFromFile(samplesCommon::locateFile(mParams.onnxFileName, mParams.dataDirs).c_str(),
        static_cast<int32_t>(sample::gLogger.getReportableSeverity()));
    if (!parsed)
    {
        return false;
    }

    samplesCommon::enableDLA(builder.get(), config.get(), mParams.dlaCore);

    return true;
}

//!
//! \brief Runs the TensorRT inference engine for this sample
//!
//! \details This function is the main execution function of the sample. It allocates
//!          the buffer, sets inputs, executes the engine, and verifies the output.
//!
bool SampleIOFormats::infer(SampleBuffer& inputBuf, SampleBuffer& outputBuf)
{
    auto const devInput = mallocCudaMem<uint8_t>(inputBuf.getBufferSize());
    auto devOutput = mallocCudaMem<uint8_t>(outputBuf.getBufferSize());

    CHECK(cudaMemcpy(devInput.get(), inputBuf.buffer, inputBuf.getBufferSize(), cudaMemcpyHostToDevice));

    auto context = std::unique_ptr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    if (!context)
    {
        return false;
    }

    for (int32_t i = 0, e = mEngine->getNbIOTensors(); i < e; i++)
    {
        auto const name = mEngine->getIOTensorName(i);
        if (mEngine->getTensorIOMode(name) == TensorIOMode::kINPUT)
        {
            context->setTensorAddress(name, devInput.get());
        }
        else
        {
            context->setTensorAddress(name, devOutput.get());
        }
    }

    // Create CUDA stream for the execution of this inference.
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // Asynchronously enqueue the inference work
    if (!context->enqueueV3(stream))
    {
        return false;
    }

    // Wait for the work in the stream to complete
    CHECK(cudaStreamSynchronize(stream));

    // Release stream
    CHECK(cudaStreamDestroy(stream));

    CHECK(cudaMemcpy(outputBuf.buffer, devOutput.get(), outputBuf.getBufferSize(), cudaMemcpyDeviceToHost));

    return true;
}

//!
//! \brief Initializes members of the params struct using the command line args
//!
samplesCommon::OnnxSampleParams initializeSampleParams(samplesCommon::Args const& args)
{
    samplesCommon::OnnxSampleParams params;
    if (args.dataDirs.empty()) // Use default directories if user hasn't provided directory paths
    {
        params.dataDirs.push_back("data/mnist/");
        params.dataDirs.push_back("data/samples/mnist/");
    }
    else // Use the data directory provided by the user
    {
        params.dataDirs = args.dataDirs;
    }
    params.onnxFileName = "mnist.onnx";
    params.dlaCore = args.useDLACore;
    params.timingCacheFile = args.timingCacheFile;

    return params;
}
//!
//! \brief Prints the help information for running this sample
//!
void printHelpInfo()
{
    std::cout
        << "Usage: ./sample_onnx_mnist [-h or --help] [-d or --datadir=<path to data directory>] [--useDLACore=<int>] "
        << "[-t or --timingCacheFile=<path to timing cache file>]" << std::endl;
    std::cout << "--help             Display help information" << std::endl;
    std::cout << "--datadir          Specify path to a data directory, overriding the default. This option can be used "
                 "multiple times to add multiple directories. If no data directories are given, the default is to use "
                 "(data/samples/mnist/, data/mnist/)"
              << std::endl;
    std::cout << "--useDLACore=N     Specify a DLA engine for layers that support DLA. Value can range from 0 to n-1, "
                 "where n is the number of DLA engines on the platform."
              << std::endl;
    std::cout << "--timingCacheFile  Specify path to a timing cache file. If it does not already exist, it will be "
              << "created." << std::endl;
}
//!
//! \brief Used to run the engine build and inference/reference functions
//!
template <typename T>
bool process(SampleIOFormats& sample, sample::Logger::TestAtom const& sampleTest, SampleBuffer& inputBuf,
    SampleBuffer& outputBuf, TypeSpec& spec)
{
    sample::gLogInfo << "Building and running a GPU inference engine with specified I/O formats." << std::endl;

    if (!sample.build(sizeof(T)))
    {
        return false;
    }
    if (!sample.verify(spec))
    {
        return false;
    }

    inputBuf = SampleBuffer(sample.mInputDims, sizeof(T), sample.mTensorFormat, true);
    outputBuf = SampleBuffer(sample.mOutputDims, sizeof(T), TensorFormat::kLINEAR, false);

    if (!sample.infer(inputBuf, outputBuf))
    {
        return false;
    }
    return true;
}

int32_t main(int32_t argc, char** argv)
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

    auto sampleTest = sample::gLogger.defineTest(gSampleName, argc, argv);

    sample::gLogger.reportTestStart(sampleTest);

    samplesCommon::OnnxSampleParams params = initializeSampleParams(args);

    std::vector<TypeSpec> fp32TypeSpec = {
        TypeSpec{DataType::kFLOAT, TensorFormat::kLINEAR, "kLINEAR"},
        TypeSpec{DataType::kFLOAT, TensorFormat::kHWC, "kHWC"},
        TypeSpec{DataType::kFLOAT, TensorFormat::kCHW32, "kCHW32"},
    };

    SampleIOFormats sample(params);

    sample::gLogInfo
        << "Build TRT engine with different IO data type and formats. Ensure that built engine abide by them"
        << std::endl;

    // Test FP32 formats
    for (auto spec : fp32TypeSpec)
    {
        sample::gLogInfo << "Testing datatype FP32 with format " << spec.formatName << std::endl;
        sample.mTensorFormat = spec.format;
        SampleBuffer inputBuf, outputBuf;

        if (!process<float>(sample, sampleTest, inputBuf, outputBuf, spec))
        {
            return sample::gLogger.reportFail(sampleTest);
        }
    }

    return sample::gLogger.reportPass(sampleTest);
}
