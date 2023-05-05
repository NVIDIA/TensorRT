/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
using samplesCommon::SampleUniquePtr;

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
        else if (format == TensorFormat::kCHW2)
        {

            this->dims[0] = dims.d[0];
            this->dims[1] = divUp(dims.d[1], 2);
            this->dims[2] = dims.d[2];
            this->dims[3] = dims.d[3];
            this->dims[4] = 2;
            this->scalarPerVector = 2;
        }
        else if (format == TensorFormat::kCHW4)
        {
            this->dims[0] = dims.d[0];
            this->dims[1] = divUp(dims.d[1], 4);
            this->dims[2] = dims.d[2];
            this->dims[3] = dims.d[3];
            this->dims[4] = 4;
            this->scalarPerVector = 4;
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
        else if (format == TensorFormat::kHWC8)
        {
            this->dims[0] = dims.d[0];
            this->dims[1] = dims.d[2];
            this->dims[2] = dims.d[3];
            this->dims[3] = divUp(dims.d[1], 8) * 8;
            this->dims[4] = 1;
            this->scalarPerVector = 8;
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
    //! \brief Runs the TensorRT inference engine for this sample
    //!
    bool infer(SampleBuffer& inputBuf, SampleBuffer& outputBuf);

    //!
    //! \brief Used to run CPU reference and get result
    //!
    bool reference();

    //!
    //! \brief Used to compare the CPU reference with the TRT result
    //!
    void compareResult();

    //!
    //! \brief Reads the digit map from the file
    //!
    bool readDigits(SampleBuffer& buffer, int32_t groundTruthDigit);

    //!
    //! \brief Verifies that the output is correct and prints it
    //!
    template <typename T>
    bool verifyOutput(SampleBuffer& outputBuf, int32_t groundTruthDigit) const;

private:
    //!
    //! \brief Parses an ONNX model for MNIST and creates a TensorRT network
    //!
    bool constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
        SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
        SampleUniquePtr<nvonnxparser::IParser>& parser);

    std::shared_ptr<nvinfer1::ICudaEngine> mEngine{nullptr}; //!< The TensorRT engine used to run the network

public:
    samplesCommon::OnnxSampleParams mParams;

    nvinfer1::Dims mInputDims; //!< The dimensions of the input to the network.

    nvinfer1::Dims mOutputDims; //!< The dimensions of the output to the network.

    TensorFormat mTensorFormat{TensorFormat::kLINEAR};

    int32_t mDigit;
};

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
    auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));
    if (!builder)
    {
        return false;
    }
    auto const networkFlags = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);

    auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(networkFlags));
    if (!network)
    {
        return false;
    }

    auto config = SampleUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config)
    {
        return false;
    }

    auto parser
        = SampleUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, sample::gLogger.getTRTLogger()));
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

    if (dataWidth == 1)
    {
        config->setFlag(BuilderFlag::kINT8);
        network->getInput(0)->setType(DataType::kINT8);
        network->getOutput(0)->setType(DataType::kINT8);
        samplesCommon::setAllDynamicRanges(network.get(), 127.0F, 127.0F);
    }
    if (dataWidth == 2)
    {
        config->setFlag(BuilderFlag::kFP16);
        network->getInput(0)->setType(DataType::kHALF);
        network->getOutput(0)->setType(DataType::kHALF);
    }

    config->setFlag(BuilderFlag::kGPU_FALLBACK);

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

    SampleUniquePtr<IRuntime> runtime{createInferRuntime(sample::gLogger.getTRTLogger())};
    if (!runtime)
    {
        return false;
    }

    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
        runtime->deserializeCudaEngine(plan->data(), plan->size()), samplesCommon::InferDeleter());
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
bool SampleIOFormats::constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
    SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
    SampleUniquePtr<nvonnxparser::IParser>& parser)
{
    auto parsed = parser->parseFromFile(locateFile(mParams.onnxFileName, mParams.dataDirs).c_str(),
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

    auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    if (!context)
    {
        return false;
    }

    // Create CUDA stream for the execution of this inference.
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    void* bindings[2] = {devInput.get(), devOutput.get()};

    // Asynchronously enqueue the inference work
    if (!context->enqueueV2(bindings, stream, nullptr))
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
//! \brief Reads the digit map from file
//!
bool SampleIOFormats::readDigits(SampleBuffer& buffer, int32_t groundTruthDigit)
{
    int32_t const inputH = buffer.dims.d[2];
    int32_t const inputW = buffer.dims.d[3];

    // Read a random digit file
    std::vector<uint8_t> fileData(inputH * inputW);
    readPGMFile(
        locateFile(std::to_string(groundTruthDigit) + ".pgm", mParams.dataDirs), fileData.data(), inputH, inputW);

    // Print ASCII representation of digit
    for (int32_t i = 0; i < inputH * inputW; i++)
    {
        sample::gLogInfo << (" .:-=+*#%@"[fileData[i] / 26]) << (((i + 1) % inputW) ? "" : "\n");
    }
    sample::gLogInfo << std::endl;

    float* inputBuf = reinterpret_cast<float*>(buffer.buffer);

    for (int32_t i = 0; i < inputH * inputW; i++)
    {
        inputBuf[i] = 1.0F - static_cast<float>(fileData[i] / 255.0F);
    }

    return true;
}

//!
//! \brief Verifies that the output is correct and prints it
//!
template <typename T>
bool SampleIOFormats::verifyOutput(SampleBuffer& outputBuf, int32_t groundTruthDigit) const
{
    T const* prob = reinterpret_cast<T const*>(outputBuf.buffer);

    float val{0.0F};
    float elem{0.0F};
    int32_t idx{0};
    int32_t const kDIGITS = 10;

    for (int32_t i = 0; i < kDIGITS; i++)
    {
        elem = static_cast<float>(prob[i]);
        if (val < elem)
        {
            val = elem;
            idx = i;
        }
    }
    sample::gLogInfo << "Predicted Output: " << idx << std::endl;

    return (idx == groundTruthDigit && val > 0.9F);
}

int32_t calcIndex(SampleBuffer& buffer, int32_t c, int32_t h, int32_t w)
{
    int32_t index;

    if (!buffer.desc.channelPivot)
    {
        index = c / buffer.desc.dims[4] * buffer.desc.dims[2] * buffer.desc.dims[3] * buffer.desc.dims[4]
            + h * buffer.desc.dims[3] * buffer.desc.dims[4] + w * buffer.desc.dims[4] + c % buffer.desc.dims[4];
    }
    else
    {
        index = h * buffer.desc.dims[3] * buffer.desc.dims[2] + w * buffer.desc.dims[3] + c;
    }

    return index;
}

//!
//! \brief Reformats the buffer. Src and dst buffers should be of same datatype and dims.
//!
template <typename T>
void reformat(SampleBuffer& src, SampleBuffer& dst)
{
    if (src.format == dst.format)
    {
        memcpy(dst.buffer, src.buffer, src.getBufferSize());
        return;
    }

    int32_t srcIndex, dstIndex;

    T* srcBuf = reinterpret_cast<T*>(src.buffer);
    T* dstBuf = reinterpret_cast<T*>(dst.buffer);

    for (int32_t c = 0; c < src.dims.d[1]; c++)
    {
        for (int32_t h = 0; h < src.dims.d[2]; h++)
        {
            for (int32_t w = 0; w < src.dims.d[3]; w++)
            {
                srcIndex = calcIndex(src, c, h, w);
                dstIndex = calcIndex(dst, c, h, w);
                dstBuf[dstIndex] = srcBuf[srcIndex];
            }
        }
    }
}

template <typename T>
void convertGoldenData(SampleBuffer& goldenInput, SampleBuffer& dstInput)
{
    SampleBuffer tmpBuf(goldenInput.dims, sizeof(T), goldenInput.format, true);

    float* golden = reinterpret_cast<float*>(goldenInput.buffer);
    T* tmp = reinterpret_cast<T*>(tmpBuf.buffer);

    for (int32_t i = 0; i < goldenInput.desc.getElememtSize(); i++)
    {
        if (std::is_same<T, int8_t>::value)
        {
            tmp[i] = static_cast<T>(1 - ((1.0F - golden[i]) * 255.0F - 128) / 255.0F);
        }
        else
        {
            tmp[i] = static_cast<T>(golden[i]);
        }
    }

    reformat<T>(tmpBuf, dstInput);
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

    return params;
}
//!
//! \brief Prints the help information for running this sample
//!
void printHelpInfo()
{
    std::cout
        << "Usage: ./sample_onnx_mnist [-h or --help] [-d or --datadir=<path to data directory>] [--useDLACore=<int>]"
        << std::endl;
    std::cout << "--help          Display help information" << std::endl;
    std::cout << "--datadir       Specify path to a data directory, overriding the default. This option can be used "
                 "multiple times to add multiple directories. If no data directories are given, the default is to use "
                 "(data/samples/mnist/, data/mnist/)"
              << std::endl;
    std::cout << "--useDLACore=N  Specify a DLA engine for layers that support DLA. Value can range from 0 to n-1, "
                 "where n is the number of DLA engines on the platform."
              << std::endl;
}
//!
//! \brief Used to run the engine build and inference/reference functions
//!
template <typename T>
bool process(SampleIOFormats& sample, sample::Logger::TestAtom const& sampleTest, SampleBuffer& inputBuf,
    SampleBuffer& outputBuf, SampleBuffer& goldenInput)
{
    sample::gLogInfo << "Building and running a GPU inference engine with specified I/O formats." << std::endl;

    inputBuf = SampleBuffer(sample.mInputDims, sizeof(T), sample.mTensorFormat, true);
    outputBuf = SampleBuffer(sample.mOutputDims, sizeof(float), TensorFormat::kLINEAR, false);
    if (!sample.build(sizeof(T)))
    {
        return false;
    }
    convertGoldenData<T>(goldenInput, inputBuf);

    if (!sample.infer(inputBuf, outputBuf))
    {
        return false;
    }

    if (!sample.verifyOutput<T>(outputBuf, sample.mDigit))
    {
        return false;
    }

    return true;
}

bool runFP32Reference(SampleIOFormats& sample, sample::Logger::TestAtom const& sampleTest, SampleBuffer& goldenInput,
    SampleBuffer& goldenOutput)
{
    sample::gLogInfo << "Building and running a FP32 GPU inference to get golden input/output" << std::endl;

    if (!sample.build(sizeof(float)))
    {
        return false;
    }

    goldenInput = SampleBuffer(sample.mInputDims, sizeof(float), TensorFormat::kLINEAR, true);
    goldenOutput = SampleBuffer(sample.mOutputDims, sizeof(float), TensorFormat::kLINEAR, false);

    sample.readDigits(goldenInput, sample.mDigit);

    if (!sample.infer(goldenInput, goldenOutput))
    {
        return false;
    }

    if (!sample.verifyOutput<float>(goldenOutput, sample.mDigit))
    {
        return false;
    }

    return true;
}

//! Specification for a network I/O tensor.
class IOSpec
{
public:
    TensorFormat format;    //!< format
    std::string formatName; //!< name of the format
};

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

    std::vector<IOSpec> vecFP16TensorFmt = {
        IOSpec{TensorFormat::kLINEAR, "kLINEAR"},
        IOSpec{TensorFormat::kCHW2, "kCHW2"},
        IOSpec{TensorFormat::kHWC8, "kHWC8"},
    };
    std::vector<IOSpec> vecINT8TensorFmt = {
        IOSpec{TensorFormat::kLINEAR, "kLINEAR"},
        IOSpec{TensorFormat::kCHW4, "kCHW4"},
        IOSpec{TensorFormat::kCHW32, "kCHW32"},
    };

    SampleBuffer goldenInput, goldenOutput;

    SampleIOFormats sample(params);

    srand(unsigned(time(nullptr)));
    sample.mDigit = rand() % 10;

    sample::gLogInfo << "The test chooses MNIST as the network and recognizes a randomly generated digit" << std::endl;
    sample::gLogInfo
        << "Firstly it runs the FP32 as the golden data, then INT8/FP16 with different formats will be tested"
        << std::endl
        << std::endl;

    if (!runFP32Reference(sample, sampleTest, goldenInput, goldenOutput))
    {
        return sample::gLogger.reportFail(sampleTest);
    }

    // Test FP16 formats
    for (auto spec : vecFP16TensorFmt)
    {
        sample::gLogInfo << "Testing datatype FP16 with format " << spec.formatName << std::endl;
        sample.mTensorFormat = spec.format;
        SampleBuffer inputBuf, outputBuf;

        if (!process<half_float::half>(sample, sampleTest, inputBuf, outputBuf, goldenInput))
        {
            return sample::gLogger.reportFail(sampleTest);
        }
    }

    // Test INT8 formats
    for (auto spec : vecINT8TensorFmt)
    {
        sample::gLogInfo << "Testing datatype INT8 with format " << spec.formatName << std::endl;
        sample.mTensorFormat = spec.format;
        SampleBuffer inputBuf, outputBuf;

        if (!process<int8_t>(sample, sampleTest, inputBuf, outputBuf, goldenInput))
        {
            return sample::gLogger.reportFail(sampleTest);
        }
    }

    return sample::gLogger.reportPass(sampleTest);
}
