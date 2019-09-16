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

//! \file SampleReformatFreeIO.cpp
//! \brief This file contains the implementation of the reformat free I/O sample.
//!
//! It builds a TensorRT engine by constructing a conv layer. It uses the engine to run
//! a conv layer with random input and weights.
//! The goal of this sample is to show how to specify allowed I/O formats.
//! It can be run with the following command line:
//! Command: ./sample_reformat_free_io

#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "half.h"
#include "logger.h"

#include "NvCaffeParser.h"
#include "NvInfer.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iostream>
#include <sstream>

#include <array>
#include <random>
#include <string>
#include <utility>
#include <vector>

const std::string gSampleName = "TensorRT.sample_reformat_free_io";

int divUp(int a, int b)
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

    BufferDesc(nvinfer1::Dims dims, int dataWidth, TensorFormat format)
    {
        this->dataWidth = dataWidth;
        if (format == TensorFormat::kLINEAR)
        {
            this->dims[0] = dims.d[0];
            this->dims[1] = dims.d[1];
            this->dims[2] = dims.d[2];
            this->dims[3] = 1;
        }
        else if (format == TensorFormat::kCHW2)
        {
            this->dims[0] = divUp(dims.d[0], 2);
            this->dims[1] = dims.d[1];
            this->dims[2] = dims.d[2];
            this->dims[3] = 2;
            this->scalarPerVector = 2;
        }
        else if (format == TensorFormat::kCHW4)
        {
            this->dims[0] = divUp(dims.d[0], 4);
            this->dims[1] = dims.d[1];
            this->dims[2] = dims.d[2];
            this->dims[3] = 4;
            this->scalarPerVector = 4;
        }
        else if (format == TensorFormat::kCHW32)
        {
            this->dims[0] = divUp(dims.d[0], 32);
            this->dims[1] = dims.d[1];
            this->dims[2] = dims.d[2];
            this->dims[3] = 32;
            this->scalarPerVector = 32;
        }
        else if (format == TensorFormat::kHWC8)
        {
            this->dims[0] = dims.d[1];
            this->dims[1] = dims.d[2];
            this->dims[2] = divUp(dims.d[0], 8) * 8;
            this->dims[3] = 1;
            this->scalarPerVector = 8;
            this->channelPivot = true;
        }
    }

    // [(C+x-1)/x][H][W][x]
    // or
    // [H][W][(C+x-1)/x*x][1]
    int dims[4] = {1, 1, 1, 1};
    int dataWidth = 1;
    int scalarPerVector = 1;

    bool channelPivot = false;

    int getElememtSize()
    {
        return dims[0] * dims[1] * dims[2] * dims[3];
    }
    int getBufferSize()
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
    }

    SampleBuffer(nvinfer1::Dims dims, int dataWidth, TensorFormat format)
        : dims(dims)
        , dataWidth(dataWidth)
        , format(format)
        , desc(dims, dataWidth, format)
    {
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

    int dataWidth{1};

    TensorFormat format{TensorFormat::kLINEAR};

    BufferDesc desc;

    uint8_t* buffer = nullptr;

    int getBufferSize()
    {
        return desc.getBufferSize();
    }
};

//!
//! \brief  The SampleReformatFreeIO class implements the reformat free I/O sample
//!
//! \details It creates the network using a single conv layer
//!
class SampleReformatFreeIO
{
    template <typename T>
    using SampleUniquePtr = std::unique_ptr<T, samplesCommon::InferDeleter>;

public:
    SampleReformatFreeIO(const samplesCommon::CaffeSampleParams& params)
        : mParams(params)
    {
    }

    //!
    //! \brief Builds the network engine
    //!
    bool build(int dataWidth);

    //!
    //! \brief Runs the TensorRT inference engine for this sample
    //!
    bool infer(SampleBuffer& inputBuf, SampleBuffer& outputBuf);

    //!
    //! \brief Used to clean up any state created in the sample class
    //!
    bool teardown();

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
    bool readDigits(SampleBuffer& buffer, int groundTruthDigit);

    //!
    //! \brief Verifies that the output is correct and prints it
    //!
    template <typename T>
    bool verifyOutput(SampleBuffer& outputBuf, int groundTruthDigit) const;

private:
    //!
    //! \brief uses a Caffe parser to create the single layer Network and marks the
    //!        output layers
    //!
    void constructNetwork(
        SampleUniquePtr<nvcaffeparser1::ICaffeParser>& parser, SampleUniquePtr<nvinfer1::INetworkDefinition>& network);

    std::shared_ptr<nvinfer1::ICudaEngine> mEngine{nullptr}; //!< The TensorRT engine used to run the network

public:
    samplesCommon::CaffeSampleParams mParams;

    nvinfer1::Dims mInputDims; //!< The dimensions of the input to the network.

    nvinfer1::Dims mOutputDims; //!< The dimensions of the output to the network.

    TensorFormat mTensorFormat{TensorFormat::kLINEAR};

    SampleUniquePtr<nvcaffeparser1::IBinaryProtoBlob> mMeanBlob;

    int mDigit;
};

//!
//! \brief Creates the network, configures the builder and creates the network engine
//!
//! \details This function creates the single layer network by manual insertion and builds
//!          the engine
//!
//! \return Returns true if the engine was created successfully and false otherwise
//!
bool SampleReformatFreeIO::build(int dataWidth)
{
    auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger.getTRTLogger()));
    if (!builder)
    {
        return false;
    }

    auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetwork());
    if (!network)
    {
        return false;
    }

    auto parser = SampleUniquePtr<nvcaffeparser1::ICaffeParser>(nvcaffeparser1::createCaffeParser());
    if (!parser)
    {
        return false;
    }

    auto config = SampleUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config)
    {
        return false;
    }

    constructNetwork(parser, network);

    network->getInput(0)->setAllowedFormats(static_cast<TensorFormats>(1 << static_cast<int>(mTensorFormat)));
    network->getOutput(0)->setAllowedFormats(static_cast<TensorFormats>(1 << static_cast<int>(mTensorFormat)));

    builder->setMaxBatchSize(1);

    mEngine.reset();

    config->setMaxWorkspaceSize(256_MiB);
    if (dataWidth == 1)
    {
        config->setFlag(BuilderFlag::kINT8);
        network->getInput(0)->setType(DataType::kINT8);
        network->getOutput(0)->setType(DataType::kINT8);
    }
    if (dataWidth == 2)
    {
        config->setFlag(BuilderFlag::kFP16);
        network->getInput(0)->setType(DataType::kHALF);
        network->getOutput(0)->setType(DataType::kHALF);
    }

    config->setFlag(BuilderFlag::kGPU_FALLBACK);
    config->setFlag(BuilderFlag::kSTRICT_TYPES);

    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
        builder->buildEngineWithConfig(*network, *config), samplesCommon::InferDeleter());

    if (!mEngine)
        return false;

    assert(network->getNbInputs() == 1);
    mInputDims = network->getInput(0)->getDimensions();
    assert(mInputDims.nbDims == 3);

    assert(network->getNbOutputs() == 1);
    mOutputDims = network->getOutput(0)->getDimensions();
    assert(mOutputDims.nbDims == 3);

    return true;
}

//!
//! \brief Uses a caffe parser to create the single layer Network and marks the
//!        output layers
//!
void SampleReformatFreeIO::constructNetwork(
    SampleUniquePtr<nvcaffeparser1::ICaffeParser>& parser, SampleUniquePtr<nvinfer1::INetworkDefinition>& network)
{
    const nvcaffeparser1::IBlobNameToTensor* blobNameToTensor = parser->parse(
        mParams.prototxtFileName.c_str(), mParams.weightsFileName.c_str(), *network, nvinfer1::DataType::kFLOAT);

    for (auto& s : mParams.outputTensorNames)
    {
        network->markOutput(*blobNameToTensor->find(s.c_str()));
    }

    nvinfer1::Dims inputDims = network->getInput(0)->getDimensions();
    // add mean subtraction to the beginning of the network
    mMeanBlob
        = SampleUniquePtr<nvcaffeparser1::IBinaryProtoBlob>(parser->parseBinaryProto(mParams.meanFileName.c_str()));
    nvinfer1::Weights meanWeights{nvinfer1::DataType::kFLOAT, mMeanBlob->getData(), inputDims.d[1] * inputDims.d[2]};
    // For this sample, a large range based on the mean data is chosen and applied to the entire network.
    // The preferred method is use scales computed based on a representative data set
    // and apply each one individually based on the tensor. The range here is large enough for the
    // network, but is chosen for example purposes only.
    float maxMean
        = samplesCommon::getMaxValue(static_cast<const float*>(meanWeights.values), samplesCommon::volume(inputDims));

    auto mean = network->addConstant(nvinfer1::Dims3(1, inputDims.d[1], inputDims.d[2]), meanWeights);
    auto meanSub = network->addElementWise(*network->getInput(0), *mean->getOutput(0), ElementWiseOperation::kSUB);
    network->getLayer(0)->setInput(0, *meanSub->getOutput(0));
    samplesCommon::setAllTensorScales(network.get(), maxMean / maxMean * 128, 128);
}

//!
//! \brief Runs the TensorRT inference engine for this sample
//!
//! \details This function is the main execution function of the sample. It allocates
//!          the buffer, sets inputs, executes the engine, and verifies the output.
//!
bool SampleReformatFreeIO::infer(SampleBuffer& inputBuf, SampleBuffer& outputBuf)
{
    const auto devInput = mallocCudaMem<uint8_t>(inputBuf.getBufferSize());
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
    if (!context->enqueue(1, bindings, stream, nullptr))
    {
        return false;
    }

    // Wait for the work in the stream to complete
    cudaStreamSynchronize(stream);

    // Release stream
    cudaStreamDestroy(stream);

    CHECK(cudaMemcpy(outputBuf.buffer, devOutput.get(), outputBuf.getBufferSize(), cudaMemcpyDeviceToHost));

    return true;
}

//!
//! \brief Used to clean up any state created in the sample class
//!
bool SampleReformatFreeIO::teardown()
{
    //! Clean up the libprotobuf files as the parsing is complete
    //! \note It is not safe to use any other part of the protocol buffers library after
    //! ShutdownProtobufLibrary() has been called.
    nvcaffeparser1::shutdownProtobufLibrary();
    return true;
}

//!
//! \brief Reads the digit map from file
//!
bool SampleReformatFreeIO::readDigits(SampleBuffer& buffer, int groundTruthDigit)
{
    const int inputH = buffer.dims.d[1];
    const int inputW = buffer.dims.d[2];

    // Read a random digit file
    std::vector<uint8_t> fileData(inputH * inputW);
    readPGMFile(
        locateFile(std::to_string(groundTruthDigit) + ".pgm", mParams.dataDirs), fileData.data(), inputH, inputW);

    // Print ASCII representation of digit
    gLogInfo << "Input:\n";
    for (int i = 0; i < inputH * inputW; i++)
    {
        gLogInfo << (" .:-=+*#%@"[fileData[i] / 26]) << (((i + 1) % inputW) ? "" : "\n");
    }
    gLogInfo << std::endl;

    float* inputBuf = reinterpret_cast<float*>(buffer.buffer);

    for (int i = 0; i < inputH * inputW; i++)
    {
        inputBuf[i] = float(fileData[i]);
    }

    return true;
}

//!
//! \brief Verifies that the output is correct and prints it
//!
template <typename T>
bool SampleReformatFreeIO::verifyOutput(SampleBuffer& outputBuf, int groundTruthDigit) const
{
    const T* prob = reinterpret_cast<const T*>(outputBuf.buffer);

    // Print histogram of the output distribution
    gLogInfo << "Output:\n";
    float val{0.0f};
    float elem{0.0f};
    int idx{0};
    const int kDIGITS = 10;

    for (int i = 0; i < kDIGITS; i++)
    {
        elem = static_cast<float>(prob[i]);
        if (val < elem)
        {
            val = elem;
            idx = i;
        }

        gLogInfo << i << ": " << std::string(int(std::floor(elem * 10 + 0.5f)), '*') << "\n";
    }
    gLogInfo << std::endl;

    return (idx == groundTruthDigit && val > 0.9f);
}

int calcIndex(SampleBuffer& buffer, int c, int h, int w)
{
    int index;

    if (!buffer.desc.channelPivot)
    {
        index = c / buffer.desc.dims[3] * buffer.desc.dims[1] * buffer.desc.dims[2] * buffer.desc.dims[3]
            + h * buffer.desc.dims[2] * buffer.desc.dims[3] + w * buffer.desc.dims[3] + c % buffer.desc.dims[3];
    }
    else
    {
        index = h * buffer.desc.dims[2] * buffer.desc.dims[1] + w * buffer.desc.dims[2]
            + c / buffer.desc.scalarPerVector * buffer.desc.scalarPerVector + c % buffer.desc.scalarPerVector;
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

    int srcIndex, dstIndex;

    T* srcBuf = reinterpret_cast<T*>(src.buffer);
    T* dstBuf = reinterpret_cast<T*>(dst.buffer);

    for (int c = 0; c < src.dims.d[0]; c++)
    {
        for (int h = 0; h < src.dims.d[1]; h++)
        {
            for (int w = 0; w < src.dims.d[2]; w++)
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
    SampleBuffer tmpBuf(goldenInput.dims, sizeof(T), goldenInput.format);

    float* golden = reinterpret_cast<float*>(goldenInput.buffer);
    T* tmp = reinterpret_cast<T*>(tmpBuf.buffer);

    for (int i = 0; i < goldenInput.desc.getElememtSize(); i++)
    {
        if (std::is_same<T, int8_t>::value)
        {
            tmp[i] = static_cast<T>(golden[i] - 128);
        }
        else
        {
            tmp[i] = static_cast<T>(golden[i]);
        }
    }

    reformat<T>(tmpBuf, dstInput);
}

//!
//! \brief Used to randomly initialize buffers
//!
void randomInitBuff(SampleBuffer& buffer)
{
    srand(time(NULL));

    float* tmpBuf = reinterpret_cast<float*>(buffer.buffer);

    for (int i = 0; i < buffer.getBufferSize() / buffer.dataWidth; i++)
    {
        tmpBuf[i] = static_cast<float>((rand() % 256) - 128);
    }
}

//!
//! \brief Initializes members of the params struct using the command line args
//!
samplesCommon::CaffeSampleParams initializeSampleParams(const samplesCommon::Args& args)
{
    samplesCommon::CaffeSampleParams params;
    if (args.dataDirs.empty())
    {
        params.dataDirs.push_back("data/mnist/");
        params.dataDirs.push_back("data/samples/mnist/");
    }
    else
    {
        params.dataDirs = args.dataDirs;
    }

    params.prototxtFileName = locateFile("mnist.prototxt", params.dataDirs);
    params.weightsFileName = locateFile("mnist.caffemodel", params.dataDirs);
    params.meanFileName = locateFile("mnist_mean.binaryproto", params.dataDirs);
    params.inputTensorNames.push_back("data");
    params.batchSize = 1;
    params.outputTensorNames.push_back("prob");
    params.dlaCore = args.useDLACore;

    return params;
}
//!
//! \brief Prints the help information for running this sample
//!
void printHelpInfo()
{
    std::cout << "Usage: ./sample_reformat_free_io [-h or --help] [-d or --datadir=<path to data directory>] "
                 "[--useDLACore=<int>]\n";
    std::cout << "--help          Display help information\n";
    std::cout << "--datadir       Specify path to a data directory, overriding the default. This option can be used "
                 "multiple times to add multiple directories. If no data directories are given, the default is to use "
                 "data/samples/googlenet/ and data/googlenet/"
              << std::endl;
    std::cout << "--useDLACore=N  Specify a DLA engine for layers that support DLA. Value can range from 0 to n-1, "
                 "where n is the number of DLA engines on the platform."
              << std::endl;
}

//!
//! \brief Used to run the engine build and inference/reference functions
//!
template <typename T>
int process(SampleReformatFreeIO& sample, const Logger::TestAtom& sampleTest, SampleBuffer& inputBuf,
    SampleBuffer& outputBuf, SampleBuffer& goldenInput, SampleBuffer& goldenOutput)
{
    gLogInfo << "Building and running a GPU inference engine for reformat free I/O" << std::endl;

    inputBuf = SampleBuffer(sample.mInputDims, sizeof(T), sample.mTensorFormat);
    outputBuf = SampleBuffer(sample.mOutputDims, sizeof(T), sample.mTensorFormat);

    if (!sample.build(sizeof(T)))
    {
        return gLogger.reportFail(sampleTest);
    }

    convertGoldenData<T>(goldenInput, inputBuf);

    if (!sample.infer(inputBuf, outputBuf))
    {
        return gLogger.reportFail(sampleTest);
    }

    SampleBuffer linearOutputBuf(sample.mOutputDims, sizeof(T), TensorFormat::kLINEAR);

    reformat<T>(outputBuf, linearOutputBuf);

    if (!sample.verifyOutput<T>(linearOutputBuf, sample.mDigit))
    {
        return gLogger.reportFail(sampleTest);
    }

    return 0;
}

int runFP32Reference(SampleReformatFreeIO& sample, const Logger::TestAtom& sampleTest, SampleBuffer& goldenInput,
    SampleBuffer& goldenOutput)
{
    gLogInfo << "Building and running a FP32 GPU inference to get golden input/output" << std::endl;

    if (!sample.build(sizeof(float)))
    {
        return gLogger.reportFail(sampleTest);
    }

    goldenInput = SampleBuffer(sample.mInputDims, sizeof(float), TensorFormat::kLINEAR);
    goldenOutput = SampleBuffer(sample.mOutputDims, sizeof(float), TensorFormat::kLINEAR);

    // randomInitBuff(goldenInput);
    sample.readDigits(goldenInput, sample.mDigit);

    if (!sample.infer(goldenInput, goldenOutput))
    {
        return gLogger.reportFail(sampleTest);
    }

    if (!sample.verifyOutput<float>(goldenOutput, sample.mDigit))
    {
        return gLogger.reportFail(sampleTest);
    }

    return 0;
}

int main(int argc, char** argv)
{
    samplesCommon::Args args;
    bool argsOK = samplesCommon::parseArgs(args, argc, argv);
    if (!argsOK)
    {
        gLogError << "Invalid arguments" << std::endl;
        printHelpInfo();
        return EXIT_FAILURE;
    }
    if (args.help)
    {
        printHelpInfo();
        return EXIT_SUCCESS;
    }

    auto sampleTest = gLogger.defineTest(gSampleName, argc, argv);

    gLogger.reportTestStart(sampleTest);

    samplesCommon::CaffeSampleParams params = initializeSampleParams(args);

    std::vector<std::pair<TensorFormat, std::string>> vecFP16TensorFmt = {
        std::make_pair(TensorFormat::kLINEAR, "kLINEAR"),
        std::make_pair(TensorFormat::kCHW2, "kCHW2"),
        std::make_pair(TensorFormat::kHWC8, "kHWC8"),
    };
    std::vector<std::pair<TensorFormat, std::string>> vecINT8TensorFmt = {
        std::make_pair(TensorFormat::kLINEAR, "kLINEAR"),
        std::make_pair(TensorFormat::kCHW4, "kCHW4"),
        std::make_pair(TensorFormat::kCHW32, "kCHW32"),
    };

    SampleBuffer goldenInput, goldenOutput;

    SampleReformatFreeIO sample(params);

    srand(unsigned(time(nullptr)));
    sample.mDigit = rand() % 10;

    gLogInfo << "The test chooses MNIST as the network and recognizes a randomly generated digit" << std::endl;
    gLogInfo << "Firstly it runs the FP32 as the golden data, then INT8/FP16 with different formats will be tested"
             << std::endl
             << std::endl;

    runFP32Reference(sample, sampleTest, goldenInput, goldenOutput);

    // Test INT8 formats
    for (auto elem : vecINT8TensorFmt)
    {
        gLogInfo << "Testing datatype INT8 with format " << elem.second << std::endl;
        sample.mTensorFormat = elem.first;
        SampleBuffer inputBuf, outputBuf;

        process<int8_t>(sample, sampleTest, inputBuf, outputBuf, goldenInput, goldenOutput);
    }

    // Test FP16 formats
    for (auto elem : vecFP16TensorFmt)
    {
        gLogInfo << "Testing datatype FP16 with format " << elem.second << std::endl;
        sample.mTensorFormat = elem.first;
        SampleBuffer inputBuf, outputBuf;

        process<half_float::half>(sample, sampleTest, inputBuf, outputBuf, goldenInput, goldenOutput);
    }

    if (!sample.teardown())
    {
        return gLogger.reportFail(sampleTest);
    }

    return gLogger.reportPass(sampleTest);
}
