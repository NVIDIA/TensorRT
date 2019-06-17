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

//!
//! sampleGoogleNet.cpp
//! This file contains the implementation of the GoogleNet sample. It creates the network using
//! the GoogleNet caffe model.
//! It can be run with the following command line:
//! Command: ./sample_googlenet [-h or --help] [-d=/path/to/data/dir or --datadir=/path/to/data/dir]
//!

#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"

#include "NvCaffeParser.h"
#include "NvInfer.h"
#include <cuda_runtime_api.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>

const std::string gSampleName = "TensorRT.sample_googlenet";

//!
//! \brief  The SampleGoogleNet class implements the GoogleNet sample
//!
//! \details It creates the network using a caffe model
//!
class SampleGoogleNet
{
    template <typename T>
    using SampleUniquePtr = std::unique_ptr<T, samplesCommon::InferDeleter>;

public:
    SampleGoogleNet(const samplesCommon::CaffeSampleParams& params)
        : mParams(params)
    {
    }

    //!
    //! \brief Builds the network engine
    //!
    bool build();

    //!
    //! \brief Runs the TensorRT inference engine for this sample
    //!
    bool infer();

    //!
    //! \brief Used to clean up any state created in the sample class
    //!
    bool teardown();

    samplesCommon::CaffeSampleParams mParams;

private:
    //!
    //! \brief Parses a Caffe model for GoogleNet and creates a TensorRT network
    //!
    void constructNetwork(
        SampleUniquePtr<nvcaffeparser1::ICaffeParser>& parser, SampleUniquePtr<nvinfer1::INetworkDefinition>& network);

    std::shared_ptr<nvinfer1::ICudaEngine> mEngine{nullptr}; //!< The TensorRT engine used to run the network
};

//!
//! \brief Creates the network, configures the builder and creates the network engine
//!
//! \details This function creates the GoogleNet network by parsing the caffe model and builds
//!          the engine that will be used to run GoogleNet (mEngine)
//!
//! \return Returns true if the engine was created successfully and false otherwise
//!
bool SampleGoogleNet::build()
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

    constructNetwork(parser, network);
    builder->setMaxBatchSize(mParams.batchSize);
    builder->setMaxWorkspaceSize(16_MB);
    samplesCommon::enableDLA(builder.get(), mParams.dlaCore);

    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(builder->buildCudaEngine(*network), samplesCommon::InferDeleter());
    if (!mEngine)
        return false;

    return true;
}

//!
//! \brief Uses a caffe parser to create the googlenet Network and marks the
//!        output layers
//!
//! \param network Pointer to the network that will be populated with the googlenet network
//!
//! \param builder Pointer to the engine builder
//!
void SampleGoogleNet::constructNetwork(
    SampleUniquePtr<nvcaffeparser1::ICaffeParser>& parser, SampleUniquePtr<nvinfer1::INetworkDefinition>& network)
{
    const nvcaffeparser1::IBlobNameToTensor* blobNameToTensor = parser->parse(
        mParams.prototxtFileName.c_str(), mParams.weightsFileName.c_str(), *network, nvinfer1::DataType::kFLOAT);

    for (auto& s : mParams.outputTensorNames)
    {
        network->markOutput(*blobNameToTensor->find(s.c_str()));
    }
}

//!
//! \brief Runs the TensorRT inference engine for this sample
//!
//! \details This function is the main execution function of the sample. It allocates the buffer,
//!          sets inputs and executes the engine.
//!
bool SampleGoogleNet::infer()
{
    // Create RAII buffer manager object
    samplesCommon::BufferManager buffers(mEngine, mParams.batchSize);

    auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    if (!context)
    {
        return false;
    }

    // Fetch host buffers and set host input buffers to all zeros
    for (auto& input : mParams.inputTensorNames)
    {
        const auto bufferSize = buffers.size(input);
        if (bufferSize == samplesCommon::BufferManager::kINVALID_SIZE_VALUE)
        {
            gLogError << "input tensor missing: " << input << "\n";
            return EXIT_FAILURE;
        }
        memset(buffers.getHostBuffer(input), 0, bufferSize);
    }

    // Memcpy from host input buffers to device input buffers
    buffers.copyInputToDevice();

    bool status = context->execute(mParams.batchSize, buffers.getDeviceBindings().data());
    if (!status)
    {
        return false;
    }

    // Memcpy from device output buffers to host output buffers
    buffers.copyOutputToHost();

    return true;
}

//!
//! \brief Used to clean up any state created in the sample class
//!
bool SampleGoogleNet::teardown()
{
    //! Clean up the libprotobuf files as the parsing is complete
    //! \note It is not safe to use any other part of the protocol buffers library after
    //! ShutdownProtobufLibrary() has been called.
    nvcaffeparser1::shutdownProtobufLibrary();
    return true;
}

//!
//! \brief Initializes members of the params struct using the command line args
//!
samplesCommon::CaffeSampleParams initializeSampleParams(const samplesCommon::Args& args)
{
    samplesCommon::CaffeSampleParams params;
    if (args.dataDirs.empty())
    {
        params.dataDirs.push_back("data/googlenet/");
        params.dataDirs.push_back("data/samples/googlenet/");
    }
    else
    {
        params.dataDirs = args.dataDirs;
    }

    params.prototxtFileName = locateFile("googlenet.prototxt", params.dataDirs);
    params.weightsFileName = locateFile("googlenet.caffemodel", params.dataDirs);
    params.inputTensorNames.push_back("data");
    params.batchSize = 4;
    params.outputTensorNames.push_back("prob");
    params.dlaCore = args.useDLACore;

    return params;
}
//!
//! \brief Prints the help information for running this sample
//!
void printHelpInfo()
{
    std::cout
        << "Usage: ./sample_googlenet [-h or --help] [-d or --datadir=<path to data directory>] [--useDLACore=<int>]\n";
    std::cout << "--help          Display help information\n";
    std::cout << "--datadir       Specify path to a data directory, overriding the default. This option can be used "
                 "multiple times to add multiple directories. If no data directories are given, the default is to use "
                 "data/samples/googlenet/ and data/googlenet/"
              << std::endl;
    std::cout << "--useDLACore=N  Specify a DLA engine for layers that support DLA. Value can range from 0 to n-1, "
                 "where n is the number of DLA engines on the platform."
              << std::endl;
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
    SampleGoogleNet sample(params);

    gLogInfo << "Building and running a GPU inference engine for GoogleNet" << std::endl;

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

    gLogInfo << "Ran " << argv[0] << " with: " << std::endl;

    std::stringstream ss;

    ss << "Input(s): ";
    for (auto& input : sample.mParams.inputTensorNames)
    {
        ss << input << " ";
    }

    gLogInfo << ss.str() << std::endl;

    ss.str(std::string());

    ss << "Output(s): ";
    for (auto& output : sample.mParams.outputTensorNames)
    {
        ss << output << " ";
    }

    gLogInfo << ss.str() << std::endl;

    return gLogger.reportPass(sampleTest);
}
