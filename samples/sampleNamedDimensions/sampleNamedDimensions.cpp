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

//!
//! sampleNamedDimensions.cpp
//! This file contains the implementation of the named dimensions sample. It creates the network using
//! a synthetic ONNX model with named input dimensions.
//! It can be run with the following command line:
//! Command: ./sample_named_dimensions [-h or --help] [-d=/path/to/data/dir or --datadir=/path/to/data/dir]
//!

// Define TRT entrypoints used in common code
#define DEFINE_TRT_ENTRYPOINTS 1
#define DEFINE_TRT_LEGACY_PARSER_ENTRYPOINT 0

#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"
#include "parserOnnxConfig.h"

#include "NvInfer.h"
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
using namespace nvinfer1;
using samplesCommon::SampleUniquePtr;

std::string const gSampleName = "TensorRT.sample_named_dimensions";

//! \brief  The SampleNamedDimensions class implements a sample with named input dimensions
//!
//! \details It creates the network using an ONNX model
//!
class SampleNamedDimensions
{
public:
    SampleNamedDimensions(samplesCommon::OnnxSampleParams const& params)
        : mParams(params)
        , mEngine(nullptr)
    {
    }

    //! \brief Adds an optimization profile for dynamic shapes
    void setNamedDimension(int32_t dim);

    //!
    //! \brief Function builds the network engine
    //!
    bool build();

    //!
    //! \brief Runs the TensorRT inference engine for this sample
    //!
    bool infer();

private:
    samplesCommon::OnnxSampleParams mParams; //!< The parameters for the sample.

    std::vector<nvinfer1::Dims> mInputDims;  //!< The dimensions of the inputs to the network.
    std::vector<nvinfer1::Dims> mOutputDims; //!< The dimensions of the outputs to the network.

    int32_t mNamedDimension; //!< The value of the named dimension.

    //! Input Tensors.
    std::vector<float> mInput0;
    std::vector<float> mInput1;

    std::shared_ptr<nvinfer1::ICudaEngine> mEngine; //!< The TensorRT engine used to run the network

    //!
    //! \brief Parses a synthetic ONNX model and creates a TensorRT network
    //!
    bool constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
        SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
        SampleUniquePtr<nvonnxparser::IParser>& parser);

    //!
    //! \brief Adds an optimization profile for dynamic shapes
    //!
    void addOptimizationProfile(SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
        SampleUniquePtr<nvinfer1::IBuilder>& builder);

    //!
    //! \brief Reads the input  and stores the result in a managed buffer
    //!
    bool processInput(samplesCommon::BufferManager const& buffers);

    //!
    //! \brief Classifies digits and verify result
    //!
    bool verifyOutput(samplesCommon::BufferManager const& buffers);
};

//!
//! \brief Sets the value of the named input dimension
//!
void SampleNamedDimensions::setNamedDimension(int32_t dim)
{
    mNamedDimension = dim;
}

//!
//! \brief Creates the network, configures the builder and creates the network engine
//!
//! \details This function creates the network definition by parsing the Onnx model and builds
//!          the engine that will be used to run the model (mEngine)
//!
//! \return true if the engine was created successfully and false otherwise
//!
bool SampleNamedDimensions::build()
{
    auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));
    if (!builder)
    {
        return false;
    }

    auto const explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
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

    ASSERT(network->getNbInputs() == 2);
    mInputDims.push_back(network->getInput(0)->getDimensions());
    mInputDims.push_back(network->getInput(1)->getDimensions());
    ASSERT(mInputDims[0].nbDims == 2);
    ASSERT(mInputDims[1].nbDims == 2);

    ASSERT(network->getNbOutputs() == 1);
    mOutputDims.push_back(network->getOutput(0)->getDimensions());
    ASSERT(mOutputDims[0].nbDims == 2);

    // CUDA stream used for profiling by the builder.
    auto profileStream = samplesCommon::makeCudaStream();
    if (!profileStream)
    {
        return false;
    }
    config->setProfileStream(*profileStream);

    addOptimizationProfile(config, builder);

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

    return true;
}

//!
//! \brief Uses ONNX parser to create the ONNX Network and marks the output layers
//!
bool SampleNamedDimensions::constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
    SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
    SampleUniquePtr<nvonnxparser::IParser>& parser)
{
    auto parsed = parser->parseFromFile(locateFile(mParams.onnxFileName, mParams.dataDirs).c_str(),
        static_cast<int32_t>(sample::gLogger.getReportableSeverity()));
    if (!parsed)
    {
        return false;
    }

    return true;
}

//!
//! \brief Adds an optimization profile for dynamic shapes
//!
void SampleNamedDimensions::addOptimizationProfile(SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
    SampleUniquePtr<nvinfer1::IBuilder>& builder)
{
    auto const input0ProfileDims = Dims2(mNamedDimension, mInputDims[0].d[1]);
    auto profile = builder->createOptimizationProfile();
    profile->setDimensions("input0", OptProfileSelector::kMIN, input0ProfileDims);
    profile->setDimensions("input0", OptProfileSelector::kMAX, input0ProfileDims);
    profile->setDimensions("input0", OptProfileSelector::kOPT, input0ProfileDims);

    auto input1ProfileDims = Dims2(mNamedDimension, mInputDims[1].d[1]);
    profile->setDimensions("input1", OptProfileSelector::kMIN, input1ProfileDims);
    profile->setDimensions("input1", OptProfileSelector::kMAX, input1ProfileDims);
    profile->setDimensions("input1", OptProfileSelector::kOPT, input1ProfileDims);

    config->addOptimizationProfile(profile);
}

//!
//! \brief Runs the TensorRT inference engine for this sample
//!
//! \details This function is the main execution function of the sample. It allocates the buffer,
//!          sets inputs and executes the engine.
//!
bool SampleNamedDimensions::infer()
{
    // Create RAII buffer manager object
    samplesCommon::BufferManager buffers(mEngine);

    auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    if (!context)
    {
        return false;
    }

    // Read the input data into the managed buffers
    ASSERT(mParams.inputTensorNames.size() == 2);
    if (!processInput(buffers))
    {
        return false;
    }

    // Memcpy from host input buffers to device input buffers
    buffers.copyInputToDevice();

    bool status = context->executeV2(buffers.getDeviceBindings().data());
    if (!status)
    {
        return false;
    }

    // Memcpy from device output buffers to host output buffers
    buffers.copyOutputToHost();

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
bool SampleNamedDimensions::processInput(samplesCommon::BufferManager const& buffers)
{
    int32_t const input0H = mNamedDimension;
    int32_t const input0W = mInputDims[0].d[1];
    int32_t const input1H = mNamedDimension;
    int32_t const input1W = mInputDims[1].d[1];

    // Generate random input
    mInput0.resize(input0H * input0W);
    mInput1.resize(input1H * input1W);
    std::default_random_engine generator(static_cast<uint32_t>(time(nullptr)));
    std::uniform_real_distribution<float> unif_real_distr(-10., 10.);

    sample::gLogInfo << "Input0:\n";
    for (int32_t i = 0; i < input0H * input0W; i++)
    {
        mInput0[i] = unif_real_distr(generator);
        sample::gLogInfo << mInput0[i] << (((i + 1) % input0W) ? " " : "\n");
    }
    sample::gLogInfo << std::endl;

    sample::gLogInfo << "Input1:\n";
    for (int32_t i = 0; i < input1H * input1W; i++)
    {
        mInput1[i] = unif_real_distr(generator);
        sample::gLogInfo << mInput1[i] << (((i + 1) % input1W) ? " " : "\n");
    }
    sample::gLogInfo << std::endl;

    auto* hostInput0Buffer = static_cast<float*>(buffers.getHostBuffer(mParams.inputTensorNames[0]));
    std::copy(mInput0.begin(), mInput0.begin() + input0H * input0W, hostInput0Buffer);

    auto* hostInput1Buffer = static_cast<float*>(buffers.getHostBuffer(mParams.inputTensorNames[1]));
    std::copy(mInput1.begin(), mInput1.begin() + input1H * input1W, hostInput1Buffer);

    return true;
}

//!
//! \brief Verify the result of concatenation
//!
//! \return whether the concatenated tesnor matches reference
//!
bool SampleNamedDimensions::verifyOutput(samplesCommon::BufferManager const& buffers)
{
    int32_t const outputH = 2 * mNamedDimension;
    int32_t const outputW = mOutputDims[0].d[1];
    int32_t const outputSize = outputH * outputW;

    auto* output = static_cast<float*>(buffers.getHostBuffer(mParams.outputTensorNames[0]));

    sample::gLogInfo << "Output:\n";
    for (int32_t i = 0; i < outputSize; i++)
    {
        sample::gLogInfo << output[i] << (((i + 1) % outputW) ? " " : "\n");
    }
    sample::gLogInfo << std::endl;

    mInput0.insert(mInput0.end(), mInput1.begin(), mInput1.end());

    for (int32_t i = 0; i < outputH * outputW; i++)
    {
        auto const reference_value = i > outputSize / 2 ? mInput1[i - outputSize / 2] : mInput0[i];
        if (fabs(output[i] - reference_value) > std::numeric_limits<float>::epsilon())
        {
            return false;
        }
    }
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
        params.dataDirs.push_back("trt/samples/sampleNamedDimensions/");
    }
    else // Use the data directory provided by the user
    {
        params.dataDirs = args.dataDirs;
    }
    params.onnxFileName = "concat_layer.onnx";
    params.inputTensorNames.push_back("input0");
    params.inputTensorNames.push_back("input1");
    params.outputTensorNames.push_back("output");

    return params;
}

//!
//! \brief Prints the help information for running this sample
//!
void printHelpInfo()
{
    std::cout
        << "Usage: ./sample_named_dimensions [-h or --help] [-d or --datadir=<path to data directory>]"
        << std::endl;
    std::cout << "--help          Display help information" << std::endl;
    std::cout << "--datadir       Specify path to a data directory, overriding the default. This option can be used "
                 "multiple times to add multiple directories. If no data directories are given, the default is to use "
                 "(trt/samples/sampleNamedDimensions)"
              << std::endl;
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

    SampleNamedDimensions sample(initializeSampleParams(args));

    sample::gLogInfo << "Building and running a GPU inference engine for synthetic ONNX model" << std::endl;

    sample.setNamedDimension(2);

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
