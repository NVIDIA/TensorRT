/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

//! \file sampleSafeMNISTBuild.cpp
//! \brief This file contains the implementation of the MNIST sample.
//!
//! It builds a TensorRT safe engine by importing a trained MNIST ONNX model.
//! It can be run with the following command line:
//! Command: ./sample_mnist_safe_build [-h or --help] [-d=/path/to/data/dir or --datadir=/path/to/data/dir]

#define DEFINE_TRT_ENTRYPOINTS 1
#define DEFINE_TRT_ONNX_PARSER_ENTRYPOINT 0
#define DEFINE_TRT_BUILDER_ENTRYPOINT 0
#define DEFINE_TRT_REFITTER_ENTRYPOINT 0
#define DEFINE_TRT_RUNTIME_ENTRYPOINT 0
#define DEFINE_TRT_LEGACY_PARSER_ENTRYPOINT 0
#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"
#include "parserOnnxConfig.h"
#include "sampleUtils.h"

#include "NvInfer.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

using namespace nvinfer1;

const std::string gSampleName = "TensorRT.sample_mnist_safe_build";

//!
//! \brief The SampleSafeMNISTBuildArgs struct stores the additional arguments required by the sample
//!
struct SampleSafeMNISTBuildArgs : public samplesCommon::Args
{
    std::string engineFileName{"safe_mnist.engine"};
    bool verbose{false};
    std::string remoteAutoTuningConfig{};
};

//!
//! \brief This function parses arguments specific to the sample
//!
bool parseSampleSafeMNISTBuildArgs(SampleSafeMNISTBuildArgs& args, int32_t argc, char* argv[])
{
    for (int32_t i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];

        // Check for help flag first
        if (arg == "--help" || arg == "-h")
        {
            args.help = true;
        }
        else if (arg == "--verbose")
        {
            args.verbose = true;
        }
        // Check for flags with values (simple parsing)
        else if (arg.find("--saveEngine=") == 0)
        {
            std::string value = arg.substr(13); // Remove "--saveEngine="
            if (value.empty())
            {
                sample::gLogError << "Engine filename cannot be empty" << std::endl;
                return false;
            }
            args.engineFileName = value;
        }
        else if (arg.find("--remoteAutoTuningConfig=") == 0)
        {
            std::string value = arg.substr(25); // Remove "--remoteAutoTuningConfig="
            if (value.empty())
            {
                sample::gLogError << "Remote auto tuning config cannot be empty" << std::endl;
                return false;
            }
            args.remoteAutoTuningConfig = value;
        }
        else if (arg.find("--datadir=") == 0 || arg.find("-d=") == 0)
        {
            std::string value;
            if (arg.find("--datadir=") == 0)
            {
                value = arg.substr(10); // Remove "--datadir="
            }
            else
            {
                value = arg.substr(3); // Remove "-d="
            }

            if (value.empty())
            {
                sample::gLogError << "Data directory path cannot be empty" << std::endl;
                return false;
            }

            std::string dirPath = value;
            if (!dirPath.empty() && dirPath.back() != '/')
            {
                dirPath += '/';
            }
            args.dataDirs.push_back(dirPath);
        }
        else
        {
            sample::gLogError << "Invalid Argument: " << argv[i] << std::endl;
            return false;
        }
    }

    return true;
}

//!
//! \brief The SampleSafeMNISTBuildParams struct stores the additional parameters required by the sample
//!
struct SampleSafeMNISTBuildParams : public samplesCommon::OnnxSampleParams
{
    std::string engineFileName{"safe_mnist.engine"};
    std::string remoteAutoTuningConfig{};
};

//!
//! \brief Initialize members of the params struct using the command line args.
//!
SampleSafeMNISTBuildParams initializeSampleParams(const SampleSafeMNISTBuildArgs& args)
{
    SampleSafeMNISTBuildParams params;
    if (args.dataDirs.empty()) // Use default directories if user hasn't provided directory paths.
    {
        params.dataDirs.push_back("data/mnist/");
        params.dataDirs.push_back("data/samples/mnist/");
    }
    else // Use the data directory provided by the user.
    {
        params.dataDirs = args.dataDirs;
    }

    params.onnxFileName = "safe_mnist.onnx";
    params.engineFileName = args.engineFileName;
    params.remoteAutoTuningConfig = args.remoteAutoTuningConfig;

    return params;
}

//!
//! \brief  The SampleSafeMNIST class implements the MNIST sample.
//!
//! \details It creates the network using a trained ONNX MNIST classification model.
//!
class SampleSafeMNIST
{
public:
    SampleSafeMNIST(const SampleSafeMNISTBuildParams& params)
        : mParams(params)
    {
    }

    //!
    //! \brief Builds the network engine.
    //!
    bool build();

private:
    //!
    //! \brief Uses an ONNX parser to create the MNIST Network and marks the
    //!        output layers.
    //!
    bool constructNetwork(std::unique_ptr<nvonnxparser::IParser>& parser);

    SampleSafeMNISTBuildParams mParams; //!< The parameters for the sample.

    nvinfer1::Dims mInputDims; //!< The dimensions of the input to the network.
};

//!
//! \brief Creates the network, configures the builder and creates the network engine.
//!
//! \details This function creates the MNIST network by parsing the ONNX model and builds
//!          the engine that will be used to run MNIST.
//!
//! \return true if the engine was created successfully and false otherwise.
//!
bool SampleSafeMNIST::build()
{
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));
    if (!builder)
    {
        return false;
    }

    NetworkDefinitionCreationFlags flags = (1 << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH))
        | (1 << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kSTRONGLY_TYPED));
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(flags));
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

    auto constructed = constructNetwork(parser);
    if (!constructed)
    {
        return false;
    }
    config->setEngineCapability(EngineCapability::kSAFETY);
    config->setFlag(BuilderFlag::kGPU_FALLBACK);

    // Set remote auto tuning config if provided
    if (!mParams.remoteAutoTuningConfig.empty())
    {
        config->setRemoteAutoTuningConfig(mParams.remoteAutoTuningConfig.c_str());
    }

    auto buffer = std::unique_ptr<nvinfer1::IHostMemory>(builder->buildSerializedNetwork(*network, *config));

    if (!buffer)
    {
        return false;
    }

    ASSERT(network->getNbInputs() == 1);
    mInputDims = network->getInput(0)->getDimensions();
    ASSERT(mInputDims.nbDims == 4);

    // Save the engine
    std::string engineFile = mParams.engineFileName;
    std::ofstream file(engineFile, std::ios::binary);
    if (!file)
    {
        sample::gLogError << "Failed to open file to save engine: " << engineFile << std::endl;
        return false;
    }
    file.write(reinterpret_cast<const char*>(buffer->data()), buffer->size());
    file.close();

    return true;
}

//!
//! \brief Uses an ONNX parser to create the MNIST Network and marks the
//!        output layers.
//!
//! \param network Pointer to the network that will be populated with the MNIST network.
//!
//! \param builder Pointer to the engine builder.
//!
bool SampleSafeMNIST::constructNetwork(std::unique_ptr<nvonnxparser::IParser>& parser)
{
    return parser->parseFromFile(samplesCommon::locateFile(mParams.onnxFileName, mParams.dataDirs).c_str(),
        static_cast<int32_t>(sample::gLogger.getReportableSeverity()));
}

//!
//! \brief Prints the help information for running this sample.
//!
void printHelpInfo()
{
    std::cout << "Usage: ./sample_mnist_safe_build [-h or --help] [--datadir=<path to data directory>]\n";
    std::cout << "--help or -h    Display help information\n";
    std::cout << "--datadir       Specify path to a data directory, overriding the default. This option can be used "
                 "multiple times to add multiple directories. If no data directories are given, the default is to use "
                 "(data/samples/mnist/, data/mnist/)"
              << std::endl;
    std::cout << "--verbose       Use verbose logging.\n";
    std::cout << "--saveEngine    Save the serialized engine to the file (default = safe_mnist.engine).\n";
    std::cout << "--remoteAutoTuningConfig  Set the remote auto tuning config. Format: "
                 "protocol://username[:password]@hostname[:port]?param1=value1&param2=value2\n";
    std::cout
        << "                Example: "
           "ssh://user:pass@192.0.2.100:22?remote_exec_path=/opt/tensorrt/bin&remote_lib_path=/opt/tensorrt/lib\n";
}

int main(int argc, char** argv)
{
    SampleSafeMNISTBuildArgs args;
    bool argsOK = parseSampleSafeMNISTBuildArgs(args, argc, argv);
    if (!argsOK)
    {
        printHelpInfo();
        return EXIT_FAILURE;
    }
    if (args.help)
    {
        printHelpInfo();
        return EXIT_SUCCESS;
    }
    if (args.verbose)
    {
        sample::setReportableSeverity(ILogger::Severity::kVERBOSE);
    }

    // Log remoteAutoTuningConfig usage
    if (!args.remoteAutoTuningConfig.empty())
    {
        sample::gLogInfo << "Remote auto tuning config specified: "
                         << sample::sanitizeRemoteAutoTuningConfig(args.remoteAutoTuningConfig) << std::endl;
        sample::gLogInfo << "This is a safety sample and will build in remote mode automatically." << std::endl;
    }

    if (!samplesCommon::isSmSafe())
    {
        sample::gLogInfo << "Skip safe mode test on unsupported platforms." << std::endl;
        return EXIT_SUCCESS;
    }

    // Create sanitized argv for logging to avoid exposing credentials in test reports
    auto sanitizedArgs = sample::sanitizeArgv(argc, argv);
    std::vector<char const*> sanitizedArgv;
    sanitizedArgv.reserve(sanitizedArgs.size());
    for (auto const& s : sanitizedArgs)
    {
        sanitizedArgv.push_back(s.c_str());
    }

    auto sampleTest
        = sample::gLogger.defineTest(gSampleName, static_cast<int32_t>(sanitizedArgv.size()), sanitizedArgv.data());

    sample::gLogger.reportTestStart(sampleTest);

    SampleSafeMNISTBuildParams params = initializeSampleParams(args);

    SampleSafeMNIST sample(params);
    sample::gLogInfo << "Building a GPU inference engine for MNIST" << std::endl;

    if (!sample.build())
    {
        return sample::gLogger.reportFail(sampleTest);
    }

    return sample::gLogger.reportPass(sampleTest);
}
