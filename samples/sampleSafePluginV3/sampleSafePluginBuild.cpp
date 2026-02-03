/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#define DEFINE_TRT_ENTRYPOINTS 1
#define DEFINE_TRT_ONNX_PARSER_ENTRYPOINT 0
#define DEFINE_TRT_BUILDER_ENTRYPOINT 0
#define DEFINE_TRT_REFITTER_ENTRYPOINT 0
#define DEFINE_TRT_RUNTIME_ENTRYPOINT 0

#include "NvInfer.h"
#include "NvInferSafeRuntime.h"
#include "NvOnnxParser.h"

#include "argsParser.h"
#include "buffers.h"
#include "logger.h"
#include "maxPoolPluginCreator.h"
#include "parserOnnxConfig.h"
#include "safeErrorRecorder.h"
#include "sampleUtils.h"
#include <fstream>
#include <ios>
#include <iostream>
#include <memory>

std::string const gSampleName = "TensorRT.sample_safe_plugin_build";
std::string const gOnnxFileName = "mnist_safe_plugin_ds.onnx";

using namespace nvinfer1;

static sample::SampleSafeRecorder g_recorder{nvinfer2::safe::Severity::kDEBUG};

namespace
{

//!
//! \brief The SampleSafePluginBuildArgs struct stores the additional arguments required by the sample
//!
struct SampleSafePluginBuildArgs : public samplesCommon::Args
{
    std::string onnx{""};
    std::string engineFileName{"safe_plugin.engine"};
    std::string remoteAutoTuningConfig{};
};

//!
//! \brief This function parses arguments specific to the sample
//!
bool parseSampleSafePluginBuildArgs(SampleSafePluginBuildArgs& args, int32_t argc, char* argv[])
{
    for (int32_t i = 1; i < argc; ++i)
    {
        std::string arg{argv[i]};

        // Handle flags with values
        auto const flagValue = sample::parseFlag(arg);
        if (!flagValue.first.empty())
        {
            if (flagValue.first == "saveEngine")
            {
                if (!sample::validateNonEmpty(flagValue.second, "Engine filename"))
                {
                    return false;
                }
                args.engineFileName = flagValue.second;
            }
            else if (flagValue.first == "remoteAutoTuningConfig")
            {
                if (!sample::validateNonEmpty(flagValue.second, "Remote auto tuning config"))
                {
                    return false;
                }
                if (!sample::validateRemoteAutoTuningConfig(flagValue.second))
                {
                    return false;
                }
                args.remoteAutoTuningConfig = flagValue.second;
            }
            else if (flagValue.first == "datadir" || flagValue.first == "d")
            {
                if (!sample::validateNonEmpty(flagValue.second, "Data directory path"))
                {
                    return false;
                }
                args.dataDirs.push_back(sample::normalizeDirectoryPath(flagValue.second));
            }
            else if (flagValue.first == "onnx")
            {
                args.onnx = flagValue.second;
            }
            else
            {
                sample::gLogError << "Unknown flag: " << flagValue.first << std::endl;
                return false;
            }
        }
        // Handle boolean flags
        else
        {
            auto const boolFlag = sample::parseBooleanFlag(arg);
            if (!boolFlag.empty())
            {
                if (boolFlag == "help" || boolFlag == "h")
                {
                    args.help = true;
                }
                else
                {
                    sample::gLogError << "Unknown flag: " << boolFlag << std::endl;
                    return false;
                }
            }
            else
            {
                sample::gLogError << "Invalid Argument: " << argv[i] << std::endl;
                return false;
            }
        }
    }

    return true;
}

//!
//! \brief The SampleSafePluginBuildParams struct stores the additional parameters required by the sample
//!
struct SampleSafePluginBuildParams : public samplesCommon::OnnxSampleParams
{
    std::string engineFileName{"safe_plugin.engine"};
    std::string remoteAutoTuningConfig{};
    bool std{false};
};

//!
//! \brief Initialize members of the params struct using the command line args.
//!
SampleSafePluginBuildParams initializeSampleParams(SampleSafePluginBuildArgs const& args)
{
    SampleSafePluginBuildParams params;
    if (args.dataDirs.empty()) // Use default directories if user hasn't provided directory paths.
    {
        params.dataDirs.push_back("data/");
        params.dataDirs.push_back("data/safe_plugin/");
        params.dataDirs.push_back("data/samples/safe_plugin/");
    }
    else // Use the data directory provided by the user.
    {
        params.dataDirs = args.dataDirs;
    }

    params.onnxFileName = args.onnx == "" ? gOnnxFileName : args.onnx;
    params.batchSize = 1;
    params.engineFileName = args.engineFileName;
    params.remoteAutoTuningConfig = args.remoteAutoTuningConfig;
    return params;
}

//!
//! \brief  The SampleSafePlugin class implements the sample.
//!
//! \details It creates the network using a trained ONNX MNIST classification model.
//!
class SampleSafePlugin
{
public:
    explicit SampleSafePlugin(SampleSafePluginBuildParams const& params)
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
    bool constructNetwork(nvonnxparser::IParser* parser);

    SampleSafePluginBuildParams mParams; //!< The parameters for the sample.

    nvinfer1::Dims mInputDims; //!< The dimensions of the input to the network.

    nvinfer1::plugin::MaxPoolCreator maxPoolPluginCreator{};
};

//!
//! \brief Creates the network, configures the builder and creates the network engine.
//!
//! \details This function creates the MNIST network by parsing the ONNX model and builds
//!          the engine that will be used to run MNIST.
//!
//! \return true if the engine was created successfully and false otherwise.
//!
bool SampleSafePlugin::build()
{
    // Register custom plugin creator for Max pooling before building

    auto safePluginRegistry = nvinfer2::safe::getSafePluginRegistry(g_recorder);
    if (!safePluginRegistry)
    {
        return false;
    }
    safePluginRegistry->registerCreator(maxPoolPluginCreator, "", g_recorder);

    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));
    if (!builder)
    {
        return false;
    }

    NetworkDefinitionCreationFlags flags
        = (1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kSTRONGLY_TYPED));
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

    auto constructed = constructNetwork(parser.get());
    if (!constructed)
    {
        return false;
    }

    // Set the input shape for the whole neural network by adding optimization profiles
    constexpr int64_t kBATCH_SIZE0 = 1;
    auto profile0 = builder->createOptimizationProfile();
    ASSERT(profile0);
    profile0->setDimensions(network->getInput(0)->getName(), OptProfileSelector::kMIN, Dims4{kBATCH_SIZE0, 1, 28, 28});
    profile0->setDimensions(network->getInput(0)->getName(), OptProfileSelector::kOPT, Dims4{kBATCH_SIZE0, 1, 28, 28});
    profile0->setDimensions(network->getInput(0)->getName(), OptProfileSelector::kMAX, Dims4{kBATCH_SIZE0, 1, 28, 28});
    config->addOptimizationProfile(profile0);
    constexpr int64_t kBATCH_SIZE1 = 5;
    auto profile1 = builder->createOptimizationProfile();
    ASSERT(profile1);
    profile1->setDimensions(network->getInput(0)->getName(), OptProfileSelector::kMIN, Dims4{kBATCH_SIZE1, 1, 28, 28});
    profile1->setDimensions(network->getInput(0)->getName(), OptProfileSelector::kOPT, Dims4{kBATCH_SIZE1, 1, 28, 28});
    profile1->setDimensions(network->getInput(0)->getName(), OptProfileSelector::kMAX, Dims4{kBATCH_SIZE1, 1, 28, 28});
    config->addOptimizationProfile(profile1);
    config->setEngineCapability(nvinfer1::EngineCapability::kSAFETY);

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
    const std::string engineFile = mParams.engineFileName;
    std::ofstream file(engineFile, std::ios::binary);
    if (!file)
    {
        sample::gLogError << "Failed to open file to save engine: " << engineFile << std::endl;
        return false;
    }
    file.write(reinterpret_cast<char const*>(buffer->data()), buffer->size());
    file.close();

    return true;
}

//!
//! \brief Uses an ONNX parser to create the MNIST Network and marks the
//!        output layers.
//!
//! \param parser ONNX parser used to parse the network
//!
bool SampleSafePlugin::constructNetwork(nvonnxparser::IParser* parser)
{
    return parser->parseFromFile(locateFile(mParams.onnxFileName, mParams.dataDirs).c_str(),
        static_cast<int32_t>(sample::gLogger.getReportableSeverity()));
}
} // namespace

//!
//! \brief Prints the help information for running this sample.
//!
void printHelpInfo()
{
    std::cout << "Usage: ./sample_plugin_safe_build [-h or --help] [-d or --datadir=<path to data directory>]\n";
    std::cout << "--help or -h    Display help information\n";
    std::cout << "--datadir or -d Specify path to a data directory, overriding the default. This option can be used "
                 "multiple times to add multiple directories. If no data directories are given, the default is to use "
                 "(data/samples/safe_plugin/, data/safe_plugin/)"
              << std::endl;
    std::cout << "--saveEngine    Save the serialized engine to the file (default = safe_plugin.engine).\n";
    std::cout << "--onnx          Specify the path to the ONNX file to use. If not specified, the default is to use "
                 "what is specified by --datadir/mnist_safe_plugin_ds.onnx.\n";
    std::cout << "--remoteAutoTuningConfig  Set the remote auto tuning config. Format: "
                 "protocol://username[:password]@hostname[:port]?param1=value1&param2=value2\n";
    std::cout
        << "                Example: "
           "ssh://user:pass@192.0.2.100:22?remote_exec_path=/opt/tensorrt/bin&remote_lib_path=/opt/tensorrt/lib\n";
}

int main(int argc, char** argv)
{
    SampleSafePluginBuildArgs args;
    bool const argsOK = parseSampleSafePluginBuildArgs(args, argc, argv);
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

    SampleSafePluginBuildParams params = initializeSampleParams(args);

    SampleSafePlugin sample(params);
    sample::gLogInfo << "Building a GPU inference engine for MNIST with plugins" << std::endl;

    if (!sample.build())
    {
        return sample::gLogger.reportFail(sampleTest);
    }

    return sample::gLogger.reportPass(sampleTest);
}
