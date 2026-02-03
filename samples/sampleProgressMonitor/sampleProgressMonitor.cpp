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

//! \file sampleProgressMonitor.cpp
//! \brief This file contains the implementation of the Progress Monitor sample.
//!
//! It demonstrates the usage of IProgressMonitor for displaying engine build progress on the user's terminal.
//! It builds a TensorRT engine by importing a trained MNIST ONNX model and runs inference on an input image of a
//! digit.
//! It can be run with the following command line:
//! Command: ./sample_progress_monitor [-h or --help] [-d=/path/to/data/dir or --datadir=/path/to/data/dir]

// Define TRT entrypoints used in common code
#define DEFINE_TRT_ENTRYPOINTS 1

#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"

#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "parserOnnxConfig.h"

#include <algorithm>
#include <cmath>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

using namespace nvinfer1;
std::string const gSampleName = "TensorRT.sample_progress_monitor";

//!
//! \brief The ConsoleProgressMonitor class displays a simple progress graph for each step of the build process.
//!
class ConsoleProgressMonitor : public IProgressMonitor
{
public:
    void phaseStart(char const* phaseName, char const* parentPhase, int32_t nbSteps) noexcept final
    {
        PhaseEntry newPhase;
        newPhase.title = phaseName;
        newPhase.nbSteps = nbSteps;

        PhaseIter iParent = mPhases.end();
        if (parentPhase)
        {
            iParent = findPhase(parentPhase);
            newPhase.nbIndents = 1 + iParent->nbIndents;
            do
            {
                ++iParent;
            } while (iParent != mPhases.end() && iParent->nbIndents >= newPhase.nbIndents);
        }
        mPhases.insert(iParent, newPhase);
        redraw();
    }

    bool stepComplete(char const* phaseName, int32_t step) noexcept final
    {
        PhaseIter const iPhase = findPhase(phaseName);
        iPhase->steps = step;
        redraw();
        return true;
    }

    void phaseFinish(char const* phaseName) noexcept final
    {
        PhaseIter const iPhase = findPhase(phaseName);
        iPhase->active = false;
        redraw();
        mPhases.erase(iPhase);
    }

private:
    struct PhaseEntry
    {
        std::string title;
        int32_t steps{0};
        int32_t nbSteps{0};
        int32_t nbIndents{0};
        bool active{true};
    };
    using PhaseIter = std::vector<PhaseEntry>::iterator;

    std::vector<PhaseEntry> mPhases;

    static int32_t constexpr kPROGRESS_INNER_WIDTH = 10;

    void redraw()
    {
        auto const moveToStartOfLine = []() { std::cout << "\x1b[0G"; };
        auto const clearCurrentLine = []() { std::cout << "\x1b[2K"; };

        moveToStartOfLine();

        int32_t inactivePhases = 0;
        for (PhaseEntry const& phase : mPhases)
        {
            clearCurrentLine();

            if (phase.nbIndents > 0)
            {
                for (int32_t indent = 0; indent < phase.nbIndents; ++indent)
                {
                    std::cout << ' ';
                }
            }

            if (phase.active)
            {
                std::cout << progressBar(phase.steps, phase.nbSteps) << ' ' << phase.title << ' ' << phase.steps << '/'
                          << phase.nbSteps << std::endl;
            }
            else
            {
                // Don't draw anything at this time, but prepare to emit blank lines later.
                // This ensures that stale phases are removed from display rather than lingering.
                ++inactivePhases;
            }
        }

        for (int32_t phase = 0; phase < inactivePhases; ++phase)
        {
            clearCurrentLine();
            std::cout << std::endl;
        }

        // Move (mPhases.size()) lines up so that logger output can overwrite the progress bars.
        std::cout << "\x1b[" << mPhases.size() << "A";
    }

    std::string progressBar(int32_t steps, int32_t nbSteps) const
    {
        std::ostringstream bar;
        bar << '[';
        int32_t const completedChars
            = static_cast<int32_t>(kPROGRESS_INNER_WIDTH * steps / static_cast<float>(nbSteps));
        for (int32_t i = 0; i < completedChars; ++i)
        {
            bar << '=';
        }
        for (int32_t i = completedChars; i < kPROGRESS_INNER_WIDTH; ++i)
        {
            bar << '-';
        }
        bar << ']';
        return bar.str();
    }

    PhaseIter findPhase(std::string const& title)
    {
        return std::find_if(mPhases.begin(), mPhases.end(),
            [title](PhaseEntry const& phase) { return phase.title == title && phase.active; });
    }
};

//!
//! \brief The SampleProgressMonitor class implements the SampleProgressReporter sample.
//!
//! \details It creates the network using a trained ONNX MNIST classification model.
//!
class SampleProgressMonitor
{
public:
    explicit SampleProgressMonitor(samplesCommon::OnnxSampleParams const& params)
        : mParams(params)
    {
    }

    //!
    //! \brief Builds the network engine.
    //!
    bool build(IProgressMonitor* monitor);

    //!
    //! \brief Runs the TensorRT inference engine for this sample.
    //!
    bool infer();

private:
    //!
    //! \brief uses a Onnx parser to create the MNIST Network and marks the output layers.
    //!
    bool constructNetwork(std::unique_ptr<nvinfer1::IBuilder>& builder,
        std::unique_ptr<nvinfer1::INetworkDefinition>& network, std::unique_ptr<nvinfer1::IBuilderConfig>& config,
        std::unique_ptr<nvonnxparser::IParser>& parser);
    //!
    //! \brief Reads the input and mean data, preprocesses, and stores the result in a managed buffer.
    //!
    bool processInput(
        samplesCommon::BufferManager const& buffers, std::string const& inputTensorName, int32_t inputFileIdx) const;

    //!
    //! \brief Verifies that the output is correct and prints it.
    //!
    bool verifyOutput(samplesCommon::BufferManager const& buffers, std::string const& outputTensorName,
        int32_t groundTruthDigit) const;

    std::unique_ptr<IRuntime> mRuntime{};
    std::shared_ptr<nvinfer1::ICudaEngine> mEngine{nullptr}; //!< The TensorRT engine used to run the network.

    samplesCommon::OnnxSampleParams mParams; //!< The parameters for the sample.

    nvinfer1::Dims mInputDims; //!< The dimensions of the input to the network.
};

//!
//! \brief Creates the network, configures the builder and creates the network engine.
//!
//! \details This function creates the MNIST network by parsing the ONNX model and builds
//!          the engine that will be used to run MNIST (mEngine).
//!
//! \return true if the engine was created successfully and false otherwise.
//!
bool SampleProgressMonitor::build(IProgressMonitor* monitor)
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

    config->setProgressMonitor(monitor);

    samplesCommon::enableDLA(builder.get(), config.get(), mParams.dlaCore, true /*GPUFallback*/);

    if (!mRuntime)
    {
        mRuntime = std::unique_ptr<IRuntime>(createInferRuntime(sample::gLogger.getTRTLogger()));
    }
    if (!mRuntime)
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

    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(mRuntime->deserializeCudaEngine(plan->data(), plan->size()));
    if (!mEngine)
    {
        return false;
    }

    ASSERT(network->getNbInputs() == 1);
    mInputDims = network->getInput(0)->getDimensions();
    ASSERT(mInputDims.nbDims == 4);

    return true;
}

//!
//! \brief Reads the input and mean data, preprocesses, and stores the result in a managed buffer.
//!
bool SampleProgressMonitor::processInput(
    samplesCommon::BufferManager const& buffers, std::string const& inputTensorName, int32_t inputFileIdx) const
{
    int32_t const inputH = mInputDims.d[2];
    int32_t const inputW = mInputDims.d[3];

    // Read a random digit file.
    srand(unsigned(time(nullptr)));
    std::vector<uint8_t> fileData(inputH * inputW);
    samplesCommon::readPGMFile(samplesCommon::locateFile(std::to_string(inputFileIdx) + ".pgm", mParams.dataDirs),
        fileData.data(), inputH, inputW);

    // Print ASCII representation of digit.
    sample::gLogInfo << "Input:\n";
    for (int32_t i = 0; i < inputH * inputW; i++)
    {
        sample::gLogInfo << (" .:-=+*#%@"[fileData[i] / 26]) << (((i + 1) % inputW) ? "" : "\n");
    }
    sample::gLogInfo << std::endl;

    float* hostInputBuffer = static_cast<float*>(buffers.getHostBuffer(inputTensorName));

    for (int32_t i = 0; i < inputH * inputW; i++)
    {
        hostInputBuffer[i] = 1.0F - static_cast<float>(fileData[i]) / 255.0F;
    }

    return true;
}

//!
//! \brief Verifies that the output is correct and prints it.
//!
bool SampleProgressMonitor::verifyOutput(
    samplesCommon::BufferManager const& buffers, std::string const& outputTensorName, int32_t groundTruthDigit) const
{
    float* prob = static_cast<float*>(buffers.getHostBuffer(outputTensorName));
    int32_t constexpr kDIGITS = 10;

    std::for_each(prob, prob + kDIGITS, [](float& n) { n = exp(n); });

    float const sum = std::accumulate(prob, prob + kDIGITS, 0.F);

    std::for_each(prob, prob + kDIGITS, [sum](float& n) { n = n / sum; });

    auto max_ele = std::max_element(prob, prob + kDIGITS);

    float const val = *max_ele;

    int32_t const idx = max_ele - prob;

    // Print histogram of the output probability distribution.
    sample::gLogInfo << "Output:\n";
    for (int32_t i = 0; i < kDIGITS; i++)
    {
        sample::gLogInfo << " Prob " << i << "  " << std::fixed << std::setw(5) << std::setprecision(4) << prob[i]
                         << " "
                         << "Class " << i << ": " << std::string(int32_t(std::floor(prob[i] * 10 + 0.5F)), '*')
                         << std::endl;
    }
    sample::gLogInfo << std::endl;

    return (idx == groundTruthDigit && val > 0.9F);
}

//!
//! \brief Uses an ONNX parser to create the MNIST Network and marks the
//!        output layers.
//!
//! \param network Pointer to the network that will be populated with the MNIST network.
//!
//! \param builder Pointer to the engine builder.
//!
bool SampleProgressMonitor::constructNetwork(std::unique_ptr<nvinfer1::IBuilder>& builder,
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
//! \brief Runs the TensorRT inference engine for this sample.
//!
//! \details This function is the main execution function of the sample. It allocates
//!          the buffer, sets inputs, executes the engine, and verifies the output.
//!
bool SampleProgressMonitor::infer()
{
    // Create RAII buffer manager object.
    samplesCommon::BufferManager buffers(mEngine);

    auto context = std::unique_ptr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    if (!context)
    {
        return false;
    }

    // Pick a random digit to try to infer.
    srand(time(NULL));
    int32_t const digit = rand() % 10;

    // Read the input data into the managed buffers.
    // There should be just 1 input tensor.
    ASSERT(mParams.inputTensorNames.size() == 1);

    if (!processInput(buffers, mParams.inputTensorNames[0], digit))
    {
        return false;
    }
    // Create CUDA stream for the execution of this inference.
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // Asynchronously copy data from host input buffers to device input buffers
    buffers.copyInputToDeviceAsync(stream);

    for (int32_t i = 0, e = mEngine->getNbIOTensors(); i < e; i++)
    {
        auto const& name = mEngine->getIOTensorName(i);
        context->setTensorAddress(name, buffers.getDeviceBuffer(name));
    }

    // Asynchronously enqueue the inference work
    if (!context->enqueueV3(stream))
    {
        return false;
    }
    // Asynchronously copy data from device output buffers to host output buffers.
    buffers.copyOutputToHostAsync(stream);

    // Wait for the work in the stream to complete.
    CHECK(cudaStreamSynchronize(stream));

    // Release stream.
    CHECK(cudaStreamDestroy(stream));

    // Check and print the output of the inference.
    // There should be just one output tensor.
    ASSERT(mParams.outputTensorNames.size() == 1);
    bool outputCorrect = verifyOutput(buffers, mParams.outputTensorNames[0], digit);
    return outputCorrect;
}

//!
//! \brief Initializes members of the params struct using the command line args
//!
samplesCommon::OnnxSampleParams initializeSampleParams(samplesCommon::Args const& args)
{
    samplesCommon::OnnxSampleParams params;
    if (args.dataDirs.empty()) // Use default directories if user hasn't provided directory paths.
    {
        params.dataDirs.push_back("data/mnist/");
        params.dataDirs.push_back("data/samples/mnist/");
    }
    else // Use the data directory provided by the user.
    {
        params.dataDirs = args.dataDirs;
    }

    params.dlaCore = args.useDLACore;

    params.onnxFileName = "mnist.onnx";
    params.inputTensorNames.push_back("Input3");
    params.outputTensorNames.push_back("Plus214_Output_0");
    params.timingCacheFile = args.timingCacheFile;

    return params;
}

//!
//! \brief Prints the help information for running this sample.
//!
void printHelpInfo()
{
    std::cout << "Usage: ./sample_progress_monitor [-h or --help] [-d or --datadir=<path to data directory>] "
                 "[--useDLACore=<int>] [--timingCacheFile=<path to timing cache file>]\n";
    std::cout << "--help          Display help information\n";
    std::cout << "--datadir       Specify path to a data directory, overriding the default. This option can be used "
                 "multiple times to add multiple directories. If no data directories are given, the default is to use "
                 "(data/samples/mnist/, data/mnist/)"
              << std::endl;
    std::cout << "--useDLACore=N  Specify a DLA engine for layers that support DLA. Value can range from 0 to n-1, "
                 "where n is the number of DLA engines on the platform."
              << std::endl;
    std::cout << "--timingCacheFile  Specify path to a timing cache file. If it does not already exist, it will be "
              << "created." << std::endl;
}

int32_t main(int32_t argc, char** argv)
{
    samplesCommon::Args args;
    bool const argsOK = samplesCommon::parseArgs(args, argc, argv);
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

    auto sampleTest = sample::Logger::defineTest(gSampleName, argc, argv);

    sample::Logger::reportTestStart(sampleTest);

    samplesCommon::OnnxSampleParams params = initializeSampleParams(args);

    SampleProgressMonitor sampleProgressMonitor(params);
    {
        sample::gLogInfo << "Building and running a GPU inference engine for MNIST." << std::endl;
        ConsoleProgressMonitor progressMonitor;

        if (!sampleProgressMonitor.build(&progressMonitor))
        {
            return sample::Logger::reportFail(sampleTest);
        }

        if (!sampleProgressMonitor.infer())
        {
            return sample::Logger::reportFail(sampleTest);
        }
    }

    return sample::Logger::reportPass(sampleTest);
}
