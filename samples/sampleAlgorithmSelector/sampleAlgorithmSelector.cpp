/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

//! \file sampleAlgorithmSelector.cpp
//! \brief This file contains the implementation of Algorithm Selector sample.
//!
//! It demonstrates the usage of IAlgorithmSelector to cache the algorithms used in a network.
//! It also shows the usage of IAlgorithmSelector::selectAlgorithms to define heuristics for selection of algorithms.
//! It builds a TensorRT engine by importing a trained MNIST ONNX model and runs inference on an input image of a
//! digit.
//! It can be run with the following command line:
//! Command: ./sample_algorithm_selector [-h or --help] [-d=/path/to/data/dir or --datadir=/path/to/data/dir]

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
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
using namespace nvinfer1;
using samplesCommon::SampleUniquePtr;

std::string const gSampleName = "TensorRT.sample_algorithm_selector";
std::string const gCacheFileName = "AlgorithmCache.txt";
//!
//! \brief Writes the default algorithm choices made by TensorRT into a file.
//!
class AlgorithmCacheWriter : public IAlgorithmSelector
{
public:
    //!
    //! \brief Return value in [0, nbChoices] for a valid algorithm.
    //!
    //! \details Lets TRT use its default tactic selection method.
    //! Writes all the possible choices to the selection buffer and returns the length of it.
    //! If BuilderFlag::kREJECT_EMPTY_ALGORITHMS is not set, just returning 0 forces default tactic selection.
    //!
    int32_t selectAlgorithms(nvinfer1::IAlgorithmContext const& context, const nvinfer1::IAlgorithm* const* choices,
        int32_t nbChoices, int32_t* selection) noexcept override
    {
        // TensorRT always provides more than zero number of algorithms in selectAlgorithms.
        ASSERT(nbChoices > 0);

        std::iota(selection, selection + nbChoices, 0);
        return nbChoices;
    }

    //!
    //! \brief called by TensorRT to report choices it made.
    //!
    //! \details Writes the TensorRT algorithm choices into a file.
    //!
    void reportAlgorithms(const nvinfer1::IAlgorithmContext* const* algoContexts,
        const nvinfer1::IAlgorithm* const* algoChoices, int32_t nbAlgorithms) noexcept override
    {
        std::ofstream algorithmFile(mCacheFileName);
        if (!algorithmFile.good())
        {
            sample::gLogError << "Cannot open algorithm cache file: " << mCacheFileName << " to write." << std::endl;
            abort();
        }

        for (int32_t i = 0; i < nbAlgorithms; i++)
        {
            algorithmFile << algoContexts[i]->getName() << "\n";
            algorithmFile << algoChoices[i]->getAlgorithmVariant().getImplementation() << "\n";
            algorithmFile << algoChoices[i]->getAlgorithmVariant().getTactic() << "\n";

            // Write number of inputs and outputs.
            int32_t const nbInputs = algoContexts[i]->getNbInputs();
            algorithmFile << nbInputs << "\n";
            int32_t const nbOutputs = algoContexts[i]->getNbOutputs();
            algorithmFile << nbOutputs << "\n";

            // Write input and output formats.
            for (int32_t j = 0; j < nbInputs + nbOutputs; j++)
            {
                algorithmFile << static_cast<int32_t>(algoChoices[i]->getAlgorithmIOInfoByIndex(j)->getTensorFormat())
                              << "\n";
                algorithmFile << static_cast<int32_t>(algoChoices[i]->getAlgorithmIOInfoByIndex(j)->getDataType())
                              << "\n";
            }
        }
        algorithmFile.close();
    }

    AlgorithmCacheWriter(std::string const& cacheFileName)
        : mCacheFileName(cacheFileName)
    {
    }

private:
    std::string mCacheFileName;
};

//!
//! \brief Replicates the algorithm selection using a cache file.
//!
class AlgorithmCacheReader : public IAlgorithmSelector
{
public:
    //!
    //! \brief Return value in [0, nbChoices] for a valid algorithm.
    //!
    //! \details Use the map created from cache to select algorithms.
    //!
    int32_t selectAlgorithms(nvinfer1::IAlgorithmContext const& algoContext,
        const nvinfer1::IAlgorithm* const* algoChoices, int32_t nbChoices, int32_t* selection) noexcept override
    {
        // TensorRT always provides more than zero number of algorithms in selectAlgorithms.
        ASSERT(nbChoices > 0);

        std::string const layerName(algoContext.getName());
        auto it = choiceMap.find(layerName);

        // The layerName can be used as a unique identifier for a layer.
        // Since the network and config has not been changed (between the cache and cache read),
        // This map must contain layerName.
        ASSERT(it != choiceMap.end());
        auto& algoItem = it->second;

        ASSERT(algoItem.nbInputs == algoContext.getNbInputs());
        ASSERT(algoItem.nbOutputs == algoContext.getNbOutputs());

        int32_t nbSelections = 0;
        for (auto i = 0; i < nbChoices; i++)
        {
            // The combination of implementation, tactic and input/output formats is unique to an algorithm,
            // and can be used to reproduce the same algorithm. Since the network and config has not been changed
            // (between the cache and cache read), there must be exactly one algorithm match for each layerName.
            if (areSame(algoItem, *algoChoices[i]))
            {
                selection[nbSelections++] = i;
            }
        }

        //! There must be only one algorithm selected.
        ASSERT(nbSelections == 1);
        return nbSelections;
    }

    //!
    //! \brief Called by TensorRT to report choices it made.
    //!
    //! \details Verifies that the algorithm used by TensorRT conform to the cache.
    //!
    void reportAlgorithms(const nvinfer1::IAlgorithmContext* const* algoContexts,
        const nvinfer1::IAlgorithm* const* algoChoices, int32_t nbAlgorithms) noexcept override
    {
        for (auto i = 0; i < nbAlgorithms; i++)
        {
            std::string const layerName(algoContexts[i]->getName());
            ASSERT(choiceMap.find(layerName) != choiceMap.end());
            auto const& algoItem = choiceMap[layerName];
            ASSERT(algoItem.nbInputs == algoContexts[i]->getNbInputs());
            ASSERT(algoItem.nbOutputs == algoContexts[i]->getNbOutputs());
            ASSERT(algoChoices[i]->getAlgorithmVariant().getImplementation() == algoItem.implementation);
            ASSERT(algoChoices[i]->getAlgorithmVariant().getTactic() == algoItem.tactic);
            auto nbFormats = algoItem.nbInputs + algoItem.nbOutputs;
            for (auto j = 0; j < nbFormats; j++)
            {
                ASSERT(algoItem.formats[j].first
                    == static_cast<int32_t>(algoChoices[i]->getAlgorithmIOInfoByIndex(j)->getTensorFormat()));
                ASSERT(algoItem.formats[j].second
                    == static_cast<int32_t>(algoChoices[i]->getAlgorithmIOInfoByIndex(j)->getDataType()));
            }
        }
    }

    AlgorithmCacheReader(std::string const& cacheFileName)
    {
        //! Use the cache file to create a map of algorithm choices.
        std::ifstream algorithmFile(cacheFileName);
        if (!algorithmFile.good())
        {
            sample::gLogError << "Cannot open algorithm cache file: " << cacheFileName << " to read." << std::endl;
            abort();
        }

        std::string line;
        while (getline(algorithmFile, line))
        {
            std::string layerName;
            layerName = line;

            AlgorithmCacheItem algoItem;
            getline(algorithmFile, line);
            algoItem.implementation = std::stoll(line);

            getline(algorithmFile, line);
            algoItem.tactic = std::stoll(line);

            getline(algorithmFile, line);
            algoItem.nbInputs = std::stoi(line);

            getline(algorithmFile, line);
            algoItem.nbOutputs = std::stoi(line);

            int32_t const nbFormats = algoItem.nbInputs + algoItem.nbOutputs;
            algoItem.formats.resize(nbFormats);
            for (int32_t i = 0; i < nbFormats; i++)
            {
                getline(algorithmFile, line);
                algoItem.formats[i].first = std::stoi(line);
                getline(algorithmFile, line);
                algoItem.formats[i].second = std::stoi(line);
            }
            choiceMap[layerName] = std::move(algoItem);
        }
        algorithmFile.close();
    }

private:
    struct AlgorithmCacheItem
    {
        int64_t implementation;
        int64_t tactic;
        int32_t nbInputs;
        int32_t nbOutputs;
        std::vector<std::pair<int32_t, int32_t>> formats;
    };
    std::unordered_map<std::string, AlgorithmCacheItem> choiceMap;

    //! The combination of implementation, tactic and input/output formats is unique to an algorithm,
    //! and can be used to check if two algorithms are same.
    static bool areSame(AlgorithmCacheItem const& algoCacheItem, IAlgorithm const& algoChoice) noexcept
    {
        if (algoChoice.getAlgorithmVariant().getImplementation() != algoCacheItem.implementation
            || algoChoice.getAlgorithmVariant().getTactic() != algoCacheItem.tactic)
        {
            return false;
        }

        // Loop over all the AlgorithmIOInfos to see if all of them match to the formats in algo item.
        auto const nbFormats = algoCacheItem.nbInputs + algoCacheItem.nbOutputs;
        for (auto j = 0; j < nbFormats; j++)
        {
            if (algoCacheItem.formats[j].first
                    != static_cast<int32_t>(algoChoice.getAlgorithmIOInfoByIndex(j)->getTensorFormat())
                || algoCacheItem.formats[j].second
                    != static_cast<int32_t>(algoChoice.getAlgorithmIOInfoByIndex(j)->getDataType()))
            {
                return false;
            }
        }

        return true;
    }
};

//!
//! \brief Selects Algorithms with minimum workspace requirements.
//!
class MinimumWorkspaceAlgorithmSelector : public IAlgorithmSelector
{
public:
    //!
    //! \brief Return value in [0, nbChoices] for a valid algorithm.
    //!
    //! \details Use the map created from cache to select algorithms.
    //!
    int32_t selectAlgorithms(nvinfer1::IAlgorithmContext const& algoContext,
        const nvinfer1::IAlgorithm* const* algoChoices, int32_t nbChoices, int32_t* selection) noexcept override
    {
        // TensorRT always provides more than zero number of algorithms in selectAlgorithms.
        ASSERT(nbChoices > 0);

        auto const* it = std::min_element(
            algoChoices, algoChoices + nbChoices, [](const nvinfer1::IAlgorithm* x, const nvinfer1::IAlgorithm* y) {
                return x->getWorkspaceSize() < y->getWorkspaceSize();
            });
        selection[0] = static_cast<int32_t>(it - algoChoices);
        return 1;
    }

    //!
    //! \brief Called by TensorRT to report choices it made.
    //!
    void reportAlgorithms(const nvinfer1::IAlgorithmContext* const* algoContexts,
        const nvinfer1::IAlgorithm* const* algoChoices, int32_t nbAlgorithms) noexcept override
    {
        // do nothing
    }
};

//!
//! \brief  The SampleAlgorithmSelector class implements the SampleAlgorithmSelector sample.
//!
//! \details It creates the network using a trained ONNX MNIST classification model.
//!
class SampleAlgorithmSelector
{
public:
    SampleAlgorithmSelector(samplesCommon::OnnxSampleParams const& params)
        : mParams(params)
    {
    }

    //!
    //! \brief Builds the network engine.
    //!
    bool build(IAlgorithmSelector* selector);

    //!
    //! \brief Runs the TensorRT inference engine for this sample.
    //!
    bool infer();

private:
    //!
    //! \brief uses a Onnx parser to create the MNIST Network and marks the output layers.
    //!
    bool constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
        SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
        SampleUniquePtr<nvonnxparser::IParser>& parser);
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
bool SampleAlgorithmSelector::build(IAlgorithmSelector* selector)
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

    builder->setMaxBatchSize(mParams.batchSize);
    config->setAlgorithmSelector(selector);

    if (mParams.fp16)
    {
        config->setFlag(BuilderFlag::kFP16);
    }
    if (mParams.int8)
    {
        config->setFlag(BuilderFlag::kINT8);
    }

    samplesCommon::enableDLA(builder.get(), config.get(), mParams.dlaCore, true /*GPUFallback*/);

    if (mParams.int8)
    {
        // The sample fails for Int8 with kREJECT_EMPTY_ALGORITHMS flag set.
        config->clearFlag(BuilderFlag::kREJECT_EMPTY_ALGORITHMS);
    }

    SampleUniquePtr<IRuntime> runtime{createInferRuntime(sample::gLogger.getTRTLogger())};
    if (!runtime)
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

    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
        runtime->deserializeCudaEngine(plan->data(), plan->size()), samplesCommon::InferDeleter());
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
bool SampleAlgorithmSelector::processInput(
    samplesCommon::BufferManager const& buffers, std::string const& inputTensorName, int32_t inputFileIdx) const
{

    int32_t const inputH = mInputDims.d[2];
    int32_t const inputW = mInputDims.d[3];

    // Read a random digit file.
    srand(unsigned(time(nullptr)));
    std::vector<uint8_t> fileData(inputH * inputW);
    readPGMFile(locateFile(std::to_string(inputFileIdx) + ".pgm", mParams.dataDirs), fileData.data(), inputH, inputW);

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
bool SampleAlgorithmSelector::verifyOutput(
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
                         << "Class " << i << ": " << std::string(int32_t(std::floor(prob[i] * 10 + 0.5f)), '*')
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
bool SampleAlgorithmSelector::constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
    SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
    SampleUniquePtr<nvonnxparser::IParser>& parser)
{
    auto parsed = parser->parseFromFile(locateFile(mParams.onnxFileName, mParams.dataDirs).c_str(),
        static_cast<int32_t>(sample::gLogger.getReportableSeverity()));
    if (!parsed)
    {
        return false;
    }

    if (mParams.fp16)
    {
        config->setFlag(BuilderFlag::kFP16);
    }
    if (mParams.int8)
    {
        config->setFlag(BuilderFlag::kINT8);
        samplesCommon::setAllDynamicRanges(network.get(), 127.0f, 127.0f);
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
bool SampleAlgorithmSelector::infer()
{
    // Create RAII buffer manager object.
    samplesCommon::BufferManager buffers(mEngine);

    auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
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

    // Asynchronously enqueue the inference work
    if (!context->enqueueV2(buffers.getDeviceBindings().data(), stream, nullptr))
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

    params.batchSize = 1;
    params.dlaCore = args.useDLACore;
    params.int8 = args.runInInt8;
    params.fp16 = args.runInFp16;

    params.onnxFileName = "mnist.onnx";
    params.inputTensorNames.push_back("Input3");
    params.outputTensorNames.push_back("Plus214_Output_0");

    return params;
}

//!
//! \brief Prints the help information for running this sample.
//!
void printHelpInfo()
{
    std::cout << "Usage: ./sample_algorithm_selector [-h or --help] [-d or --datadir=<path to data directory>] "
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

    auto sampleTest = sample::Logger::defineTest(gSampleName, argc, argv);

    sample::Logger::reportTestStart(sampleTest);

    samplesCommon::OnnxSampleParams params = initializeSampleParams(args);

    // Write Algorithm Cache.
    SampleAlgorithmSelector sampleAlgorithmSelector(params);

    {
        sample::gLogInfo << "Building and running a GPU inference engine for MNIST." << std::endl;
        sample::gLogInfo << "Writing Algorithm Cache for MNIST." << std::endl;
        AlgorithmCacheWriter algorithmCacheWriter(gCacheFileName);

        if (!sampleAlgorithmSelector.build(&algorithmCacheWriter))
        {
            return sample::Logger::reportFail(sampleTest);
        }

        if (!sampleAlgorithmSelector.infer())
        {
            return sample::Logger::reportFail(sampleTest);
        }
    }

    {
        // Build network using Cache from previous run.
        sample::gLogInfo << "Building a GPU inference engine for MNIST using Algorithm Cache." << std::endl;
        AlgorithmCacheReader algorithmCacheReader(gCacheFileName);

        if (!sampleAlgorithmSelector.build(&algorithmCacheReader))
        {
            return sample::Logger::reportFail(sampleTest);
        }

        if (!sampleAlgorithmSelector.infer())
        {
            return sample::Logger::reportFail(sampleTest);
        }
    }

    {
        // Build network using MinimumWorkspaceAlgorithmSelector.
        sample::gLogInfo
            << "Building a GPU inference engine for MNIST using Algorithms with minimum workspace requirements."
            << std::endl;
        MinimumWorkspaceAlgorithmSelector minimumWorkspaceAlgorithmSelector;
        if (!sampleAlgorithmSelector.build(&minimumWorkspaceAlgorithmSelector))
        {
            return sample::Logger::reportFail(sampleTest);
        }

        if (!sampleAlgorithmSelector.infer())
        {
            return sample::Logger::reportFail(sampleTest);
        }
    }

    return sample::Logger::reportPass(sampleTest);
}
