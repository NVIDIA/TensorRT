/*
 * Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
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

#include "NvInfer.h"
#include "common.h"
#include "logger.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <ctime>
#include <cuda_profiler_api.h>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <map>
#include <sstream>
#include <sys/stat.h>
#include <vector>

#include "bertUtils.h"

#include "dataUtils.h"

#include "bertEncoder.h"
#include "embLayerNormPlugin.h"
#include "squad.h"

using namespace bert;

Args gArgs;

const std::string gSampleName = "TensorRT.sample_bert";
const std::string TEST_INPUT_NAME = "test_inputs.weights_int32";
const std::string TEST_OUTPUT_NAME = "test_outputs.weights";
const std::string BERT_WEIGHTS_NAME = "bert.weights";
const int NUM_RUNS = 10;

void doInference(IExecutionContext& context, const std::map<std::string, nvinfer1::Weights>& inCfg,
    std::map<std::string, std::vector<float>>& outCfg, const int batchSize, cudaStream_t stream,
    std::vector<float>& timesTotal, std::vector<float>& timesCompute, int verbose = 1)
{

    const int numRuns = timesTotal.size();
    assert(numRuns == timesCompute.size());
    assert(numRuns > 0);

    const ICudaEngine& engine = context.getEngine();
    const int numBindings = engine.getNbBindings();
    assert(numBindings == inCfg.size() + outCfg.size());
    std::vector<void*> buffers(numBindings);
    allocBindingsFromWeights(engine, buffers, batchSize, inCfg, verbose);
    allocBindingsFromVectors(engine, buffers, batchSize, outCfg, verbose);

    void** bs = buffers.data();

    std::vector<cudaEvent_t> startsTotal(numRuns);
    std::vector<cudaEvent_t> stopsTotal(numRuns);
    std::vector<cudaEvent_t> startsCompute(numRuns);
    std::vector<cudaEvent_t> stopsCompute(numRuns);

    for (int it = 0; it < numRuns; it++)
    {
        cudaEventCreate(&startsTotal[it]);
        cudaEventCreate(&stopsTotal[it]);

        cudaEventCreate(&startsCompute[it]);
        cudaEventCreate(&stopsCompute[it]);
    }

    cudaProfilerStart();
    for (int it = 0; it < numRuns; it++)
    {
        CHECK(cudaEventRecord(startsTotal[it], stream));
        copyToDeviceBindings(engine, buffers, batchSize, inCfg, stream);
        CHECK(cudaEventRecord(startsCompute[it], stream));
        context.enqueue(batchSize, bs, stream, nullptr);
        CHECK(cudaEventRecord(stopsCompute[it], stream));
        copyFromDeviceBindings(engine, buffers, batchSize, outCfg, stream);
        CHECK(cudaEventRecord(stopsTotal[it], stream));
    }
    CHECK(cudaDeviceSynchronize());

    cudaProfilerStop();
    float milliseconds = 0;
    for (int it = 0; it < numRuns; it++)
    {
        cudaEventElapsedTime(&milliseconds, startsTotal[it], stopsTotal[it]);
        timesTotal[it] = milliseconds;
        cudaEventElapsedTime(&milliseconds, startsCompute[it], stopsCompute[it]);
        timesCompute[it] = milliseconds;

        cudaEventDestroy(startsTotal[it]);
        cudaEventDestroy(stopsTotal[it]);
        cudaEventDestroy(startsCompute[it]);
        cudaEventDestroy(stopsCompute[it]);

        printf("Run %d; Total: %fms Comp.only: %fms\n", it, timesTotal[it], timesCompute[it]);
    }

    cudaProfilerStop();

    for (auto& devptr : buffers)
    {
        CHECK(cudaFree(devptr));
    }
}

// Create the Engine using only the API and not any parser.
nvinfer1::ICudaEngine* fromAPIToModel(nvinfer1::IBuilder* builder, const int numHeads, const int B, const int S)
{

    builder->setMaxBatchSize(B);
    builder->setMaxWorkspaceSize(5000_MB);
    builder->setFp16Mode(gArgs.runInFp16);
    if (gArgs.runInFp16)
    {
        gLogInfo << ("Running in FP 16 Mode\n");
        builder->setStrictTypeConstraints(true);
    }

    nvinfer1::INetworkDefinition* network = builder->createNetwork();

    WeightMap weightMap;

    const std::string weightsPath(locateFile(BERT_WEIGHTS_NAME, gArgs.dataDirs));

    loadWeights(weightsPath, weightMap);

    // infer these from the parameters
    int intermediateSize = 0;
    int numHiddenLayers = 0;
    int hiddenSize = 0;

    inferNetworkSizes(weightMap, hiddenSize, intermediateSize, numHiddenLayers);

    assert(intermediateSize);
    assert(hiddenSize);
    assert(numHiddenLayers);

    /// Embeddings Layer

    ITensor* inputIds = network->addInput("input_ids", DataType::kINT32, Dims{1, S});

    ITensor* segmentIds = network->addInput("segment_ids", DataType::kINT32, Dims{1, S});

    ITensor* inputMask = network->addInput("input_mask", DataType::kINT32, Dims{1, S});

    const Weights& wBeta = weightMap.at("bert_embeddings_layernorm_beta");
    const Weights& wGamma = weightMap.at("bert_embeddings_layernorm_gamma");
    const Weights& wWordEmb = weightMap.at("bert_embeddings_word_embeddings");
    const Weights& wTokEmb = weightMap.at("bert_embeddings_token_type_embeddings");
    const Weights& wPosEmb = weightMap.at("bert_embeddings_position_embeddings");
    ITensor* inputs[3] = {inputIds, segmentIds, inputMask};

    auto embPlugin = EmbLayerNormPlugin("embeddings", gArgs.runInFp16, wBeta, wGamma, wWordEmb, wPosEmb, wTokEmb);
    IPluginV2Layer* embLayer = network->addPluginV2(inputs, 3, embPlugin);
    setOutputName(embLayer, "embeddings_", "output");

    ITensor* embeddings = embLayer->getOutput(0);
    ITensor* maskIdx = embLayer->getOutput(1);

    /// BERT Encoder

    const BertConfig config(numHeads, hiddenSize, intermediateSize, numHiddenLayers, gArgs.runInFp16);

    ILayer* bertLayer = bertModel(config, weightMap, network, embeddings, maskIdx);

    /// SQuAD Output Layer

    ILayer* squadLayer = squad("cls_", config, weightMap, network, bertLayer->getOutput(0));

    network->markOutput(*squadLayer->getOutput(0));

    // Build the engine

    auto engine = builder->buildCudaEngine(*network);
    // we don't need the network any more
    network->destroy();

    // Once we have built the cuda engine, we can release all of our held memory.
    for (auto& w : weightMap)
        free(const_cast<void*>(w.second.values));
    return engine;
}

nvinfer1::ICudaEngine* APIToModel(const int numHeads, const int B, const int S)
{
    // create the builder
    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(gLogger.getTRTLogger());
    assert(builder != nullptr);

    // create the model to populate the network, then set the outputs and create an engine
    nvinfer1::ICudaEngine* engine = fromAPIToModel(builder, numHeads, B, S);

    assert(engine != nullptr);

    builder->destroy();
    return engine;
}

//!
//! \brief This function prints the help information for running this sample
//!
void printHelpInfo()
{
    std::cout << "Usage: ./sample_bert [-h or --help] [-d or --datadir=<path to data directory>] [--fp1 ]\n";
    std::cout << "--help          Display help information\n";
    std::cout << "--datadir       Specify path to a data directory, overriding the default. This option can be used "
                 "multiple times to add multiple directories. The given path(s) must contain the weights and test "
                 "inputs/outputs."
              << std::endl;
    std::cout << "--nheads        Number of attention heads." << std::endl;
    std::cout << "--fp16          OPTIONAL: Run in FP16 mode." << std::endl;
    std::cout << "--saveEngine    The path at which to write a serialized engine." << std::endl;
}

int main(int argc, char* argv[])
{

    bool argsOK = parseArgs(gArgs, argc, argv);
    if (gArgs.help)
    {
        printHelpInfo();
        return EXIT_SUCCESS;
    }
    if (!argsOK)
    {
        gLogError << "Invalid arguments" << std::endl;
        printHelpInfo();
        return EXIT_FAILURE;
    }
    if (gArgs.dataDirs.empty())
    {
        gLogError << "No datadirs given" << std::endl;
        printHelpInfo();
        return EXIT_FAILURE;
    }
    if (gArgs.numHeads <= 0)
    {
        gLogError << "invalid number of heads" << std::endl;
        printHelpInfo();
        return EXIT_FAILURE;
    }
    const std::string weightsPath(locateFile(TEST_OUTPUT_NAME, gArgs.dataDirs));
    std::map<std::string, nvinfer1::Weights> testOutputs;
    loadWeights(weightsPath, testOutputs);

    std::vector<nvinfer1::Weights> inputIds;
    std::vector<nvinfer1::Weights> inputMasks;
    std::vector<nvinfer1::Weights> segmentIds;
    std::vector<nvinfer1::Dims> inputDims;
    std::string inputPath(locateFile(TEST_INPUT_NAME, gArgs.dataDirs));

    int S = 0;
    int Bmax = 0;
    loadInputs(inputPath, Bmax, S, inputIds, inputMasks, segmentIds, inputDims);
    assert(inputIds.size() > 0);

    auto sampleTest = gLogger.defineTest(gSampleName, argc, const_cast<const char**>(argv));

    gLogger.reportTestStart(sampleTest);

    const int numHeads = gArgs.numHeads;
    nvinfer1::ICudaEngine* engine = APIToModel(numHeads, Bmax, S);
    if (engine == nullptr)
    {
        gLogError << "Unable to build engine." << std::endl;
        return gLogger.reportFail(sampleTest);
    }

    if (!gArgs.saveEngine.empty())
    {
        std::ofstream engineFile(gArgs.saveEngine, std::ios::binary);
        if (!engineFile)
        {
            gLogError << "Cannot open engine file: " << gArgs.saveEngine << std::endl;
            return gLogger.reportFail(sampleTest);
        }

        nvinfer1::IHostMemory* serializedEngine{engine->serialize()};
        if (serializedEngine == nullptr)
        {
            gLogError << "Engine serialization failed" << std::endl;
            return false;
        }

        engineFile.write(static_cast<char*>(serializedEngine->data()), serializedEngine->size());
        serializedEngine->destroy();
    }

    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(gLogger);
    if (runtime == nullptr)
    {
        gLogError << "Unable to create runtime." << std::endl;
        return gLogger.reportFail(sampleTest);
    }

    nvinfer1::IExecutionContext* context = engine->createExecutionContext();
    if (context == nullptr)
    {
        gLogError << "Unable to create context." << std::endl;
        return gLogger.reportFail(sampleTest);
    }

    const std::map<std::string, nvinfer1::Weights> inCfg{std::make_pair("input_ids", inputIds[0]),
        std::make_pair("input_mask", inputMasks[0]), std::make_pair("segment_ids", segmentIds[0])};
    const int B = inputDims[0].d[0];

    const std::string outputName("cls_squad_logits");
    std::map<std::string, std::vector<float>> outCfg = {make_pair(outputName, std::vector<float>(2 * B * S))};

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    std::vector<float> timesTotal(NUM_RUNS);   // total time
    std::vector<float> timesCompute(NUM_RUNS); // computation time

    doInference(*context, inCfg, outCfg, B, stream, timesTotal, timesCompute);

    cudaStreamDestroy(stream);
    context->destroy();
    engine->destroy();
    runtime->destroy();
    auto& output = outCfg[outputName];
    transposeLogits(output, B, S);
    const float* test = reinterpret_cast<const float*>(testOutputs["logits"].values);

    float mae = 0;
    float maxdiff = 0;
    for (int it = 0; it < testOutputs["logits"].count; it++)
    {
        const float diff = std::abs(test[it] - output[it]);
        mae += diff;
        maxdiff = std::max(diff, maxdiff);
    }
    const float avgTotal
        = std::accumulate(timesTotal.begin(), timesTotal.end(), 0.f, std::plus<float>()) / timesTotal.size();
    const float avgCompute
        = std::accumulate(timesCompute.begin(), timesCompute.end(), 0.f, std::plus<float>()) / timesCompute.size();

    printf("B=%d S=%d MAE=%.12e MaxDiff=%.12e ", B, S, (mae) / output.size(), maxdiff);
    printf(" Runtime(total avg)=%.6fms Runtime(comp ms)=%.6f\n", avgTotal, avgCompute);

    // destroy the engine
    bool pass{true};

    return gLogger.reportTest(sampleTest, pass);
}
