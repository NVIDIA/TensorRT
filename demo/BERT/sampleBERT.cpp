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

#include "bert.h"
#include "bertUtils.h"

#include "dataUtils.h"

#include "bertEncoder.h"
#include "embLayerNormPlugin.h"
#include "squad.h"

using namespace bert;
using namespace nvinfer1;

Args gArgs;

constexpr const char* gSampleName = "TensorRT.sample_bert";
constexpr const char* kTEST_INPUT_FNAME = "test_inputs.weights_int32";
constexpr const char* kTEST_OUTPUT_FNAME = "test_outputs.weights";
constexpr const char* kBERT_WEIGHTS_FNAME = "bert.weights";
constexpr int kNUM_RUNS = 10;

//!
//! \brief This function prints the help information for running this sample
//!
void printHelpInfo()
{
    std::cout << "Usage: ./sample_bert --nheads=<number of heads> [-h or --help] [-d or --datadir=<path to data directory>] [--fp16] [--saveEngine=<path>]\n";
    std::cout << "--help          Display help information\n";
    std::cout << "--datadir       Specify path to a data directory, overriding the default. This option can be used "
            "multiple times to add multiple directories. The given path(s) must contain the weights and test "
            "inputs/outputs.\n";
    std::cout << "--nheads        Number of attention heads.\n";
    std::cout << "--fp16          Run in FP16 mode.\n";
    std::cout << "--saveEngine    The path at which to write a serialized engine." << endl;
}

int main(int argc, char* argv[])
{
    const bool argsOK = parseArgs(gArgs, argc, argv);
    if (gArgs.help)
    {
        printHelpInfo();
        return EXIT_SUCCESS;
    }
    if (!argsOK)
    {
        gLogError << "Invalid arguments" << endl;
        printHelpInfo();
        return EXIT_FAILURE;
    }
    if (gArgs.dataDirs.empty())
    {
        gLogError << "No datadirs given" << endl;
        printHelpInfo();
        return EXIT_FAILURE;
    }
    if (gArgs.numHeads <= 0)
    {
        gLogError << "invalid number of heads" << endl;
        printHelpInfo();
        return EXIT_FAILURE;
    }

    gLogger.setReportableSeverity(Logger::Severity::kINFO);
    auto sampleTest = gLogger.defineTest(gSampleName, argc, const_cast<const char**>(argv));
    gLogger.reportTestStart(sampleTest);

    // Load weights and golden files
    const std::string outputPath(locateFile(kTEST_OUTPUT_FNAME, gArgs.dataDirs));
    WeightMap testOutputs;
    loadWeights(outputPath, testOutputs);

    vector<Weights> inputIds;
    vector<Weights> inputMasks;
    vector<Weights> segmentIds;
    vector<Dims> inputDims;
    const std::string inputPath(locateFile(kTEST_INPUT_FNAME, gArgs.dataDirs));

    int S = 0;
    int Bmax = 0;
    loadInputs(inputPath, Bmax, S, inputIds, inputMasks, segmentIds, inputDims);
    assert(inputIds.size() > 0 && "No inputs found in supplied golden file");

    // Create optimization profiles. In this case, we only create a single profile for the shape we care about.
    const int numHeads = gArgs.numHeads;

    const auto profile = std::make_tuple(Dims{2, Bmax, S}, Dims{2, Bmax, S}, Dims{2, Bmax, S});
    OptProfileMap optProfileMap = {std::make_pair(kMODEL_INPUT0_NAME, profile),
        std::make_pair(kMODEL_INPUT1_NAME, profile), std::make_pair(kMODEL_INPUT2_NAME, profile)};

    OptProfiles optProfiles = {optProfileMap};

    // Prepare the TRT Network
    BERTDriver bertDriver(numHeads, gArgs.runInFp16, 5000_MiB, optProfiles);

    const std::vector<size_t> inputShape(inputDims[0].d, inputDims[0].d + 2);
    const HostTensorMap inCfg{
        std::make_pair(kMODEL_INPUT0_NAME,
            make_shared<HostTensor>(const_cast<void*>(inputIds[0].values), inputIds[0].type, inputShape)),
        std::make_pair(kMODEL_INPUT1_NAME,
            make_shared<HostTensor>(const_cast<void*>(segmentIds[0].values), segmentIds[0].type, inputShape)),
        std::make_pair(kMODEL_INPUT2_NAME,
            make_shared<HostTensor>(const_cast<void*>(inputMasks[0].values), inputMasks[0].type, inputShape))};

    const int B = inputDims[0].d[0];

    WeightMap weightMap;

    const std::string weightsPath(locateFile(kBERT_WEIGHTS_FNAME, gArgs.dataDirs));
    loadWeights(weightsPath, weightMap);

    HostTensorMap params;
    for (auto& kv : weightMap)
    {
        std::vector<size_t> shape{static_cast<size_t>(kv.second.count)};
        params[kv.first] = make_shared<HostTensor>(const_cast<void*>(kv.second.values), kv.second.type, shape);
    }

    // Build the TRT Engine
    bertDriver.init(params);

    if (!gArgs.saveEngine.empty())
    {
        bertDriver.serializeEngine(gArgs.saveEngine);
    }

    // Benchmark inference
    const std::string outputName("cls_squad_logits");
    std::vector<float> output(2 * B * S);
    HostTensorMap outCfg
        = {make_pair(outputName, make_shared<HostTensor>(output.data(), DataType::kFLOAT, std::vector<size_t>{2, static_cast<size_t>(B), static_cast<size_t>(S)}))};

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    std::vector<float> timesTotal(kNUM_RUNS);   // Total time
    std::vector<float> timesCompute(kNUM_RUNS); // Computation time

    bertDriver.benchmark(inCfg, outCfg, B, stream, timesTotal, timesCompute);

    cudaStreamDestroy(stream);
    transposeLogits(output, B, S);
    const float* test = reinterpret_cast<const float*>(testOutputs["logits"].values);

    // Analyze benchmark results
    float meanAbsErr = 0;
    float maxdiff = 0;
    for (int it = 0; it < testOutputs["logits"].count; it++)
    {
        const float diff = abs(test[it] - output[it]);
        meanAbsErr += diff;
        maxdiff = max(diff, maxdiff);
    }
    meanAbsErr /= output.size();

    const float avgTotal = accumulate(timesTotal.begin(), timesTotal.end(), 0.f, plus<float>()) / timesTotal.size();
    const float avgCompute
        = accumulate(timesCompute.begin(), timesCompute.end(), 0.f, plus<float>()) / timesCompute.size();


    printf("B=%d S=%d MAE=%.12e MaxDiff=%.12e ", B, S, meanAbsErr, maxdiff);
    printf(" Runtime(total avg)=%.6fms Runtime(comp ms)=%.6f\n", avgTotal, avgCompute);

    bool pass{false};
    if (gArgs.runInFp16)
    {
        pass = meanAbsErr < 2e-2;
    }
    else
    {
        pass = meanAbsErr < 1e-5;
    }
    return gLogger.reportTest(sampleTest, pass);
}
