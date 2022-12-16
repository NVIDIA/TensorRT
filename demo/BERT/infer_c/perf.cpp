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

#include "bert_infer.h"
#include "common.h"
#include <limits>
#include <random>

using namespace nvinfer1;

void printHelpInfo()
{
    std::cout << "usage: ./perf [-h] [-e ENGINE] [-b BATCH_SIZE] [-s SEQUENCE_LENGTH]\n";
    std::cout << "               [-i ITERATIONS] [-w WARM_UP_RUNS] [-r RANDOM_SEED] [--enable_graph]\n";
    std::cout << "\n";
    std::cout << "BERT Inference Benchmark\n";
    std::cout << "\n";
    std::cout << "optional arguments:\n";
    std::cout << "  -h, --help            show this help message and exit\n";
    std::cout << "  -e ENGINE, --engine ENGINE\n";
    std::cout << "                        Path to BERT TensorRT engine\n";
    std::cout << "  -b BATCH_SIZE, --batch_size BATCH_SIZE\n";
    std::cout << "                        Batch size(s) to benchmark. Can be specified multiple\n";
    std::cout << "                        times for more than one batch size. This script\n";
    std::cout << "                        assumes that the engine has been built with one\n";
    std::cout << "                        optimization profile for each batch size, and that\n";
    std::cout << "                        these profiles are in order of increasing batch size.\n";
    std::cout << "  -s SEQUENCE_LENGTH, --sequence_length SEQUENCE_LENGTH\n";
    std::cout << "                        Sequence length of the BERT model\n";
    std::cout << "  -i ITERATIONS, --iterations ITERATIONS\n";
    std::cout << "                        Number of iterations to run when benchmarking.\n";
    std::cout << "  -w WARM_UP_RUNS, --warm_up_runs WARM_UP_RUNS\n";
    std::cout << "                        Number of iterations to run prior to benchmarking.\n";
    std::cout << "  -r RANDOM_SEED, --random_seed RANDOM_SEED\n";
    std::cout << "                        Random seed.\n";
    std::cout << "  --enable_graph\n";
    std::cout << "                        Enable CUDA Graph.\n";
    std::cout << std::endl;
}

void printDeviceInfo()
{
    int32_t device{};
    gpuErrChk(cudaGetDevice(&device));

    cudaDeviceProp properties{};
    gpuErrChk(cudaGetDeviceProperties(&properties, device));

    std::cout << "=== Device Information ===" << std::endl;
    std::cout << "Selected Device: " << properties.name << std::endl;
    std::cout << "Compute Capability: " << properties.major << "." << properties.minor << std::endl;
    std::cout << "SMs: " << properties.multiProcessorCount << std::endl;
    std::cout << "Compute Clock Rate: " << properties.clockRate / 1000000.0F << " GHz" << std::endl;
    std::cout << "Device Global Memory: " << (properties.totalGlobalMem >> 20) << " MiB" << std::endl;
    std::cout << "Shared Memory per SM: " << (properties.sharedMemPerMultiprocessor >> 10) << " KiB" << std::endl;
    std::cout << "Memory Bus Width: " << properties.memoryBusWidth << " bits"
              << " (ECC " << (properties.ECCEnabled != 0 ? "enabled" : "disabled") << ")" << std::endl;
    std::cout << "Memory Clock Rate: " << properties.memoryClockRate / 1000000.0F << " GHz" << std::endl;
    std::cout << "=== Software Information ===" << std::endl;
    std::cout << "Build time TensorRT version: " << NV_TENSORRT_MAJOR << "." << NV_TENSORRT_MINOR << "." << NV_TENSORRT_PATCH << std::endl;
    std::cout << "Runtime linked TensorRT version: " << getInferLibVersion() << std::endl;
}

int main(int argc, char* argv[])
{

    Args args;

    const bool argsOK = parseArgs(args, argc, argv);
    if (args.help)
    {
        printHelpInfo();
        return EXIT_SUCCESS;
    }

    if (!argsOK)
    {
        std::cerr << "Invalid arguments" << std::endl;
        printHelpInfo();
        return EXIT_FAILURE;
    }
    printDeviceInfo();

    if (args.batchSize.empty())
    {
        args.batchSize.push_back(1);
    }

    const int maxBatchSize = *std::max_element(args.batchSize.begin(), args.batchSize.end());

    BertInference bert(args.engine, maxBatchSize, args.sequenceLength, args.enableGraph);

    std::default_random_engine generator(args.randomSeed);
    std::uniform_int_distribution<int> distribution(0, std::numeric_limits<int>::max());

    const int pseudoVocabSize = 30522;
    const int pseudoTypeVocabSize = 2;
    const int maxInputSize = args.sequenceLength * maxBatchSize;

    std::vector<int> testWordIds(maxInputSize);
    std::vector<int> testSegmentIds(maxInputSize);
    std::vector<int> testInputMask(maxInputSize);
    std::generate(
        testWordIds.begin(), testWordIds.end(), [&] { return distribution(generator) % pseudoVocabSize; });
    std::generate(testSegmentIds.begin(), testSegmentIds.end(),
        [&] { return distribution(generator) % pseudoTypeVocabSize; });
    std::generate(testInputMask.begin(), testInputMask.end(), [&] { return 1; });

    for (int i = 0; i < args.batchSize.size(); i++)
    {
        bert.run(i, args.batchSize[i], (void*) (testWordIds.data()), (void*) (testSegmentIds.data()),
            (void*) (testInputMask.data()), args.warmUpRuns, args.iterations);
    }

    for (int i = 0; i < args.batchSize.size(); i++)
    {
        bert.reportTiming(i, args.batchSize[i]);
    }

    return EXIT_SUCCESS;
}
