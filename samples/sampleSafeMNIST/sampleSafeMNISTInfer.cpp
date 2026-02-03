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

//! \file sampleSafeMNISTInfer.cpp
//! \brief This file contains the implementation of the MNIST sample.
//!
//! It uses the prebuilt TensorRT engine to run inference on an input image of a digit.
//! It can be run with the following command line:
//! Command: ./sample_mnist_safe_infer

#include "NvInferSafeRuntime.h"
#include "safeCommon.h"
#include "safeErrorRecorder.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstring>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <thread>
#include <vector>

using namespace samplesSafeCommon;

namespace
{

//!
//! \brief Locate path to file by its filename. Will walk back MAX_DEPTH dirs from CWD to check for such a file path.
//!
std::string locateFile(std::string const& fileName, nvinfer2::safe::ISafeRecorder& recorder)
{
    constexpr uint32_t MAX_DEPTH{10U};
    std::array<const std::string, 2> const dirPatterns
        = {std::string{"data/samples/mnist/"}, std::string{"data/mnist/"}};
    std::string foundFile{};

    for (auto const& dir : dirPatterns)
    {
        std::string file{dir + fileName};
        bool found{false};
        for (uint32_t i = 0U; i < MAX_DEPTH; i++)
        {
            std::ifstream checkFile(file);
            found = checkFile.is_open();
            if (found)
            {
                break;
            }
            file = "../" + file; // Try again in parent dir.
        }
        if (found)
        {
            foundFile = file;
            break;
        }
    }

    if (foundFile.empty())
    {
        safeLogError(recorder, "Could not find " + fileName + " in data/samples/mnist/ or data/mnist.");
        safeLogError(recorder, "&&&& FAILED");
        exit(EXIT_FAILURE);
    }

    return foundFile;
}

//!
//! \brief Reads the input data, preprocesses, and stores the result in a managed buffer.
//!
bool processInput(void* input, int32_t const inputFileIdx, nvinfer2::safe::ISafeRecorder& recorder)
{
    std::stringstream ss;
    constexpr int32_t kINPUT_H{28};
    constexpr int32_t kINPUT_W{28};

    // Read the digit file according to the inputFileIdx.
    std::vector<uint8_t> fileData(kINPUT_H * kINPUT_W);
    readPGMFile(locateFile(std::to_string(inputFileIdx) + ".pgm", recorder), fileData.data(), kINPUT_H, kINPUT_W);

    // Print ASCII representation of digit.
    ss << "Input:\n";
    for (int32_t i = 0; i < kINPUT_H * kINPUT_W; i++)
    {
        ss << (" .:-=+*#%@"[fileData[i] / 26U]) << (((i + 1) % kINPUT_W) ? "" : "\n");
    }
    safeLogInfo(recorder, ss.str());

    float* hostInputBuffer = static_cast<float*>(input);
    std::copy(fileData.begin(), fileData.end(), hostInputBuffer);
    // Normalize to 0-1 with background at 0
    std::transform(hostInputBuffer, hostInputBuffer + kINPUT_H * kINPUT_W, hostInputBuffer,
        [](float v) -> float { return 1.0f - v / 255.0f; });

    return true;
}

//!
//! \brief Verifies that the output is correct and prints it.
//!
bool verifyOutput(void* output, int32_t groundTruthDigit, nvinfer2::safe::ISafeRecorder& recorder)
{
    float* prob = static_cast<float*>(output);

    // Print histogram of the output distribution.
    safeLogInfo(recorder, "Output:");
    float val{0.0f};
    int32_t idx{0};
    constexpr int32_t kDIGITS{10};

    // Calculate Softmax
    float sum{0.0f};
    for (int32_t i = 0; i < kDIGITS; ++i)
    {
        prob[i] = exp(prob[i]);
        sum += prob[i];
    }

    for (int32_t i = 0; i < kDIGITS; ++i)
    {
        std::stringstream ss;

        prob[i] /= sum;
        if (val < prob[i])
        {
            val = prob[i];
            idx = i;
        }

        ss << " Prob " << i << "  " << std::fixed << std::setw(5) << std::setprecision(4) << prob[i] << " Class " << i
           << ": " << std::string(int32_t(std::floor(prob[i] * 10 + 0.5f)), '*');
        safeLogInfo(recorder, ss.str());
    }

    return (idx == groundTruthDigit && val > 0.9f);
}

//!
//! \brief Loads the enginePlanFile from engineFile and returns it.
//!
std::vector<char> loadEnginePlanFile(std::string const& engineFile, int& size, nvinfer2::safe::ISafeRecorder& recorder)
{
    std::string const& filename = engineFile;
    std::vector<char> gieModelStream;
    std::ifstream file(filename, std::ios::binary);
    if (!file.good())
    {
        safeLogError(recorder, "Could not open input engine file or file is empty. File name: " + filename);
        return {};
    }
    file.seekg(0, std::ifstream::end);
    size = file.tellg();
    file.seekg(0, std::ifstream::beg);
    gieModelStream.resize(size);
    file.read(gieModelStream.data(), size);
    file.close();

    return gieModelStream;
}

//!
//! \brief Returns a random digit between 0 and 9
//!
int32_t getRandomDigit()
{
    std::random_device rd;
    std::default_random_engine generator{rd()};
    std::uniform_int_distribution<int32_t> distribution(0, 9);
    return distribution(generator);
}

//!
//! \brief Structure representing memory allocation for CUDA
//!
struct CudaMemory
{
    void* hostPtr = nullptr;
    void* devicePtr = nullptr;
    size_t size = 0;
};

//!
//! \brief Do inference
//!
void doInferenceThread(nvinfer2::safe::ITRTGraph* graph, int8_t& ret_status, nvinfer2::safe::ISafeRecorder* recorder)
{
    // Initialize to success; will be set to 0 on any error.
    ret_status = 1;

    int64_t nbIOs{};
    SAFE_API_CALL(graph->getNbIOTensors(nbIOs), *recorder);
    // This sample only has one input and one output.
    SAFE_ASSERT(nbIOs == 2);
    CudaMemory inputCudaMemory;
    CudaMemory outputCudaMemory;

    // Initialize main stream
    cudaStream_t stream;
    CUDA_CALL(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), *recorder);

    // Pick a random digit to try to infer.
    int32_t digit = getRandomDigit();

    // Iterate through all input/output tensors
    for (int64_t i = 0; i < nbIOs; ++i)
    {
        // Get the tensor name for the current I/O tensor
        const char* tensorName;
        SAFE_API_CALL(graph->getIOTensorName(tensorName, i), *recorder);
        // Get tensor descriptor which contains metadata like size and I/O mode
        nvinfer2::safe::TensorDescriptor desc;
        SAFE_API_CALL(graph->getIOTensorDescriptor(desc, tensorName), *recorder);

        // Allocate device and host memory for this tensor
        void* deviceBuf = nullptr;
        void* hostBuf = nullptr;
        CUDA_CALL(cudaMalloc(&deviceBuf, desc.sizeInBytes), *recorder);
        CUDA_CHECK(cudaHostAlloc(&hostBuf, desc.sizeInBytes, cudaHostAllocDefault));

        if (desc.ioMode == TensorIOMode::kINPUT)
        {
            // Read the input data into the managed buffers.
            processInput(hostBuf, digit, *recorder);

            // Asynchronously copy data from host input buffers to device input buffers.
            CUDA_CHECK(cudaMemcpyAsync(deviceBuf, hostBuf, desc.sizeInBytes, cudaMemcpyHostToDevice, stream));
            inputCudaMemory = {hostBuf, deviceBuf, desc.sizeInBytes};
        }
        else if (desc.ioMode == TensorIOMode::kOUTPUT)
        {
            CUDA_CALL(cudaMemsetAsync(deviceBuf, 0, desc.sizeInBytes, stream), *recorder);
            outputCudaMemory = {hostBuf, deviceBuf, desc.sizeInBytes};
        }
        else
        {
            safeLogError(*recorder, "Unexpected tensor IO mode");
            ret_status = 0;
        }
        SAFE_ASSERT(desc.dataType == DataType::kFLOAT);
        // Create a typed array for the tensor
        nvinfer2::safe::TypedArray tensor
            = nvinfer2::safe::TypedArray(static_cast<float*>(deviceBuf), desc.sizeInBytes);

        SAFE_API_CALL(graph->setIOTensorAddress(tensorName, tensor), *recorder);
    }

    cudaEvent_t inputConsumedEvent;
    cudaEventCreate(&inputConsumedEvent);
    SAFE_API_CALL(graph->setInputConsumedEvent(inputConsumedEvent), *recorder);

    // Run the graph
    SAFE_API_CALL(graph->executeAsync(stream), *recorder);

    cudaEvent_t retrievedEvent;
    SAFE_API_CALL(graph->getInputConsumedEvent(retrievedEvent), *recorder);
    SAFE_ASSERT(retrievedEvent != nullptr);
    cudaEventSynchronize(retrievedEvent);

    // Synchronize the network
    SAFE_API_CALL(graph->sync(), *recorder);

    // Asynchronously copy data from device output buffers to host output buffers.
    CUDA_CHECK(cudaMemcpyAsync(
        outputCudaMemory.hostPtr, outputCudaMemory.devicePtr, outputCudaMemory.size, cudaMemcpyDeviceToHost, stream));

    // Wait for the work in the stream to complete.
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Check and print the output of the inference.
    if (!verifyOutput(outputCudaMemory.hostPtr, digit, *recorder))
    {
        safeLogError(*recorder, "Failed to verify output");
        ret_status = 0;
    }

    // Release stream and buffers.
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFreeHost(inputCudaMemory.hostPtr));
    CUDA_CHECK(cudaFreeHost(outputCudaMemory.hostPtr));
    CUDA_CHECK(cudaFree(inputCudaMemory.devicePtr));
    CUDA_CHECK(cudaFree(outputCudaMemory.devicePtr));
}

//!
//! \brief Runs the TensorRT inference engine for this sample.
//!
//! \details This function is the main execution function of the sample. It allocates
//!          the buffer, sets inputs, executes the engine, and verifies the output.
//!
bool doInference(std::string const& engineFileName, int32_t const& nbThreads)
{
    std::vector<int8_t> ret_status(nbThreads);
    std::vector<std::unique_ptr<sample::SampleSafeRecorder>> recorders(nbThreads);
    for (int32_t i = 0; i < nbThreads; ++i)
    {
        recorders[i] = std::make_unique<sample::SampleSafeRecorder>(nvinfer2::safe::Severity::kINFO, i);
    }
    // Load safe engine blob
    int32_t engineFileSize = 0;
    auto gieModelStream = loadEnginePlanFile(engineFileName, engineFileSize, *recorders[0]);
    SAFE_ASSERT(engineFileSize != 0);

    // Configure executor(s)
    std::vector<nvinfer2::safe::ITRTGraph*> graphs(nbThreads);
    SAFE_API_CALL(nvinfer2::safe::createTRTGraph(graphs[0], gieModelStream.data(), engineFileSize, *recorders[0], true),
        *recorders[0]);

    for (int32_t i = 1; i < nbThreads; ++i)
    {
        SAFE_API_CALL(graphs[0]->clone(graphs[i], *recorders[i]), *recorders[0]);
    }

    // Run the graphs in independent threads
    std::vector<std::thread> threads(nbThreads);
    for (int32_t i = 0; i < nbThreads; ++i)
    {
        threads[i] = std::thread{doInferenceThread, graphs[i], std::ref(ret_status[i]), recorders[i].get()};
    }

    for (int32_t i = 0; i < nbThreads; ++i)
    {
        threads[i].join();
        if (!ret_status[i])
        {
            return false;
        }
    }

    for (int32_t i = 0; i < nbThreads; ++i)
    {
        SAFE_API_CALL(nvinfer2::safe::destroyTRTGraph(graphs[i]), *recorders[i]);
        graphs[i] = nullptr;
    }

    return true;
}

//!
//! \brief The SampleSafeMNISTInferArgs struct stores the additional arguments required by the sample
//!
struct SampleSafeMNISTInferArgs
{
    std::string engineFileName{"safe_mnist.engine"};
    int32_t threads{1};
    bool help{false};
};

//!
//! \brief This function parses arguments specific to the sample
//!
bool parseSampleSafeMNISTInferArgs(SampleSafeMNISTInferArgs& args, int32_t argc, char* argv[])
{
    for (int32_t i = 1; i < argc; ++i)
    {
        if (!strncmp(argv[i], "--loadEngine=", 13))
        {
            args.engineFileName = (argv[i] + 13);
        }
        else if (!strncmp(argv[i], "--threads=", 10))
        {
            args.threads = std::stoi(argv[i] + 10);
            if (args.threads <= 0)
            {
                SAFE_LOG << "Invalid Argument: " << argv[i] << std::endl;
                return false;
            }
        }
        else if (!strncmp(argv[i], "--help", 6) || !strncmp(argv[i], "-h", 2))
        {
            args.help = true;
        }
        else
        {
            SAFE_LOG << "Invalid Argument: " << argv[i] << std::endl;
            return false;
        }
    }
    return true;
}

//!
//! \brief Prints the help information for running this sample.
//!
void printHelpInfo()
{
    std::cout << "Usage: ./sample_mnist_safe_infer [-h or --help] [--loadEngine=<path to engine file>] "
                 "[--threads=<number of threads to run>]\n";
    std::cout << "--help or -h    Display help information\n";
    std::cout << "--loadEngine    load the serialized engine from the file (default = safe_mnist.engine).\n";
    std::cout << "--threads       Number of threads (default = 1)\n";
}
} // namespace

int32_t main(int32_t argc, char** argv)
{
    safetyCompliance::setPromgrAbility();
    SampleSafeMNISTInferArgs args;
    bool argsOK = parseSampleSafeMNISTInferArgs(args, argc, argv);
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

    // Initialize SafeCuda before any other Cuda APIs are called. This may be skipped if createInferRuntime() is called
    // first as per DEEPLRN_RES_116
    safetyCompliance::initSafeCuda();

    if (!isSmSafe())
    {
        SAFE_LOG << "Skip safe mode test on unsupported platforms." << std::endl;
        return EXIT_SUCCESS;
    }

    TestResult result = doInference(args.engineFileName, args.threads) ? TestResult::kPASSED : TestResult::kFAILED;
    reportTestResult("TensorRT.sample_mnist_safe_infer", result, argc, argv);

    return EXIT_SUCCESS;
}
