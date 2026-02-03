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
#include "NvInferSafeRuntime.h"
#include "maxPoolPluginRuntimeCreator.h"
#include "safeCommon.h"
#include "safeErrorRecorder.h"
#include <fstream>
#include <ios>
#include <iostream>
#include <memory>
#include <random>
#include <type_traits>

using namespace nvinfer1;
using namespace samplesSafeCommon;
const std::string gSampleName = "TensorRT.sample_safe_plugin_infer";

static sample::SampleSafeRecorder g_recorder{nvinfer2::safe::Severity::kINFO};

//!
//! \brief The SampleSafeMNISTInferArgs struct stores the additional arguments required by the sample
//!
class SampleSafeMNISTInferArgs
{
public:
    std::string engineFileName{"safe_plugin.engine"};
    bool help{false};
};

//!
//! \brief This function parses arguments specific to the sample
//!
bool parseSampleSafePluginInferArgs(SampleSafeMNISTInferArgs& args, int32_t const argc, char const* const argv[])
{
    for (int32_t i = 1; i < argc; ++i)
    {
        if (strncmp(argv[i], "--loadEngine=", 13) == 0)
        {
            args.engineFileName = (argv[i] + 13);
        }
        else if (strncmp(argv[i], "--help", 6) == 0 || strncmp(argv[i], "-h", 2) == 0)
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

nvinfer2::safe::TypedArray createTypedArray(
    void* const ptr, DataType type, uint64_t bufferSize, nvinfer2::safe::ISafeRecorder& recorder)
{
    switch (type)
    {
    case DataType::kFLOAT: return nvinfer2::safe::TypedArray(static_cast<float*>(ptr), bufferSize);
    case DataType::kHALF: return nvinfer2::safe::TypedArray(static_cast<nvinfer2::safe::half_t*>(ptr), bufferSize);
    case DataType::kINT32: return nvinfer2::safe::TypedArray(static_cast<int32_t*>(ptr), bufferSize);
    case DataType::kINT8: return nvinfer2::safe::TypedArray(static_cast<int8_t*>(ptr), bufferSize);
    default:
    {
        SAFE_LOG << "Invalid tensor DataType encountered." << std::endl;
        return nvinfer2::safe::TypedArray{};
    }
    }
}

//!
//! \brief Allocate memory and memset it to zero using safe CUDA-compatible APIs.
//!
void* allocateAndMemset(uint64_t sizeInBytes, nvinfer2::safe::ISafeRecorder& recorder)
{
    void* deviceBuf{nullptr};
    cudaStream_t stream;
    CUDA_CALL(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), recorder);
    CUDA_CALL(cudaMalloc(&deviceBuf, sizeInBytes), recorder);
    CUDA_CALL(cudaMemsetAsync(deviceBuf, 0, sizeInBytes, stream), recorder);
    CUDA_CALL(cudaStreamSynchronize(stream), recorder);
    CUDA_CALL(cudaStreamDestroy(stream), recorder);
    return deviceBuf;
}

//!
//! \brief Helper function to get the volume.
//!
inline int64_t volume(nvinfer1::Dims const& d)
{
    return std::accumulate(d.d, d.d + d.nbDims, 1L, std::multiplies<int64_t>());
}

///!
//! \brief Loads the enginePlanFile from engineFile and returns it.
//!
std::vector<char> loadEnginePlanFile(std::string const& engineFile, int32_t& size)
{
    std::string const& filename = engineFile;
    std::vector<char> engineStream;
    std::ifstream file(filename, std::ios::binary);
    if (!file.good())
    {
        SAFE_LOG << "Could not open input engine file or file is empty. File name: " << filename << std::endl;
        return {};
    }
    file.seekg(0, std::ifstream::end);
    size = file.tellg();
    file.seekg(0, std::ifstream::beg);
    engineStream.resize(size);
    file.read(engineStream.data(), size);
    file.close();

    return engineStream;
}

//!
//! \brief Reads the input data, preprocesses, and stores the result in a managed buffer.
//!
bool processInput(void* const input, int32_t const inputFileIdx, int64_t const kBATCH_SIZE, int64_t const index)
{
    constexpr int32_t kINPUT_H{28};
    constexpr int32_t kINPUT_W{28};

    // Read the digit file according to the inputFileIdx.
    std::vector<uint8_t> fileData(static_cast<size_t>(kINPUT_H * kINPUT_W));
    std::vector<std::string> dataDirs;
    dataDirs.push_back("data/samples/mnist/");
    readPGMFile(locateFile(std::to_string(inputFileIdx) + ".pgm", dataDirs), fileData.data(), kINPUT_H, kINPUT_W);

    // Print ASCII representation of digit.
    SAFE_LOG << "Input:\n";
    for (int32_t i = 0; i < kINPUT_H * kINPUT_W; i++)
    {
        SAFE_LOG << (" .:-=+*#%@"[fileData[i] / 26]) << (((i + 1) % kINPUT_W) ? "" : "\n");
    }
    SAFE_LOG << std::endl;

    float* const hostInputBuffer = static_cast<float*>(input) + index * kINPUT_H * kINPUT_W;
    static_cast<void>(std::copy(fileData.begin(), fileData.end(), hostInputBuffer));
    // Normalize to 0-1 with background at 0
    static_cast<void>(std::transform(hostInputBuffer, hostInputBuffer + kINPUT_H * kINPUT_W, hostInputBuffer,
        [](float v) noexcept -> float { return 1.0F - v / 255.0F; }));

    return true;
}

//!
//! \brief Verifies that the output is correct and prints it.
//!
bool verifyOutput(void* const output, std::vector<int32_t> const& groundTruthDigits, int64_t const batchSize)
{
    bool result{true};
    constexpr int32_t kDIGITS{10};
    for (int64_t j = 0; j < batchSize; ++j)
    {
        float* const prob = static_cast<float*>(output) + j * kDIGITS;

        // Print histogram of the output distribution.
        SAFE_LOG << "Output:" << std::endl;
        float val{0.0F};
        int32_t idx{0};

        // Calculate Softmax
        float sum{0.0F};
        for (int32_t i = 0; i < kDIGITS; i++)
        {
            prob[i] = exp(prob[i]);
            sum += prob[i];
        }

        for (int32_t i = 0; i < kDIGITS; i++)
        {
            prob[i] /= sum;
            if (val < prob[i])
            {
                val = prob[i];
                idx = i;
            }

            SAFE_LOG << " Prob " << i << "  " << std::fixed << std::setw(5) << std::setprecision(4) << prob[i]
                     << " Class " << i << ": "
                     << std::string(static_cast<int32_t>(std::floor(prob[i] * 10 + 0.5F)), '*') << std::endl;
        }
        result &= (idx == groundTruthDigits[j]) && (val > 0.9F);
    }
    return result;
}

//!
//! \brief Set I/O tensor buffer.
//!
void setTensorBuffer(nvinfer2::safe::ITRTGraph* graph, nvinfer2::safe::ISafeRecorder& recorder,
    std::string const& tensorName, void*& tensorAddress)
{
    nvinfer2::safe::TensorDescriptor desc;
    SAFE_API_CALL(graph->getIOTensorDescriptor(desc, tensorName.c_str()), recorder);

    void* deviceBuf = allocateAndMemset(desc.sizeInBytes, recorder);
    tensorAddress = deviceBuf;
    nvinfer2::safe::TypedArray tensor = createTypedArray(deviceBuf, desc.dataType, desc.sizeInBytes, recorder);
    SAFE_API_CALL(graph->setIOTensorAddress(tensorName.c_str(), tensor), recorder);
    SAFE_LOG << "Set address of " << tensorName << " on device at " << std::hex << (uint64_t) deviceBuf << std::dec
             << std::endl;
}

//!
//! \brief Runs the TensorRT inference engine for this sample.
//!
//! \details This function is the main execution function of the sample. It allocates
//!          the buffer, sets inputs, executes the engine, and verifies the output.
//!
bool doInference(SampleSafeMNISTInferArgs const& args)
{
    // Create the engine by loading from a local saved plan
    int32_t engineFileSize = 0;
    auto engineFile = loadEnginePlanFile(args.engineFileName, engineFileSize);
    SAFE_ASSERT(engineFileSize != 0);

    // Inference
    nvinfer1::plugin::MaxPoolRuntimeCreator creator;
    ITRTGraph* graph = nullptr;

    getSafePluginRegistry(g_recorder)->registerCreator(creator, "", g_recorder);
    createTRTGraph(graph, engineFile.data(), engineFileSize, g_recorder, true, nullptr);
    SAFE_ASSERT(graph != nullptr);

    bool outputCorrect = true;
    int64_t nbIOProfile = 0;
    SAFE_API_CALL(graph->getNbIOProfiles(nbIOProfile), g_recorder);
    SAFE_ASSERT(nbIOProfile == 2);

    auto descToString = [](nvinfer2::safe::TensorDescriptor const& desc) {
        std::stringstream ss;
        ss << desc.tensorName << " {";
        for (int64_t i = 0; i < desc.shape.nbDims; ++i)
        {
            ss << desc.shape.d[i];
            if (i < desc.shape.nbDims - 1)
            {
                ss << ", ";
            }
        }
        ss << "}";
        return ss.str();
    };

    for (int64_t k = 0; k < nbIOProfile; ++k)
    {
        graph->setIOProfile(k);
        // Memory Config
        int64_t nbIOs{};
        SAFE_API_CALL(graph->getNbIOTensors(nbIOs), g_recorder);

        // This sample only has one input and one output.
        SAFE_ASSERT(nbIOs == 2);
        constexpr int32_t inputIndex{0};
        constexpr int32_t outputIndex{1};

        // Get the binding dimensions according to the input/output index.
        char const* inputBindingName = nullptr;
        char const* outputBindingName = nullptr;
        nvinfer2::safe::TensorDescriptor inputDesc;
        nvinfer2::safe::TensorDescriptor outputDesc;

        graph->getIOTensorName(inputBindingName, inputIndex);
        graph->getIOTensorName(outputBindingName, outputIndex);
        graph->getIOTensorDescriptor(inputDesc, inputBindingName);
        graph->getIOTensorDescriptor(outputDesc, outputBindingName);
        SAFE_ASSERT(inputDesc.ioMode == nvinfer1::TensorIOMode::kINPUT);
        SAFE_ASSERT(outputDesc.ioMode == nvinfer1::TensorIOMode::kOUTPUT);
        SAFE_ASSERT(inputDesc.shape.nbDims > 0);
        SAFE_LOG << "Set IO profile to " << k << std::endl;
        SAFE_LOG << descToString(inputDesc) << std::endl;
        int64_t kBATCH_SIZE = inputDesc.shape.d[0];
        SAFE_ASSERT(0 < kBATCH_SIZE && kBATCH_SIZE <= 9);
        // Create host buffers
        std::vector<void*> hostBuffers(nbIOs, nullptr);

        hostBuffers[inputIndex] = malloc(inputDesc.sizeInBytes);
        hostBuffers[outputIndex] = malloc(outputDesc.sizeInBytes);

        std::vector<int32_t> groundTruthDigits(kBATCH_SIZE);
        for (int64_t j = 0; j < kBATCH_SIZE; ++j)
        {
            // Pick a random digit to try to infer.
            std::random_device rd;
            std::default_random_engine generator{rd()};
            std::uniform_int_distribution<int32_t> distribution(0, 9);
            int32_t const digit = distribution(generator);
            groundTruthDigits[j] = digit;
            // Read the input data into the managed buffers.
            if (!processInput(hostBuffers[inputIndex], digit, kBATCH_SIZE, j))
            {
                return false;
            }
        }
        std::vector<void*> buffers(nbIOs, nullptr);
        // Set input tensor values
        for (int64_t i = 0; i < nbIOs; ++i)
        {
            char const* tensor;
            SAFE_API_CALL(graph->getIOTensorName(tensor, i), g_recorder);
            setTensorBuffer(graph, g_recorder, tensor, buffers[i]);
        }

        // Initialize main stream
        cudaStream_t stream;
        CUDA_CALL(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), g_recorder);

        // Asynchronously copy data from host input buffers to device input buffers.
        CUDA_CALL(cudaMemcpyAsync(buffers[inputIndex], hostBuffers[inputIndex], inputDesc.sizeInBytes,
                      cudaMemcpyHostToDevice, stream),
            g_recorder);

        // Run the graph
        SAFE_API_CALL(graph->executeAsync(stream), g_recorder);

        // Asynchronously copy data from device output buffers to host output buffers.
        CUDA_CALL(cudaMemcpyAsync(hostBuffers[outputIndex], buffers[outputIndex], outputDesc.sizeInBytes,
                      cudaMemcpyDeviceToHost, stream),
            g_recorder);

        graph->sync();

        // Check and print the output of the inference.
        outputCorrect &= verifyOutput(hostBuffers[outputIndex], groundTruthDigits, kBATCH_SIZE);

        // free host&device buffers
        free(hostBuffers[inputIndex]);
        free(hostBuffers[outputIndex]);
        CUDA_CALL(cudaFree(buffers[inputIndex]), g_recorder);
        CUDA_CALL(cudaFree(buffers[outputIndex]), g_recorder);
    }

    destroyTRTGraph(graph);
    return outputCorrect;
}

//!
//! \brief Prints the help information for running this sample.
//!
void printHelpInfo()
{
    std::cout << "Usage: ./sample_safe_plugin_infer [-h or --help] [--loadEngine=<path to engine file>]\n";
    std::cout << "--help or -h    Display help information\n";
    std::cout << "--loadEngine    Load the serialized engine from the file (default = safe_plugin.engine).\n";
}

int main(int32_t argc, char** argv)
{
    safetyCompliance::setPromgrAbility();
    SampleSafeMNISTInferArgs args;
    bool const argsOK = parseSampleSafePluginInferArgs(args, argc, argv);
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

    // Initialize SafeCuda before any other Cuda APIs are called. This may be skipped if createInferRuntime() is
    // called first as per DEEPLRN_RES_116
    safetyCompliance::initSafeCuda();

    if (!samplesSafeCommon::isSmSafe())
    {
        SAFE_LOG << "Skip safe mode test on unsupported platforms." << std::endl;
        return EXIT_SUCCESS;
    }

    TestResult result = TestResult::kPASSED;
    try
    {
        if (!doInference(args))
        {
            result = TestResult::kFAILED;
        }
    }
    catch (std::runtime_error& e)
    {
        SAFE_LOG << e.what() << std::endl;
        result = TestResult::kFAILED;
    }

    reportTestResult("TensorRT.sample_plugin_safe_infer", result, argc, argv);
    return EXIT_SUCCESS;
}
