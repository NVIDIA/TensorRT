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

//! \file SampleCuDLA.cpp
//! \brief This file contains the implementation of the cuDLA sample.
//!

// Define TRT entrypoints used in common code
#define DEFINE_TRT_ENTRYPOINTS 1
#define DEFINE_TRT_ONNX_PARSER_ENTRYPOINT 0

#include "NvInfer.h"
#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "half.h"
#include "logger.h"

#include "cudla.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

using namespace samplesCommon;

#define CHECK_CUDLA(expr)                                                             \
    do {                                                                              \
        auto const status = (expr);                                                   \
        if (status != cudlaSuccess)                                                   \
        {                                                                             \
            sample::gLogError << "Error in " << expr << " = " << status << std::endl; \
            exit(EXIT_FAILURE);                                                                  \
        }                                                                             \
    } while (0)

std::string const gSampleName = "TensorRT.sample_cudla";

bool isDLAHeader(void const* ptr)
{
    CHECK_RETURN(ptr, false);
    char const* p = static_cast<char const*>(ptr);
    return p[4] == 'N' && p[5] == 'V' && p[6] == 'D' && p[7] == 'A';
}

#if ENABLE_DLA

class DlaContext
{
public:
    DlaContext(void* data, size_t const size)
    {
        // Initialize CUDA.
        CHECK(cudaFree(0));
        CHECK(cudaSetDevice(0));
        CHECK_CUDLA(cudlaCreateDevice(0, &mDevHandle, CUDLA_CUDA_DLA));

        // Get available devices.
        uint64_t numEngines{0};
        CHECK_CUDLA(cudlaDeviceGetCount(&numEngines));
        ASSERT(numEngines >= 1);

        // Create CUDA stream.
        CHECK(cudaStreamCreateWithFlags(&mStream, cudaStreamNonBlocking));

        // Load the module with cuDLA from the loadable data.
        deserialize(data, size);

        // Create and allocate I/O tensors with the GPU momory registered with cuDLA.
        createIoTensors();
    }

    ~DlaContext()
    {
        // Unregister the memory with cuDLA and clear vectors.
        for (auto& elem : mTensorIn)
        {
            CHECK_CUDLA(cudlaMemUnregister(mDevHandle, elem));
        }
        mTensorIn.clear();

        for (auto& elem : mTensorOut)
        {
            CHECK_CUDLA(cudlaMemUnregister(mDevHandle, elem));
        }
        mTensorOut.clear();

        // Free the buffers on GPU and clear vectors.
        for (auto& elem : mBufferGPUIn)
        {
            CHECK(cudaFree(elem));
        }
        mBufferGPUIn.clear();

        for (auto& elem : mBufferGPUOut)
        {
            CHECK(cudaFree(elem));
        }
        mBufferGPUOut.clear();

        mInputTensorDesc.clear();
        mOutputTensorDesc.clear();
        CHECK(cudaStreamDestroy(mStream));

        // Unload the module with cuDLA and destroy the device.
        CHECK_CUDLA(cudlaModuleUnload(mModuleHandle, 0));
        sample::gLogInfo << "Successfully unloaded module" << std::endl;
        CHECK_CUDLA(cudlaDestroyDevice(mDevHandle));
        sample::gLogInfo << "Device destroyed successfully" << std::endl;
    }

    //!
    //! \brief Enqueue and execute the current task.
    //!
    using BufRef = std::reference_wrapper<std::vector<half_float::half>>;
    using BufRefConst = std::reference_wrapper<std::vector<half_float::half> const>;
    void submit(std::vector<BufRefConst> const& inputBufVec, std::vector<BufRef> const& outputBufVec)
    {
        // Copy data from CPU buffers to GPU buffers.
        uint32_t const inputBufVecSize = inputBufVec.size();
        for (uint32_t i = 0; i < inputBufVecSize; ++i)
        {
            auto const& inputBuf = inputBufVec.at(i).get();
            void* inputBufferGPU = mBufferGPUIn.at(i);
            CHECK(cudaMemcpyAsync(inputBufferGPU, inputBuf.data(), mInputTensorDesc.at(i).size,
                                        cudaMemcpyHostToDevice, mStream));
        }
        // Consist of the input and output tensors in the form of the addresses registered with the DLA.
        // Ensure the registered pointers are visible to the DLA before the execution.
        mTask.moduleHandle = mModuleHandle;
        mTask.outputTensor = mTensorOut.data();
        mTask.numOutputTensors = getNbOutputTensors();
        mTask.numInputTensors = getNbInputTensors();
        mTask.inputTensor = mTensorIn.data();
        mTask.waitEvents = NULL;
        mTask.signalEvents = NULL;
        CHECK_CUDLA(cudlaSubmitTask(mDevHandle, &mTask, 1, mStream, 0));
        sample::gLogInfo << "Submitted task to DLA successfully" << std::endl;
        // Bring output buffer to CPU.
        uint32_t const outputBufVecSize = outputBufVec.size();
        for (uint32_t i = 0; i < outputBufVecSize; ++i)
        {
            auto& outputBuf = outputBufVec.at(i).get();
            void* outputBufferGPU = mBufferGPUOut.at(i);
            CHECK(cudaMemcpyAsync(outputBuf.data(), outputBufferGPU, mOutputTensorDesc.at(i).size,
                                cudaMemcpyDeviceToHost, mStream));
        }
    }

    //!
    //! \brief Wait for stream operations to finish.
    //!
    void synchronize()
    {
        CHECK(cudaStreamSynchronize(mStream));
    }

private:
    cudlaDevHandle mDevHandle;                                      //!< Device handler
    cudlaModule mModuleHandle;                                      //!< Module handler
    cudlaTask mTask;                                                //!< CuDLA task
    cudaStream_t mStream;                                           //!< CUDA stream
    std::vector<cudlaModuleTensorDescriptor> mInputTensorDesc;      //!< Input tensor descriptors
    std::vector<cudlaModuleTensorDescriptor> mOutputTensorDesc;     //!< Output tensor descriptors
    std::vector<uint64_t*> mTensorIn;                               //!< Input registered pointers
    std::vector<uint64_t*> mTensorOut;                              //!< Output registered pointers
    std::vector<void*> mBufferGPUIn;                                //!< Input allocated buffers
    std::vector<void*> mBufferGPUOut;                               //!< Output allocated buffers

    //!
    //! \brief Load the module with cuDLA from the loadable data.
    //!
    void deserialize(void* data, size_t const size)
    {
        ASSERT(isDLAHeader(static_cast<char*>(data)));
        CHECK_CUDLA(cudlaModuleLoadFromMemory(mDevHandle, static_cast<unsigned char*>(data), size, &mModuleHandle, 0));
        sample::gLogInfo << "Successfully loaded module" << std::endl;
    }

    //!
    //! \brief Get the number of input tensors.
    //!
    uint32_t getNbInputTensors() const
    {
        cudlaModuleAttribute attribute;
        CHECK_CUDLA(cudlaModuleGetAttributes(mModuleHandle, CUDLA_NUM_INPUT_TENSORS, &attribute));
        return attribute.numInputTensors;
    }

    //!
    //! \brief Get the number of output tensors.
    //!
    uint32_t getNbOutputTensors() const
    {
        cudlaModuleAttribute attribute;
        CHECK_CUDLA(cudlaModuleGetAttributes(mModuleHandle, CUDLA_NUM_OUTPUT_TENSORS, &attribute));
        return attribute.numOutputTensors;
    }

    //!
    //! \brief Allocate memory for a buffer on GPU and register the required pointer with cuDLA.
    //!
    void createMemDLA(std::vector<uint64_t*>& mTensor, std::vector<void*>& mBufferGPU, uint64_t const size, int32_t const idx)
    {
        void* bufferGPU = nullptr;
        uint64_t* bufferRegisteredPtr = nullptr;
        // Allocate memory on GPU.
        CHECK(cudaMalloc(&bufferGPU, size));
        // Register the CUDA-allocated buffers.
        CHECK_CUDLA(cudlaMemRegister(mDevHandle, static_cast<uint64_t*>(bufferGPU), size, &bufferRegisteredPtr, 0));
        CHECK(cudaMemsetAsync(bufferGPU, 0, size, mStream));
        mTensor.emplace_back(bufferRegisteredPtr);
        mBufferGPU.emplace_back(bufferGPU);
    }

    //!
    //! \brief Create and allocate I/O tensors with the GPU momory registered with cuDLA.
    //!
    void createIoTensors()
    {
        // Prepare I/O tensors
        uint32_t const numInputTensors = getNbInputTensors();
        uint32_t const numOutputTensors = getNbOutputTensors();

        // Allocate memory for input and output tensor descriptors.
        mInputTensorDesc.resize(numInputTensors);
        mOutputTensorDesc.resize(numOutputTensors);

        // Get module attributes from the loaded module.
        // Fill in the input and output tensor descriptors.
        cudlaModuleAttribute attribute;
        attribute.inputTensorDesc = mInputTensorDesc.data();
        CHECK_CUDLA(cudlaModuleGetAttributes(mModuleHandle, CUDLA_INPUT_TENSOR_DESCRIPTORS, &attribute));

        attribute.outputTensorDesc = mOutputTensorDesc.data();
        CHECK_CUDLA(cudlaModuleGetAttributes(mModuleHandle, CUDLA_OUTPUT_TENSOR_DESCRIPTORS, &attribute));

        // Get each tensor size, allocate GPU memory and register that memory with cuDLA.
        for (uint32_t i = 0; i < numInputTensors; ++i)
        {
            uint64_t const inputDescSize = mInputTensorDesc.at(i).size;
            createMemDLA(mTensorIn, mBufferGPUIn, inputDescSize, i);
        }

        for (uint32_t i = 0; i < numOutputTensors; ++i)
        {
            uint64_t const outputDescSize = mOutputTensorDesc.at(i).size;
            createMemDLA(mTensorOut, mBufferGPUOut, outputDescSize, i);
        }
    }
};

//!
//! \brief Create the single layer Network and marks the output layers.
//!
void constructNetwork(nvinfer1::INetworkDefinition& network)
{
    nvinfer1::Dims const inputDims{4, {1, 32, 32, 32}};

    auto inA = network.addInput("inputA", nvinfer1::DataType::kHALF, inputDims);
    auto inB = network.addInput("inputB", nvinfer1::DataType::kHALF, inputDims);

    auto layer = network.addElementWise(*inA, *inB, nvinfer1::ElementWiseOperation::kSUM);
    nvinfer1::ITensor* out = layer->getOutput(0);

    out->setName("output");
    network.markOutput(*out);
}

//!
//! \brief Explicitly set network I/O formats.
//!
void setNetworkIOFormats(nvinfer1::INetworkDefinition& network, bool isInt8)
{
    nvinfer1::TensorFormat const formats = isInt8 ? nvinfer1::TensorFormat::kCHW32 : nvinfer1::TensorFormat::kCHW16;
    nvinfer1::DataType const dataType = isInt8 ? nvinfer1::DataType::kINT8 : nvinfer1::DataType::kHALF;
    uint32_t const numInputs = network.getNbInputs();
    for (uint32_t i = 0; i < numInputs; i++)
    {
        auto input = network.getInput(i);
        input->setType(dataType);
        input->setAllowedFormats(static_cast<nvinfer1::TensorFormats>(1U << static_cast<int32_t>(formats)));
    }

    uint32_t const numOutputs = network.getNbOutputs();
    for (uint32_t i = 0; i < numOutputs; i++)
    {
        auto output = network.getOutput(i);
        output->setType(dataType);
        output->setAllowedFormats(static_cast<nvinfer1::TensorFormats>(1U << static_cast<int32_t>(formats)));
    }
}

//!
//! \brief Randomly initializes buffer.
//!
template <typename T>
void randomInit(std::vector<T>& buffer)
{
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_int_distribution<int32_t> dist(0, 63);

    auto gen = [&dist, &mt]() { return T(dist(mt)); };
    std::generate(buffer.begin(), buffer.end(), gen);
}

//!
//! \brief Verifies that the output is correct.
//!
template <typename T>
bool verifyOutput(std::vector<T> const& ref, std::vector<T> const& output)
{
    return std::equal(ref.begin(), ref.end(), output.begin());
}

//!
//! \brief Creates the network, configures the builder, and creates the network engine.
//!
//! \details This function creates a network and builds an engine to run in DLA safe mode.
//! The network consists of only one elementwise sum layer with FP16 precision.
//!
//! \return true if the engine was created successfully and false otherwise.
//!
bool build(std::unique_ptr<nvinfer1::IHostMemory>& mLoadable, nvinfer1::Dims& mInputDims, nvinfer1::Dims& mOutputDims,
    std::string const& timingCacheFile)
{
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));
    CHECK_RETURN(builder.get(), false);

    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(0));
    CHECK_RETURN(network.get(), false);

    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    CHECK_RETURN(config.get(), false);

    constructNetwork(*network);
    setNetworkIOFormats(*network, false);

    config->setFlag(nvinfer1::BuilderFlag::kFP16);

    samplesCommon::enableDLA(builder.get(), config.get(), 0);

    config->clearFlag(nvinfer1::BuilderFlag::kGPU_FALLBACK);
    config->setEngineCapability(nvinfer1::EngineCapability::kDLA_STANDALONE);

    std::unique_ptr<nvinfer1::ITimingCache> timingCache{};

    // Load timing cache
    if (!timingCacheFile.empty())
    {
        timingCache = samplesCommon::buildTimingCacheFromFile(sample::gLogger.getTRTLogger(), *config, timingCacheFile);
    }

    mLoadable = std::unique_ptr<nvinfer1::IHostMemory>(builder->buildSerializedNetwork(*network, *config));
    if (!mLoadable)
    {
        return false;
    }

    if (timingCache != nullptr && !timingCacheFile.empty())
    {
        samplesCommon::updateTimingCacheFile(
            sample::gLogger.getTRTLogger(), timingCacheFile, timingCache.get(), *builder);
    }

    mInputDims = network->getInput(0)->getDimensions();
    mOutputDims = network->getOutput(0)->getDimensions();

    return true;
}
#endif // ENABLE_DLA

//!
//! \brief Prints the help information for running this sample.
//!
void printHelpInfo()
{
    sample::gLogInfo << "Usage: ./sample_cudla [-h or --help] [--timingCacheFile=<path to timing cache file>]\n";
    sample::gLogInfo << "--help             Display help information\n";
    sample::gLogInfo
        << "--timingCacheFile  Specify path to a timing cache file. If it does not already exist, it will be "
        << "created." << std::endl;
}

int main(int argc, char** argv)
{
    samplesCommon::Args args;
    bool argsOK = samplesCommon::parseArgs(args, argc, argv);
    if (!argsOK)
    {
        sample::gLogError << "Invalid arguments " << std::endl;
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

#if ENABLE_DLA
    std::unique_ptr<nvinfer1::IHostMemory> mLoadable{nullptr}; //!< The DLA loadable.
    nvinfer1::Dims mInputDims;                                 //!< The dimensions of the input to the network.
    nvinfer1::Dims mOutputDims;                                //!< The dimensions of the output to the network.
    CHECK_RETURN(
        build(mLoadable, mInputDims, mOutputDims, args.timingCacheFile), sample::gLogger.reportFail(sampleTest));

    int64_t const inputBufSize = samplesCommon::volume(mInputDims, 0, mInputDims.nbDims);
    int64_t const expectedOutputSize = samplesCommon::volume(mOutputDims, 0, mOutputDims.nbDims);

    // Allocate and initialize input and output buffers
    // The allocation and initialization only needs to be done once
    std::vector<half_float::half> inputBufA(inputBufSize);
    std::vector<half_float::half> inputBufB(inputBufSize);
    std::vector<half_float::half> referenceBuf(expectedOutputSize);
    std::vector<half_float::half> outputBuf(expectedOutputSize);

    randomInit(inputBufA);
    randomInit(inputBufB);

    // Create a DlaContext
    DlaContext context(mLoadable->data(), mLoadable->size());

    // The cuDLA task can be submitted more than once
    context.submit({inputBufA, inputBufB}, {outputBuf});

    context.synchronize();

    // Compute the reference output for comparision
    std::transform(
        inputBufA.begin(), inputBufA.end(), inputBufB.begin(), referenceBuf.begin(), std::plus<half_float::half>());
    CHECK_RETURN(verifyOutput(referenceBuf, outputBuf), sample::gLogger.reportFail(sampleTest));
#else  // ENABLE_DLA
    sample::gLogError << "DLA is not enabled, please compile with ENABLE_DLA=1" << std::endl;
#endif // ENABLE_DLA

    return sample::gLogger.reportPass(sampleTest);
}
