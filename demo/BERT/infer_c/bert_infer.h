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

#ifndef INFER_C_BERT_INFER_H
#define INFER_C_BERT_INFER_H

#include "common.h"
#include "logging.h"
#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <algorithm>
#include <cuda_runtime.h>
#include <fstream>
#include <numeric>
#include <string.h>
#include <vector>

using namespace nvinfer1;

struct BertInference
{
    BertInference(
        const std::string& enginePath, const int maxBatchSize, const int seqLength, const bool enableGraph = false)
        : mSeqLength(seqLength)
        , mEnableGraph(enableGraph)
    {
        gLogInfo << "--------------------\n";
        gLogInfo << "Using BERT inference C++\n";
        if (enableGraph)
        {
            gLogInfo << "CUDA Graph is enabled\n";
        }
        else
        {
            gLogInfo << "CUDA Graph is disabled\n";
        }

        gLogInfo << "--------------------\n";

        initLibNvInferPlugins(&gLogger, "");

        gLogInfo << "Loading BERT Inference Engine ... \n";
        std::ifstream input(enginePath, std::ios::binary);
        if (!input)
        {
            gLogError << "Error opening engine file: " << enginePath << "\n";
            exit(-1);
        }

        input.seekg(0, input.end);
        const size_t fsize = input.tellg();
        input.seekg(0, input.beg);

        std::vector<char> bytes(fsize);
        input.read(bytes.data(), fsize);

        mRuntime = TrtUniquePtr<IRuntime>(createInferRuntime(gLogger));
        if (mRuntime == nullptr)
        {
            gLogError << "Error creating TRT mRuntime\n";
            exit(-1);
        }

        mEngine = TrtUniquePtr<ICudaEngine>(mRuntime->deserializeCudaEngine(bytes.data(), bytes.size()));
        if (mEngine == nullptr)
        {
            gLogError << "Error deserializing CUDA engine\n";
            exit(-1);
        }
        gLogInfo << "Done\n";

        const int numBindingPerProfile = mEngine->getNbBindings() / mEngine->getNbOptimizationProfiles();
        mEnableVariableLen = numBindingPerProfile == kBERT_INPUT_NUM + 1 ? false : true;
        if (mEnableVariableLen)
        {
            gLogInfo << "Variable length is enabled\n";
        }
        else
        {
            gLogInfo << "Variable length is disabled\n";
        }

        mContext = TrtUniquePtr<IExecutionContext>(mEngine->createExecutionContext());
        if (!mContext)
        {
            gLogError << "Error creating execution context\n";
            exit(-1);
        }

        gpuErrChk(cudaStreamCreate(&mStream));

        allocateBindings(maxBatchSize);
    }

    void allocateBindings(const int maxBatchSize)
    {
        const size_t allocationSize = mSeqLength * maxBatchSize * sizeof(int32_t);

        // Static sizes with implicit batch size: allocation sizes known to engine
        if (mEnableVariableLen)
        {
            const size_t allocationSizes[] = {allocationSize, allocationSize,
                                              sizeof(int32_t) * (maxBatchSize + 1),
                                              sizeof(int32_t) * (mSeqLength)};
            for (int i = 0; i < sizeof(allocationSizes) / sizeof(allocationSizes[0]); i++)
            {
                void* devBuf;
                gpuErrChk(cudaMalloc(&devBuf, allocationSizes[i]));
                gpuErrChk(cudaMemset(devBuf, 0, allocationSizes[i]));
                mDeviceBuffers.emplace_back(devBuf);
                mInputSizes.emplace_back(allocationSizes[i]);
            }
        }
        else
        {
            for (int i = 0; i < kBERT_INPUT_NUM; i++)
            {
                void* devBuf;
                gpuErrChk(cudaMalloc(&devBuf, allocationSize));
                gpuErrChk(cudaMemset(devBuf, 0, allocationSize));
                mDeviceBuffers.emplace_back(devBuf);
                mInputSizes.emplace_back(allocationSize);
            }
        }

        const size_t numOutputItems = maxBatchSize * mSeqLength * 2;
        mOutputSize = numOutputItems * sizeof(float);
        if (mEnableVariableLen)
        {
            mOutputDims = {maxBatchSize * mSeqLength * 2};
        }
        else
        {
            mOutputDims = {maxBatchSize, mSeqLength, 2, 1, 1};
        }
        void* devBuf;
        gpuErrChk(cudaMalloc(&devBuf, mOutputSize));
        gpuErrChk(cudaMemset(devBuf, 0, mOutputSize));
        mDeviceBuffers.emplace_back(devBuf);
        mHostOutput.resize(numOutputItems);

        mBindings.resize(mEngine->getNbBindings());
    }

    void prepare(int profIdx, int batchSize)
    {

        mContext->setOptimizationProfile(profIdx);
        const int numBindingPerProfile = mEngine->getNbBindings() / mEngine->getNbOptimizationProfiles();
        const int bindingIdxOffset = profIdx * numBindingPerProfile;
        std::copy(mDeviceBuffers.begin(), mDeviceBuffers.end(), mBindings.begin() + bindingIdxOffset);

        if (mEnableVariableLen)
        {
            const int allocationSizes[] = {mSeqLength * batchSize, mSeqLength * batchSize, batchSize + 1, mSeqLength};
            for (int i = 0; i < sizeof(allocationSizes)/sizeof(allocationSizes[0]); i++)
            {
                mContext->setBindingDimensions(i + bindingIdxOffset, Dims{1, {allocationSizes[i]}});
            }
        }
        else
        {
            for (int i = 0; i < kBERT_INPUT_NUM; i++)
            {
                mContext->setBindingDimensions(i + bindingIdxOffset, Dims2(batchSize, mSeqLength));
            }
        }

        if (!mContext->allInputDimensionsSpecified())
        {
            gLogError << "Not all input dimensions are specified for the exeuction context\n";
            exit(-1);
        }

        if (mEnableGraph)
        {
            cudaGraph_t graph;
            cudaGraphExec_t exec;
            // warm up and let mContext do cublas initialization
            bool status = mContext->enqueueV2(mBindings.data(), mStream, nullptr);
            if (!status)
            {
                gLogError << "Enqueue failed\n";
                exit(-1);
            }
            gLogVerbose << "Capturing graph\n";

            gpuErrChk(cudaStreamBeginCapture(mStream, cudaStreamCaptureModeRelaxed));
            status = mContext->enqueueV2(mBindings.data(), mStream, nullptr);
            if (!status)
            {
                gLogError << "Enqueue failed\n";
                exit(-1);
            }

            gpuErrChk(cudaStreamEndCapture(mStream, &graph));
            gpuErrChk(cudaStreamSynchronize(mStream));

            gpuErrChk(cudaGraphInstantiate(&exec, graph, NULL, NULL, 0));
            mExecGraph = exec;
        }
        mCuSeqlens.resize(batchSize + 1);
        std::generate(mCuSeqlens.begin(), mCuSeqlens.end(), [pos = -mSeqLength, this]() mutable{ pos += mSeqLength; return pos; });
    }

    void run(const void* const* inputBuffers, int warmUps, int iterations)
    {
        for (int i = 0; i < kBERT_INPUT_NUM; i++)
        {
            gpuErrChk(
                cudaMemcpyAsync(mDeviceBuffers[i], inputBuffers[i], mInputSizes[i], cudaMemcpyHostToDevice, mStream));
        }

        gLogInfo << "Warming up " << warmUps << " iterations ...\n";
        for (int it = 0; it < warmUps; it++)
        {
            if (mEnableGraph)
            {
                gpuErrChk(cudaGraphLaunch(mExecGraph, mStream));
            }
            else
            {
                bool status = mContext->enqueueV2(mBindings.data(), mStream, nullptr);
                if (!status)
                {
                    gLogError << "Enqueue failed\n";
                    exit(-1);
                }
            }
        }
        gpuErrChk(cudaStreamSynchronize(mStream));

        cudaEvent_t start, stop;
        gpuErrChk(cudaEventCreate(&start));
        gpuErrChk(cudaEventCreate(&stop));

        std::vector<float> times;
        gLogInfo << "Running " << iterations << " iterations ...\n";
        for (int it = 0; it < iterations; it++)
        {
            gpuErrChk(cudaEventRecord(start, mStream));
            if (mEnableGraph)
            {
                gpuErrChk(cudaGraphLaunch(mExecGraph, mStream));
            }
            else
            {
                bool status = mContext->enqueueV2(mBindings.data(), mStream, nullptr);
                if (!status)
                {
                    gLogError << "Enqueue failed\n";
                    exit(-1);
                }
            }
            gpuErrChk(cudaEventRecord(stop, mStream));
            gpuErrChk(cudaStreamSynchronize(mStream));
            float time;
            gpuErrChk(cudaEventElapsedTime(&time, start, stop));
            times.push_back(time);
        }

        gpuErrChk(cudaMemcpyAsync(
            mHostOutput.data(), mDeviceBuffers[mEnableVariableLen ? kBERT_INPUT_NUM + 1 : kBERT_INPUT_NUM], mOutputSize, cudaMemcpyDeviceToHost, mStream));

        gpuErrChk(cudaStreamSynchronize(mStream));

        mTimes.push_back(times);
    }

    void run(const void* inputIds, const void* segmentIds, const void* inputMask, int warmUps, int iterations)
    {
        if (mEnableVariableLen)
        {
            const std::vector<const void*> inputBuffers = {inputIds, segmentIds, mCuSeqlens.data()};
            run(inputBuffers.data(), warmUps, iterations);
        }
        else
        {
            const std::vector<const void*> inputBuffers = {inputIds, segmentIds, inputMask};
            run(inputBuffers.data(), warmUps, iterations);
        }
    }

    void run(int profIdx, int batchSize, const void* inputIds, const void* segmentIds, const void* inputMask,
        int warmUps, int iterations)
    {

        prepare(profIdx, batchSize);
        run(inputIds, segmentIds, inputMask, warmUps, iterations);
    }

    void reportTiming(int batchIndex, int batchSize)
    {

        std::vector<float>& times = mTimes[batchIndex];
        const float totalTime = std::accumulate(times.begin(), times.end(), 0.0);
        const float avgTime = totalTime / times.size();

        sort(times.begin(), times.end());
        const float percentile95 = times[(int) ((float) times.size() * 0.95)];
        const float percentile99 = times[(int) ((float) times.size() * 0.99)];
        const int throughput = (int) ((float) batchSize * (1000.0 / avgTime));
        gLogInfo << "Running " << times.size() << " iterations with Batch Size: " << batchSize << "\n";
        gLogInfo << "\tTotal Time: " << totalTime << " ms \n";
        gLogInfo << "\tAverage Time: " << avgTime << " ms\n";
        gLogInfo << "\t95th Percentile Time: " << percentile95 << " ms\n";
        gLogInfo << "\t99th Percentile Time: " << percentile99 << " ms\n";
        gLogInfo << "\tThroughput: " << throughput << " sentences/s\n";
    }

    ~BertInference()
    {

        gpuErrChk(cudaStreamDestroy(mStream));

        for (auto& buf : mDeviceBuffers)
        {
            gpuErrChk(cudaFree(buf));
        }
    }

    static const int kBERT_INPUT_NUM = 3;

    const int mSeqLength;
    const bool mEnableGraph;

    TrtUniquePtr<IRuntime> mRuntime{nullptr};
    TrtUniquePtr<ICudaEngine> mEngine{nullptr};
    TrtUniquePtr<IExecutionContext> mContext{nullptr};
    std::vector<void*> mBindings;
    bool mEnableVariableLen;
    std::vector<int> mCuSeqlens;

    cudaStream_t mStream{NULL};
    std::vector<void*> mDeviceBuffers;
    std::vector<float> mHostOutput;
    std::vector<size_t> mInputSizes;
    size_t mOutputSize;
    std::vector<int> mOutputDims;

    std::vector<std::vector<float>> mTimes;

    cudaGraphExec_t mExecGraph;
};

#endif // INFER_C_BERT_INFER_H
