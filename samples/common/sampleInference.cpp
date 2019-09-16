/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#include <vector>
#include <chrono>
#include <numeric>
#include <utility> 
#include <thread>
#include <mutex>
#include <functional>
#include <limits>

#include "NvInfer.h"

#include "sampleUtils.h"
#include "sampleOptions.h"
#include "sampleReporting.h"
#include "sampleInference.h"

namespace sample
{

void setUpInference(InferenceEnvironment& iEnv, const InferenceOptions& inference)
{
    if (iEnv.profiler)
    {
        iEnv.context.front()->setProfiler(iEnv.profiler.get());
    }
    for (int s = 0; s < inference.streams; ++s)
    {
        iEnv.context.emplace_back(iEnv.engine->createExecutionContext());
    }
    iEnv.bindings.resize(inference.streams);

    // Set all input dimensions before all bindings can be allocated
    for (int b = 0; b < iEnv.engine->getNbBindings(); ++b)
    {
        if (iEnv.engine->bindingIsInput(b))
        {
            auto dims = iEnv.context.front()->getBindingDimensions(b);
            if (std::any_of(dims.d, dims.d + dims.nbDims, [](int d) { return d == -1; }))
            {
                auto shape = inference.shapes.find(iEnv.engine->getBindingName(b));
                std::copy(shape->second.d, shape->second.d + shape->second.nbDims, dims.d);
                for (auto& c : iEnv.context)
                {
                    c->setBindingDimensions(b, dims);
                }
            }
        }
    }

    for (int b = 0; b < iEnv.engine->getNbBindings(); ++b)
    {
        auto dims = iEnv.context.front()->getBindingDimensions(b);
        auto vecDim = iEnv.engine->getBindingVectorizedDim(b);
        if (vecDim != -1)
        {
            dims.d[vecDim] = roundUp(dims.d[vecDim], iEnv.engine->getBindingComponentsPerElement(b));
        }
        auto name = iEnv.engine->getBindingName(b);
        auto vol = volume(dims) * std::max(inference.batch, 1);
        vol *= dataTypeSize(iEnv.engine->getBindingDataType(b));
        for (auto& bin : iEnv.bindings)
        {
            bin.addBinding(b, name, vol);
        }
    }
}

namespace {

struct SynchStruct
{
    std::mutex mutex;
    TrtCudaStream mainStream;
    TrtCudaEvent mainStart{cudaEventBlockingSync};
    int sleep{0};
    InferenceTime totalTime;
};

struct IterStruct
{
    TrtCudaStream stream;
    TrtCudaEvent start{cudaEventBlockingSync};
    TrtCudaEvent end{cudaEventBlockingSync};
    nvinfer1::IExecutionContext* context{nullptr};
    void** buffers{nullptr};
};

inline
void enqueue(nvinfer1::IExecutionContext& context, int batch, void** buffers, TrtCudaStream& stream)
{
    if (batch)
    {
        context.enqueue(batch, buffers, stream.get(), nullptr);
    }
    else
    {
        context.enqueueV2(buffers, stream.get(), nullptr);
    }
}

void inferenceLoop(std::vector<IterStruct>& streamItor, SynchStruct& synch, int batch, int iterations, float maxDuration, float warmup, std::vector<InferenceTime>& times)
{
    float duration = 0;
    float gpuStart = 0;
    int skip = 0;

    auto loopStart = std::chrono::high_resolution_clock::now();
    auto timingStart{loopStart};
    auto timingEnd{loopStart};

    for (int i = 0; i - skip < iterations || duration < maxDuration; ++i)
    {
        auto iterStart = std::chrono::high_resolution_clock::now();

        for (auto& s : streamItor)
        {
            s.start.record(s.stream);
            enqueue(*s.context, batch, s.buffers, s.stream);
            s.end.record(s.stream);
        }
        float currentGpuStart = std::numeric_limits<float>::max();
        for (auto& s : streamItor)
        {
            s.end.synchronize();
            currentGpuStart = std::min(currentGpuStart, s.start - synch.mainStart);
        }

        auto iterEnd = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration<float, std::milli>(iterEnd - loopStart).count();

        if (duration < warmup)
        {
            ++skip;
            timingStart = std::chrono::high_resolution_clock::now();
            gpuStart = currentGpuStart;
            continue;
        }
        else
        {
            timingEnd = iterEnd;
        }

        float latency = std::chrono::duration<float, std::milli>(iterEnd - iterStart).count();
        for (auto& s : streamItor)
        {
            float gpuTime = s.end - s.start;
            times.emplace_back(latency, gpuTime);
        }
    }

    float totalLatency = std::chrono::duration<float, std::milli>(timingEnd - timingStart).count();
    float totalGpuTime = 0;
    for (auto& s : streamItor)
    {
        totalGpuTime = std::max(totalGpuTime, s.end - synch.mainStart);
    }
    totalGpuTime -= gpuStart;
    times.emplace_back(totalLatency, totalGpuTime);
}

void inferenceExecution(const InferenceOptions& inference, InferenceEnvironment& iEnv, SynchStruct& synch, int offset, int streams, std::vector<InferenceTime>& trace)
{
    float warmup = static_cast<float>(inference.warmup);
    float duration = static_cast<float>(inference.duration * 1000 + inference.warmup);

    std::vector<IterStruct> streamItor(streams);
    for (auto& s : streamItor)
    {
        if (inference.spin)
        {
            s.start.reset(cudaEventDefault);
            s.end.reset(cudaEventDefault);
        }

        s.context = iEnv.context[offset].get();
        s.buffers = iEnv.bindings[offset].getDeviceBuffers();
        ++offset;
    }

    // Allocate enough space for all iterations and the duration assuming 1ms inference
    // to avoid allocations during timing
    std::vector<InferenceTime> times;
    times.reserve(static_cast<size_t>(std::max(inference.iterations, static_cast<int>(duration * 1000))));

    for (auto& s : streamItor)
    {
        s.stream.wait(synch.mainStart);
    }

    inferenceLoop(streamItor, synch, inference.batch, inference.iterations, duration, warmup, times);

    synch.mutex.lock();
    trace.insert(trace.end(), times.begin(), times.end() - 1);
    synch.totalTime.latency = std::max(synch.totalTime.latency, times.back().latency);
    synch.totalTime.gpuTime = std::max(synch.totalTime.gpuTime, times.back().gpuTime);
    synch.mutex.unlock();
}

inline
std::thread makeThread(const InferenceOptions& inference, InferenceEnvironment& iEnv, SynchStruct& synch, int thread, int streamsPerThread, std::vector<InferenceTime>& trace)
{
    return std::thread(inferenceExecution, std::cref(inference), std::ref(iEnv), std::ref(synch), thread, streamsPerThread, std::ref(trace));
}

} // namespace

void runInference(const InferenceOptions& inference, InferenceEnvironment& iEnv, std::vector<InferenceTime>& trace)
{
    trace.resize(0);

    SynchStruct synch;
    synch.sleep = inference.sleep;
    synch.mainStream.sleep(&synch.sleep);
    synch.mainStart.record(synch.mainStream);

    int threadsNum = inference.threads ? inference.streams : 1;
    int streamsPerThread  = inference.streams / threadsNum;

    std::vector<std::thread> threads;
    for (int t = 0; t < threadsNum; ++t)
    {
        threads.emplace_back(makeThread(inference, iEnv, synch, t, streamsPerThread, trace));
    }
    for (auto& th : threads)
    {
        th.join();
    }

    trace.emplace_back(synch.totalTime);
}

} // namespace sample
