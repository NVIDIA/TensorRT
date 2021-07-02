/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

#include <algorithm>
#include <array>
#include <chrono>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <thread>
#include <utility>
#include <vector>

#include "NvInfer.h"

#include "logger.h"
#include "sampleDevice.h"
#include "sampleInference.h"
#include "sampleOptions.h"
#include "sampleReporting.h"
#include "sampleUtils.h"

namespace sample
{

bool setUpInference(InferenceEnvironment& iEnv, const InferenceOptions& inference)
{
    for (int s = 0; s < inference.streams; ++s)
    {
        iEnv.context.emplace_back(iEnv.engine->createExecutionContext());
        iEnv.bindings.emplace_back(new Bindings);
    }
    if (iEnv.profiler)
    {
        iEnv.context.front()->setProfiler(iEnv.profiler.get());
    }

    const int nOptProfiles = iEnv.engine->getNbOptimizationProfiles();
    const int nBindings = iEnv.engine->getNbBindings();
    const int bindingsInProfile = nOptProfiles > 0 ? nBindings / nOptProfiles : 0;
    const int endBindingIndex = bindingsInProfile ? bindingsInProfile : iEnv.engine->getNbBindings();

    if (nOptProfiles > 1)
    {
        sample::gLogWarning << "Multiple profiles are currently not supported. Running with one profile." << std::endl;
    }

    // Set all input dimensions before all bindings can be allocated
    for (int b = 0; b < endBindingIndex; ++b)
    {
        if (iEnv.engine->bindingIsInput(b))
        {
            auto dims = iEnv.context.front()->getBindingDimensions(b);
            const bool isScalar = dims.nbDims == 0;
            const bool isDynamicInput = std::any_of(dims.d, dims.d + dims.nbDims, [](int dim) { return dim == -1; })
                || iEnv.engine->isShapeBinding(b);
            if (isDynamicInput)
            {
                auto shape = inference.shapes.find(iEnv.engine->getBindingName(b));

                // If no shape is provided, set dynamic dimensions to 1.
                std::vector<int> staticDims;
                if (shape == inference.shapes.end())
                {
                    constexpr int DEFAULT_DIMENSION = 1;
                    if (iEnv.engine->isShapeBinding(b))
                    {
                        if (isScalar)
                        {
                            staticDims.push_back(1);
                        }
                        else
                        {
                            staticDims.resize(dims.d[0]);
                            std::fill(staticDims.begin(), staticDims.end(), DEFAULT_DIMENSION);
                        }
                    }
                    else
                    {
                        staticDims.resize(dims.nbDims);
                        std::transform(dims.d, dims.d + dims.nbDims, staticDims.begin(),
                            [&](int dimension) { return dimension >= 0 ? dimension : DEFAULT_DIMENSION; });
                    }
                    sample::gLogWarning << "Dynamic dimensions required for input: " << iEnv.engine->getBindingName(b)
                                        << ", but no shapes were provided. Automatically overriding shape to: "
                                        << staticDims << std::endl;
                }
                else
                {
                    staticDims = shape->second;
                }

                for (auto& c : iEnv.context)
                {
                    if (iEnv.engine->isShapeBinding(b))
                    {
                        if (!c->setInputShapeBinding(b, staticDims.data()))
                        {
                            return false;
                        }
                    }
                    else
                    {
                        if (!c->setBindingDimensions(b, toDims(staticDims)))
                        {
                            return false;
                        }
                    }
                }
            }
        }
    }

    for (int b = 0; b < endBindingIndex; ++b)
    {
        const auto dims = iEnv.context.front()->getBindingDimensions(b);
        const auto vecDim = iEnv.engine->getBindingVectorizedDim(b);
        const auto comps = iEnv.engine->getBindingComponentsPerElement(b);
        const auto dataType = iEnv.engine->getBindingDataType(b);
        const auto strides = iEnv.context.front()->getStrides(b);
        const int batch = iEnv.engine->hasImplicitBatchDimension() ? inference.batch : 1;
        const auto vol = volume(dims, strides, vecDim, comps, batch);
        const auto name = iEnv.engine->getBindingName(b);
        const auto isInput = iEnv.engine->bindingIsInput(b);
        for (auto& bindings : iEnv.bindings)
        {
            const auto input = inference.inputs.find(name);
            if (isInput && input != inference.inputs.end())
            {
                bindings->addBinding(b, name, isInput, vol, dataType, input->second);
            }
            else
            {
                bindings->addBinding(b, name, isInput, vol, dataType);
            }

            if (isInput)
            {
                sample::gLogInfo << "Created input binding for " << name << " with dimensions " << dims << std::endl;
            }
            else
            {
                sample::gLogInfo << "Created output binding for " << name << " with dimensions " << dims << std::endl;
            }
        }
    }

    return true;
}

namespace
{

using TimePoint = std::chrono::time_point<std::chrono::high_resolution_clock>;

//!
//! \struct SyncStruct
//! \brief Threads synchronization structure
//!
struct SyncStruct
{
    std::mutex mutex;
    TrtCudaStream mainStream;
    TrtCudaEvent gpuStart{cudaEventBlockingSync};
    TimePoint cpuStart{};
    int sleep{0};
};

struct Enqueue
{
    explicit Enqueue(nvinfer1::IExecutionContext& context, void** buffers)
        : mContext(context)
        , mBuffers(buffers)
    {
    }

    nvinfer1::IExecutionContext& mContext;
    void** mBuffers{};
};

//!
//! \class EnqueueImplicit
//! \brief Functor to enqueue inference with implict batch
//!
class EnqueueImplicit : private Enqueue
{

public:
    explicit EnqueueImplicit(nvinfer1::IExecutionContext& context, void** buffers, int batch)
        : Enqueue(context, buffers)
        , mBatch(batch)
    {
    }

    bool operator()(TrtCudaStream& stream) const
    {
        return mContext.enqueue(mBatch, mBuffers, stream.get(), nullptr);
    }

private:
    int mBatch;
};

//!
//! \class EnqueueExplicit
//! \brief Functor to enqueue inference with explict batch
//!
class EnqueueExplicit : private Enqueue
{

public:
    explicit EnqueueExplicit(nvinfer1::IExecutionContext& context, void** buffers)
        : Enqueue(context, buffers)
    {
    }

    bool operator()(TrtCudaStream& stream) const
    {
        return mContext.enqueueV2(mBuffers, stream.get(), nullptr);
    }
};

//!
//! \class EnqueueGraph
//! \brief Functor to enqueue inference from CUDA Graph
//!
class EnqueueGraph
{

public:
    explicit EnqueueGraph(TrtCudaGraph& graph)
        : mGraph(graph)
    {
    }

    bool operator()(TrtCudaStream& stream) const
    {
        return mGraph.launch(stream);
    }

    TrtCudaGraph& mGraph;
};

using EnqueueFunction = std::function<bool(TrtCudaStream&)>;

enum class StreamType : int
{
    kINPUT = 0,
    kCOMPUTE = 1,
    kOUTPUT = 2,
    kNUM = 3
};

enum class EventType : int
{
    kINPUT_S = 0,
    kINPUT_E = 1,
    kCOMPUTE_S = 2,
    kCOMPUTE_E = 3,
    kOUTPUT_S = 4,
    kOUTPUT_E = 5,
    kNUM = 6
};

using MultiStream = std::array<TrtCudaStream, static_cast<int>(StreamType::kNUM)>;

using MultiEvent = std::array<std::unique_ptr<TrtCudaEvent>, static_cast<int>(EventType::kNUM)>;

using EnqueueTimes = std::array<TimePoint, 2>;

//!
//! \class Iteration
//! \brief Inference iteration and streams management
//!
class Iteration
{

public:
    Iteration(int id, const InferenceOptions& inference, nvinfer1::IExecutionContext& context, Bindings& bindings)
        : mBindings(bindings)
        , mStreamId(id)
        , mDepth(1 + inference.overlap)
        , mActive(mDepth)
        , mEvents(mDepth)
        , mEnqueueTimes(mDepth)
    {
        for (int d = 0; d < mDepth; ++d)
        {
            for (int e = 0; e < static_cast<int>(EventType::kNUM); ++e)
            {
                mEvents[d][e].reset(new TrtCudaEvent(!inference.spin));
            }
        }
        createEnqueueFunction(inference, context, bindings);
    }

    bool query(bool skipTransfers)
    {
        if (mActive[mNext])
        {
            return true;
        }

        if (!skipTransfers)
        {
            record(EventType::kINPUT_S, StreamType::kINPUT);
            mBindings.transferInputToDevice(getStream(StreamType::kINPUT));
            record(EventType::kINPUT_E, StreamType::kINPUT);
            wait(EventType::kINPUT_E, StreamType::kCOMPUTE); // Wait for input DMA before compute
        }

        record(EventType::kCOMPUTE_S, StreamType::kCOMPUTE);
        recordEnqueueTime();
        if (!mEnqueue(getStream(StreamType::kCOMPUTE)))
        {
            return false;
        }
        recordEnqueueTime();
        record(EventType::kCOMPUTE_E, StreamType::kCOMPUTE);

        if (!skipTransfers)
        {
            wait(EventType::kCOMPUTE_E, StreamType::kOUTPUT); // Wait for compute before output DMA
            record(EventType::kOUTPUT_S, StreamType::kOUTPUT);
            mBindings.transferOutputToHost(getStream(StreamType::kOUTPUT));
            record(EventType::kOUTPUT_E, StreamType::kOUTPUT);
        }

        mActive[mNext] = true;
        moveNext();
        return true;
    }

    float sync(
        const TimePoint& cpuStart, const TrtCudaEvent& gpuStart, std::vector<InferenceTrace>& trace, bool skipTransfers)
    {
        if (mActive[mNext])
        {
            if (skipTransfers)
            {
                getEvent(EventType::kCOMPUTE_E).synchronize();
            }
            else
            {
                getEvent(EventType::kOUTPUT_E).synchronize();
            }
            trace.emplace_back(getTrace(cpuStart, gpuStart, skipTransfers));
            mActive[mNext] = false;
            return getEvent(EventType::kCOMPUTE_S) - gpuStart;
        }
        return 0;
    }

    void syncAll(
        const TimePoint& cpuStart, const TrtCudaEvent& gpuStart, std::vector<InferenceTrace>& trace, bool skipTransfers)
    {
        for (int d = 0; d < mDepth; ++d)
        {
            sync(cpuStart, gpuStart, trace, skipTransfers);
            moveNext();
        }
    }

    void wait(TrtCudaEvent& gpuStart)
    {
        getStream(StreamType::kINPUT).wait(gpuStart);
    }

    void setInputData()
    {
        mBindings.transferInputToDevice(getStream(StreamType::kINPUT));
    }

    void fetchOutputData()
    {
        mBindings.transferOutputToHost(getStream(StreamType::kOUTPUT));
    }

private:
    void moveNext()
    {
        mNext = mDepth - 1 - mNext;
    }

    TrtCudaStream& getStream(StreamType t)
    {
        return mStream[static_cast<int>(t)];
    }

    TrtCudaEvent& getEvent(EventType t)
    {
        return *mEvents[mNext][static_cast<int>(t)];
    }

    void record(EventType e, StreamType s)
    {
        getEvent(e).record(getStream(s));
    }

    void recordEnqueueTime()
    {
        mEnqueueTimes[mNext][enqueueStart] = std::chrono::high_resolution_clock::now();
        enqueueStart = 1 - enqueueStart;
    }

    TimePoint getEnqueueTime(bool start)
    {
        return mEnqueueTimes[mNext][start ? 0 : 1];
    }

    void wait(EventType e, StreamType s)
    {
        getStream(s).wait(getEvent(e));
    }

    InferenceTrace getTrace(const TimePoint& cpuStart, const TrtCudaEvent& gpuStart, bool skipTransfers)
    {
        float is
            = skipTransfers ? getEvent(EventType::kCOMPUTE_S) - gpuStart : getEvent(EventType::kINPUT_S) - gpuStart;
        float ie
            = skipTransfers ? getEvent(EventType::kCOMPUTE_S) - gpuStart : getEvent(EventType::kINPUT_E) - gpuStart;
        float os
            = skipTransfers ? getEvent(EventType::kCOMPUTE_E) - gpuStart : getEvent(EventType::kOUTPUT_S) - gpuStart;
        float oe
            = skipTransfers ? getEvent(EventType::kCOMPUTE_E) - gpuStart : getEvent(EventType::kOUTPUT_E) - gpuStart;

        return InferenceTrace(mStreamId,
            std::chrono::duration<float, std::milli>(getEnqueueTime(true) - cpuStart).count(),
            std::chrono::duration<float, std::milli>(getEnqueueTime(false) - cpuStart).count(), is, ie,
            getEvent(EventType::kCOMPUTE_S) - gpuStart, getEvent(EventType::kCOMPUTE_E) - gpuStart, os, oe);
    }

    void createEnqueueFunction(
        const InferenceOptions& inference, nvinfer1::IExecutionContext& context, Bindings& bindings)
    {
        if (inference.batch)
        {
            mEnqueue = EnqueueFunction(EnqueueImplicit(context, mBindings.getDeviceBuffers(), inference.batch));
        }
        else
        {
            mEnqueue = EnqueueFunction(EnqueueExplicit(context, mBindings.getDeviceBuffers()));
        }
        if (inference.graph)
        {
            TrtCudaStream& stream = getStream(StreamType::kCOMPUTE);
            // Avoid capturing initialization calls by executing the enqueue function at least
            // once before starting CUDA graph capture.
            const auto ret = mEnqueue(stream);
            assert(ret);
            stream.synchronize();

            mGraph.beginCapture(stream);
            // The built TRT engine may contain operations that are not permitted under CUDA graph capture mode.
            // When the stream is capturing, the enqueue call may return false if the current CUDA graph capture fails.
            if (mEnqueue(stream))
            {
                mGraph.endCapture(stream);
                mEnqueue = EnqueueFunction(EnqueueGraph(mGraph));
            }
            else
            {
                mGraph.endCaptureOnError(stream);
                // Ensure any CUDA error has been cleaned up.
                cudaCheck(cudaGetLastError());
                sample::gLogWarning << "The built TensorRT engine contains operations that are not permitted under "
                                       "CUDA graph capture mode."
                                    << std::endl;
                sample::gLogWarning << "The specified --useCudaGraph flag has been ignored. The inference will be "
                                       "launched without using CUDA graph launch."
                                    << std::endl;
            }
        }
    }

    Bindings& mBindings;

    TrtCudaGraph mGraph;
    EnqueueFunction mEnqueue;

    int mStreamId{0};
    int mNext{0};
    int mDepth{2}; // default to double buffer to hide DMA transfers

    std::vector<bool> mActive;
    MultiStream mStream;
    std::vector<MultiEvent> mEvents;

    int enqueueStart{0};
    std::vector<EnqueueTimes> mEnqueueTimes;
};

using IterationStreams = std::vector<std::unique_ptr<Iteration>>;

bool inferenceLoop(IterationStreams& iStreams, const TimePoint& cpuStart, const TrtCudaEvent& gpuStart, int iterations,
    float maxDurationMs, float warmupMs, std::vector<InferenceTrace>& trace, bool skipTransfers)
{
    float durationMs = 0;
    int skip = 0;

    for (int i = 0; i < iterations + skip || durationMs < maxDurationMs; ++i)
    {
        for (auto& s : iStreams)
        {
            if (!s->query(skipTransfers))
            {
                return false;
            }
        }
        for (auto& s : iStreams)
        {
            durationMs = std::max(durationMs, s->sync(cpuStart, gpuStart, trace, skipTransfers));
        }
        if (durationMs < warmupMs) // Warming up
        {
            if (durationMs) // Skip complete iterations
            {
                ++skip;
            }
            continue;
        }
    }
    for (auto& s : iStreams)
    {
        s->syncAll(cpuStart, gpuStart, trace, skipTransfers);
    }
    return true;
}

void inferenceExecution(const InferenceOptions& inference, InferenceEnvironment& iEnv, SyncStruct& sync, int offset,
    int streams, int device, std::vector<InferenceTrace>& trace)
{
    float warmupMs = static_cast<float>(inference.warmup);
    float durationMs = static_cast<float>(inference.duration) * 1000 + warmupMs;

    cudaCheck(cudaSetDevice(device));

    IterationStreams iStreams;
    for (int s = 0; s < streams; ++s)
    {
        Iteration* iteration = new Iteration(offset + s, inference, *iEnv.context[offset], *iEnv.bindings[offset]);
        if (inference.skipTransfers)
        {
            iteration->setInputData();
        }
        iStreams.emplace_back(iteration);
    }

    for (auto& s : iStreams)
    {
        s->wait(sync.gpuStart);
    }

    std::vector<InferenceTrace> localTrace;
    if (!inferenceLoop(iStreams, sync.cpuStart, sync.gpuStart, inference.iterations, durationMs, warmupMs, localTrace,
            inference.skipTransfers))
    {
        iEnv.error = true;
    }

    if (inference.skipTransfers)
    {
        for (auto& s : iStreams)
        {
            s->fetchOutputData();
        }
    }

    sync.mutex.lock();
    trace.insert(trace.end(), localTrace.begin(), localTrace.end());
    sync.mutex.unlock();
}

inline std::thread makeThread(const InferenceOptions& inference, InferenceEnvironment& iEnv, SyncStruct& sync,
    int thread, int streamsPerThread, int device, std::vector<InferenceTrace>& trace)
{
    return std::thread(inferenceExecution, std::cref(inference), std::ref(iEnv), std::ref(sync), thread,
        streamsPerThread, device, std::ref(trace));
}

} // namespace

bool runInference(
    const InferenceOptions& inference, InferenceEnvironment& iEnv, int device, std::vector<InferenceTrace>& trace)
{
    trace.resize(0);

    SyncStruct sync;
    sync.sleep = inference.sleep;
    sync.mainStream.sleep(&sync.sleep);
    sync.cpuStart = std::chrono::high_resolution_clock::now();
    sync.gpuStart.record(sync.mainStream);

    int threadsNum = inference.threads ? inference.streams : 1;
    int streamsPerThread = inference.streams / threadsNum;

    std::vector<std::thread> threads;
    for (int t = 0; t < threadsNum; ++t)
    {
        threads.emplace_back(makeThread(inference, iEnv, sync, t, streamsPerThread, device, trace));
    }
    for (auto& th : threads)
    {
        th.join();
    }

    auto cmpTrace = [](const InferenceTrace& a, const InferenceTrace& b) { return a.h2dStart < b.h2dStart; };
    std::sort(trace.begin(), trace.end(), cmpTrace);

    return !iEnv.error;
}
namespace
{
size_t reportGpuMemory()
{
    static size_t prevFree{0};
    size_t free{0};
    size_t total{0};
    size_t newlyAllocated{0};
    cudaCheck(cudaMemGetInfo(&free, &total));
    sample::gLogInfo << "Free GPU memory = " << free / 1024.0_MiB << " GiB";
    if (prevFree != 0)
    {
        newlyAllocated = (prevFree - free);
        sample::gLogInfo << ", newly allocated GPU memory = " << newlyAllocated / 1024.0_MiB << " GiB";
    }
    sample::gLogInfo << ", total GPU memory = " << total / 1024.0_MiB << " GiB" << std::endl;
    prevFree = free;
    return newlyAllocated;
}
} // namespace

//! Returns true if deserialization is slower than expected or fails.
bool timeDeserialize(InferenceEnvironment& iEnv)
{
    TrtUniquePtr<IRuntime> rt{createInferRuntime(sample::gLogger.getTRTLogger())};
    constexpr int32_t kNB_ITERS{20};
    TrtUniquePtr<ICudaEngine> engine;
    TrtUniquePtr<IHostMemory> serializedEngine{iEnv.engine->serialize()};

    sample::gLogInfo << "Begin deserialization engine..." << std::endl;
    auto startClock = std::chrono::high_resolution_clock::now();
    engine.reset(rt->deserializeCudaEngine(serializedEngine->data(), serializedEngine->size(), nullptr));
    auto endClock = std::chrono::high_resolution_clock::now();
    auto const first = std::chrono::duration<float, std::milli>(endClock - startClock).count();
    sample::gLogInfo << "First deserialization time = " << first << " milliseconds" << std::endl;

    // Check if first deserialization suceeded.
    if (engine == nullptr)
    {
        sample::gLogError << "Engine deserialization failed." << std::endl;
        return true;
    }

    // Record initial gpu memory state.
    reportGpuMemory();

    float totalTime{0.F};
    for (int32_t i = 0; i < kNB_ITERS; ++i)
    {
        engine.reset(nullptr);

        startClock = std::chrono::high_resolution_clock::now();
        engine.reset(rt->deserializeCudaEngine(serializedEngine->data(), serializedEngine->size(), nullptr));
        endClock = std::chrono::high_resolution_clock::now();
        totalTime += std::chrono::duration<float, std::milli>(endClock - startClock).count();
    }
    const auto averageTime = totalTime / kNB_ITERS;
    // reportGpuMemory sometimes reports zero after a single deserialization of a small engine,
    // so use the size of memory for all the iterations.
    const auto totalEngineSizeGpu = reportGpuMemory();
    sample::gLogInfo << "Total deserialization time = " << totalTime << " milliseconds, average time = " << averageTime
                     << ", first time = " << first << "." << std::endl;
    sample::gLogInfo << "Deserialization Bandwidth = " << 1E-6 * totalEngineSizeGpu / totalTime << " GB/s" << std::endl;

    // If the first deserialization is more than tolerance slower than
    // the average deserialization, return true, which means an error occurred.
    const auto tolerance = 1.50F;
    const bool isSlowerThanExpected = first > averageTime * tolerance;
    if (isSlowerThanExpected)
    {
        sample::gLogInfo << "First deserialization time divided by average time is " << (first / averageTime)
                         << ". Exceeds tolerance of " << tolerance << "x." << std::endl;
    }
    return isSlowerThanExpected;
}
} // namespace sample
