/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
                            [&](int dim) { return dim >= 0 ? dim : DEFAULT_DIMENSION; });
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
        const auto vol = volume(dims, vecDim, comps, inference.batch);
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

    void operator()(TrtCudaStream& stream) const
    {
        mContext.enqueue(mBatch, mBuffers, stream.get(), nullptr);
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

    void operator()(TrtCudaStream& stream) const
    {
        mContext.enqueueV2(mBuffers, stream.get(), nullptr);
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

    void operator()(TrtCudaStream& stream) const
    {
        mGraph.launch(stream);
    }

    TrtCudaGraph& mGraph;
};

using EnqueueFunction = std::function<void(TrtCudaStream&)>;

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

    void query()
    {
        if (mActive[mNext])
        {
            return;
        }

        record(EventType::kINPUT_S, StreamType::kINPUT);
        mBindings.transferInputToDevice(getStream(StreamType::kINPUT));
        record(EventType::kINPUT_E, StreamType::kINPUT);

        wait(EventType::kINPUT_E, StreamType::kCOMPUTE); // Wait for input DMA before compute
        record(EventType::kCOMPUTE_S, StreamType::kCOMPUTE);
        recordEnqueueTime();
        mEnqueue(getStream(StreamType::kCOMPUTE));
        recordEnqueueTime();
        record(EventType::kCOMPUTE_E, StreamType::kCOMPUTE);

        wait(EventType::kCOMPUTE_E, StreamType::kOUTPUT); // Wait for compute before output DMA
        record(EventType::kOUTPUT_S, StreamType::kOUTPUT);
        mBindings.transferOutputToHost(getStream(StreamType::kOUTPUT));
        record(EventType::kOUTPUT_E, StreamType::kOUTPUT);

        mActive[mNext] = true;
        moveNext();
    }

    float sync(const TimePoint& cpuStart, const TrtCudaEvent& gpuStart, std::vector<InferenceTrace>& trace)
    {
        if (mActive[mNext])
        {
            getEvent(EventType::kOUTPUT_E).synchronize();
            trace.emplace_back(getTrace(cpuStart, gpuStart));
            mActive[mNext] = false;
            return getEvent(EventType::kCOMPUTE_S) - gpuStart;
        }
        return 0;
    }

    void syncAll(const TimePoint& cpuStart, const TrtCudaEvent& gpuStart, std::vector<InferenceTrace>& trace)
    {
        for (int d = 0; d < mDepth; ++d)
        {
            sync(cpuStart, gpuStart, trace);
            moveNext();
        }
    }

    void wait(TrtCudaEvent& gpuStart)
    {
        getStream(StreamType::kINPUT).wait(gpuStart);
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

    InferenceTrace getTrace(const TimePoint& cpuStart, const TrtCudaEvent& gpuStart)
    {
        return InferenceTrace(mStreamId,
            std::chrono::duration<float, std::milli>(getEnqueueTime(true) - cpuStart).count(),
            std::chrono::duration<float, std::milli>(getEnqueueTime(false) - cpuStart).count(),
            getEvent(EventType::kINPUT_S) - gpuStart, getEvent(EventType::kINPUT_E) - gpuStart,
            getEvent(EventType::kCOMPUTE_S) - gpuStart, getEvent(EventType::kCOMPUTE_E) - gpuStart,
            getEvent(EventType::kOUTPUT_S) - gpuStart, getEvent(EventType::kOUTPUT_E) - gpuStart);
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
            mEnqueue(stream);
            stream.synchronize();
            mGraph.beginCapture(stream);
            mEnqueue(stream);
            mGraph.endCapture(stream);
            mEnqueue = EnqueueFunction(EnqueueGraph(mGraph));
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

void inferenceLoop(IterationStreams& iStreams, const TimePoint& cpuStart, const TrtCudaEvent& gpuStart, int iterations,
    float maxDurationMs, float warmupMs, std::vector<InferenceTrace>& trace)
{
    float durationMs = 0;
    int skip = 0;

    for (int i = 0; i < iterations + skip || durationMs < maxDurationMs; ++i)
    {
        for (auto& s : iStreams)
        {
            s->query();
        }
        for (auto& s : iStreams)
        {
            durationMs = std::max(durationMs, s->sync(cpuStart, gpuStart, trace));
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
        s->syncAll(cpuStart, gpuStart, trace);
    }
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
        iStreams.emplace_back(new Iteration(offset + s, inference, *iEnv.context[offset], *iEnv.bindings[offset]));
    }

    for (auto& s : iStreams)
    {
        s->wait(sync.gpuStart);
    }

    std::vector<InferenceTrace> localTrace;
    inferenceLoop(iStreams, sync.cpuStart, sync.gpuStart, inference.iterations, durationMs, warmupMs, localTrace);

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

void runInference(
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

    auto cmpTrace = [](const InferenceTrace& a, const InferenceTrace& b) { return a.inStart < b.inStart; };
    std::sort(trace.begin(), trace.end(), cmpTrace);
}

} // namespace sample
