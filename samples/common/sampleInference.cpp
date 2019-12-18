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

#include <array>
#include <vector>
#include <numeric>
#include <algorithm>
#include <utility>
#include <thread>
#include <mutex>
#include <functional>
#include <limits>
#include <memory>

#include "NvInfer.h"

#include "logger.h"
#include "sampleDevice.h"
#include "sampleUtils.h"
#include "sampleOptions.h"
#include "sampleReporting.h"
#include "sampleInference.h"

namespace sample
{

void setUpInference(InferenceEnvironment& iEnv, const InferenceOptions& inference)
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

    // Set all input dimensions before all bindings can be allocated
    for (int b = 0; b < iEnv.engine->getNbBindings(); ++b)
    {
        if (iEnv.engine->bindingIsInput(b))
        {
            auto dims = iEnv.context.front()->getBindingDimensions(b);
            const bool isDynamicInput = std::any_of(dims.d, dims.d + dims.nbDims, [](int dim){ return dim == -1; }) || iEnv.engine->isShapeBinding(b);
            if (isDynamicInput)
            {
                auto shape = inference.shapes.find(iEnv.engine->getBindingName(b));

                // If no shape is provided, set dynamic dimensions to 1.
                nvinfer1::Dims staticDims{};
                if (shape == inference.shapes.end())
                {
                    constexpr int DEFAULT_DIMENSION = 1;
                    if (iEnv.engine->isShapeBinding(b))
                    {
                        staticDims.nbDims = dims.d[0];
                        std::fill(staticDims.d, staticDims.d + staticDims.nbDims, DEFAULT_DIMENSION);
                    }
                    else
                    {
                        staticDims.nbDims = dims.nbDims;
                        std::transform(dims.d, dims.d + dims.nbDims, staticDims.d, [&](int dim) { return dim > 0 ? dim : DEFAULT_DIMENSION; });
                    }
                    gLogWarning << "Dynamic dimensions required for input: " << iEnv.engine->getBindingName(b) << ", but no shapes were provided. Automatically overriding shape to: " << staticDims << std::endl;
                }
                else
                {
                    staticDims = shape->second;
                }

                for (auto& c : iEnv.context)
                {
                    if (iEnv.engine->isShapeBinding(b))
                    {
                        c->setInputShapeBinding(b, staticDims.d);
                    }
                    else
                    {
                        c->setBindingDimensions(b, staticDims);
                    }
                }
            }
        }
    }

    for (int b = 0; b < iEnv.engine->getNbBindings(); ++b)
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
}

namespace {

//!
//! \struct SyncStruct
//! \brief Threads synchronization structure
//!
struct SyncStruct
{
    std::mutex mutex;
    TrtCudaStream mainStream;
    TrtCudaEvent mainStart{cudaEventBlockingSync};
    int sleep{0};
};

//!
//! \class EnqueueImplicit
//! \brief Functor to enqueue inference with implict batch
//!
class EnqueueImplicit
{

public:

    explicit EnqueueImplicit(int batch): mBatch(batch) {}

    void operator() (nvinfer1::IExecutionContext& context, void** buffers, TrtCudaStream& stream) const
    {
        context.enqueue(mBatch, buffers, stream.get(), nullptr);
    }

private:

    int mBatch{};
};

//!
//! \class EnqueueExplicit
//! \brief Functor to enqueue inference with explict batch
//!
class EnqueueExplicit
{

public:

    void operator() (nvinfer1::IExecutionContext& context, void** buffers, TrtCudaStream& stream) const
    {
        context.enqueueV2(buffers, stream.get(), nullptr);
    }
};

using EnqueueFunction = std::function<void(nvinfer1::IExecutionContext&, void**, TrtCudaStream&)>;

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

//!
//! \class Iteration
//! \brief Inference iteration and streams management
//!
class Iteration
{

public:

    Iteration(int id, bool overlap, bool spin, nvinfer1::IExecutionContext& context, Bindings& bindings,
               EnqueueFunction enqueue): mContext(context), mBindings(bindings), mEnqueue(enqueue),
               mStreamId(id), mDepth(1 + overlap), mActive(mDepth), mEvents(mDepth)
    {
        for (int d = 0; d < mDepth; ++d)
        {
            for (int e = 0; e < static_cast<int>(EventType::kNUM); ++e)
            {
                mEvents[d][e].reset(new TrtCudaEvent(!spin));
            }
        }
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
        mEnqueue(mContext, mBindings.getDeviceBuffers(), getStream(StreamType::kCOMPUTE));
        record(EventType::kCOMPUTE_E, StreamType::kCOMPUTE);

        wait(EventType::kCOMPUTE_E, StreamType::kOUTPUT); // Wait for compute before output DMA
        record(EventType::kOUTPUT_S, StreamType::kOUTPUT);
        mBindings.transferOutputToHost(getStream(StreamType::kOUTPUT));
        record(EventType::kOUTPUT_E, StreamType::kOUTPUT);

        mActive[mNext] = true;
        moveNext();
    }

    float sync(const TrtCudaEvent& start, std::vector<InferenceTrace>& trace)
    {
        if (mActive[mNext])
        {
            getEvent(EventType::kOUTPUT_E).synchronize();
            trace.emplace_back(getTrace(start));
            mActive[mNext] = false;
            return getEvent(EventType::kCOMPUTE_S) - start;
        }
        return 0;
    }

    void syncAll(const TrtCudaEvent& start, std::vector<InferenceTrace>& trace)
    {
        for (int d = 0; d < mDepth; ++d)
        {
            sync(start, trace);
            moveNext();
        }
    }

    void wait(TrtCudaEvent& start)
    {
        getStream(StreamType::kINPUT).wait(start);
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

    void wait(EventType e, StreamType s)
    {
        getStream(s).wait(getEvent(e));
    }

    InferenceTrace getTrace(const TrtCudaEvent& start)
    {
        return InferenceTrace(mStreamId, getEvent(EventType::kINPUT_S) - start, getEvent(EventType::kINPUT_E) - start,
                                         getEvent(EventType::kCOMPUTE_S) - start, getEvent(EventType::kCOMPUTE_E) - start,
                                         getEvent(EventType::kOUTPUT_S)- start, getEvent(EventType::kOUTPUT_E)- start);
    }

    nvinfer1::IExecutionContext& mContext;
    Bindings& mBindings;

    EnqueueFunction mEnqueue;

    int mStreamId{0};
    int mNext{0};
    int mDepth{2}; // default to double buffer to hide DMA transfers

    std::vector<bool> mActive;
    MultiStream mStream;
    std::vector<MultiEvent> mEvents;
};

using IterationStreams = std::vector<std::unique_ptr<Iteration>>;

void inferenceLoop(IterationStreams& iStreams, const TrtCudaEvent& mainStart, int batch, int iterations, float maxDurationMs, float warmupMs, std::vector<InferenceTrace>& trace)
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
            durationMs = std::max(durationMs, s->sync(mainStart, trace));
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
        s->syncAll(mainStart, trace);
    }
}

void inferenceExecution(const InferenceOptions& inference, InferenceEnvironment& iEnv, SyncStruct& sync, int offset, int streams, std::vector<InferenceTrace>& trace)
{
    float warmupMs = static_cast<float>(inference.warmup);
    float durationMs = static_cast<float>(inference.duration) * 1000 + warmupMs;

    auto enqueue = inference.batch ? EnqueueFunction(EnqueueImplicit(inference.batch)) : EnqueueFunction(EnqueueExplicit());

    IterationStreams iStreams;
    for (int s = 0; s < streams; ++s)
    {
        iStreams.emplace_back(new Iteration(offset + s, inference.overlap, inference.spin, *iEnv.context[offset], *iEnv.bindings[offset], enqueue));
    }

    for (auto& s : iStreams)
    {
        s->wait(sync.mainStart);
    }

    std::vector<InferenceTrace> localTrace;
    inferenceLoop(iStreams, sync.mainStart, inference.batch, inference.iterations, durationMs, warmupMs, localTrace);

    sync.mutex.lock();
    trace.insert(trace.end(), localTrace.begin(), localTrace.end());
    sync.mutex.unlock();
}

inline
std::thread makeThread(const InferenceOptions& inference, InferenceEnvironment& iEnv, SyncStruct& sync, int thread, int streamsPerThread, std::vector<InferenceTrace>& trace)
{
    return std::thread(inferenceExecution, std::cref(inference), std::ref(iEnv), std::ref(sync), thread, streamsPerThread, std::ref(trace));
}

} // namespace

void runInference(const InferenceOptions& inference, InferenceEnvironment& iEnv, std::vector<InferenceTrace>& trace)
{
    trace.resize(0);

    SyncStruct sync;
    sync.sleep = inference.sleep;
    sync.mainStream.sleep(&sync.sleep);
    sync.mainStart.record(sync.mainStream);

    int threadsNum = inference.threads ? inference.streams : 1;
    int streamsPerThread  = inference.streams / threadsNum;

    std::vector<std::thread> threads;
    for (int t = 0; t < threadsNum; ++t)
    {
        threads.emplace_back(makeThread(inference, iEnv, sync, t, streamsPerThread, trace));
    }
    for (auto& th : threads)
    {
        th.join();
    }

    auto cmpTrace = [](const InferenceTrace& a, const InferenceTrace& b) { return a.inStart < b.inStart; };
    std::sort(trace.begin(), trace.end(), cmpTrace);
}

} // namespace sample
