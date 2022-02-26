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

#if defined(__QNX__)
#include <sys/neutrino.h>
#include <sys/syspage.h>
#endif

#include "NvInfer.h"

#include "ErrorRecorder.h"
#include "logger.h"
#include "sampleDevice.h"
#include "sampleEngines.h"
#include "sampleInference.h"
#include "sampleOptions.h"
#include "sampleReporting.h"
#include "sampleUtils.h"

namespace sample
{

template <class MapType, class EngineType>
bool validateTensorNames(
    const MapType& map, const EngineType* engine, const int32_t endBindingIndex)
{
    // Check if the provided input tensor names match the input tensors of the engine.
    // Throw an error if the provided input tensor names cannot be found because it implies a potential typo.
    for (const auto& item : map)
    {
        bool tensorNameFound{false};
        for (int32_t b = 0; b < endBindingIndex; ++b)
        {
            if (engine->bindingIsInput(b) && engine->getBindingName(b) == item.first)
            {
                tensorNameFound = true;
                break;
            }
        }
        if (!tensorNameFound)
        {
            sample::gLogError << "Cannot find input tensor with name \"" << item.first << "\" in the engine bindings! "
                              << "Please make sure the input tensor names are correct." << std::endl;
            return false;
        }
    }
    return true;
}

template <class EngineType, class ContextType>
class FillBindingClosure
{
private:
    using InputsMap = std::unordered_map<std::string, std::string>;
    using BindingsVector = std::vector<std::unique_ptr<Bindings>>;

    EngineType const* engine;
    ContextType const* context;
    InputsMap const& inputs;
    BindingsVector& bindings;
    int32_t batch;
    int32_t endBindingIndex;

    void fillOneBinding(int32_t bindingIndex, int64_t vol)
    {
        auto const dims = getDims(bindingIndex);
        auto const name = engine->getBindingName(bindingIndex);
        auto const isInput = engine->bindingIsInput(bindingIndex);
        auto const dataType = engine->getBindingDataType(bindingIndex);
        auto const *bindingInOutStr = isInput ? "input" : "output";
        for (auto& binding : bindings)
        {
            const auto input = inputs.find(name);
            if (isInput && input != inputs.end())
            {
                sample::gLogInfo << "Using values loaded from " << input->second << " for input " << name << std::endl;
                binding->addBinding(bindingIndex, name, isInput, vol, dataType, input->second);
            }
            else
            {
                sample::gLogInfo << "Using random values for " << bindingInOutStr << " " << name << std::endl;
                binding->addBinding(bindingIndex, name, isInput, vol, dataType);
            }
            sample::gLogInfo << "Created " << bindingInOutStr <<" binding for " << name << " with dimensions " << dims << std::endl;
        }
    }

    bool fillAllBindings(int32_t batch, int32_t endBindingIndex)
    {
        if (!validateTensorNames(inputs, engine, endBindingIndex))
        {
            sample::gLogError << "Invalid tensor names found in --loadInputs flag." << std::endl;
            return false;
        }

        for (int32_t b = 0; b < endBindingIndex; b++)
        {
            auto const dims = getDims(b);
            auto const comps = engine->getBindingComponentsPerElement(b);
            auto const strides = context->getStrides(b);
            int32_t const vectorDimIndex = engine->getBindingVectorizedDim(b);
            auto const vol = volume(dims, strides, vectorDimIndex, comps, batch);
            fillOneBinding(b, vol);
        }
        return true;
    }

    Dims getDims(int32_t bindingIndex);

public:
    FillBindingClosure(EngineType const* _engine, ContextType const* _context, InputsMap const& _inputs, BindingsVector& _bindings, int32_t _batch, int32_t _endBindingIndex)
        : engine(_engine)
        , context(_context)
        , inputs(_inputs)
        , bindings(_bindings)
        , batch(_batch)
        , endBindingIndex(_endBindingIndex)
    {
    }

    bool operator()()
    {
        return fillAllBindings(batch, endBindingIndex);
    }
};

template <>
Dims FillBindingClosure<nvinfer1::ICudaEngine, nvinfer1::IExecutionContext>::getDims(int32_t bindingIndex)
{
    return context->getBindingDimensions(bindingIndex);
}

template <>
Dims FillBindingClosure<nvinfer1::safe::ICudaEngine, nvinfer1::safe::IExecutionContext>::getDims(int32_t bindingIndex)
{
    return engine->getBindingDimensions(bindingIndex);
}

bool setUpInference(InferenceEnvironment& iEnv, const InferenceOptions& inference)
{
    int32_t device{};
    cudaCheck(cudaGetDevice(&device));

    cudaDeviceProp properties;
    cudaCheck(cudaGetDeviceProperties(&properties, device));
    // Use managed memory on integrated devices when transfers are skipped
    // and when it is explicitly requested on the commandline.
    bool useManagedMemory{(inference.skipTransfers && properties.integrated) || inference.useManaged};
    using FillSafeBindings = FillBindingClosure<nvinfer1::safe::ICudaEngine, nvinfer1::safe::IExecutionContext>;
    if (iEnv.safe)
    {
        ASSERT(sample::hasSafeRuntime());
        auto* safeEngine = iEnv.safeEngine.get();
        for (int32_t s = 0; s < inference.streams; ++s)
        {
            iEnv.safeContext.emplace_back(safeEngine->createExecutionContext());
            iEnv.bindings.emplace_back(new Bindings(useManagedMemory));
        }
        const int32_t nBindings = safeEngine->getNbBindings();
        auto const* safeContext = iEnv.safeContext.front().get();
        // batch is set to 1 because safety only support explicit batch.
        return FillSafeBindings(iEnv.safeEngine.get(), safeContext, inference.inputs, iEnv.bindings, 1, nBindings)();
    }

    using FillStdBindings = FillBindingClosure<nvinfer1::ICudaEngine, nvinfer1::IExecutionContext>;

    for (int32_t s = 0; s < inference.streams; ++s)
    {
        auto ec = iEnv.engine->createExecutionContext();
        if (ec == nullptr)
        {
            sample::gLogError << "Unable to create execution context for stream " << s << "." << std::endl;
            return false;
        }
        iEnv.context.emplace_back(ec);
        iEnv.bindings.emplace_back(new Bindings(useManagedMemory));
    }
    if (iEnv.profiler)
    {
        iEnv.context.front()->setProfiler(iEnv.profiler.get());
        // Always run reportToProfiler() after enqueue launch
        iEnv.context.front()->setEnqueueEmitsProfile(false);
    }

    const int32_t nOptProfiles = iEnv.engine->getNbOptimizationProfiles();
    const int32_t nBindings = iEnv.engine->getNbBindings();
    const int32_t bindingsInProfile = nOptProfiles > 0 ? nBindings / nOptProfiles : 0;
    const int32_t endBindingIndex = bindingsInProfile ? bindingsInProfile : iEnv.engine->getNbBindings();

    if (nOptProfiles > 1)
    {
        sample::gLogWarning << "Multiple profiles are currently not supported. Running with one profile." << std::endl;
    }

    // Make sure that the tensor names provided in command-line args actually exist in any of the engine bindings
    // to avoid silent typos.
    if (!validateTensorNames(inference.shapes, iEnv.engine.get(), endBindingIndex))
    {
        sample::gLogError << "Invalid tensor names found in --shapes flag." << std::endl;
        return false;
    }

    // Set all input dimensions before all bindings can be allocated
    for (int32_t b = 0; b < endBindingIndex; ++b)
    {
        if (iEnv.engine->bindingIsInput(b))
        {
            auto dims = iEnv.context.front()->getBindingDimensions(b);
            const bool isScalar = dims.nbDims == 0;
            const bool isDynamicInput = std::any_of(dims.d, dims.d + dims.nbDims, [](int32_t dim) { return dim == -1; })
                || iEnv.engine->isShapeBinding(b);
            if (isDynamicInput)
            {
                auto shape = inference.shapes.find(iEnv.engine->getBindingName(b));

                // If no shape is provided, set dynamic dimensions to 1.
                std::vector<int32_t> staticDims;
                if (shape == inference.shapes.end())
                {
                    constexpr int32_t DEFAULT_DIMENSION = 1;
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
                            [&](int32_t dimension) { return dimension >= 0 ? dimension : DEFAULT_DIMENSION; });
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

    auto* engine = iEnv.engine.get();
    auto const* context = iEnv.context.front().get();
    int32_t const batch = engine->hasImplicitBatchDimension() ? inference.batch : 1;
    return FillStdBindings(engine, context, inference.inputs, iEnv.bindings, batch, endBindingIndex)();
}

namespace
{

#if defined(__QNX__)
using TimePoint = double;
#else
using TimePoint = std::chrono::time_point<std::chrono::high_resolution_clock>;
#endif

TimePoint getCurrentTime()
{
#if defined(__QNX__)
    uint64_t const currentCycles = ClockCycles();
    uint64_t const cyclesPerSecond = SYSPAGE_ENTRY(qtime)->cycles_per_sec;
    // Return current timestamp in ms.
    return static_cast<TimePoint>(currentCycles) * 1000. / cyclesPerSecond;
#else
    return std::chrono::high_resolution_clock::now();
#endif
}

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
    int32_t sleep{0};
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
    explicit EnqueueImplicit(nvinfer1::IExecutionContext& context, void** buffers, int32_t batch)
        : Enqueue(context, buffers)
        , mBatch(batch)
    {
    }

    bool operator()(TrtCudaStream& stream) const
    {
        if (mContext.enqueue(mBatch, mBuffers, stream.get(), nullptr))
        {
            // Collecting layer timing info from current profile index of execution context
            if (mContext.getProfiler() && !mContext.getEnqueueEmitsProfile() && !mContext.reportToProfiler())
            {
                gLogWarning << "Failed to collect layer timing info from previous enqueue()" << std::endl;
            }
            return true;
        }
        return false;
    }

private:
    int32_t mBatch;
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
        if (mContext.enqueueV2(mBuffers, stream.get(), nullptr))
        {
            // Collecting layer timing info from current profile index of execution context
            if (mContext.getProfiler() && !mContext.getEnqueueEmitsProfile() && !mContext.reportToProfiler())
            {
                gLogWarning << "Failed to collect layer timing info from previous enqueueV2()" << std::endl;
            }
            return true;
        }
        return false;
    }
};

//!
//! \class EnqueueGraph
//! \brief Functor to enqueue inference from CUDA Graph
//!
class EnqueueGraph
{

public:
    explicit EnqueueGraph(nvinfer1::IExecutionContext& context, TrtCudaGraph& graph)
        : mGraph(graph)
        , mContext(context)
    {
    }

    bool operator()(TrtCudaStream& stream) const
    {
        if (mGraph.launch(stream))
        {
            // Collecting layer timing info from current profile index of execution context
            if (mContext.getProfiler() && !mContext.reportToProfiler())
            {
                gLogWarning << "Failed to collect layer timing info from previous CUDA graph launch" << std::endl;
            }
            return true;
        }
        return false;
    }

    TrtCudaGraph& mGraph;
    nvinfer1::IExecutionContext& mContext;
};

//!
//! \class EnqueueSafe
//! \brief Functor to enqueue safe execution context
//!
class EnqueueSafe
{
public:
    explicit EnqueueSafe(nvinfer1::safe::IExecutionContext& context, void** buffers)
        : mContext(context)
        , mBuffers(buffers)
    {
    }

    bool operator()(TrtCudaStream& stream) const
    {
        if (mContext.enqueueV2(mBuffers, stream.get(), nullptr))
        {
            return true;
        }
        return false;
    }

    nvinfer1::safe::IExecutionContext& mContext;
    void** mBuffers{};
};

using EnqueueFunction = std::function<bool(TrtCudaStream&)>;

enum class StreamType : int32_t
{
    kINPUT = 0,
    kCOMPUTE = 1,
    kOUTPUT = 2,
    kNUM = 3
};

enum class EventType : int32_t
{
    kINPUT_S = 0,
    kINPUT_E = 1,
    kCOMPUTE_S = 2,
    kCOMPUTE_E = 3,
    kOUTPUT_S = 4,
    kOUTPUT_E = 5,
    kNUM = 6
};

using MultiStream = std::array<TrtCudaStream, static_cast<int32_t>(StreamType::kNUM)>;

using MultiEvent = std::array<std::unique_ptr<TrtCudaEvent>, static_cast<int32_t>(EventType::kNUM)>;

using EnqueueTimes = std::array<TimePoint, 2>;

//!
//! \class Iteration
//! \brief Inference iteration and streams management
//!
template <class ContextType>
class Iteration
{

public:
    Iteration(int32_t id, const InferenceOptions& inference, ContextType& context, Bindings& bindings)
        : mBindings(bindings)
        , mStreamId(id)
        , mDepth(1 + inference.overlap)
        , mActive(mDepth)
        , mEvents(mDepth)
        , mEnqueueTimes(mDepth)
        , mContext(&context)
    {
        for (int32_t d = 0; d < mDepth; ++d)
        {
            for (int32_t e = 0; e < static_cast<int32_t>(EventType::kNUM); ++e)
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
        for (int32_t d = 0; d < mDepth; ++d)
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
        return mStream[static_cast<int32_t>(t)];
    }

    TrtCudaEvent& getEvent(EventType t)
    {
        return *mEvents[mNext][static_cast<int32_t>(t)];
    }

    void record(EventType e, StreamType s)
    {
        getEvent(e).record(getStream(s));
    }

    void recordEnqueueTime()
    {
        mEnqueueTimes[mNext][enqueueStart] = getCurrentTime();
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
                mEnqueue = EnqueueFunction(EnqueueGraph(context, mGraph));
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

    void createEnqueueFunction(const InferenceOptions&, nvinfer1::safe::IExecutionContext& context, Bindings&)
    {
        mEnqueue = EnqueueFunction(EnqueueSafe(context, mBindings.getDeviceBuffers()));
    }

    Bindings& mBindings;

    TrtCudaGraph mGraph;
    EnqueueFunction mEnqueue;

    int32_t mStreamId{0};
    int32_t mNext{0};
    int32_t mDepth{2}; // default to double buffer to hide DMA transfers

    std::vector<bool> mActive;
    MultiStream mStream;
    std::vector<MultiEvent> mEvents;

    int32_t enqueueStart{0};
    std::vector<EnqueueTimes> mEnqueueTimes;
    ContextType* mContext{nullptr};
};

template <class ContextType>
bool inferenceLoop(std::vector<std::unique_ptr<Iteration<ContextType>>>& iStreams, const TimePoint& cpuStart,
    const TrtCudaEvent& gpuStart, int iterations, float maxDurationMs, float warmupMs,
    std::vector<InferenceTrace>& trace, bool skipTransfers)
{
    float durationMs = 0;
    int32_t skip = 0;

    for (int32_t i = 0; i < iterations + skip || durationMs < maxDurationMs; ++i)
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

template <class ContextType>
void inferenceExecution(const InferenceOptions& inference, InferenceEnvironment& iEnv, SyncStruct& sync,
    const int32_t threadIdx, const int32_t streamsPerThread, int32_t device, std::vector<InferenceTrace>& trace)
{
    float warmupMs = static_cast<float>(inference.warmup);
    float durationMs = static_cast<float>(inference.duration) * 1000 + warmupMs;

    cudaCheck(cudaSetDevice(device));

    std::vector<std::unique_ptr<Iteration<ContextType>>> iStreams;

    for (int32_t s = 0; s < streamsPerThread; ++s)
    {
        const int32_t streamId{threadIdx * streamsPerThread + s};
        auto* iteration = new Iteration<ContextType>(
            streamId, inference, *iEnv.template getContext<ContextType>(streamId), *iEnv.bindings[streamId]);
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
    int32_t threadIdx, int32_t streamsPerThread, int32_t device, std::vector<InferenceTrace>& trace)
{

    if (iEnv.safe)
    {
        ASSERT(sample::hasSafeRuntime());
        return std::thread(inferenceExecution<nvinfer1::safe::IExecutionContext>, std::cref(inference), std::ref(iEnv),
            std::ref(sync), threadIdx, streamsPerThread, device, std::ref(trace));
    }

    return std::thread(inferenceExecution<nvinfer1::IExecutionContext>, std::cref(inference), std::ref(iEnv),
        std::ref(sync), threadIdx, streamsPerThread, device, std::ref(trace));
}

} // namespace

bool runInference(
    const InferenceOptions& inference, InferenceEnvironment& iEnv, int32_t device, std::vector<InferenceTrace>& trace)
{
    trace.resize(0);

    SyncStruct sync;
    sync.sleep = inference.sleep;
    sync.mainStream.sleep(&sync.sleep);
    sync.cpuStart = getCurrentTime();
    sync.gpuStart.record(sync.mainStream);

    // When multiple streams are used, trtexec can run inference in two modes:
    // (1) if inference.threads is true, then run each stream on each thread.
    // (2) if inference.threads is false, then run all streams on the same thread.
    const int32_t numThreads = inference.threads ? inference.streams : 1;
    const int32_t streamsPerThread = inference.threads ? 1 : inference.streams;

    std::vector<std::thread> threads;
    for (int32_t threadIdx = 0; threadIdx < numThreads; ++threadIdx)
    {
        threads.emplace_back(makeThread(inference, iEnv, sync, threadIdx, streamsPerThread, device, trace));
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
    constexpr int32_t kNB_ITERS{20};
    std::unique_ptr<IHostMemory> serializedEngine{iEnv.engine->serialize()};
    std::unique_ptr<IRuntime> rt{createInferRuntime(sample::gLogger.getTRTLogger())};
    std::unique_ptr<ICudaEngine> engine;

    std::unique_ptr<safe::IRuntime> safeRT{sample::createSafeInferRuntime(sample::gLogger.getTRTLogger())};
    std::unique_ptr<safe::ICudaEngine> safeEngine;

    if (iEnv.safe)
    {
        ASSERT(sample::hasSafeRuntime() && safeRT != nullptr);
        safeRT->setErrorRecorder(&gRecorder);
    }

    auto timeDeserializeFn = [&]() -> float {
        bool deserializeOK{false};
        auto startClock = std::chrono::high_resolution_clock::now();
        engine.reset(nullptr);
        safeEngine.reset(nullptr);
        if (iEnv.safe)
        {
            safeEngine.reset(safeRT->deserializeCudaEngine(serializedEngine->data(), serializedEngine->size()));
            deserializeOK = (safeEngine != nullptr);
        }
        else
        {
            engine.reset(rt->deserializeCudaEngine(serializedEngine->data(), serializedEngine->size(), nullptr));
            deserializeOK = (engine != nullptr);
        }
        auto endClock = std::chrono::high_resolution_clock::now();
        // return NAN if deserialization failed.
        return deserializeOK ? std::chrono::duration<float, std::milli>(endClock - startClock).count() : NAN;
    };

    // Warmup the caches to make sure that cache thrashing isn't throwing off the results
    {
        sample::gLogInfo << "Begin deserialization warmup..." << std::endl;
        for (int32_t i = 0, e = 2; i < e; ++i)
        {
            timeDeserializeFn();
        }
    }
    sample::gLogInfo << "Begin deserialization engine timing..." << std::endl;
    float const first = timeDeserializeFn();

    // Check if first deserialization suceeded.
    if (std::isnan(first))
    {
        sample::gLogError << "Engine deserialization failed." << std::endl;
        return true;
    }

    sample::gLogInfo << "First deserialization time = " << first << " milliseconds" << std::endl;

    // Record initial gpu memory state.
    reportGpuMemory();

    float totalTime{0.F};
    for (int32_t i = 0; i < kNB_ITERS; ++i)
    {
        totalTime += timeDeserializeFn();
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
    // The tolerance is set to 2x since the deserialization time is quick and susceptible
    // to caching issues causing problems in the first timing.
    const auto tolerance = 2.0F;
    const bool isSlowerThanExpected = first > averageTime * tolerance;
    if (isSlowerThanExpected)
    {
        sample::gLogInfo << "First deserialization time divided by average time is " << (first / averageTime)
                         << ". Exceeds tolerance of " << tolerance << "x." << std::endl;
    }
    return isSlowerThanExpected;
}

std::string getLayerInformation(const InferenceEnvironment& iEnv, nvinfer1::LayerInformationFormat format)
{
    auto runtime = std::unique_ptr<IRuntime>(createInferRuntime(sample::gLogger.getTRTLogger()));
    auto inspector = std::unique_ptr<IEngineInspector>(iEnv.engine->createEngineInspector());
    std::string result = inspector->getEngineInformation(format);
    return result;
}

} // namespace sample
