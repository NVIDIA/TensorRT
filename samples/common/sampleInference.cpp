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

#include <algorithm>
#include <array>
#include <chrono>
#include <cuda_profiler_api.h>
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
using namespace nvinfer1;
namespace sample
{

template <class MapType, class EngineType>
bool validateTensorNames(
    MapType const& map, EngineType const* engine, int32_t const endBindingIndex)
{
    // Check if the provided input tensor names match the input tensors of the engine.
    // Throw an error if the provided input tensor names cannot be found because it implies a potential typo.
    for (auto const& item : map)
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

    void fillOneBinding(TensorInfo const& tensorInfo)
    {
        auto const name = tensorInfo.name;
        auto const* bindingInOutStr = tensorInfo.isInput ? "input" : "output";
        for (auto& binding : bindings)
        {
            auto const input = inputs.find(name);
            if (tensorInfo.isInput && input != inputs.end())
            {
                sample::gLogInfo << "Using values loaded from " << input->second << " for input " << name << std::endl;
                binding->addBinding(tensorInfo, input->second);
            }
            else
            {
                sample::gLogInfo << "Using random values for " << bindingInOutStr << " " << name << std::endl;
                binding->addBinding(tensorInfo);
            }
            sample::gLogInfo << "Created " << bindingInOutStr << " binding for " << name << " with dimensions "
                             << tensorInfo.dims << std::endl;
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
            TensorInfo tensorInfo;
            tensorInfo.bindingIndex = b;
            getTensorInfo(tensorInfo);
            tensorInfo.updateVolume(batch);
            fillOneBinding(tensorInfo);
        }
        return true;
    }

    void getTensorInfo(TensorInfo& tensorInfo);

public:
    FillBindingClosure(EngineType const* _engine, ContextType const* _context, InputsMap const& _inputs,
        BindingsVector& _bindings, int32_t _batch, int32_t _endBindingIndex)
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
void FillBindingClosure<nvinfer1::ICudaEngine, nvinfer1::IExecutionContext>::getTensorInfo(TensorInfo& tensorInfo)
{
    auto const b = tensorInfo.bindingIndex;
    auto const name = engine->getBindingName(b);
    tensorInfo.name = name;
    if (engine->hasImplicitBatchDimension())
    {
        tensorInfo.dims = context->getBindingDimensions(b);
        tensorInfo.comps = engine->getBindingComponentsPerElement(b);
        tensorInfo.strides = context->getStrides(b);
        tensorInfo.vectorDimIndex = engine->getBindingVectorizedDim(b);
        tensorInfo.isInput = engine->bindingIsInput(b);
        tensorInfo.dataType = engine->getBindingDataType(b);
    }
    else
    {
        // Use enqueueV3.
        tensorInfo.dims = context->getTensorShape(name);
        tensorInfo.isDynamic = std::any_of(
            tensorInfo.dims.d, tensorInfo.dims.d + tensorInfo.dims.nbDims, [](int32_t dim) { return dim == -1; });
        tensorInfo.comps = engine->getTensorComponentsPerElement(name);
        tensorInfo.strides = context->getTensorStrides(name);
        tensorInfo.vectorDimIndex = engine->getTensorVectorizedDim(name);
        tensorInfo.isInput = engine->getTensorIOMode(name) == TensorIOMode::kINPUT;
        tensorInfo.dataType = engine->getTensorDataType(name);
    }
}

template <>
void FillBindingClosure<nvinfer1::safe::ICudaEngine, nvinfer1::safe::IExecutionContext>::getTensorInfo(
    TensorInfo& tensorInfo)
{
    // Use enqueueV3 for safe engine/context
    auto const b = tensorInfo.bindingIndex;
    auto const name = engine->getIOTensorName(b);
    tensorInfo.name = name;
    tensorInfo.dims = engine->getTensorShape(name);
    tensorInfo.isDynamic = false;
    tensorInfo.comps = engine->getTensorComponentsPerElement(name);
    tensorInfo.strides = context->getTensorStrides(name);
    tensorInfo.vectorDimIndex = engine->getTensorVectorizedDim(name);
    tensorInfo.isInput = engine->getTensorIOMode(name) == TensorIOMode::kINPUT;
    tensorInfo.dataType = engine->getTensorDataType(name);
}

bool setUpInference(InferenceEnvironment& iEnv, InferenceOptions const& inference, SystemOptions const& system)
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

        auto* safeEngine = iEnv.engine.getSafe();
        SMP_RETVAL_IF_FALSE(safeEngine != nullptr, "Got invalid safeEngine!", false, sample::gLogError);

        // Release serialized blob to save memory space.
        iEnv.engine.releaseBlob();

        for (int32_t s = 0; s < inference.streams; ++s)
        {
            auto ec = safeEngine->createExecutionContext();
            if (ec == nullptr)
            {
                sample::gLogError << "Unable to create execution context for stream " << s << "." << std::endl;
                return false;
            }
            iEnv.safeContexts.emplace_back(ec);
            iEnv.bindings.emplace_back(new Bindings(useManagedMemory));
        }
        int32_t const nbBindings = safeEngine->getNbBindings();
        auto const* safeContext = iEnv.safeContexts.front().get();
        // batch is set to 1 because safety only support explicit batch.
        return FillSafeBindings(safeEngine, safeContext, inference.inputs, iEnv.bindings, 1, nbBindings)();
    }

    using FillStdBindings = FillBindingClosure<nvinfer1::ICudaEngine, nvinfer1::IExecutionContext>;

    auto* engine = iEnv.engine.get();
    SMP_RETVAL_IF_FALSE(engine != nullptr, "Got invalid engine!", false, sample::gLogError);

    bool const hasDLA = system.DLACore >= 0;
    if (engine->hasImplicitBatchDimension() && hasDLA && inference.batch != engine->getMaxBatchSize())
    {
        sample::gLogError << "When using DLA with an implicit batch engine, the inference batch size must be the same "
                             "as the engine's maximum batch size. Please specify the batch size by adding: '--batch="
                          << engine->getMaxBatchSize() << "' to your command." << std::endl;
        return false;
    }

    // Release serialized blob to save memory space.
    iEnv.engine.releaseBlob();

    for (int32_t s = 0; s < inference.streams; ++s)
    {
        auto ec = engine->createExecutionContext();
        if (ec == nullptr)
        {
            sample::gLogError << "Unable to create execution context for stream " << s << "." << std::endl;
            return false;
        }
        ec->setNvtxVerbosity(inference.nvtxVerbosity);

        int32_t const persistentCacheLimit
            = samplesCommon::getMaxPersistentCacheSize() * inference.persistentCacheRatio;
        sample::gLogInfo << "Setting persistentCacheLimit to " << persistentCacheLimit << " bytes." << std::endl;
        ec->setPersistentCacheLimit(persistentCacheLimit);

        iEnv.contexts.emplace_back(ec);
        iEnv.bindings.emplace_back(new Bindings(useManagedMemory));
    }
    if (iEnv.profiler)
    {
        iEnv.contexts.front()->setProfiler(iEnv.profiler.get());
        // Always run reportToProfiler() after enqueue launch
        iEnv.contexts.front()->setEnqueueEmitsProfile(false);
    }

    int32_t const nbOptProfiles = engine->getNbOptimizationProfiles();
    int32_t const endBindingIndex = engine->getNbIOTensors();

    if (nbOptProfiles > 1)
    {
        sample::gLogWarning << "Multiple profiles are currently not supported. Running with one profile." << std::endl;
    }

    // Make sure that the tensor names provided in command-line args actually exist in any of the engine bindings
    // to avoid silent typos.
    if (!validateTensorNames(inference.shapes, engine, endBindingIndex))
    {
        sample::gLogError << "Invalid tensor names found in --shapes flag." << std::endl;
        return false;
    }

    // Set all input dimensions before all bindings can be allocated
    bool const useEnqueueV3 = !engine->hasImplicitBatchDimension();
    if (useEnqueueV3)
    {
        sample::gLogVerbose << "Using enqueueV3." << std::endl;
    }
    for (int32_t b = 0; b < endBindingIndex; ++b)
    {
        auto const& name = engine->getIOTensorName(b);
        auto const& mode = engine->getTensorIOMode(name);
        if (mode == TensorIOMode::kINPUT)
        {
            Dims const dims = iEnv.contexts.front()->getTensorShape(name);
            bool isShapeInferenceIO{false};
            if (useEnqueueV3)
            {
                isShapeInferenceIO = engine->isShapeInferenceIO(name);
            }
            else
            {
                isShapeInferenceIO = engine->isShapeBinding(b);
            }
            bool const hasRuntimeDim = std::any_of(dims.d, dims.d + dims.nbDims, [](int32_t dim) { return dim == -1; });
            if (hasRuntimeDim || isShapeInferenceIO)
            {
                // Set shapeData to either dimensions of the input (if it has a dynamic shape)
                // or set to values of the input (if it is an input shape tensor).
                std::vector<int32_t> shapeData;

                auto const shape = inference.shapes.find(name);
                if (shape == inference.shapes.end())
                {
                    // No information provided. Use default value for missing data.
                    constexpr int32_t kDEFAULT_VALUE = 1;
                    if (isShapeInferenceIO)
                    {
                        // Set shape tensor to all ones.
                        shapeData.assign(volume(dims, 0, dims.nbDims), kDEFAULT_VALUE);
                        sample::gLogWarning << "Values missing for input shape tensor: " << engine->getBindingName(b)
                                            << "Automatically setting values to: " << shapeData << std::endl;
                    }
                    else
                    {
                        // Use default value for unspecified runtime dimensions.
                        shapeData.resize(dims.nbDims);
                        std::transform(dims.d, dims.d + dims.nbDims, shapeData.begin(),
                            [&](int32_t dimension) { return dimension >= 0 ? dimension : kDEFAULT_VALUE; });
                        sample::gLogWarning
                            << "Shape missing for input with dynamic shape: " << engine->getBindingName(b)
                            << "Automatically setting shape to: " << shapeData << std::endl;
                    }
                }
                else if (inference.inputs.count(shape->first) && isShapeInferenceIO)
                {
                    // Load shape tensor from file.
                    int64_t const size = volume(dims, 0, dims.nbDims);
                    shapeData.resize(size);
                    auto const& filename = inference.inputs.at(shape->first);
                    auto dst = reinterpret_cast<char*>(shapeData.data());
                    loadFromFile(filename, dst, size * sizeof(decltype(shapeData)::value_type));
                }
                else
                {
                    shapeData = shape->second;
                }

                int32_t* shapeTensorData{nullptr};
                if (isShapeInferenceIO)
                {
                    // Save the data in iEnv, in a way that it's address does not change
                    // before enqueueV2 or enqueueV3 is called.
                    iEnv.inputShapeTensorValues.emplace_back(std::move(shapeData));
                    shapeTensorData = iEnv.inputShapeTensorValues.back().data();
                }

                for (auto& c : iEnv.contexts)
                {
                    if (useEnqueueV3)
                    {
                        if (isShapeInferenceIO)
                        {
                            if (!c->setTensorAddress(name, shapeTensorData))
                            {
                                return false;
                            }
                        }
                        else
                        {
                            if (!c->setInputShape(name, toDims(shapeData)))
                            {
                                return false;
                            }
                        }
                    }
                    else
                    {
                        if (isShapeInferenceIO)
                        {
                            if (!c->setInputShapeBinding(b, shapeTensorData))
                            {
                                return false;
                            }
                        }
                        else
                        {
                            if (!c->setBindingDimensions(b, toDims(shapeData)))
                            {
                                return false;
                            }
                        }
                    }
                }
            }
        }
    }

    auto const* context = iEnv.contexts.front().get();
    int32_t const batch = engine->hasImplicitBatchDimension() ? inference.batch : 1;
    return FillStdBindings(engine, context, inference.inputs, iEnv.bindings, batch, endBindingIndex)();
}

TaskInferenceEnvironment::TaskInferenceEnvironment(
    std::string engineFile, InferenceOptions inference, int32_t deviceId, int32_t DLACore, int32_t bs)
    : iOptions(inference)
    , device(deviceId)
    , batch(bs)
{
    BuildEnvironment bEnv(false, DLACore);
    loadEngineToBuildEnv(engineFile, false, bEnv, sample::gLogError);
    std::unique_ptr<InferenceEnvironment> tmp(new InferenceEnvironment(bEnv));
    iEnv = std::move(tmp);

    cudaCheck(cudaSetDevice(device));
    SystemOptions system{};
    system.device = device;
    system.DLACore = DLACore;
    if (!setUpInference(*iEnv, iOptions, system))
    {
        sample::gLogError << "Inference set up failed" << std::endl;
    }
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
    float sleep{};
};

struct Enqueue
{
    explicit Enqueue(nvinfer1::IExecutionContext& context)
        : mContext(context)
    {
    }

    nvinfer1::IExecutionContext& mContext;
};

//!
//! \class EnqueueImplicit
//! \brief Functor to enqueue inference with implicit batch
//!
class EnqueueImplicit : private Enqueue
{

public:
    explicit EnqueueImplicit(nvinfer1::IExecutionContext& context, void** buffers, int32_t batch)
        : Enqueue(context)
        , mBuffers(buffers)
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
    void** mBuffers{};
    int32_t mBatch{};
};

//!
//! \class EnqueueExplicit
//! \brief Functor to enqueue inference with explict batch
//!
class EnqueueExplicit : private Enqueue
{

public:
    explicit EnqueueExplicit(nvinfer1::IExecutionContext& context, Bindings const& bindings)
        : Enqueue(context)
        , mBindings(bindings)
    {
        ASSERT(mBindings.setTensorAddresses(mContext));
    }

    bool operator()(TrtCudaStream& stream) const
    {
        if (mContext.enqueueV3(stream.get()))
        {
            // Collecting layer timing info from current profile index of execution context
            if (mContext.getProfiler() && !mContext.getEnqueueEmitsProfile() && !mContext.reportToProfiler())
            {
                gLogWarning << "Failed to collect layer timing info from previous enqueueV3()" << std::endl;
            }
            return true;
        }
        return false;
    }

private:
    Bindings const& mBindings;
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
            if (mContext.getProfiler() && !mContext.getEnqueueEmitsProfile() && !mContext.reportToProfiler())
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
//! \class EnqueueGraphSafe
//! \brief Functor to enqueue inference from CUDA Graph
//!
class EnqueueGraphSafe
{

public:
    explicit EnqueueGraphSafe(TrtCudaGraph& graph)
        : mGraph(graph)
    {
    }

    bool operator()(TrtCudaStream& stream) const
    {
        return mGraph.launch(stream);
    }

    TrtCudaGraph& mGraph;
};

//!
//! \class EnqueueSafe
//! \brief Functor to enqueue safe execution context
//!
class EnqueueSafe
{
public:
    explicit EnqueueSafe(nvinfer1::safe::IExecutionContext& context, Bindings const& bindings)
        : mContext(context)
        , mBindings(bindings)
    {
        ASSERT(mBindings.setSafeTensorAddresses(mContext));
    }

    bool operator()(TrtCudaStream& stream) const
    {
        if (mContext.enqueueV3(stream.get()))
        {
            return true;
        }
        return false;
    }

    nvinfer1::safe::IExecutionContext& mContext;
private:
    Bindings const& mBindings;
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
    Iteration(int32_t id, InferenceOptions const& inference, ContextType& context, Bindings& bindings)
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
            setInputData(false);
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
            fetchOutputData(false);
            record(EventType::kOUTPUT_E, StreamType::kOUTPUT);
        }

        mActive[mNext] = true;
        moveNext();
        return true;
    }

    float sync(
        TimePoint const& cpuStart, TrtCudaEvent const& gpuStart, std::vector<InferenceTrace>& trace, bool skipTransfers)
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
        TimePoint const& cpuStart, TrtCudaEvent const& gpuStart, std::vector<InferenceTrace>& trace, bool skipTransfers)
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

    void setInputData(bool sync)
    {
        mBindings.transferInputToDevice(getStream(StreamType::kINPUT));
        // additional sync to avoid overlapping with inference execution.
        if (sync)
        {
            getStream(StreamType::kINPUT).synchronize();
        }
    }

    void fetchOutputData(bool sync)
    {
        mBindings.transferOutputToHost(getStream(StreamType::kOUTPUT));
        // additional sync to avoid overlapping with inference execution.
        if (sync)
        {
            getStream(StreamType::kOUTPUT).synchronize();
        }
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

    InferenceTrace getTrace(TimePoint const& cpuStart, TrtCudaEvent const& gpuStart, bool skipTransfers)
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
        InferenceOptions const& inference, nvinfer1::IExecutionContext& context, Bindings& bindings)
    {
        if (context.getEngine().hasImplicitBatchDimension())
        {
            mEnqueue = EnqueueFunction(EnqueueImplicit(context, mBindings.getDeviceBuffers(), inference.batch));
        }
        else
        {
            mEnqueue = EnqueueFunction(EnqueueExplicit(context, mBindings));
        }
        if (inference.graph)
        {
            TrtCudaStream& stream = getStream(StreamType::kCOMPUTE);
            // Avoid capturing initialization calls by executing the enqueue function at least
            // once before starting CUDA graph capture.
            auto const ret = mEnqueue(stream);
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

    void createEnqueueFunction(InferenceOptions const& inference, nvinfer1::safe::IExecutionContext& context, Bindings&)
    {
        mEnqueue = EnqueueFunction(EnqueueSafe(context, mBindings));
        if (inference.graph)
        {
            TrtCudaStream& stream = getStream(StreamType::kCOMPUTE);
            ASSERT(mEnqueue(stream));
            stream.synchronize();
            mGraph.beginCapture(stream);
            ASSERT(mEnqueue(stream));
            mGraph.endCapture(stream);
            mEnqueue = EnqueueFunction(EnqueueGraphSafe(mGraph));
        }

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
bool inferenceLoop(std::vector<std::unique_ptr<Iteration<ContextType>>>& iStreams, TimePoint const& cpuStart,
    TrtCudaEvent const& gpuStart, int iterations, float maxDurationMs, float warmupMs,
    std::vector<InferenceTrace>& trace, bool skipTransfers, float idleMs)
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
        if (idleMs != 0.F)
        {
            std::this_thread::sleep_for(std::chrono::duration<float, std::milli>(idleMs));
        }
    }
    for (auto& s : iStreams)
    {
        s->syncAll(cpuStart, gpuStart, trace, skipTransfers);
    }
    return true;
}

template <class ContextType>
void inferenceExecution(InferenceOptions const& inference, InferenceEnvironment& iEnv, SyncStruct& sync,
    int32_t const threadIdx, int32_t const streamsPerThread, int32_t device, std::vector<InferenceTrace>& trace)
{
    float warmupMs = inference.warmup;
    float durationMs = inference.duration * 1000.F + warmupMs;

    cudaCheck(cudaSetDevice(device));

    std::vector<std::unique_ptr<Iteration<ContextType>>> iStreams;

    for (int32_t s = 0; s < streamsPerThread; ++s)
    {
        int32_t const streamId{threadIdx * streamsPerThread + s};
        auto* iteration = new Iteration<ContextType>(
            streamId, inference, *iEnv.template getContext<ContextType>(streamId), *iEnv.bindings[streamId]);
        if (inference.skipTransfers)
        {
            iteration->setInputData(true);
        }
        iStreams.emplace_back(iteration);
    }

    for (auto& s : iStreams)
    {
        s->wait(sync.gpuStart);
    }

    std::vector<InferenceTrace> localTrace;
    if (!inferenceLoop(iStreams, sync.cpuStart, sync.gpuStart, inference.iterations, durationMs, warmupMs, localTrace,
            inference.skipTransfers, inference.idle))
    {
        iEnv.error = true;
    }

    if (inference.skipTransfers)
    {
        for (auto& s : iStreams)
        {
            s->fetchOutputData(true);
        }
    }

    sync.mutex.lock();
    trace.insert(trace.end(), localTrace.begin(), localTrace.end());
    sync.mutex.unlock();
}

inline std::thread makeThread(InferenceOptions const& inference, InferenceEnvironment& iEnv, SyncStruct& sync,
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
    InferenceOptions const& inference, InferenceEnvironment& iEnv, int32_t device, std::vector<InferenceTrace>& trace)
{
    cudaCheck(cudaProfilerStart());

    trace.resize(0);

    SyncStruct sync;
    sync.sleep = inference.sleep;
    sync.mainStream.sleep(&sync.sleep);
    sync.cpuStart = getCurrentTime();
    sync.gpuStart.record(sync.mainStream);

    // When multiple streams are used, trtexec can run inference in two modes:
    // (1) if inference.threads is true, then run each stream on each thread.
    // (2) if inference.threads is false, then run all streams on the same thread.
    int32_t const numThreads = inference.threads ? inference.streams : 1;
    int32_t const streamsPerThread = inference.threads ? 1 : inference.streams;

    std::vector<std::thread> threads;
    for (int32_t threadIdx = 0; threadIdx < numThreads; ++threadIdx)
    {
        threads.emplace_back(makeThread(inference, iEnv, sync, threadIdx, streamsPerThread, device, trace));
    }
    for (auto& th : threads)
    {
        th.join();
    }

    cudaCheck(cudaProfilerStop());

    auto cmpTrace = [](InferenceTrace const& a, InferenceTrace const& b) { return a.h2dStart < b.h2dStart; };
    std::sort(trace.begin(), trace.end(), cmpTrace);

    return !iEnv.error;
}

bool runMultiTasksInference(std::vector<std::unique_ptr<TaskInferenceEnvironment>>& tEnvList)
{
    cudaCheck(cudaProfilerStart());
    cudaSetDeviceFlags(cudaDeviceScheduleSpin);

    SyncStruct sync;
    sync.sleep = 0;
    sync.mainStream.sleep(&sync.sleep);
    sync.cpuStart = getCurrentTime();
    sync.gpuStart.record(sync.mainStream);

    std::vector<std::thread> threads;
    for (size_t i = 0; i < tEnvList.size(); ++i)
    {
        auto& tEnv = tEnvList[i];
        threads.emplace_back(makeThread(
            tEnv->iOptions, *(tEnv->iEnv), sync, /*threadIdx*/ 0, /*streamsPerThread*/ 1, tEnv->device, tEnv->trace));
    }
    for (auto& th : threads)
    {
        th.join();
    }

    cudaCheck(cudaProfilerStop());

    auto cmpTrace = [](InferenceTrace const& a, InferenceTrace const& b) { return a.h2dStart < b.h2dStart; };
    for (auto& tEnv : tEnvList)
    {
        std::sort(tEnv->trace.begin(), tEnv->trace.end(), cmpTrace);
    }

    return std::none_of(tEnvList.begin(), tEnvList.end(),
        [](std::unique_ptr<TaskInferenceEnvironment>& tEnv) { return tEnv->iEnv->error; });
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
        engine.reset(nullptr);
        safeEngine.reset(nullptr);
        auto startClock = std::chrono::high_resolution_clock::now();
        if (iEnv.safe)
        {
            safeEngine.reset(safeRT->deserializeCudaEngine(iEnv.engine.getBlob().data(), iEnv.engine.getBlob().size()));
            deserializeOK = (safeEngine != nullptr);
        }
        else
        {
            engine.reset(rt->deserializeCudaEngine(iEnv.engine.getBlob().data(), iEnv.engine.getBlob().size(), nullptr));
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

    // Check if first deserialization succeeded.
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
    auto const averageTime = totalTime / kNB_ITERS;
    // reportGpuMemory sometimes reports zero after a single deserialization of a small engine,
    // so use the size of memory for all the iterations.
    auto const totalEngineSizeGpu = reportGpuMemory();
    sample::gLogInfo << "Total deserialization time = " << totalTime << " milliseconds in " << kNB_ITERS
                     << " iterations, average time = " << averageTime << " milliseconds, first time = " << first
                     << " milliseconds." << std::endl;
    sample::gLogInfo << "Deserialization Bandwidth = " << 1E-6 * totalEngineSizeGpu / totalTime << " GB/s" << std::endl;

    // If the first deserialization is more than tolerance slower than
    // the average deserialization, return true, which means an error occurred.
    // The tolerance is set to 2x since the deserialization time is quick and susceptible
    // to caching issues causing problems in the first timing.
    auto const tolerance = 2.0F;
    bool const isSlowerThanExpected = first > averageTime * tolerance;
    if (isSlowerThanExpected)
    {
        sample::gLogInfo << "First deserialization time divided by average time is " << (first / averageTime)
                         << ". Exceeds tolerance of " << tolerance << "x." << std::endl;
    }
    return isSlowerThanExpected;
}

std::string getLayerInformation(
    nvinfer1::ICudaEngine* engine, nvinfer1::IExecutionContext* context, nvinfer1::LayerInformationFormat format)
{
    auto runtime = std::unique_ptr<IRuntime>(createInferRuntime(sample::gLogger.getTRTLogger()));
    auto inspector = std::unique_ptr<IEngineInspector>(engine->createEngineInspector());
    if (context != nullptr)
    {
        inspector->setExecutionContext(context);
    }
    std::string result = inspector->getEngineInformation(format);
    return result;
}

void Binding::fill(std::string const& fileName)
{
    loadFromFile(fileName, static_cast<char*>(buffer->getHostBuffer()), buffer->getSize());
}

void Binding::fill()
{
    switch (dataType)
    {
    case nvinfer1::DataType::kBOOL:
    {
        fillBuffer<bool>(buffer->getHostBuffer(), volume, 0, 1);
        break;
    }
    case nvinfer1::DataType::kINT32:
    {
        fillBuffer<int32_t>(buffer->getHostBuffer(), volume, -128, 127);
        break;
    }
    case nvinfer1::DataType::kINT8:
    {
        fillBuffer<int8_t>(buffer->getHostBuffer(), volume, -128, 127);
        break;
    }
    case nvinfer1::DataType::kFLOAT:
    {
        fillBuffer<float>(buffer->getHostBuffer(), volume, -1.0F, 1.0F);
        break;
    }
    case nvinfer1::DataType::kHALF:
    {
        fillBuffer<__half>(buffer->getHostBuffer(), volume, -1.0F, 1.0F);
        break;
    }
    case nvinfer1::DataType::kUINT8:
    {
        fillBuffer<uint8_t>(buffer->getHostBuffer(), volume, 0, 255);
        break;
    }
    }
}

void Binding::dump(std::ostream& os, Dims dims, Dims strides, int32_t vectorDim, int32_t spv,
    std::string const separator /*= " "*/) const
{
    void* outputBuffer{};
    if (outputAllocator != nullptr)
    {
        outputBuffer = outputAllocator->getBuffer()->getHostBuffer();
    }
    else
    {
        outputBuffer = buffer->getHostBuffer();
    }
    switch (dataType)
    {
    case nvinfer1::DataType::kBOOL:
    {
        dumpBuffer<bool>(outputBuffer, separator, os, dims, strides, vectorDim, spv);
        break;
    }
    case nvinfer1::DataType::kINT32:
    {
        dumpBuffer<int32_t>(outputBuffer, separator, os, dims, strides, vectorDim, spv);
        break;
    }
    case nvinfer1::DataType::kINT8:
    {
        dumpBuffer<int8_t>(outputBuffer, separator, os, dims, strides, vectorDim, spv);
        break;
    }
    case nvinfer1::DataType::kFLOAT:
    {
        dumpBuffer<float>(outputBuffer, separator, os, dims, strides, vectorDim, spv);
        break;
    }
    case nvinfer1::DataType::kHALF:
    {
        dumpBuffer<__half>(outputBuffer, separator, os, dims, strides, vectorDim, spv);
        break;
    }
    case nvinfer1::DataType::kUINT8:
    {
        dumpBuffer<uint8_t>(outputBuffer, separator, os, dims, strides, vectorDim, spv);
        break;
    }
    }
}

void Bindings::addBinding(TensorInfo const& tensorInfo, std::string const& fileName /*= ""*/)
{
    auto const b = tensorInfo.bindingIndex;
    while (mBindings.size() <= static_cast<size_t>(b))
    {
        mBindings.emplace_back();
        mDevicePointers.emplace_back();
    }
    mNames[tensorInfo.name] = b;
    mBindings[b].isInput = tensorInfo.isInput;
    mBindings[b].volume = tensorInfo.vol;
    mBindings[b].dataType = tensorInfo.dataType;
    if (tensorInfo.isDynamic)
    {
        ASSERT(!tensorInfo.isInput); // Only output shape can be possibly unknown because of DDS.
        if (mBindings[b].outputAllocator == nullptr)
        {
            if (mUseManaged)
            {
                mBindings[b].outputAllocator.reset(new OutputAllocator(new UnifiedMirroredBuffer));
            }
            else
            {
                mBindings[b].outputAllocator.reset(new OutputAllocator(new DiscreteMirroredBuffer));
            }
        }
    }
    else
    {
        if (mBindings[b].buffer == nullptr)
        {
            if (mUseManaged)
            {
                mBindings[b].buffer.reset(new UnifiedMirroredBuffer);
            }
            else
            {
                mBindings[b].buffer.reset(new DiscreteMirroredBuffer);
            }
        }
        // Some memory allocators return nullptr when allocating zero bytes, but TensorRT requires a non-null ptr
        // even for empty tensors, so allocate a dummy byte.
        if (tensorInfo.vol == 0)
        {
            mBindings[b].buffer->allocate(1);
        }
        else
        {
            mBindings[b].buffer->allocate(
                static_cast<size_t>(tensorInfo.vol) * static_cast<size_t>(dataTypeSize(tensorInfo.dataType)));
        }
        mDevicePointers[b] = mBindings[b].buffer->getDeviceBuffer();
    }
    if (tensorInfo.isInput)
    {
        if (fileName.empty())
        {
            fill(b);
        }
        else
        {
            fill(b, fileName);
        }
    }
}

void** Bindings::getDeviceBuffers()
{
    return mDevicePointers.data();
}

void Bindings::transferInputToDevice(TrtCudaStream& stream)
{
    for (auto& b : mNames)
    {
        if (mBindings[b.second].isInput)
        {
            mBindings[b.second].buffer->hostToDevice(stream);
        }
    }
}

void Bindings::transferOutputToHost(TrtCudaStream& stream)
{
    for (auto& b : mNames)
    {
        if (!mBindings[b.second].isInput)
        {
            if (mBindings[b.second].outputAllocator != nullptr)
            {
                mBindings[b.second].outputAllocator->getBuffer()->deviceToHost(stream);
            }
            else
            {
                mBindings[b.second].buffer->deviceToHost(stream);
            }
        }
    }
}

template <>
void Bindings::dumpBindingValues<nvinfer1::IExecutionContext>(nvinfer1::IExecutionContext const& context, int32_t binding, std::ostream& os,
    std::string const& separator /*= " "*/, int32_t batch /*= 1*/) const
{
    Dims dims = context.getBindingDimensions(binding);
    Dims strides = context.getStrides(binding);
    int32_t vectorDim = context.getEngine().getBindingVectorizedDim(binding);
    int32_t const spv = context.getEngine().getBindingComponentsPerElement(binding);

    if (context.getEngine().hasImplicitBatchDimension())
    {
        auto const insertN = [](Dims& d, int32_t bs) {
            int32_t const nbDims = d.nbDims;
            ASSERT(nbDims < Dims::MAX_DIMS);
            std::copy_backward(&d.d[0], &d.d[nbDims], &d.d[nbDims + 1]);
            d.d[0] = bs;
            d.nbDims = nbDims + 1;
        };
        int32_t batchStride = 0;
        for (int32_t i = 0; i < strides.nbDims; ++i)
        {
            if (strides.d[i] * dims.d[i] > batchStride)
            {
                batchStride = strides.d[i] * dims.d[i];
            }
        }
        insertN(dims, batch);
        insertN(strides, batchStride);
        vectorDim = (vectorDim == -1) ? -1 : vectorDim + 1;
    }

    mBindings[binding].dump(os, dims, strides, vectorDim, spv, separator);
}

template <>
void Bindings::dumpBindingDimensions<nvinfer1::IExecutionContext>(int binding, nvinfer1::IExecutionContext const& context, std::ostream& os) const
{
    auto const dims = context.getBindingDimensions(binding);
    // Do not add a newline terminator, because the caller may be outputting a JSON string.
    os << dims;
}

template <>
void Bindings::dumpBindingDimensions<nvinfer1::safe::IExecutionContext>(int binding, nvinfer1::safe::IExecutionContext const& context, std::ostream& os) const
{
    auto const dims = context.getEngine().getBindingDimensions(binding);
    // Do not add a newline terminator, because the caller may be outputting a JSON string.
    os << dims;
}

template <>
void Bindings::dumpBindingValues<nvinfer1::safe::IExecutionContext>(nvinfer1::safe::IExecutionContext const& context, int32_t binding, std::ostream& os,
    std::string const& separator /*= " "*/, int32_t batch /*= 1*/) const
{
    Dims const dims = context.getEngine().getBindingDimensions(binding);
    Dims const strides = context.getStrides(binding);
    int32_t const vectorDim = context.getEngine().getBindingVectorizedDim(binding);
    int32_t const spv = context.getEngine().getBindingComponentsPerElement(binding);

    mBindings[binding].dump(os, dims, strides, vectorDim, spv, separator);
}

std::unordered_map<std::string, int> Bindings::getBindings(std::function<bool(Binding const&)> predicate) const
{
    std::unordered_map<std::string, int> bindings;
    for (auto const& n : mNames)
    {
        auto const binding = n.second;
        if (predicate(mBindings[binding]))
        {
            bindings.insert(n);
        }
    }
    return bindings;
}

bool Bindings::setTensorAddresses(nvinfer1::IExecutionContext& context) const
{
    for (auto const& b : mNames)
    {
        auto const name = b.first.c_str();
        auto const location = context.getEngine().getTensorLocation(name);
        if (location == TensorLocation::kDEVICE)
        {
            if (mBindings[b.second].outputAllocator != nullptr)
            {
                if (!context.setOutputAllocator(name, mBindings[b.second].outputAllocator.get()))
                {
                    return false;
                }
            }
            else
            {
                if (!context.setTensorAddress(name, mDevicePointers[b.second]))
                {
                    return false;
                }
            }
        }
    }
    return true;
}

bool Bindings::setSafeTensorAddresses(nvinfer1::safe::IExecutionContext& context) const
{
    for (auto const& b : mNames)
    {
        auto const name = b.first.c_str();
        if (context.getEngine().getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT)
        {
            if (!context.setInputTensorAddress(name, static_cast<void const*>(mDevicePointers[b.second])))
            {
                return false;
            }
        }
        else
        {
            if (!context.setOutputTensorAddress(name, mDevicePointers[b.second]))
            {
                return false;
            }
        }
    }
    return true;
}

} // namespace sample
