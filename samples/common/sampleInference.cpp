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

#include <algorithm>
#include <array>
#include <chrono>
#include <cuda.h>
#include <iomanip>
#include <optional>
#include <cuda_profiler_api.h>
#include <functional>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#if defined(__QNX__)
#include <sys/neutrino.h>
#include <sys/syspage.h>
#endif

#include "NvInferRuntime.h"
#include "bfloat16.h"
#include "common.h"
#include "debugTensorWriter.h"
#include "half.h"
#include "logger.h"
#include "sampleDevice.h"
#include "sampleEngines.h"
#include "sampleInference.h"
#include "sampleOptions.h"
#include "sampleReporting.h"
#include "sampleUtils.h"
#include <cuda.h>

#if CUDA_VERSION >= 11060
#include <cuda_fp8.h>
#endif

using namespace nvinfer1;
#if ENABLE_UNIFIED_BUILDER
using namespace nvinfer2::safe;
// Provide a weak default definition that can be overridden
__attribute__((weak)) std::shared_ptr<sample::SampleSafeRecorder> gSafeRecorder
    = std::make_shared<sample::SampleSafeRecorder>(nvinfer2::safe::Severity::kINFO);
#endif

namespace sample
{
#if !TRT_STATIC
std::string const& getRuntimeLibraryName(RuntimeMode const mode)
{
    switch (mode)
    {
    case RuntimeMode::kFULL: return kNVINFER_LIBNAME;
    case RuntimeMode::kDISPATCH: return kNVINFER_DISPATCH_LIBNAME;
    case RuntimeMode::kLEAN: return kNVINFER_LEAN_LIBNAME;
    case RuntimeMode::kSAFE: return kNVINFER_SAFE_LIBNAME;
    }
    throw std::runtime_error("Unknown runtime mode");
}

#endif // !TRT_STATIC

#if ENABLE_UNIFIED_BUILDER
namespace safe
{
namespace
{
std::function<nvinfer1::ErrorCode(
    nvinfer2::safe::ITRTGraph*&, void const*, int64_t, ISafeRecorder&, bool, ISafeMemAllocator*)>
    pcreateTRTGraphInternal{};
std::function<nvinfer1::ErrorCode(nvinfer2::safe::ITRTGraph* graph)> pdestroyTRTGraphInternal{};
std::function<nvinfer2::safe::ISafePluginRegistry*(ISafeRecorder& recorder)> pgetSafePluginRegistryInternal{};
} // namespace

//! Track runtime used for the execution of trtexec.
//! Must be tracked as a global variable due to how library init functions APIs are organized.
RuntimeMode gUseRuntime = RuntimeMode::kSAFE;

//!
//! \brief Initialize the NVIDIA Inference Safe Runtime library
//!
//! This function dynamically loads the Safe TensorRT runtime library and initializes
//! function pointers for safe TensorRT operations. It is used to set up the safe runtime
//! environment for inference with safety-certified TensorRT engines.
//!
//! The function performs the following operations:
//! - Dynamically loads the safe TensorRT runtime library
//! - Retrieves and stores function pointers for:
//!   - createTRTGraph: Creates a safe TRT graph from serialized engine data
//!   - destroyTRTGraph: Destroys a safe TRT graph and releases resources
//!   - getSafePluginRegistry: Gets the safe plugin registry for loading plugins
//!
//! \return true if the safe runtime library was successfully loaded and initialized,
//!         false otherwise (e.g., in static builds or if library loading fails)
//!
bool initNvinferSafe()
{
#if !TRT_STATIC
    static LibraryPtr libnvinfersafePtr{};
    auto fetchPtrs = [](samplesCommon::DynamicLibrary* l) {
        if (gUseRuntime == RuntimeMode::kSAFE)
        {
            pcreateTRTGraphInternal = l->symbolAddress<nvinfer2::safe::ErrorCode(nvinfer2::safe::ITRTGraph*&,
                void const*, int64_t, ISafeRecorder&, bool, ISafeMemAllocator*)>("createTRTGraph");

            pdestroyTRTGraphInternal
                = l->symbolAddress<nvinfer2::safe::ErrorCode(nvinfer2::safe::ITRTGraph * graph)>("destroyTRTGraph");

            pgetSafePluginRegistryInternal
                = l->symbolAddress<nvinfer2::safe::ISafePluginRegistry*(ISafeRecorder & recorder)>(
                    "getSafePluginRegistry");
        }
    };
    return initLibrary(libnvinfersafePtr, sample::getRuntimeLibraryName(gUseRuntime), fetchPtrs);
#else
    return false;
#endif // !TRT_STATIC
}

//!
//! \brief Create a safe TRT graph from serialized engine data
//!
//! This function creates a safe TRT graph from serialized engine data. It is used to create
//! a safe TRT graph for inference with safety-certified TensorRT engines.
//!
nvinfer1::ErrorCode createSafeTRTGraph(nvinfer2::safe::ITRTGraph*& graph, void const* blob, int64_t size,
    ISafeRecorder& recorder, bool useManaged, ISafeMemAllocator* allocator)
{
    if (!initNvinferSafe())
    {
        return nvinfer1::ErrorCode::kINTERNAL_ERROR;
    }
    ASSERT(pcreateTRTGraphInternal != nullptr);
    return pcreateTRTGraphInternal(graph, blob, size, recorder, useManaged, allocator);
}

//!
//! \brief Destroy a safe TRT graph and release resources
//!
//! This function destroys a safe TRT graph and releases the associated resources. It is used to clean up
//! the safe TRT graph after inference with safety-certified TensorRT engines.
//!
nvinfer1::ErrorCode destroySafeTRTGraph(nvinfer2::safe::ITRTGraph*& graph)
{
    if (!initNvinferSafe())
    {
        return nvinfer1::ErrorCode::kINTERNAL_ERROR;
    }
    ASSERT(pdestroyTRTGraphInternal != nullptr);
    return pdestroyTRTGraphInternal(graph);
}

//!
//! \brief Get the safe plugin registry for loading plugins
//!
//! This function retrieves the safe plugin registry for loading plugins. It is used to get the safe plugin registry
//! for loading plugins with safety-certified TensorRT engines.
//!
nvinfer2::safe::ISafePluginRegistry* getSafePluginRegistry(ISafeRecorder& recorder)
{
    if (!initNvinferSafe())
    {
        return nullptr;
    }
    ASSERT(pgetSafePluginRegistryInternal != nullptr);
    return pgetSafePluginRegistryInternal(recorder);
}

namespace
{
nvinfer2::safe::TypedArray createTypedArray(void* const ptr, DataType const type, uint64_t bufferSize)
{
    switch (type)
    {
    case DataType::kFLOAT: return nvinfer2::safe::TypedArray(static_cast<float*>(ptr), bufferSize);
    case DataType::kHALF: return nvinfer2::safe::TypedArray(static_cast<nvinfer2::safe::half_t*>(ptr), bufferSize);
    case DataType::kINT32: return nvinfer2::safe::TypedArray(static_cast<int32_t*>(ptr), bufferSize);
    case DataType::kINT8: return nvinfer2::safe::TypedArray(static_cast<int8_t*>(ptr), bufferSize);
    case DataType::kUINT8: return nvinfer2::safe::TypedArray(static_cast<uint8_t*>(ptr), bufferSize);
    case DataType::kINT64: return nvinfer2::safe::TypedArray(static_cast<int64_t*>(ptr), bufferSize);
    case DataType::kBOOL: return nvinfer2::safe::TypedArray(static_cast<bool*>(ptr), bufferSize);
    default:
    {
        sample::gLogError << "Invalid tensor DataType encountered." << std::endl;
        return nvinfer2::safe::TypedArray{};
    }
    }
}
} // namespace
} // namespace safe
#endif

template <class TMapType, class TEngineType>
bool validateTensorNames(TMapType const& map, TEngineType const* engine, int32_t const endBindingIndex)
{
    // Check if the provided input tensor names match the input tensors of the engine.
    // Throw an error if the provided input tensor names cannot be found because it implies a potential typo.
    for (auto const& item : map)
    {
        bool tensorNameFound{false};
        for (int32_t b = 0; b < endBindingIndex; ++b)
        {
            auto const tensorName = engine->getIOTensorName(b);
            auto const tensorIOMode = engine->getTensorIOMode(tensorName);
            if (tensorIOMode == nvinfer1::TensorIOMode::kINPUT && matchStringWithOneWildcard(item.first, tensorName))
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

template <class TEngineType>
class FillBindingClosure
{
private:
    using InputsMap = std::unordered_map<std::string, std::string>;
    using BindingsVector = std::vector<std::unique_ptr<BindingsStd>>;

    TEngineType const* mEngine;
    nvinfer1::IExecutionContext const* mContext;
    InputsMap const& inputs;
    BindingsVector& bindings;
    int32_t batch;
    int32_t endBindingIndex;
    int32_t profileIndex;

    void fillOneBinding(TensorInfo const& tensorInfo)
    {
        auto const name = tensorInfo.name;
        auto const* bindingInOutStr = tensorInfo.isInput ? "Input" : "Output";
        for (auto& binding : bindings)
        {
            if (tensorInfo.isInput)
            {
                auto const input = findPlausible(inputs, name);
                if (input != inputs.end())
                {
                    sample::gLogInfo << "Using values loaded from " << input->second << " for input " << name
                                     << std::endl;
                    binding->addBinding(tensorInfo, input->second);
                }
                else
                {
                    sample::gLogInfo << "Using random values for input " << name << std::endl;
                    binding->addBinding(tensorInfo);
                }
            }
            else
            {
                binding->addBinding(tensorInfo, "", mEngine->getAliasedInputTensor(name));
            }
            if (tensorInfo.isDynamic)
            {
                sample::gLogInfo << bindingInOutStr << " binding for " << name
                                 << " is dynamic and will be created during execution using OutputAllocator."
                                 << std::endl;
            }
            else
            {
                sample::gLogInfo << bindingInOutStr << " binding for " << name << " with dimensions " << tensorInfo.dims
                                 << " and type " << tensorInfo.dataType << " is created." << std::endl;
            }
        }
    }

    bool fillAllBindings(int32_t batch, int32_t endBindingIndex)
    {
        if (!validateTensorNames(inputs, mEngine, endBindingIndex))
        {
            sample::gLogError << "Invalid tensor names found in --loadInputs flag." << std::endl;
            return false;
        }

        // Process inputs first to ensure aliased outputs can reference them
        std::vector<TensorInfo> outputTensors;
        outputTensors.reserve(endBindingIndex);

        for (int32_t b = 0; b < endBindingIndex; b++)
        {
            TensorInfo tensorInfo;
            tensorInfo.bindingIndex = b;
            getTensorInfo(tensorInfo);
            tensorInfo.updateVolume(batch);
            if (tensorInfo.isInput)
            {
                fillOneBinding(tensorInfo);
            }
            else
            {
                outputTensors.emplace_back(tensorInfo);
            }
        }

        // Then process outputs (may alias with inputs)
        for (auto const& tensorInfo : outputTensors)
        {
            fillOneBinding(tensorInfo);
        }
        return true;
    }

    void getTensorInfo(TensorInfo& tensorInfo);

public:
    FillBindingClosure(TEngineType const* _engine, nvinfer1::IExecutionContext const* _context,
        InputsMap const& _inputs, BindingsVector& _bindings, int32_t _batch, int32_t _endBindingIndex,
        int32_t _profileIndex)
        : mEngine(_engine)
        , mContext(_context)
        , inputs(_inputs)
        , bindings(_bindings)
        , batch(_batch)
        , endBindingIndex(_endBindingIndex)
        , profileIndex(_profileIndex)
    {
    }

    bool operator()()
    {
        return fillAllBindings(batch, endBindingIndex);
    }
};

template <>
void FillBindingClosure<nvinfer1::ICudaEngine>::getTensorInfo(TensorInfo& tensorInfo)
{
    auto const b = tensorInfo.bindingIndex;
    auto const name = mEngine->getIOTensorName(b);
    tensorInfo.name = name;
    tensorInfo.dims = mContext->getTensorShape(name);
    tensorInfo.isDynamic = std::any_of(
        tensorInfo.dims.d, tensorInfo.dims.d + tensorInfo.dims.nbDims, [](int32_t dim) { return dim == -1; });
    tensorInfo.comps = mEngine->getTensorComponentsPerElement(name, profileIndex);
    tensorInfo.strides = mContext->getTensorStrides(name);
    tensorInfo.vectorDimIndex = mEngine->getTensorVectorizedDim(name, profileIndex);
    tensorInfo.isInput = mEngine->getTensorIOMode(name) == TensorIOMode::kINPUT;
    tensorInfo.dataType = mEngine->getTensorDataType(name);
}

namespace
{
bool allocateContextMemory(InferenceEnvironmentStd& iEnv, InferenceOptions const& inference)
{
    auto* engine = iEnv.engine.get();
    iEnv.deviceMemory.resize(inference.infStreams);
    // Delay context memory allocation until input shapes are specified because runtime allocation would require actual
    // input shapes.
    for (int32_t i = 0; i < inference.infStreams; ++i)
    {
        auto const& ec = iEnv.contexts.at(i);
        if (inference.memoryAllocationStrategy == MemoryAllocationStrategy::kSTATIC)
        {
            sample::gLogInfo << "Created execution context with device memory size: "
                             << (engine->getDeviceMemorySizeV2() / 1.0_MiB) << " MiB" << std::endl;
        }
        else
        {
            size_t sizeToAlloc{0};
            const char* allocReason{nullptr};
            if (inference.memoryAllocationStrategy == MemoryAllocationStrategy::kPROFILE)
            {
                auto const p = inference.optProfileIndex;
                sizeToAlloc = static_cast<size_t>(engine->getDeviceMemorySizeForProfileV2(p));
                allocReason = "current profile";
            }
            else if (inference.memoryAllocationStrategy == MemoryAllocationStrategy::kRUNTIME)
            {
                sizeToAlloc = ec->updateDeviceMemorySizeForShapes();
                allocReason = "current input shapes";
            }
            else
            {
                sample::gLogError << "Unrecognizable memory allocation strategy." << std::endl;
                return false;
            }
            iEnv.deviceMemory.at(i) = TrtDeviceBuffer(sizeToAlloc);
            ec->setDeviceMemoryV2(iEnv.deviceMemory.at(i).get(), iEnv.deviceMemory.at(i).getSize());
            sample::gLogInfo << "Maximum device memory size across all profiles: "
                             << (engine->getDeviceMemorySizeV2() / 1.0_MiB) << " MiB" << std::endl;
            sample::gLogInfo << "Only allocated device memory enough for " << allocReason << ": "
                             << (sizeToAlloc / 1.0_MiB) << " MiB" << std::endl;
        }
    }
    return true;
}

//! \brief Transform shapeData so that it can be type-punned to array of int32_t.
//!
//! Transform shapeData so if data() is type-punned to (int32_t*), the sequence
//! of values are equal to the original elements of shapeData.
void contractInt64ToInt32(std::vector<int64_t>& shapeData)
{
    int64_t const size = shapeData.size();
    for (int64_t const& val : shapeData)
    {
        ASSERT(val <= std::numeric_limits<int32_t>::max() && val >= std::numeric_limits<int32_t>::min()
            && "Value out of range for int32_t conversion");
    }
    int64_t const* src = shapeData.data();
    int32_t* dst = reinterpret_cast<int32_t*>(shapeData.data());
    std::copy(src, src + size, dst);
    shapeData.resize((size + 1) / 2);
}

void setPersistentCacheLimit(
    nvinfer1::IExecutionContext* ec, InferenceOptions const& inference, std::optional<cudaDeviceProp> const& properties)
{
    int32_t const persistentCacheLimit = samplesCommon::getMaxPersistentCacheSize() * inference.persistentCacheRatio;
    sample::gLogInfo << "Setting persistentCacheLimit to " << persistentCacheLimit << " bytes." << std::endl;

    // try to increase the persistent cache size if it is less than the requested size
    if (properties && properties->persistingL2CacheMaxSize < persistentCacheLimit)
    {
        sample::gLogWarning << "persistentCacheLimit is greater than the device's cudaLimitPersistingL2CacheSize "
                               "limit ("
                            << properties->persistingL2CacheMaxSize << " bytes), trying to increase the device's limit."
                            << std::endl;
        cudaError_t error = cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, persistentCacheLimit);
        if (error != cudaSuccess)
        {
            sample::gLogWarning << "Failed to increase persistent cache size, continuing with the device's max "
                                   "persisting cache size configuration."
                                << std::endl;
        }
    }
    ec->setPersistentCacheLimit(persistentCacheLimit);
}

} // namespace


bool setUpInference(InferenceEnvironmentBase& iEnv, InferenceOptions const& inference, SystemOptions const& system)
{
    ASSERT(!inference.refPairs.empty() && "refPairs must have at least one element");

#if ENABLE_UNIFIED_BUILDER
    if (iEnv.safe)
    {
        return setUpSafeInference(static_cast<InferenceEnvironmentSafe&>(iEnv), inference, system);
    }
#endif

    return setUpStdInference(static_cast<InferenceEnvironmentStd&>(iEnv), inference, system);
}

#if ENABLE_UNIFIED_BUILDER
void getSafeTensorInfo(uint32_t profileIndex, nvinfer2::safe::ITRTGraph* safeGraph, TensorInfo& tensorInfo)
{
    nvinfer2::safe::TensorDescriptor desc;
    auto const b = tensorInfo.bindingIndex;
    const char* name = nullptr;
    safeGraph->getIOTensorName(name, b);
    tensorInfo.name = name;
    safeGraph->getIOTensorDescriptor(desc, name);
    std::vector<int64_t> dimsSpec{desc.shape.d, desc.shape.d + desc.shape.nbDims};
    tensorInfo.dims = toDims(dimsSpec);
    tensorInfo.isDynamic = std::any_of(
        tensorInfo.dims.d, tensorInfo.dims.d + tensorInfo.dims.nbDims, [](int32_t dim) { return dim == -1; });
    std::vector<int64_t> stridesSpec{desc.stride.d, desc.stride.d + desc.stride.nbDims};
    tensorInfo.strides = toDims(stridesSpec);
    tensorInfo.isInput = desc.ioMode == TensorIOMode::kINPUT;
    tensorInfo.dataType = desc.dataType;
}

bool setUpSafeInference(InferenceEnvironmentSafe& iEnv, InferenceOptions const& inference, SystemOptions const& system)
{
    // Always use refPairs[0] for initial binding setup; other pairs are loaded in inferenceLoop
    int64_t constexpr kPAIR_INDEX = 0;

    int32_t device{};
    CHECK(cudaGetDevice(&device));

    cudaDeviceProp properties;
    CHECK(cudaGetDeviceProperties(&properties, device));
    int32_t const isIntegrated{properties.integrated};

    ASSERT(sample::hasSafeRuntime());
    ASSERT(sample::safe::initNvinferSafe());

    auto safeEngineBlob = iEnv.engine.getBlob();
    SMP_RETVAL_IF_FALSE(safeEngineBlob.data != nullptr, "Engine blob is empty.", false, sample::gLogError);
    SMP_RETVAL_IF_FALSE(iEnv.engine.checkDLASafe(),
        "Safe DLA engine built with kDLA_STANDALONE should not be infered in TRT!", false, sample::gLogError);

    std::unique_ptr<nvinfer2::safe::ITRTGraph> safeGraph;

    // Use managed memory on integrated devices when transfers are skipped
    // and when it is explicitly requested on the commandline.
    bool useManagedMemory{(!inference.includeTransfers && isIntegrated) || inference.useManaged};

    nvinfer2::safe::ITRTGraph* tempGraph = nullptr;
    if (sample::safe::createSafeTRTGraph(
            tempGraph, safeEngineBlob.data, safeEngineBlob.size, *gSafeRecorder, true, nullptr)
        != nvinfer2::safe::ErrorCode::kSUCCESS)
    {
        sample::gLogError << "Create Safe TRT Graph Failed." << std::endl;
    }
    safeGraph.reset(tempGraph);

    // Release serialized blob to save memory space.
    iEnv.engine.releaseBlob();

    for (int32_t s = 0; s < inference.infStreams; ++s)
    {
        nvinfer2::safe::ITRTGraph* clonedGraph{nullptr};

        safeGraph->clone(clonedGraph, *gSafeRecorder); // return errorcode
        iEnv.mClonedGraphs.emplace_back(clonedGraph);
        iEnv.bindings.emplace_back(std::make_unique<BindingsSafe>(useManagedMemory));
        iEnv.mAuxStreamsDeleters.push_back(samplesSafeCommon::setUpAuxStreamsOn(*clonedGraph, *gSafeRecorder));
    }

    int64_t endBindingIndex = 0;
    safeGraph->getNbIOTensors(endBindingIndex);

    ASSERT(!inference.refPairs.empty() && "refPairs must have at least one element");
    auto const& inferenceInputs = inference.refPairs[kPAIR_INDEX].first;

    for (int32_t b = 0; b < endBindingIndex; b++)
    {
        TensorInfo tensorInfo;
        tensorInfo.bindingIndex = b;
        getSafeTensorInfo(inference.optProfileIndex, safeGraph.get(), tensorInfo);
        tensorInfo.updateVolume(1);
        auto const name = tensorInfo.name;
        auto const* bindingInOutStr = tensorInfo.isInput ? "Input" : "Output";
        for (auto& binding : iEnv.bindings)
        {
            auto const input = findPlausible(inferenceInputs, name);
            if (tensorInfo.isInput && input != inferenceInputs.end())
            {
                sample::gLogInfo << "Using values loaded from " << input->second << " for input " << name << std::endl;
                binding->addBinding(tensorInfo, input->second);
            }
            else
            {
                if (tensorInfo.isInput)
                {
                    sample::gLogInfo << "Using random values for input " << name << std::endl;
                }
                binding->addBinding(tensorInfo);
            }
            if (tensorInfo.isDynamic)
            {
                sample::gLogInfo << bindingInOutStr << " binding for " << name
                                 << " is dynamic and will be created during execution using OutputAllocator."
                                 << std::endl;
            }
            else
            {
                sample::gLogInfo << bindingInOutStr << " binding for " << name << " with dimensions " << tensorInfo.dims
                                 << " is created." << std::endl;
            }
        }
    }
    return true;
}
#endif

IExecutionContext* setupExecutionContext(
    nvinfer1::ICudaEngine* engine, InferenceOptions const& inference, std::optional<cudaDeviceProp> const& properties)
{
    IExecutionContext* ec{nullptr};

    //! \return the `ExecutionContextAllocationStrategy` to use for the given allocation strategy, \p s.
    auto getExecutionContextAllocationStrategy = [](MemoryAllocationStrategy s) {
        return s == MemoryAllocationStrategy::kSTATIC
            // Let TRT pre-allocate and manage the memory.
            ? ExecutionContextAllocationStrategy::kSTATIC
            // Allocate based on the current profile or runtime shapes.
            : ExecutionContextAllocationStrategy::kUSER_MANAGED;
    };

    ec = engine->createExecutionContext(getExecutionContextAllocationStrategy(inference.memoryAllocationStrategy));
    if (ec == nullptr)
    {
        sample::gLogError << "Unable to create execution context. " << std::endl;
        return nullptr;
    }
    ec->setNvtxVerbosity(inference.nvtxVerbosity);

    setPersistentCacheLimit(ec, inference, properties);

    cudaStream_t s;
    CHECK(cudaStreamCreate(&s));
    auto setProfile = ec->setOptimizationProfileAsync(inference.optProfileIndex, s);
    CHECK(cudaStreamSynchronize(s));
    CHECK(cudaStreamDestroy(s));

    if (!setProfile)
    {
        sample::gLogError << "Set optimization profile failed. " << std::endl;
        return nullptr;
    }
    return ec;
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
bool setUpStdInference(InferenceEnvironmentStd& iEnv, InferenceOptions const& inference, SystemOptions const& system)
{
    // Always use refPairs[0] for initial binding setup; other pairs are loaded in inferenceLoop
    int64_t constexpr kPAIR_INDEX = 0;

    ASSERT(!inference.refPairs.empty() && "refPairs must have at least one element");
    auto const& inferenceInputs = inference.refPairs[kPAIR_INDEX].first;

    std::optional<cudaDeviceProp> properties{};
    int32_t device{};
    CHECK(cudaGetDevice(&device));

    properties = std::make_optional<cudaDeviceProp>();
    CHECK(cudaGetDeviceProperties(&properties.value(), device));
    int32_t const isIntegrated{properties.value().integrated};
    // Use managed memory on integrated devices when transfers are skipped
    // and when it is explicitly requested on the commandline.
    bool useManagedMemory{(!inference.includeTransfers && isIntegrated) || inference.useManaged};

    using FillStdBindings = FillBindingClosure<nvinfer1::ICudaEngine>;

    auto* engine = iEnv.engine.get();
    SMP_RETVAL_IF_FALSE(engine != nullptr, "Got invalid engine!", false, sample::gLogError);

    // Release serialized blob to save memory space.
    iEnv.engine.releaseBlob();


    // Setup weight streaming if enabled
    if (engine->getStreamableWeightsSize() > 0)
    {
        auto const& budget = inference.weightStreamingBudget;
        int64_t wsBudget = budget.bytes;
        if (budget.percent != 100.0)
        {
            double const percent = budget.percent;
            ASSERT(percent < 100.0);
            auto const max = engine->getStreamableWeightsSize();
            wsBudget = (max >= 0) ? (percent / 100) * (max) : WeightStreamingBudget::kDISABLE;
        }

        if (wsBudget == WeightStreamingBudget::kDISABLE)
        {
            wsBudget = engine->getStreamableWeightsSize();
        }
        else if (wsBudget == WeightStreamingBudget::kAUTOMATIC)
        {
            wsBudget = engine->getWeightStreamingAutomaticBudget();
        }
        ASSERT(wsBudget >= 0);
        bool success = engine->setWeightStreamingBudgetV2(wsBudget);
        SMP_RETVAL_IF_FALSE(success, "Failed to set weight streaming limit!", false, sample::gLogError);
        switch (wsBudget)
        {
        case WeightStreamingBudget::kDISABLE:
        {
            sample::gLogInfo << "Weight streaming has been disabled at runtime." << std::endl;
            break;
        }

        case WeightStreamingBudget::kAUTOMATIC:
        {
            sample::gLogInfo << "The weight streaming budget will automatically be chosen by TensorRT." << std::endl;
            break;
        }
        default:
        {
            sample::gLogInfo << "Weight streaming is enabled with a device memory limit of " << wsBudget << " bytes."
                             << std::endl;
            break;
        }
        }
    }

    int32_t const nbOptProfiles = engine->getNbOptimizationProfiles();

    if (inference.optProfileIndex >= nbOptProfiles)
    {
        sample::gLogError << "Selected profile index " << inference.optProfileIndex
                          << " exceeds the number of profiles that the engine holds. " << std::endl;
        return false;
    }

    if (nbOptProfiles > 1 && !inference.setOptProfile)
    {
        sample::gLogWarning << nbOptProfiles
                            << " profiles detected but not set. Running with profile 0. Please use "
                               "--dumpOptimizationProfile to see all available profiles."
                            << std::endl;
    }

    for (int32_t s = 0; s < inference.infStreams; ++s)
    {
        IExecutionContext* ec = setupExecutionContext(engine, inference, properties);
        if (ec == nullptr)
        {
            sample::gLogError << "Unable to create execution context for inference stream " << s << ". " << std::endl;
            return false;
        }
        iEnv.contexts.emplace_back(ec);
        iEnv.bindings.emplace_back(std::make_unique<BindingsStd>(useManagedMemory));
    }

    if (iEnv.profiler)
    {
        iEnv.contexts.front()->setProfiler(iEnv.profiler.get());
        // Always run reportToProfiler() after enqueue launch
        iEnv.contexts.front()->setEnqueueEmitsProfile(false);
    }

    int32_t const endBindingIndex = engine->getNbIOTensors();

    // Make sure that the tensor names provided in command-line args actually exist in any of the engine bindings
    // to avoid silent typos.
    if (!validateTensorNames(inference.shapes, engine, endBindingIndex))
    {
        sample::gLogError << "Invalid tensor names found in --shapes flag." << std::endl;
        return false;
    }

    for (int32_t b = 0; b < endBindingIndex; ++b)
    {
        auto const& name = engine->getIOTensorName(b);
        auto const& mode = engine->getTensorIOMode(name);
        if (mode == TensorIOMode::kINPUT)
        {
            Dims const dims = iEnv.contexts.front()->getTensorShape(name);
            bool isShapeInferenceIO{false};
            isShapeInferenceIO = engine->isShapeInferenceIO(name);
            bool const hasRuntimeDim = std::any_of(dims.d, dims.d + dims.nbDims, [](int32_t dim) { return dim == -1; });
            auto const shape = findPlausible(inference.shapes, name);
            if (hasRuntimeDim || isShapeInferenceIO)
            {
                // Set shapeData to either dimensions of the input (if it has a dynamic shape)
                // or set to values of the input (if it is an input shape tensor).
                std::vector<int64_t> shapeData;

                if (shape == inference.shapes.end())
                {
                    // No information provided. Use default value for missing data.
                    constexpr int32_t kDEFAULT_VALUE = 1;
                    if (isShapeInferenceIO)
                    {
                        // Set shape tensor to all ones.
                        shapeData.assign(volume(dims, 0, dims.nbDims), kDEFAULT_VALUE);
                        sample::gLogWarning << "Values missing for input shape tensor: " << name
                                            << "Automatically setting values to: " << shapeData << std::endl;
                    }
                    else
                    {
                        // Use default value for unspecified runtime dimensions.
                        shapeData.resize(dims.nbDims);
                        std::transform(dims.d, dims.d + dims.nbDims, shapeData.begin(),
                            [&](int32_t dimension) { return dimension >= 0 ? dimension : kDEFAULT_VALUE; });
                        sample::gLogWarning << "Shape missing for input with dynamic shape: " << name
                                            << "Automatically setting shape to: " << shapeData << std::endl;
                    }
                }
                else if (inferenceInputs.count(shape->first) && isShapeInferenceIO)
                {
                    // Load shape tensor from file.
                    int64_t const size = volume(dims, 0, dims.nbDims);
                    shapeData.resize(size);
                    auto const& filename = inferenceInputs.at(shape->first);
                    auto dst = reinterpret_cast<char*>(shapeData.data());
                    loadFromFile(filename, dst, size * sizeof(decltype(shapeData)::value_type));
                }
                else
                {
                    shapeData = shape->second;
                }

                int64_t* shapeTensorData{nullptr};
                if (isShapeInferenceIO)
                {
                    // Save the data in iEnv, in a way that its address does not change
                    // before enqueueV3 is called.
                    DataType const type = engine->getTensorDataType(name);
                    switch (type)
                    {
                    case DataType::kINT64: break;
                    case DataType::kINT32: contractInt64ToInt32(shapeData); break;
                    default:
                        sample::gLogError << "Shape tensor " << name << " has unexpected type " << type << std::endl;
                        return false;
                    }
                    iEnv.inputShapeTensorValues.emplace_back(shapeData);
                    shapeTensorData = iEnv.inputShapeTensorValues.back().data();
                }

                for (auto& c : iEnv.contexts)
                {
                    if (isShapeInferenceIO)
                    {
                        sample::gLogInfo << "Set input shape tensor " << name << " to: " << shapeData << std::endl;
                        if (!c->setTensorAddress(name, shapeTensorData))
                        {
                            return false;
                        }
                    }
                    else
                    {
                        sample::gLogInfo << "Set shape of input tensor " << name << " to: " << shapeData << std::endl;
                        if (!c->setInputShape(name, toDims(shapeData)))
                        {
                            return false;
                        }
                    }
                }
            }
            else if (nbOptProfiles && shape != inference.shapes.end())
            {
                // Check if the provided shape matches the static dimensions in the engine.
                for (auto& c : iEnv.contexts)
                {
                    if (!c->setInputShape(name, toDims(shape->second)))
                    {
                        sample::gLogError << "The engine was built with static shapes for input tensor " << name
                                          << " but the provided shapes do not match the static shapes!" << std::endl;
                        return false;
                    }
                }
            }
        }
    }

    // Create Debug Listener and turn on debug states if client requested dumping debug tensors.
    if (!inference.debugTensorFileNames.empty() || !inference.dumpAlldebugTensorFormats.empty())
    {
        iEnv.listener = std::make_unique<DebugTensorWriter>(
            inference.debugTensorFileNames, inference.dumpAlldebugTensorFormats, engine->getName(), iEnv.cmdline);
        iEnv.contexts.front()->setDebugListener(iEnv.listener.get());
        for (auto const& s : inference.debugTensorFileNames)
        {
            iEnv.contexts.front()->setTensorDebugState(s.first.c_str(), true);
        }
        if (!inference.dumpAlldebugTensorFormats.empty())
        {
            iEnv.contexts.front()->setUnfusedTensorsDebugState(true);
        }
    }

    if (!allocateContextMemory(iEnv, inference))
    {
        return false;
    }

    auto const* context = iEnv.contexts.front().get();
    bool fillBindingsSuccess = FillStdBindings(
        engine, context, inferenceInputs, iEnv.bindings, 1, endBindingIndex, inference.optProfileIndex)();


    return fillBindingsSuccess;
}

TaskInferenceEnvironment::TaskInferenceEnvironment(std::string engineFile, InferenceOptions const& inference,
    ReportingOptions const& reporting, int32_t deviceId, int32_t DLACore, int32_t bs)
    : iOptions(inference)
    , rOptions(reporting)
    , device(deviceId)
    , batch(bs)
{
    BuildEnvironment bEnv(/* isSafe */ false, /* versionCompatible */ false, DLACore, "", getTempfileControlDefaults());
    SystemOptions system{};
    system.device = device;
    system.DLACore = DLACore;
    loadEngineToBuildEnv(engineFile, bEnv, sample::gLogError, system, false);
    iEnv = std::make_unique<InferenceEnvironmentStd>(bEnv);

    CHECK(cudaSetDevice(device));

    if (!setUpStdInference(*iEnv, iOptions, system))
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

#if ENABLE_UNIFIED_BUILDER
struct SafeEnqueue
{
    explicit SafeEnqueue(nvinfer2::safe::ITRTGraph& graph)
        : mGraph(graph)
    {
    }

    nvinfer2::safe::ITRTGraph& mGraph;
};
#endif

//!
//! \class EnqueueExplicit
//! \brief Functor to enqueue inference with explict batch
//!
class EnqueueExplicit : private Enqueue
{

public:
    explicit EnqueueExplicit(nvinfer1::IExecutionContext& context, BindingsStd const& bindings)
        : Enqueue(context)
        , mBindings(bindings)
    {
        ASSERT(mBindings.setTensorAddresses(mContext));
    }

    bool operator()(TrtCudaStream& stream) const
    {
        try
        {
            bool const result = mContext.enqueueV3(stream.get());
            // Collecting layer timing info from current profile index of execution context, except under capturing
            // mode.
            if (!isStreamCapturing(stream) && mContext.getProfiler() && !mContext.getEnqueueEmitsProfile()
                && !mContext.reportToProfiler())
            {
                gLogWarning << "Failed to collect layer timing info from previous enqueueV3()" << std::endl;
            }
            return result;
        }
        catch (const std::exception&)
        {
            return false;
        }
        return false;
    }

private:
    // Helper function to check if a stream is in capturing mode.
    bool isStreamCapturing(TrtCudaStream& stream) const
    {
        cudaStreamCaptureStatus status{cudaStreamCaptureStatusNone};
        CHECK(cudaStreamIsCapturing(stream.get(), &status));
        return status != cudaStreamCaptureStatusNone;
    }

    BindingsStd const& mBindings;
};

#if ENABLE_UNIFIED_BUILDER
//!
//! \class EnqueueExplicitSafe
//! \brief Functor to safeEnqueue inference with explict batch
//!
class EnqueueExplicitSafe : private SafeEnqueue
{

public:
    explicit EnqueueExplicitSafe(nvinfer2::safe::ITRTGraph& graph, BindingsSafe const& bindings)
        : SafeEnqueue(graph)
        , mBindings(bindings)
    {
        ASSERT(mBindings.setTensorAddresses(graph));
    }

    bool operator()(TrtCudaStream& stream) const
    {
        try
        {
            bool const result = (mGraph.executeAsync(stream.get()) == nvinfer1::ErrorCode::kSUCCESS);
            return result;
        }
        catch (const std::exception&)
        {
            return false;
        }
        return false;
    }

private:
    BindingsSafe const& mBindings;
};
#endif

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

#if ENABLE_UNIFIED_BUILDER
//!
//! \class EnqueueGraphSafe
//! \brief Functor to enqueue inference from CUDA Graph
//!
class EnqueueGraphSafe
{

public:
    explicit EnqueueGraphSafe(nvinfer2::safe::ITRTGraph& graph)
        : mGraph(graph)
    {
    }

    bool operator()(TrtCudaStream& stream) const
    {
        return mGraph.executeAsync(stream.get()) == nvinfer1::ErrorCode::kSUCCESS;
    }

    nvinfer2::safe::ITRTGraph& mGraph;
};
#endif

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
//! \class IterationBase
//! \brief Inference iteration and streams management
//!
class IterationBase
{

public:
    explicit IterationBase(int32_t id, InferenceOptions const& inference, BindingsBase& bindings)
        : mBindings(bindings)
        , mStreamId(id)
        , mDepth(1 + inference.overlap)
        , mActive(mDepth)
        , mEvents(mDepth)
        , mEnqueueTimes(mDepth)
    {
        for (auto& eventsAtDepth : mEvents)
        {
            std::generate(eventsAtDepth.begin(), eventsAtDepth.end(),
                [&] { return std::make_unique<TrtCudaEvent>(!inference.spin); });
        }
    }

    bool query(bool includeTransfers)
    {
        if (mActive[mNext])
        {
            return true;
        }

        if (includeTransfers)
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

        if (includeTransfers)
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

    float sync(TimePoint const& cpuStart, TrtCudaEvent const& gpuStart, std::vector<InferenceTrace>& trace,
        bool includeTransfers)
    {
        if (mActive[mNext])
        {
            if (includeTransfers)
            {
                getEvent(EventType::kOUTPUT_E).synchronize();
            }
            else
            {
                getEvent(EventType::kCOMPUTE_E).synchronize();
            }
            trace.emplace_back(getTrace(cpuStart, gpuStart, includeTransfers));
            mActive[mNext] = false;
            return getEvent(EventType::kCOMPUTE_S) - gpuStart;
        }
        return 0;
    }

    void syncAll(TimePoint const& cpuStart, TrtCudaEvent const& gpuStart, std::vector<InferenceTrace>& trace,
        bool includeTransfers)
    {
        for (int32_t d = 0; d < mDepth; ++d)
        {
            sync(cpuStart, gpuStart, trace, includeTransfers);
            moveNext();
        }
    }

    void wait(TrtCudaEvent& gpuStart)
    {
        getStream(StreamType::kINPUT).wait(gpuStart);
        getStream(StreamType::kCOMPUTE).wait(gpuStart);
        getStream(StreamType::kOUTPUT).wait(gpuStart);
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

protected:
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

    InferenceTrace getTrace(TimePoint const& cpuStart, TrtCudaEvent const& gpuStart, bool includeTransfers)
    {
        auto eventTime = [&](EventType transferEvent, EventType computeEvent) {
            return includeTransfers ? getEvent(transferEvent) - gpuStart : getEvent(computeEvent) - gpuStart;
        };
        float is = eventTime(EventType::kINPUT_S, EventType::kCOMPUTE_S);
        float ie = eventTime(EventType::kINPUT_E, EventType::kCOMPUTE_S);
        float os = eventTime(EventType::kOUTPUT_S, EventType::kCOMPUTE_E);
        float oe = eventTime(EventType::kOUTPUT_E, EventType::kCOMPUTE_E);

        return InferenceTrace(mStreamId,
            std::chrono::duration<float, std::milli>(getEnqueueTime(true) - cpuStart).count(),
            std::chrono::duration<float, std::milli>(getEnqueueTime(false) - cpuStart).count(), is, ie,
            getEvent(EventType::kCOMPUTE_S) - gpuStart, getEvent(EventType::kCOMPUTE_E) - gpuStart, os, oe);
    }

    BindingsBase& mBindings;

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
};

//!
//! \class IterationStd
//! \brief Inference iteration and streams management for standard inference
//!
class IterationStd : public IterationBase
{
public:
    explicit IterationStd(
        int32_t id, InferenceOptions const& inference, nvinfer1::IExecutionContext& context, BindingsStd& bindings)
        : IterationBase(id, inference, bindings)
    {
        createEnqueueFunction(inference, context, bindings);
    }

private:
    void createEnqueueFunction(
        InferenceOptions const& inference, nvinfer1::IExecutionContext& context, BindingsStd& bindings)
    {
        mEnqueue = EnqueueFunction(EnqueueExplicit(context, bindings));
        if (inference.graph)
        {
            sample::gLogInfo << "Capturing CUDA graph for the current execution context" << std::endl;

            TrtCudaStream& stream = getStream(StreamType::kCOMPUTE);
            // Initialize input device buffers before the warmup enqueue and graph capture.
            bindings.transferInputToDevice(stream);
            stream.synchronize();
            // Avoid capturing initialization calls by executing the enqueue function at least
            // once before starting CUDA graph capture.
            auto const ret = mEnqueue(stream);
            if (!ret)
            {
                throw std::runtime_error("Inference enqueue failed.");
            }
            stream.synchronize();

            mGraph.beginCapture(stream);
            // The built TRT engine may contain operations that are not permitted under CUDA graph capture mode.
            // When the stream is capturing, the enqueue call may return false if the current CUDA graph capture fails.
            if (mEnqueue(stream))
            {
                mGraph.endCapture(stream);
                mEnqueue = EnqueueFunction(EnqueueGraph(context, mGraph));
                sample::gLogInfo << "Successfully captured CUDA graph for the current execution context. "
                                    "When profiling with Nsight Systems, add \"--cuda-graph-trace=node\" to the "
                                    "nsys command to see per-kernel execution times."
                                 << std::endl;
            }
            else
            {
                mGraph.endCaptureOnError(stream);
                // Ensure any CUDA error has been cleaned up.
                CHECK(cudaGetLastError());
                sample::gLogWarning << "The built TensorRT engine contains operations that are not permitted under "
                                       "CUDA graph capture mode."
                                    << std::endl;
                sample::gLogWarning << "CUDA graph capture failed. The inference will be "
                                       "launched without using CUDA graph."
                                    << std::endl;
            }
        }
    }
};

#if ENABLE_UNIFIED_BUILDER
//!
//! \class IterationSafe
//! \brief Inference iteration and streams management for safe inference
//!
class IterationSafe : public IterationBase
{
public:
    explicit IterationSafe(
        int32_t id, InferenceOptions const& inference, nvinfer2::safe::ITRTGraph& graph, BindingsSafe& bindings)
        : IterationBase(id, inference, bindings)
    {
        createEnqueueFunction(inference, graph, bindings);
    }

private:
    void createEnqueueFunction(
        InferenceOptions const& inference, nvinfer2::safe::ITRTGraph& graph, BindingsSafe& bindings)
    {
        mEnqueue = EnqueueFunction(EnqueueExplicitSafe(graph, bindings));
        if (inference.graph)
        {
            sample::gLogInfo << "Capturing CUDA graph for the current execution context" << std::endl;

            TrtCudaStream& stream = getStream(StreamType::kCOMPUTE);
            // Initialize input device buffers before the warmup enqueue and graph capture.
            bindings.transferInputToDevice(stream);
            stream.synchronize();
            // Avoid capturing initialization calls by executing the enqueue function at least
            // once before starting CUDA graph capture.
            auto const ret = mEnqueue(stream);
            if (!ret)
            {
                throw std::runtime_error("Inference enqueue failed.");
            }
            stream.synchronize();

            mGraph.beginCapture(stream);
            // The built TRT engine may contain operations that are not permitted under CUDA graph capture mode.
            // When the stream is capturing, the enqueue call may return false if the current CUDA graph capture fails.
            if (mEnqueue(stream))
            {
                mGraph.endCapture(stream);
                mEnqueue = EnqueueFunction(EnqueueGraphSafe(graph));
                sample::gLogInfo << "Successfully captured CUDA graph for the current execution context. "
                                    "When profiling with Nsight Systems, add \"--cuda-graph-trace=node\" to the "
                                    "nsys command to see per-kernel execution times."
                                 << std::endl;
            }
            else
            {
                mGraph.endCaptureOnError(stream);
                // Ensure any CUDA error has been cleaned up.
                CHECK(cudaGetLastError());
                sample::gLogWarning << "The built TensorRT engine contains operations that are not permitted under "
                                       "CUDA graph capture mode."
                                    << std::endl;
                sample::gLogWarning << "CUDA graph capture failed. The inference will be "
                                       "launched without using CUDA graph."
                                    << std::endl;
            }
        }
    }
};
#endif

// Helper template function to validate accuracy - abstracts tensor info retrieval
// Note: This function is placed here (before inferenceLoop) because inferenceLoop is a template
// that calls validateAccuracy, and templates require definitions to be visible at instantiation.
template <typename TensorInfoGetter>
std::unordered_map<std::string, double> validateAccuracy(InferenceEnvironmentBase& iEnv, BindingsBase const& bindings,
    int64_t pairIndex, TensorInfoGetter getTensorInfo, InferenceOptions const& inference)
{
    std::unordered_map<std::string, double> accuracyResults;
    // Accuracy validation with reference outputs is not supported on Windows or RTX (tuner is Linux enterprise-only).
#if defined(_WIN32)
    // Early return if no reference outputs are available for validation.
    return accuracyResults;
#else
    // The checks are ordered to ensure safe array access:
    // 1. First check if refOutputsAll vector is empty
    // 2. Then check if pairIndex is within bounds
    // 3. Only after bounds are verified, check if the specific pair's map is empty
    // This ordering prevents out-of-bounds access due to short-circuit evaluation.
    bool const isRefOutputsEmpty = iEnv.refOutputsAll.empty();
    bool const isPairIndexOutOfBounds = static_cast<size_t>(pairIndex) >= iEnv.refOutputsAll.size();
    // Only access refOutputsAll[pairIndex] if pairIndex is within bounds
    bool const isPairEmpty = !isPairIndexOutOfBounds && !isRefOutputsEmpty && iEnv.refOutputsAll[pairIndex].empty();

    if (isRefOutputsEmpty || isPairIndexOutOfBounds || isPairEmpty)
    {
        sample::gLogVerbose << "iEnv.refOutputsAll.empty():" << isRefOutputsEmpty << std::endl;
        sample::gLogVerbose << "static_cast<size_t>(pairIndex) >= iEnv.refOutputsAll.size():" << isPairIndexOutOfBounds
                            << std::endl;
        sample::gLogVerbose << "pairIndex:" << pairIndex << ", refOutputsAll.size():" << iEnv.refOutputsAll.size()
                            << std::endl;
        if (!isRefOutputsEmpty && !isPairIndexOutOfBounds)
        {
            sample::gLogVerbose << "iEnv.refOutputsAll[pairIndex].empty():" << isPairEmpty << std::endl;
        }
        return accuracyResults;
    }

    auto const outputBindings = bindings.getOutputBindings();

    for (auto const& refOutput : iEnv.refOutputsAll[pairIndex])
    {
        std::string const& tensorName = refOutput.first;
        auto const* refBuffer = refOutput.second.get();

        // Find the binding for this tensor
        auto it = outputBindings.find(tensorName);
        if (it == outputBindings.end())
        {
            sample::gLogWarning << "Reference output tensor " << tensorName << " not found in output bindings"
                                << std::endl;
            continue;
        }

        int32_t const bindingIndex = it->second;
        auto const& binding = bindings.getBinding(bindingIndex);

        // Get actual output buffer
        void* actualBuffer = nullptr;
        if (binding.outputAllocator != nullptr)
        {
            actualBuffer = binding.outputAllocator->getBuffer()->getHostBuffer();
        }
        else
        {
            actualBuffer = binding.buffer->getHostBuffer();
        }

        // Get tensor info using the provided getter
        nvinfer1::Dims dims;
        nvinfer1::DataType dataType;
        getTensorInfo(tensorName.c_str(), dims, dataType);

        int64_t const volume = std::accumulate(dims.d, dims.d + dims.nbDims, 1LL, std::multiplies<int64_t>{});

        // Calculate accuracy based on data type using the selected algorithm
        double accuracy = 0.0;

        switch (dataType)
        {
        case nvinfer1::DataType::kFLOAT:
        {
            std::vector<float> actual(static_cast<float*>(actualBuffer), static_cast<float*>(actualBuffer) + volume);
            std::vector<float> reference(
                static_cast<float const*>(refBuffer->get()), static_cast<float const*>(refBuffer->get()) + volume);
            auto validator
                = createAccuracyValidator<float>(inference.accuracyValidationAlgorithm, inference.atol, inference.rtol);
            accuracy = validator->calculateAccuracy(actual, reference);
            break;
        }
        case nvinfer1::DataType::kHALF:
        {
            // Convert half to float for comparison
            auto const* actualHalf = static_cast<half_float::half const*>(actualBuffer);
            auto const* refHalf = static_cast<half_float::half const*>(refBuffer->get());
            std::vector<float> actual(volume);
            std::vector<float> reference(volume);
            for (int64_t i = 0; i < volume; ++i)
            {
                actual[i] = static_cast<float>(actualHalf[i]);
                reference[i] = static_cast<float>(refHalf[i]);
            }
            auto validator
                = createAccuracyValidator<float>(inference.accuracyValidationAlgorithm, inference.atol, inference.rtol);
            accuracy = validator->calculateAccuracy(actual, reference);
            break;
        }
        case nvinfer1::DataType::kINT32:
        {
            std::vector<int32_t> actual(
                static_cast<int32_t*>(actualBuffer), static_cast<int32_t*>(actualBuffer) + volume);
            std::vector<int32_t> reference(
                static_cast<int32_t const*>(refBuffer->get()), static_cast<int32_t const*>(refBuffer->get()) + volume);
            auto validator = createAccuracyValidator<int32_t>(
                inference.accuracyValidationAlgorithm, inference.atol, inference.rtol);
            accuracy = validator->calculateAccuracy(actual, reference);
            break;
        }
        case nvinfer1::DataType::kINT8:
        {
            std::vector<int8_t> actual(static_cast<int8_t*>(actualBuffer), static_cast<int8_t*>(actualBuffer) + volume);
            std::vector<int8_t> reference(
                static_cast<int8_t const*>(refBuffer->get()), static_cast<int8_t const*>(refBuffer->get()) + volume);
            auto validator = createAccuracyValidator<int8_t>(
                inference.accuracyValidationAlgorithm, inference.atol, inference.rtol);
            accuracy = validator->calculateAccuracy(actual, reference);
            break;
        }
        default:
            sample::gLogWarning << "Unsupported data type for accuracy validation: " << static_cast<int>(dataType)
                                << std::endl;
            continue;
        }

        accuracyResults[tensorName] = accuracy;
        sample::gLogInfo << "Accuracy loss for tensor " << tensorName << " ("
                         << getAlgorithmName(inference.accuracyValidationAlgorithm) << "): " << std::fixed
                         << std::setprecision(6) << accuracy;

        // Check if accuracy exceeds threshold (if threshold is set)
        if (inference.accuracyThresholdEndToEnd > 0 && accuracy >= inference.accuracyThresholdEndToEnd)
        {
            sample::gLogInfo << " [FAIL: >= threshold " << inference.accuracyThresholdEndToEnd << "]";
            iEnv.accuracyFailed = true;
        }
        sample::gLogInfo << std::endl;
    }

    return accuracyResults;
#endif // !(defined(_WIN32) || TRT_WINML)
}

//!
//! \brief Run the inference loop with optional accuracy validation.
//! \tparam TensorInfoGetter Callable type for retrieving tensor info (dims, dataType) by name.
//! \param iStreams Vector of iteration streams for inference execution.
//! \param cpuStart CPU timestamp at start of inference.
//! \param gpuStart GPU event recorded at start of inference.
//! \param inference Inference options containing iterations, duration, warmup, etc.
//! \param trace Output vector for inference traces.
//! \param iEnv Inference environment containing reference outputs.
//! \param bindings Bindings for input reloading and validation.
//! \param getTensorInfo Callable to get tensor dims and dataType by name.
//! \return true if inference completed successfully, false on error.
//!
template <typename TensorInfoGetter>
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
bool inferenceLoop(std::vector<std::unique_ptr<IterationBase>>& iStreams, TimePoint const& cpuStart,
    TrtCudaEvent const& gpuStart, InferenceOptions const& inference, std::vector<InferenceTrace>& trace,
    InferenceEnvironmentBase& iEnv, BindingsBase& bindings, TensorInfoGetter getTensorInfo)
{
    // Create local aliases for frequently used inference options
    int const iterations = inference.iterations;
    float const warmupMs = inference.warmup;
    bool const includeTransfers = inference.includeTransfers;
    float const idleMs = inference.idle;
    // Number of reference pairs to validate (derived from inference.refPairs.size())
    int32_t const numRefPairs = static_cast<int32_t>(inference.refPairs.size());
    float maxDurationMs = -1.F;
    if (inference.duration != -1.F)
    {
        maxDurationMs = inference.duration * 1000.F + warmupMs;
    }

    float durationMs = 0;
    int32_t skip = 0;

    if (maxDurationMs == -1.F)
    {
        sample::gLogWarning << "--duration=-1 is specified, inference will run in an endless loop until"
                            << " aborted with CTRL-C (SIGINT)" << std::endl;
        while (true)
        {
            if (!std::all_of(iStreams.begin(), iStreams.end(), [&](auto& s) { return s->query(includeTransfers); }))
            {
                return false;
            }
            for (auto& s : iStreams)
            {
                s->sync(cpuStart, gpuStart, trace, includeTransfers);
            }
        }
    }

    for (int32_t i = 0; i < iterations + skip + numRefPairs || durationMs < maxDurationMs; ++i)
    {
        // For the first numRefPairs iterations, force includeTransfers to true
        // so that input/output data transfers happen for accuracy validation.
        bool const currentIncludeTransfers = (i < numRefPairs) || includeTransfers;

        // Before each refPair iteration, load input data (including i=0 for consistency)
        if (i < numRefPairs)
        {
            bindings.fillInputsFromMap(inference.refPairs[i].first);
        }

        for (auto& s : iStreams)
        {
            if (!s->query(currentIncludeTransfers))
            {
                return false;
            }
        }
        for (auto& s : iStreams)
        {
            durationMs = std::max(durationMs, s->sync(cpuStart, gpuStart, trace, currentIncludeTransfers));
        }

        // Validate accuracy for refPair iterations (runs for first numRefPairs iterations)
        // This must happen BEFORE the warmup check to ensure validation is not skipped
        if (i < numRefPairs)
        {
            // Force output data transfer to complete before validation
            for (auto& s : iStreams)
            {
                s->fetchOutputData(true);
            }
            auto results = validateAccuracy(iEnv, bindings, i, getTensorInfo, inference);
            // Merge per-tensor accuracy values into iEnv for the tuning cache
            for (auto const& [name, value] : results)
            {
                iEnv.accuracyLossValues[name] = value;
            }
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
        s->syncAll(cpuStart, gpuStart, trace, includeTransfers);
    }
    return true;
}

void inferenceExecution(InferenceOptions const& inference, InferenceEnvironmentBase& iEnv, SyncStruct& sync,
    int32_t const threadIdx, int32_t const streamsPerThread, int32_t device, std::vector<InferenceTrace>& trace,
    ReportingOptions const& reporting) noexcept
{
    try
    {
        CHECK(cudaSetDevice(device));

#if ENABLE_UNIFIED_BUILDER
        if (iEnv.safe)
        {
            //! Function to make one iteration:
            auto makeIteration = [&](int32_t s) -> std::unique_ptr<IterationSafe> {
                int32_t const streamId{threadIdx * streamsPerThread + s};
                auto iteration = std::make_unique<IterationSafe>(streamId, inference,
                    *static_cast<InferenceEnvironmentSafe&>(iEnv).mClonedGraphs[streamId],
                    *static_cast<InferenceEnvironmentSafe&>(iEnv).bindings[streamId]);
                if (!inference.includeTransfers)
                {
                    iteration->setInputData(true);
                }
                return iteration;
            };

            std::vector<std::unique_ptr<IterationBase>> iStreams;
            for (int32_t s = 0; s < streamsPerThread; ++s)
            {
                iStreams.emplace_back(makeIteration(s));
            }

            for (auto& s : iStreams)
            {
                s->wait(sync.gpuStart);
            }
            std::vector<InferenceTrace> localTrace;

            // Get bindings and graph for accuracy validation
            auto* iEnvSafe = static_cast<InferenceEnvironmentSafe*>(&iEnv);
            int32_t const streamIdx = threadIdx * streamsPerThread;
            auto& bindingsRef = *iEnvSafe->bindings[streamIdx];
            auto& graphRef = *iEnvSafe->mClonedGraphs[streamIdx];

            // Create tensor info getter that captures safety-specific context
            auto getTensorInfo = [&graphRef](char const* name, nvinfer1::Dims& dims, nvinfer1::DataType& dataType) {
                nvinfer2::safe::TensorDescriptor desc;
                graphRef.getIOTensorDescriptor(desc, name);
                std::vector<int64_t> dimsSpec{desc.shape.d, desc.shape.d + desc.shape.nbDims};
                dims = toDims(dimsSpec);
                dataType = desc.dataType;
            };

            if (!inferenceLoop(
                    iStreams, sync.cpuStart, sync.gpuStart, inference, localTrace, iEnv, bindingsRef, getTensorInfo))
            {
                std::lock_guard<std::mutex> lock{sync.mutex};
                iEnv.error = true;
            }
            if (!inference.includeTransfers)
            {
                for (auto& s : iStreams)
                {
                    s->fetchOutputData(true);
                }
            }
            std::lock_guard<std::mutex> lock{sync.mutex};
            trace.insert(trace.end(), localTrace.begin(), localTrace.end());
            return;
        }
#endif

        //! Function to make one iteration:
        auto makeIteration = [&](int32_t s) -> std::unique_ptr<IterationStd> {
            int32_t const streamId{threadIdx * streamsPerThread + s};
            auto iteration = std::make_unique<IterationStd>(streamId, inference,
                *static_cast<InferenceEnvironmentStd&>(iEnv).getContext(streamId),
                *static_cast<InferenceEnvironmentStd&>(iEnv).bindings[streamId]);
            if (!inference.includeTransfers)
            {
                iteration->setInputData(true);
            }
            return iteration;
        };

        std::vector<std::unique_ptr<IterationBase>> iStreams;
        for (int32_t s = 0; s < streamsPerThread; ++s)
        {
            iStreams.emplace_back(makeIteration(s));
        }

        for (auto& s : iStreams)
        {
            s->wait(sync.gpuStart);
        }

        std::vector<InferenceTrace> localTrace;

        // Get context and bindings for accuracy validation
        auto* iEnvStd = static_cast<InferenceEnvironmentStd*>(&iEnv);
        int32_t const streamIdx = threadIdx * streamsPerThread;
        auto& bindingsRef = *iEnvStd->bindings[streamIdx];
        auto& contextRef = *iEnvStd->contexts[streamIdx];
        auto const* engine = &contextRef.getEngine();

        // Create tensor info getter that captures standard-specific context
        auto getTensorInfo
            = [&contextRef, engine](char const* name, nvinfer1::Dims& dims, nvinfer1::DataType& dataType) {
                  dims = contextRef.getTensorShape(name);
                  dataType = engine->getTensorDataType(name);
              };

        if (!inferenceLoop(
                iStreams, sync.cpuStart, sync.gpuStart, inference, localTrace, iEnv, bindingsRef, getTensorInfo))
        {
            std::lock_guard<std::mutex> lock{sync.mutex};
            iEnv.error = true;
        }

        auto const needOutput = reporting.output || !reporting.exportOutput.empty();
        if (!inference.includeTransfers && needOutput)
        {
            for (auto& s : iStreams)
            {
                s->fetchOutputData(true);
            }
        }

        {
            std::lock_guard<std::mutex> lock{sync.mutex};
            trace.insert(trace.end(), localTrace.begin(), localTrace.end());
        }
    }
    catch (...)
    {
        std::lock_guard<std::mutex> lock{sync.mutex};
        iEnv.error = true;
    }
}

inline std::thread makeThread(InferenceOptions const& inference, InferenceEnvironmentBase& iEnv, SyncStruct& sync,
    int32_t threadIdx, int32_t streamsPerThread, int32_t device, std::vector<InferenceTrace>& trace,
    ReportingOptions const& reporting)
{
    return std::thread(inferenceExecution, std::cref(inference), std::ref(iEnv), std::ref(sync), threadIdx,
        streamsPerThread, device, std::ref(trace), std::cref(reporting));
}

} // namespace

bool runInference(InferenceOptions const& inference, InferenceEnvironmentBase& iEnv, int32_t device,
    std::vector<InferenceTrace>& trace, ReportingOptions const& reporting)
{
    CHECK(cudaProfilerStart());

    trace.resize(0);

    SyncStruct sync;
    sync.sleep = inference.sleep;
    sync.mainStream.sleep(&sync.sleep);
    sync.cpuStart = getCurrentTime();
    sync.gpuStart.record(sync.mainStream);

    // When multiple streams are used, trtexec can run inference in two modes:
    // (1) if inference.threads is true, then run each stream on each thread.
    // (2) if inference.threads is false, then run all streams on the same thread.
    int32_t const numThreads = inference.threads ? inference.infStreams : 1;
    int32_t const streamsPerThread = inference.threads ? 1 : inference.infStreams;

    std::vector<std::thread> threads;
    for (int32_t threadIdx = 0; threadIdx < numThreads; ++threadIdx)
    {
        threads.emplace_back(makeThread(inference, iEnv, sync, threadIdx, streamsPerThread, device, trace, reporting));
    }
    for (auto& th : threads)
    {
        th.join();
    }
    CHECK(cudaProfilerStop());

    auto cmpTrace = [](InferenceTrace const& a, InferenceTrace const& b) { return a.h2dStart < b.h2dStart; };
    std::sort(trace.begin(), trace.end(), cmpTrace);


    return !iEnv.error;
}

bool runMultiTasksInference(std::vector<std::unique_ptr<TaskInferenceEnvironment>>& tEnvList)
{
    CHECK(cudaProfilerStart());
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
            tEnv->iOptions, *(tEnv->iEnv), sync, /*threadIdx*/ 0, /*streamsPerThread*/ 1, tEnv->device, tEnv->trace,
            tEnv->rOptions));
    }
    for (auto& th : threads)
    {
        th.join();
    }

    CHECK(cudaProfilerStop());

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
    CHECK(cudaMemGetInfo(&free, &total));
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
bool timeDeserialize(InferenceEnvironmentBase& iEnv, SystemOptions const& sys)
{
    constexpr int32_t kNB_ITERS{20};
    std::unique_ptr<IRuntime> rt{createRuntime()};
    std::unique_ptr<ICudaEngine> engine;

    SMP_RETVAL_IF_FALSE(!iEnv.safe, "Safe inference is not supported!", false, sample::gLogError);

    auto timeDeserializeFn = [&]() -> float {
        bool deserializeOK{false};
        engine.reset(nullptr);
        auto startClock = std::chrono::high_resolution_clock::now();

        SMP_RETVAL_IF_FALSE(!iEnv.safe, "Safe inference is not supported!", false, sample::gLogError);

        for (auto const& pluginPath : sys.dynamicPlugins)
        {
            rt->getPluginRegistry().loadLibrary(pluginPath.c_str());
        }
        auto& asyncReader = iEnv.engine.getAsyncFileReader();
        ASSERT(asyncReader.isOpen());
        if (asyncReader.isOpen())
        {
            asyncReader.reset();
            engine.reset(rt->deserializeCudaEngine(asyncReader));
        }
        deserializeOK = (engine != nullptr);
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
    auto runtime = std::unique_ptr<IRuntime>{createRuntime()};
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
    case nvinfer1::DataType::kINT64:
    {
        fillBuffer<int64_t>(buffer->getHostBuffer(), volume, -128, 127);
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
    case nvinfer1::DataType::kBF16:
    {
        fillBuffer<BFloat16>(buffer->getHostBuffer(), volume, -1.0F, 1.0F);
        break;
    }
    case nvinfer1::DataType::kUINT8:
    {
        fillBuffer<uint8_t>(buffer->getHostBuffer(), volume, 0, 255);
        break;
    }
    case nvinfer1::DataType::kFP8:
#if CUDA_VERSION < 11060
        ASSERT(false && "FP8 is not supported");
#else
    {
        fillBuffer<__nv_fp8_e4m3>(buffer->getHostBuffer(), volume, -1.0F, 1.0F);
        break;
    }
#endif
    case nvinfer1::DataType::kINT4:
    {
        // int4 is implemented as packing two elements into a single byte,
        // so all possible bit patterns of the two int4 elements coincides with all possible bit patterns of
        // an uint8.
        fillBuffer<uint8_t>(buffer->getHostBuffer(), volume, 0, 255);
        break;
    }
    case DataType::kFP4: ASSERT(false && "FP4 is not supported");
    case DataType::kE8M0: ASSERT(false && "E8M0 is not supported");
    }
}

void Binding::dump(std::ostream& os, Dims dims, Dims strides, int32_t vectorDim, int32_t spv,
    std::string const separator /*= " "*/) const
{
    void* outputBuffer{};
    if (outputAllocator != nullptr)
    {
        outputBuffer = outputAllocator->getBuffer()->getHostBuffer();
        // Overwrite dimensions with those reported by the output allocator.
        dims = outputAllocator->getFinalDims();
        os << "Final shape is " << dims << " reported by the output allocator." << std::endl;
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
    case nvinfer1::DataType::kBF16:
    {
        dumpBuffer<BFloat16>(outputBuffer, separator, os, dims, strides, vectorDim, spv);
        break;
    }
    case nvinfer1::DataType::kUINT8:
    {
        dumpBuffer<uint8_t>(outputBuffer, separator, os, dims, strides, vectorDim, spv);
        break;
    }
    case nvinfer1::DataType::kINT64:
    {
        dumpBuffer<int64_t>(outputBuffer, separator, os, dims, strides, vectorDim, spv);
        break;
    }
    case nvinfer1::DataType::kFP8:
#if CUDA_VERSION < 11060
        ASSERT(false && "FP8 is not supported");
#else
    {
        dumpBuffer<__nv_fp8_e4m3>(outputBuffer, separator, os, dims, strides, vectorDim, spv);
        break;
    }
#endif
    case nvinfer1::DataType::kINT4:
    {
        dumpInt4Buffer(outputBuffer, separator, os, dims, strides, vectorDim, spv);
        break;
    }
    case nvinfer1::DataType::kFP4: ASSERT(false && "FP4 is not supported");
    case nvinfer1::DataType::kE8M0: ASSERT(false && "E8M0 is not supported");
    }
}

void BindingsBase::addBinding(
    TensorInfo const& tensorInfo, std::string const& fileName /*= ""*/, char const* aliasedInputTensor /*= nullptr*/)
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

    // Check if this output should alias with an input
    if (!tensorInfo.isInput && aliasedInputTensor != nullptr)
    {
        // Share the same buffer as the input and skip buffer allocation
        auto const aliasedIndex = mNames.at(aliasedInputTensor);
        mBindings[b].buffer = mBindings[aliasedIndex].buffer;
        mDevicePointers[b] = mDevicePointers[aliasedIndex];
        return;
    }

    //! Make a UnifiedMirroredBuffer if useManaged or Discrete othereise:
    auto makeBuffer = [](bool useManaged) -> std::shared_ptr<IMirroredBuffer> {
        if (useManaged)
        {
            return std::make_shared<UnifiedMirroredBuffer>();
        }
        else
        {
            return std::make_shared<DiscreteMirroredBuffer>();
        }
    };
    if (tensorInfo.isDynamic)
    {
        ASSERT(!tensorInfo.isInput); // Only output shape can be possibly unknown because of DDS.
        if (mBindings[b].outputAllocator == nullptr)
        {
            mBindings[b].outputAllocator = std::make_unique<OutputAllocator>(makeBuffer(mUseManaged));
        }
    }
    else
    {
        if (mBindings[b].buffer == nullptr)
        {
            mBindings[b].buffer = makeBuffer(mUseManaged);
        }
        // Some memory allocators return nullptr when allocating zero bytes, but TensorRT requires a non-null ptr
        // even for empty tensors, so allocate a dummy byte.
        if (tensorInfo.vol == 0)
        {
            mBindings[b].buffer->allocate(1);
        }
        else
        {
            mBindings[b].buffer->allocate(samplesCommon::getNbBytes(tensorInfo.dataType, tensorInfo.vol));
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

void** BindingsBase::getDeviceBuffers()
{
    return mDevicePointers.data();
}

void BindingsBase::transferInputToDevice(TrtCudaStream& stream)
{
    for (auto& b : mNames)
    {
        if (mBindings[b.second].isInput)
        {
            mBindings[b.second].buffer->hostToDevice(stream);
        }
    }
}

void BindingsBase::transferOutputToHost(TrtCudaStream& stream)
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

void BindingsStd::dumpBindingValues(nvinfer1::IExecutionContext const& context, int32_t binding, std::ostream& os,
    std::string const& separator /*= " "*/, int32_t batch /*= 1*/) const
{
    auto const tensorName = context.getEngine().getIOTensorName(binding);
    Dims dims = context.getTensorShape(tensorName);
    Dims strides = context.getTensorStrides(tensorName);
    int32_t vectorDim = context.getEngine().getTensorVectorizedDim(tensorName);
    int32_t const spv = context.getEngine().getTensorComponentsPerElement(tensorName);

    mBindings[binding].dump(os, dims, strides, vectorDim, spv, separator);
}

namespace
{

Dims getBindingDimensions(nvinfer1::IExecutionContext const& context, std::string const& name)
{
    return context.getTensorShape(name.c_str());
}
} // namespace

void BindingsStd::dumpRawBindingToFiles(nvinfer1::IExecutionContext const& context, std::ostream& os) const
{
    os << "Dumping I/O Bindings to RAW Files:" << std::endl;
    for (auto const& n : mNames)
    {
        auto name = n.first;
        auto bIndex = n.second;
        auto const& binding = mBindings[bIndex];
        void* outputBuffer{};
        if (binding.outputAllocator != nullptr)
        {
            outputBuffer = binding.outputAllocator->getBuffer()->getHostBuffer();
        }
        else
        {
            outputBuffer = binding.buffer->getHostBuffer();
        }

        Dims dims = getBindingDimensions(context, name);
        std::string dimsStr;
        std::string dotStr;

        for (int32_t i = 0; i < dims.nbDims; i++)
        {
            dimsStr += dotStr + std::to_string(dims.d[i]);
            dotStr = ".";
        }

        std::string const bindingTypeStr = (binding.isInput ? "input" : "output");

        std::stringstream fileNameStream;
        fileNameStream << name << "." << bindingTypeStr << "." << dimsStr << "." << binding.dataType << ".raw";
        std::string fileName = genFilenameSafeString(fileNameStream.str());

        os << "Writing file for " << bindingTypeStr << " binding " << name << " (with datatype " << binding.dataType
           << " and dimensions " << dimsStr << ") to " << fileName << std::endl;

        std::ofstream f(fileName, std::ios::out | std::ios::binary);
        ASSERT(f && "Cannot open file for write");
        f.write(static_cast<char*>(outputBuffer), samplesCommon::getNbBytes(binding.dataType, binding.volume));
        f.close();
    }
}

void BindingsStd::dumpBindingDimensions(
    std::string const& name, nvinfer1::IExecutionContext const& context, std::ostream& os) const
{
    auto const dims = context.getTensorShape(name.c_str());
    // Do not add a newline terminator, because the caller may be outputting a JSON string.
    os << dims;
}

std::unordered_map<std::string, int> BindingsBase::getBindings(std::function<bool(Binding const&)> predicate) const
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

bool BindingsStd::setTensorAddresses(nvinfer1::IExecutionContext& context) const
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
#if ENABLE_DLA
                // When DLA is enabled, a tensor's address must be set to nullptr to unregister
                // the tensor with DLA before setting it to a new address.
                if (context.getTensorAddress(name) != nullptr)
                {
                    if (!context.setTensorAddress(name, nullptr))
                    {
                        return false;
                    }
                }
#endif // ENABLE_DLA
                if (!context.setTensorAddress(name, mDevicePointers[b.second]))
                {
                    return false;
                }
            }
        }
    }
    return true;
}

#if ENABLE_UNIFIED_BUILDER
namespace
{
Dims getBindingDimensions(ITRTGraph& graph, std::string const& name)
{
    nvinfer2::safe::TensorDescriptor desc;
    graph.getIOTensorDescriptor(desc, name.c_str());
    std::vector<int64_t> dimsSpec{desc.shape.d, desc.shape.d + desc.shape.nbDims};
    return toDims(dimsSpec);
}
} // namespace

void BindingsSafe::dumpBindingDimensions(std::string const& name, ITRTGraph const& graph, std::ostream& os) const
{
    // Do not add a newline terminator, because the caller may be outputting a JSON string.
    os << getBindingDimensions(const_cast<ITRTGraph&>(graph), name);
}

void BindingsSafe::dumpBindingValues(ITRTGraph const& graph, int32_t binding, std::ostream& os,
    std::string const& separator /*= " "*/, int32_t batch /*= 1*/) const
{
    char const* tensorName;
    graph.getIOTensorName(tensorName, binding);
    nvinfer2::safe::TensorDescriptor desc;
    graph.getIOTensorDescriptor(desc, tensorName);
    std::vector<int64_t> dimsSpec{desc.shape.d, desc.shape.d + desc.shape.nbDims};
    Dims dims = toDims(dimsSpec);
    std::vector<int64_t> stridesSpec{desc.stride.d, desc.stride.d + desc.stride.nbDims};
    Dims strides = toDims(stridesSpec);
    int32_t vectorDim = desc.vectorizedDim;
    int32_t const spv = desc.componentsPerVector;

    mBindings[binding].dump(os, dims, strides, vectorDim, spv, separator);
}

void BindingsSafe::dumpRawBindingToFiles(ITRTGraph& graph, std::ostream& os) const
{
    os << "Dumping I/O Bindings to RAW Files:" << std::endl;
    for (auto const& n : mNames)
    {
        auto name = n.first;
        auto bIndex = n.second;
        auto const& binding = mBindings[bIndex];
        void* outputBuffer{};
        if (binding.outputAllocator != nullptr)
        {
            outputBuffer = binding.outputAllocator->getBuffer()->getHostBuffer();
        }
        else
        {
            outputBuffer = binding.buffer->getHostBuffer();
        }

        Dims dims = getBindingDimensions(graph, name);
        std::string dimsStr;
        std::string dotStr;

        for (int32_t i = 0; i < dims.nbDims; i++)
        {
            dimsStr += dotStr + std::to_string(dims.d[i]);
            dotStr = ".";
        }

        std::string const bindingTypeStr = (binding.isInput ? "input" : "output");

        std::stringstream fileName;
        fileName << genFilenameSafeString(name) << "." << bindingTypeStr << "." << dimsStr << "." << binding.dataType
                 << ".raw";

        os << "Writing file for " << bindingTypeStr << " binding " << name << " (with datatype " << binding.dataType
           << " and dimensions " << dimsStr << ") to " << fileName.str() << std::endl;

        std::ofstream f(fileName.str(), std::ios::out | std::ios::binary);
        ASSERT(f && "Cannot open file for write");
        f.write(static_cast<char*>(outputBuffer), samplesCommon::getNbBytes(binding.dataType, binding.volume));
        f.close();
    }
}

bool BindingsSafe::setTensorAddresses(ITRTGraph& graph) const
{
    for (auto const& b : mNames)
    {
        auto const name = b.first.c_str();
        nvinfer2::safe::TensorDescriptor desc;
        graph.getIOTensorDescriptor(desc, name);
        bool onGpu = desc.memPlacement == nvinfer2::safe::MemoryPlacement::kGPU
            || desc.memPlacement == nvinfer2::safe::MemoryPlacement::kNONE;
        if (onGpu)
        {
            if (mBindings[b.second].outputAllocator != nullptr)
            {
                nvinfer2::safe::TypedArray tensor = safe::createTypedArray(
                    mBindings[b.second].outputAllocator->getBuffer(), desc.dataType, desc.sizeInBytes);
                graph.setIOTensorAddress(name, tensor);
            }
            else
            {
                nvinfer2::safe::TypedArray tensor
                    = safe::createTypedArray(mDevicePointers[b.second], desc.dataType, desc.sizeInBytes);
                graph.setIOTensorAddress(name, tensor);
            }
        }
    }
    return true;
}
#endif

#if !defined(_WIN32)
namespace
{
// Helper template function to load reference outputs - abstracts tensor info retrieval
template <typename TensorInfoGetter>
void loadRefOutputsImpl(InferenceEnvironmentBase& iEnv, InferenceOptions const& inference, int64_t pairIndex,
    TensorInfoGetter getTensorInfo)
{
    ASSERT(!inference.refPairs.empty() && "refPairs must have at least one element");
    ASSERT(pairIndex >= 0 && static_cast<size_t>(pairIndex) < inference.refPairs.size() && "pairIndex out of range");
    auto const& inferenceRefOutputs = inference.refPairs[pairIndex].second;

    if (inferenceRefOutputs.empty())
    {
        return; // No reference outputs to load
    }

    // Ensure refOutputsAll has enough space
    if (iEnv.refOutputsAll.size() <= static_cast<size_t>(pairIndex))
    {
        iEnv.refOutputsAll.resize(pairIndex + 1);
    }

    for (auto const& refOutput : inferenceRefOutputs)
    {
        std::string const& tensorName = refOutput.first;
        std::string const& fileName = refOutput.second;

        // Get tensor info using the provided getter
        nvinfer1::Dims dims;
        nvinfer1::DataType dataType;
        getTensorInfo(tensorName.c_str(), dims, dataType);

        int64_t const volume = std::accumulate(dims.d, dims.d + dims.nbDims, 1LL, std::multiplies<int64_t>{});
        size_t const nbBytes = samplesCommon::getNbBytes(dataType, volume);

        // Allocate host buffer and load data
        auto hostBuffer = std::make_unique<TrtHostBuffer>(nbBytes);
        loadFromFile(fileName, static_cast<char*>(hostBuffer->get()), nbBytes);

        sample::gLogInfo << "Loaded reference output for tensor " << tensorName << " from " << fileName
                         << " (volume=" << volume << ", bytes=" << nbBytes << ")" << std::endl;

        iEnv.refOutputsAll[pairIndex].insert_or_assign(tensorName, std::move(hostBuffer));
    }
}
} // namespace

void loadRefOutputs(InferenceEnvironmentBase& iEnv, InferenceOptions const& inference,
    nvinfer1::IExecutionContext const& context, int64_t pairIndex)
{
    auto const* engine = &context.getEngine();
    auto getTensorInfo = [&context, engine](char const* name, nvinfer1::Dims& dims, nvinfer1::DataType& dataType) {
        dims = context.getTensorShape(name);
        dataType = engine->getTensorDataType(name);
    };
    loadRefOutputsImpl(iEnv, inference, pairIndex, getTensorInfo);
}

#if ENABLE_UNIFIED_BUILDER
void loadRefOutputs(InferenceEnvironmentBase& iEnv, InferenceOptions const& inference,
    nvinfer2::safe::ITRTGraph const& graph, int64_t pairIndex)
{
    auto getTensorInfo = [&graph](char const* name, nvinfer1::Dims& dims, nvinfer1::DataType& dataType) {
        nvinfer2::safe::TensorDescriptor desc;
        graph.getIOTensorDescriptor(desc, name);
        std::vector<int64_t> dimsSpec{desc.shape.d, desc.shape.d + desc.shape.nbDims};
        dims = toDims(dimsSpec);
        dataType = desc.dataType;
    };
    loadRefOutputsImpl(iEnv, inference, pairIndex, getTensorInfo);
}
#endif
#endif // !defined(_WIN32) && !TRT_WINML

} // namespace sample
