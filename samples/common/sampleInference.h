/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef TRT_SAMPLE_INFERENCE_H
#define TRT_SAMPLE_INFERENCE_H

#include "debugTensorWriter.h"
#include "sampleDevice.h"
#include "sampleEngines.h"
#include "sampleReporting.h"
#include "sampleUtils.h"

#include <functional>
#include <iostream>
#include <list>
#include <memory>
#include <string>
#include <vector>

#if ENABLE_UNIFIED_BUILDER
#include "safeCudaAllocator.h"
#endif
namespace sample
{
using LibraryPtr = std::unique_ptr<samplesCommon::DynamicLibrary>;

std::string const TRT_NVINFER_NAME = "nvinfer";
std::string const TRT_ONNXPARSER_NAME = "nvonnxparser";
std::string const TRT_LIB_SUFFIX = "";

#if !TRT_STATIC
#if defined(_WIN32)
std::string const kNVINFER_PLUGIN_LIBNAME
    = std::string{"nvinfer_plugin_"} + std::to_string(NV_TENSORRT_MAJOR) + std::string{".dll"};
std::string const kNVINFER_LIBNAME = std::string(TRT_NVINFER_NAME) + std::string{"_"}
    + std::to_string(NV_TENSORRT_MAJOR) + TRT_LIB_SUFFIX + std::string{".dll"};
std::string const kNVINFER_SAFE_LIBNAME
    = std::string{"nvinfer_safe_"} + std::to_string(NV_TENSORRT_MAJOR) + std::string{".dll"};
std::string const kNVONNXPARSER_LIBNAME = std::string(TRT_ONNXPARSER_NAME) + std::string{"_"}
    + std::to_string(NV_TENSORRT_MAJOR) + TRT_LIB_SUFFIX + std::string{".dll"};
std::string const kNVINFER_LEAN_LIBNAME
    = std::string{"nvinfer_lean_"} + std::to_string(NV_TENSORRT_MAJOR) + std::string{".dll"};
std::string const kNVINFER_DISPATCH_LIBNAME
    = std::string{"nvinfer_dispatch_"} + std::to_string(NV_TENSORRT_MAJOR) + std::string{".dll"};
#else
std::string const kNVINFER_PLUGIN_LIBNAME = std::string{"libnvinfer_plugin.so."} + std::to_string(NV_TENSORRT_MAJOR);
std::string const kNVINFER_LIBNAME
    = std::string{"lib"} + std::string(TRT_NVINFER_NAME) + std::string{".so."} + std::to_string(NV_TENSORRT_MAJOR);
std::string const kNVINFER_SAFE_LIBNAME = std::string{"libnvinfer_safe.so."} + std::to_string(NV_TENSORRT_MAJOR);
std::string const kNVONNXPARSER_LIBNAME
    = std::string{"lib"} + std::string(TRT_ONNXPARSER_NAME) + std::string{".so."} + std::to_string(NV_TENSORRT_MAJOR);
std::string const kNVINFER_LEAN_LIBNAME = std::string{"libnvinfer_lean.so."} + std::to_string(NV_TENSORRT_MAJOR);
std::string const kNVINFER_DISPATCH_LIBNAME
    = std::string{"libnvinfer_dispatch.so."} + std::to_string(NV_TENSORRT_MAJOR);
#endif

std::string const& getRuntimeLibraryName(RuntimeMode const mode);

template <typename FetchPtrs>
bool initLibrary(LibraryPtr& libPtr, std::string const& libName, FetchPtrs fetchFunc)
{
    if (libPtr != nullptr)
    {
        return true;
    }
    try
    {
        libPtr.reset(new samplesCommon::DynamicLibrary{libName});
        fetchFunc(libPtr.get());
    }
    catch (std::exception const& e)
    {
        libPtr.reset();
        sample::gLogError << "Could not load library " << libName << ": " << e.what() << std::endl;
        return false;
    }
    catch (...)
    {
        libPtr.reset();
        sample::gLogError << "Could not load library " << libName << std::endl;
        return false;
    }

    return true;
}
#endif // !TRT_STATIC

#if ENABLE_UNIFIED_BUILDER
namespace safe
{

//!
//! \brief Initialize the NVIDIA Inference Safe Runtime library
//!
//! This function dynamically loads the Safe TensorRT runtime library and initializes
//! function pointers for safe TensorRT operations. It is used to set up the safe runtime
//! environment for inference with safety-certified TensorRT engines.
//!
//! \return true if the safe runtime library was successfully loaded and initialized,
//!         false otherwise (e.g., in static builds or if library loading fails)
//!
bool initNvinferSafe();

//!
//! \brief Create a safe TRT graph from serialized engine data
//!
//! This function creates a safe TRT graph from serialized engine data. It is used to create
//! a safe TRT graph for inference with safety-certified TensorRT engines.
//!
//! \param graph: Pointer to the safe TRT graph to be created
//! \param blob: Pointer to the serialized engine data
//! \param size: Size of the serialized engine data
//! \param recorder: Reference to the safe recorder
//! \param useManaged: Flag indicating whether to use managed memory
//! \param allocator: Pointer to the safe memory allocator
//! \return Error code indicating the success or failure of the operation
//!
nvinfer1::ErrorCode createSafeTRTGraph(nvinfer2::safe::ITRTGraph*& graph, void const* blob, int64_t size,
    ISafeRecorder& recorder, bool useManaged, ISafeMemAllocator* allocator);

//!
//! \brief Destroy a safe TRT graph and release resources
//!
//! This function destroys a safe TRT graph and releases the associated resources. It is used to clean up
//! the safe TRT graph after inference with safety-certified TensorRT engines.
//!
//! \param graph: Pointer to the safe TRT graph to be destroyed
//! \return Error code indicating the success or failure of the operation
//!
nvinfer1::ErrorCode destroySafeTRTGraph(nvinfer2::safe::ITRTGraph*& graph);

//!
//! \brief Get the safe plugin registry for loading plugins
//!
//! This function retrieves the safe plugin registry for loading plugins. It is used to get the safe plugin registry
//! for loading plugins with safety-certified TensorRT engines.
//!
//! \param recorder: Reference to the safe recorder
//! \return Pointer to the safe plugin registry
//!
nvinfer2::safe::ISafePluginRegistry* getSafePluginRegistry(ISafeRecorder& recorder);
} // namespace safe
#endif

struct InferenceEnvironmentBase
{
    InferenceEnvironmentBase() = delete;
    virtual ~InferenceEnvironmentBase() = default;
    InferenceEnvironmentBase(InferenceEnvironmentBase const& other) = delete;
    InferenceEnvironmentBase(InferenceEnvironmentBase&& other) = delete;
    InferenceEnvironmentBase(BuildEnvironment& bEnv)
        : engine(std::move(bEnv.engine))
        , safe(bEnv.engine.isSafe())
        , cmdline(bEnv.cmdline)
    {
    }

    LazilyDeserializedEngine engine;
    std::unique_ptr<Profiler> profiler;
    std::vector<TrtDeviceBuffer>
        deviceMemory; //< Device memory used for inference when the allocation strategy is not static.
    std::unique_ptr<DebugTensorWriter> listener;
    bool error{false};

    bool safe{false};
    std::string cmdline;
};

struct InferenceEnvironmentStd : public InferenceEnvironmentBase
{
    InferenceEnvironmentStd() = delete;
    InferenceEnvironmentStd(InferenceEnvironmentStd const& other) = delete;
    InferenceEnvironmentStd(InferenceEnvironmentStd&& other) = delete;
    InferenceEnvironmentStd(BuildEnvironment& bEnv)
        : InferenceEnvironmentBase(bEnv)
    {
    }
    std::vector<std::unique_ptr<nvinfer1::IExecutionContext>> contexts;
    std::vector<std::unique_ptr<BindingsStd>> bindings;

    inline nvinfer1::IExecutionContext* getContext(int32_t streamIdx);

    //! Storage for input shape tensors.
    //!
    //! It's important that the addresses of the data do not change between the calls to
    //! setTensorAddress/setInputShape (which tells TensorRT where the input shape tensor is)
    //! and enqueueV3 (when TensorRT might use the input shape tensor).
    //!
    //! The input shape tensors could alternatively be handled via member bindings,
    //! but it simplifies control-flow to store the data here since it's shared across
    //! the bindings.
    std::list<std::vector<int64_t>> inputShapeTensorValues;
};

#if ENABLE_UNIFIED_BUILDER
// Forward declaration of BindingsSafe
class BindingsSafe;

struct InferenceEnvironmentSafe : public InferenceEnvironmentBase
{
    InferenceEnvironmentSafe() = delete;
    InferenceEnvironmentSafe(InferenceEnvironmentSafe const& other) = delete;
    InferenceEnvironmentSafe(InferenceEnvironmentSafe&& other) = delete;
    InferenceEnvironmentSafe(BuildEnvironment& bEnv)
        : InferenceEnvironmentBase(bEnv)
    {
    }

    std::vector<std::unique_ptr<BindingsSafe>> bindings;
    inline void* getClonedGraph(int32_t streamIdx);

    std::vector<std::unique_ptr<nvinfer2::safe::ITRTGraph>> mClonedGraphs;
};
#endif

inline nvinfer1::IExecutionContext* InferenceEnvironmentStd::getContext(int32_t streamIdx)
{
    return contexts[streamIdx].get();
}

//!
//! \brief Set up contexts/graphs and bindings for inference
//!
bool setUpInference(InferenceEnvironmentBase& iEnv, InferenceOptions const& inference, SystemOptions const& system);

#if ENABLE_UNIFIED_BUILDER
//!
//! \brief Set up graphs and bindings for safe inference
//!
bool setUpSafeInference(InferenceEnvironmentSafe& iEnv, InferenceOptions const& inference, SystemOptions const& system);
#endif

//!
//! \brief Set up contexts and bindings for standard inference
//!
bool setUpStdInference(InferenceEnvironmentStd& iEnv, InferenceOptions const& inference, SystemOptions const& system);

//!
//! \brief Deserialize the engine and time how long it takes.
//!
bool timeDeserialize(InferenceEnvironmentBase& iEnv, SystemOptions const& sys);

//!
//! \brief Run inference and collect timing, return false if any error hit during inference
//!
bool runInference(InferenceOptions const& inference, InferenceEnvironmentBase& iEnv, int32_t device,
    std::vector<InferenceTrace>& trace, ReportingOptions const& reporting);

//!
//! \brief Get layer information of the engine.
//!
std::string getLayerInformation(
    nvinfer1::ICudaEngine* engine, nvinfer1::IExecutionContext* context, nvinfer1::LayerInformationFormat format);

struct Binding
{
    bool isInput{false};
    std::unique_ptr<IMirroredBuffer> buffer;
    std::unique_ptr<OutputAllocator> outputAllocator;
    int64_t volume{0};
    nvinfer1::DataType dataType{nvinfer1::DataType::kFLOAT};

    void fill(std::string const& fileName);

    void fill();

    void dump(std::ostream& os, nvinfer1::Dims dims, nvinfer1::Dims strides, int32_t vectorDim, int32_t spv,
        std::string const separator = " ") const;
};

struct TensorInfo
{
    int32_t bindingIndex{-1};
    char const* name{nullptr};
    nvinfer1::Dims dims{};
    bool isDynamic{};
    int32_t comps{-1};
    nvinfer1::Dims strides{};
    int32_t vectorDimIndex{-1};
    bool isInput{};
    nvinfer1::DataType dataType{};
    int64_t vol{-1};

    void updateVolume(int32_t batch)
    {
        vol = volume(dims, strides, vectorDimIndex, comps, batch);
    }
};

class BindingsBase
{
public:
    BindingsBase() = delete;
    explicit BindingsBase(bool useManaged)
        : mUseManaged(useManaged)
    {
    }

    void addBinding(TensorInfo const& tensorInfo, std::string const& fileName = "");

    void** getDeviceBuffers();

    void transferInputToDevice(TrtCudaStream& stream);

    void transferOutputToHost(TrtCudaStream& stream);

    void fill(int binding, std::string const& fileName)
    {
        mBindings[binding].fill(fileName);
    }

    void fill(int binding)
    {
        mBindings[binding].fill();
    }

    std::unordered_map<std::string, int> getInputBindings() const
    {
        auto isInput = [](Binding const& b) { return b.isInput; };
        return getBindings(isInput);
    }

    std::unordered_map<std::string, int> getOutputBindings() const
    {
        auto isOutput = [](Binding const& b) { return !b.isInput; };
        return getBindings(isOutput);
    }

    std::unordered_map<std::string, int> getBindings() const
    {
        auto all = [](Binding const& b) { return true; };
        return getBindings(all);
    }

    std::unordered_map<std::string, int> getBindings(std::function<bool(Binding const&)> predicate) const;

protected:
    std::unordered_map<std::string, int32_t> mNames;
    std::vector<Binding> mBindings;
    std::vector<void*> mDevicePointers;
    bool mUseManaged{false};
};

class BindingsStd : public BindingsBase
{
public:
    BindingsStd() = delete;
    explicit BindingsStd(bool useManaged)
        : BindingsBase(useManaged)
    {
    }

    void dumpInputs(nvinfer1::IExecutionContext const& context, std::ostream& os) const
    {
        auto isInput = [](Binding const& b) { return b.isInput; };
        dumpBindings(context, isInput, os);
    }

    void dumpOutputs(nvinfer1::IExecutionContext const& context, std::ostream& os) const
    {
        auto isOutput = [](Binding const& b) { return !b.isInput; };
        dumpBindings(context, isOutput, os);
    }

    void dumpBindings(nvinfer1::IExecutionContext const& context, std::ostream& os) const
    {
        auto all = [](Binding const& b) { return true; };
        dumpBindings(context, all, os);
    }

    void dumpBindings(nvinfer1::IExecutionContext const& context, std::function<bool(Binding const&)> predicate,
        std::ostream& os) const
    {
        for (auto const& n : mNames)
        {
            auto const name = n.first;
            auto const binding = n.second;
            if (predicate(mBindings[binding]))
            {
                os << n.first << ": (";
                dumpBindingDimensions(name, context, os);
                os << ")" << std::endl;

                dumpBindingValues(context, binding, os);
                os << std::endl;
            }
        }
    }

    void dumpBindingDimensions(
        std::string const& name, nvinfer1::IExecutionContext const& context, std::ostream& os) const;

    void dumpBindingValues(nvinfer1::IExecutionContext const& context, int32_t binding, std::ostream& os,
        std::string const& separator = " ", int32_t batch = 1) const;

    void dumpRawBindingToFiles(nvinfer1::IExecutionContext const& context, std::ostream& os) const;

    bool setTensorAddresses(nvinfer1::IExecutionContext& context) const;
};
#if ENABLE_UNIFIED_BUILDER
class BindingsSafe : public BindingsBase
{
public:
    BindingsSafe() = delete;
    explicit BindingsSafe(bool useManaged)
        : BindingsBase(useManaged)
    {
    }

    void dumpInputs(ITRTGraph const& graph, std::ostream& os) const
    {
        auto isInput = [](Binding const& b) { return b.isInput; };
        dumpBindings(graph, isInput, os);
    }

    void dumpOutputs(ITRTGraph const& graph, std::ostream& os) const
    {
        auto isOutput = [](Binding const& b) { return !b.isInput; };
        dumpBindings(graph, isOutput, os);
    }

    void dumpBindings(ITRTGraph const& graph, std::ostream& os) const
    {
        auto all = [](Binding const& b) { return true; };
        dumpBindings(graph, all, os);
    }

    void dumpBindings(ITRTGraph const& graph, std::function<bool(Binding const&)> predicate, std::ostream& os) const
    {
        for (auto const& n : mNames)
        {
            auto const name = n.first;
            auto const binding = n.second;
            if (predicate(mBindings[binding]))
            {
                os << n.first << ": (";
                dumpBindingDimensions(name, graph, os);
                os << ")" << std::endl;

                dumpBindingValues(graph, binding, os);
                os << std::endl;
            }
        }
    }

    void dumpBindingDimensions(std::string const& name, ITRTGraph const& graph, std::ostream& os) const;

    void dumpBindingValues(ITRTGraph const& graph, int32_t binding, std::ostream& os,
        std::string const& separator = " ", int32_t batch = 1) const;

    void dumpRawBindingToFiles(ITRTGraph& graph, std::ostream& os) const;

    bool setTensorAddresses(ITRTGraph& graph) const;
};
#endif

struct TaskInferenceEnvironment
{
    TaskInferenceEnvironment(std::string engineFile, InferenceOptions const& inference,
        ReportingOptions const& reporting, int32_t deviceId = 0,
        int32_t DLACore = -1, int32_t bs = batchNotProvided);
    InferenceOptions iOptions{};
    ReportingOptions rOptions{};
    int32_t device{defaultDevice};
    int32_t batch{batchNotProvided};
    std::unique_ptr<InferenceEnvironmentStd> iEnv;
    std::vector<InferenceTrace> trace;
};

bool runMultiTasksInference(std::vector<std::unique_ptr<TaskInferenceEnvironment>>& tEnvList);

} // namespace sample

#endif // TRT_SAMPLE_INFERENCE_H
