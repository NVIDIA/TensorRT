/*
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

#ifndef NV_INFER_RUNTIME_H
#define NV_INFER_RUNTIME_H

//!
//! \file NvInferRuntime.h
//!
//! This is the top-level API file for TensorRT extended runtime library.
//!

#include "NvInferImpl.h"
#include "NvInferRuntimeCommon.h"

namespace nvinfer1
{

class IExecutionContext; //!< Forward declaration of IExecutionContext for use by other interfaces.
class ICudaEngine;       //!< Forward declaration of ICudaEngine for use by other interfaces.
class IPluginFactory;    //!< Forward declaration of IPluginFactory for use by other interfaces.
class IEngineInspector;  //!< Forward declaration of IEngineInspector for use by other interfaces.

//!
//! \class INoCopy
//!
//! \brief Base class for all TensorRT interfaces that are implemented by the TensorRT libraries
//!
//! Objects of such classes are not movable or copyable, and should only be manipulated
//! via pointers.
//!

class INoCopy
{
protected:
    INoCopy() = default;
    virtual ~INoCopy() = default;
    INoCopy(const INoCopy& other) = delete;
    INoCopy& operator=(const INoCopy& other) = delete;
    INoCopy(INoCopy&& other) = delete;
    INoCopy& operator=(INoCopy&& other) = delete;
};

//!
//! \enum EngineCapability
//!
//! \brief List of supported engine capability flows.
//!
//! \details The EngineCapability determines the restrictions of a network during build time and what runtime
//! it targets. When BuilderFlag::kSAFETY_SCOPE is not set (by default), EngineCapability::kSTANDARD does not provide
//! any restrictions on functionality and the resulting serialized engine can be executed with TensorRT's standard
//! runtime APIs in the nvinfer1 namespace. EngineCapability::kSAFETY provides a restricted subset of network
//! operations that are safety certified and the resulting serialized engine can be executed with TensorRT's safe
//! runtime APIs in the nvinfer1::safe namespace. EngineCapability::kDLA_STANDALONE provides a restricted subset of
//! network operations that are DLA compatible and the resulting serialized engine can be executed using standalone
//! DLA runtime APIs. See sampleNvmedia for an example of integrating NvMediaDLA APIs with TensorRT APIs.
//!

enum class EngineCapability : int32_t
{
    //!
    //! Standard: TensorRT flow without targeting the safety runtime.
    //! This flow supports both DeviceType::kGPU and DeviceType::kDLA.
    //!
    kSTANDARD = 0,
    kDEFAULT TRT_DEPRECATED_ENUM = kSTANDARD,

    //!
    //! Safety: TensorRT flow with restrictions targeting the safety runtime.
    //! See safety documentation for list of supported layers and formats.
    //! This flow supports only DeviceType::kGPU.
    //!
    //! This flag is only supported in NVIDIA Drive(R) products.
    kSAFETY = 1,
    kSAFE_GPU TRT_DEPRECATED_ENUM = kSAFETY,

    //!
    //! DLA Standalone: TensorRT flow with restrictions targeting external, to TensorRT, DLA runtimes.
    //! See DLA documentation for list of supported layers and formats.
    //! This flow supports only DeviceType::kDLA.
    //!
    kDLA_STANDALONE = 2,
    kSAFE_DLA TRT_DEPRECATED_ENUM = kDLA_STANDALONE,
};

namespace impl
{
//! Maximum number of elements in EngineCapability enum. \see EngineCapability
template <>
struct EnumMaxImpl<EngineCapability>
{
    static constexpr int32_t kVALUE = 3;
};
} // namespace impl

//!
//! \class Weights
//!
//! \brief An array of weights used as a layer parameter.
//!
//! When using the DLA, the cumulative size of all Weights used in a network
//! must be less than 512MB in size. If the build option kGPU_FALLBACK is specified,
//! then multiple DLA sub-networks may be generated from the single original network.
//!
//! The weights are held by reference until the engine has been built. Therefore the data referenced
//! by \p values field should be preserved until the build is complete.
//!
//! The term "empty weights" refers to Weights with weight coefficients ( \p count == 0 and \p values == nullptr).
//!
class Weights
{
public:
    DataType type;      //!< The type of the weights.
    const void* values; //!< The weight values, in a contiguous array.
    int64_t count;      //!< The number of weights in the array.
};

//!
//! \class IHostMemory
//!
//! \brief Class to handle library allocated memory that is accessible to the user.
//!
//! The memory allocated via the host memory object is owned by the library and will
//! be de-allocated when the destroy method is called.
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
class IHostMemory : public INoCopy
{
public:
    virtual ~IHostMemory() noexcept = default;

    //! A pointer to the raw data that is owned by the library.
    void* data() const noexcept
    {
        return mImpl->data();
    }

    //! The size in bytes of the data that was allocated.
    std::size_t size() const noexcept
    {
        return mImpl->size();
    }

    //! The type of the memory that was allocated.
    DataType type() const noexcept
    {
        return mImpl->type();
    }
    //!
    //! Destroy the allocated memory.
    //!
    //! \deprecated Deprecated interface will be removed in TensorRT 10.0.
    //!
    //! \warning Calling destroy on a managed pointer will result in a double-free error.
    //!
    TRT_DEPRECATED void destroy() noexcept
    {
        delete this;
    }

protected:
    apiv::VHostMemory* mImpl;
};

//!
//! \enum DimensionOperation
//!
//! \brief An operation on two IDimensionExpr, which represent integer expressions used in dimension computations.
//!
//! For example, given two IDimensionExpr x and y and an IExprBuilder& eb,
//! eb.operation(DimensionOperation::kSUM, x, y) creates a representation of x+y.
//!
//! \see IDimensionExpr, IExprBuilder
//!
enum class DimensionOperation : int32_t
{
    kSUM = 0,       //!< Sum of the two operands.
    kPROD = 1,      //!< Product of the two operands.
    kMAX = 2,       //!< Maximum of the two operands.
    kMIN = 3,       //!< Minimum of the two operands.
    kSUB = 4,       //!< Substract the second element from the first.
    kEQUAL = 5,     //!< 1 if operands are equal, 0 otherwise.
    kLESS = 6,      //!< 1 if first operand is less than second operand, 0 otherwise.
    kFLOOR_DIV = 7, //!< Floor division of the first element by the second.
    kCEIL_DIV = 8   //!< Division rounding up
};

//! Maximum number of elements in DimensionOperation enum. \see DimensionOperation
template <>
constexpr inline int32_t EnumMax<DimensionOperation>() noexcept
{
    return 9;
}

//!
//! \enum TensorLocation
//! \brief The location for tensor data storage, device or host.
//!
enum class TensorLocation : int32_t
{
    kDEVICE = 0, //!< Data stored on device.
    kHOST = 1,   //!< Data stored on host.
};

namespace impl
{
//! Maximum number of elements in TensorLocation enum. \see TensorLocation
template <>
struct EnumMaxImpl<TensorLocation>
{
    static constexpr int32_t kVALUE = 2;
};
} // namespace impl

//!
//! \class IDimensionExpr
//!
//! An IDimensionExpr represents an integer expression constructed from constants,
//! input dimensions, and binary operations.  These expressions are can be used
//! in overrides of IPluginV2DynamicExt::getOutputDimensions to define output
//! dimensions in terms of input dimensions.
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
//! \see DimensionOperation, IPluginV2DynamicExt::getOutputDimensions
//!
class IDimensionExpr : public INoCopy
{
public:
    //! Return true if expression is a build-time constant.
    bool isConstant() const noexcept
    {
        return mImpl->isConstant();
    }

    //! If isConstant(), returns value of the constant.
    //! If !isConstant(), return std::numeric_limits<int32_t>::min().
    int32_t getConstantValue() const noexcept
    {
        return mImpl->getConstantValue();
    }

protected:
    apiv::VDimensionExpr* mImpl;
    virtual ~IDimensionExpr() noexcept = default;
};

//!
//! \class IExprBuilder
//!
//! Object for constructing IDimensionExpr.
//!
//! There is no public way to construct an IExprBuilder.  It appears as an argument to
//! method IPluginV2DynamicExt::getOutputDimensions().  Overrides of that method can use
//! that IExprBuilder argument to construct expressions that define output dimensions
//! in terms of input dimensions.
//!
//! Clients should assume that any values constructed by the IExprBuilder are destroyed
//! after IPluginV2DynamicExt::getOutputDimensions() returns.
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
//! \see IDimensionExpr
//!
class IExprBuilder : public INoCopy
{
public:
    //! Return pointer to IDimensionExp for given value.
    const IDimensionExpr* constant(int32_t value) noexcept
    {
        return mImpl->constant(value);
    }

    //! Return pointer to IDimensionExp that represents the given operation applied to first and second.
    //! Returns nullptr if op is not a valid DimensionOperation.
    const IDimensionExpr* operation(
        DimensionOperation op, const IDimensionExpr& first, const IDimensionExpr& second) noexcept
    {
        return mImpl->operation(op, first, second);
    }

protected:
    apiv::VExprBuilder* mImpl;
    virtual ~IExprBuilder() noexcept = default;
};

//!
//! \class DimsExprs
//!
//! Analog of class Dims with expressions instead of constants for the dimensions.
//!
class DimsExprs
{
public:
    int32_t nbDims;                          //!< The number of dimensions.
    const IDimensionExpr* d[Dims::MAX_DIMS]; //!< The extent of each dimension.
};

//!
//! \class DynamicPluginTensorDesc
//!
//! Summarizes tensors that a plugin might see for an input or output.
//!
struct DynamicPluginTensorDesc
{
    //! Information required to interpret a pointer to tensor data, except that desc.dims has -1 in place of any runtime dimension.
    PluginTensorDesc desc;

    //! Lower bounds on tensor’s dimensions
    Dims min;

    //! Upper bounds on tensor’s dimensions
    Dims max;
};

//!
//! \class IPluginV2DynamicExt
//!
//! Similar to IPluginV2Ext, but with support for dynamic shapes.
//!
//! Clients should override the public methods, including the following inherited methods:
//!
//!     virtual int32_t getNbOutputs() const noexcept = 0;
//!     virtual nvinfer1::DataType getOutputDataType(int32_t index, const nvinfer1::DataType* inputTypes, int32_t
//!     nbInputs) const noexcept = 0; virtual size_t getSerializationSize() const noexcept = 0; virtual void
//!     serialize(void* buffer) const noexcept = 0; virtual void destroy() noexcept = 0; virtual void
//!     setPluginNamespace(const char* pluginNamespace) noexcept = 0; virtual const char* getPluginNamespace() const
//!     noexcept = 0;
//!
//! For getOutputDataType, the inputTypes will always be DataType::kFLOAT or DataType::kINT32,
//! and the returned type is canonicalized to DataType::kFLOAT if it is DataType::kHALF or DataType:kINT8.
//! Details about the floating-point precision are elicited later by method supportsFormatCombination.
//!
class IPluginV2DynamicExt : public nvinfer1::IPluginV2Ext
{
public:
    IPluginV2DynamicExt* clone() const noexcept override = 0;

    //!
    //! \brief Get expressions for computing dimensions of an output tensor from dimensions of the input tensors.
    //!
    //! \param outputIndex The index of the output tensor
    //! \param inputs Expressions for dimensions of the input tensors
    //! \param nbInputs The number of input tensors
    //! \param exprBuilder Object for generating new expressions
    //!
    //! This function is called by the implementations of IBuilder during analysis of the network.
    //!
    //! Example #1: A plugin has a single output that transposes the last two dimensions of the plugin's single input.
    //! The body of the override of getOutputDimensions can be:
    //!
    //!     DimsExprs output(inputs[0]);
    //!     std::swap(output.d[output.nbDims-1], output.d[output.nbDims-2]);
    //!     return output;
    //!
    //! Example #2: A plugin concatenates its two inputs along the first dimension.
    //! The body of the override of getOutputDimensions can be:
    //!
    //!     DimsExprs output(inputs[0]);
    //!     output.d[0] = exprBuilder.operation(DimensionOperation::kSUM, *inputs[0].d[0], *inputs[1].d[0]);
    //!     return output;
    //!
    virtual DimsExprs getOutputDimensions(
        int32_t outputIndex, const DimsExprs* inputs, int32_t nbInputs, IExprBuilder& exprBuilder) noexcept
        = 0;

    //!
    //! Limit on number of format combinations accepted.
    //!
    static constexpr int32_t kFORMAT_COMBINATION_LIMIT = 100;

    //!
    //! \brief Return true if plugin supports the format and datatype for the input/output indexed by pos.
    //!
    //! For this method inputs are numbered 0..(nbInputs-1) and outputs are numbered nbInputs..(nbInputs+nbOutputs-1).
    //! Using this numbering, pos is an index into InOut, where 0 <= pos < nbInputs+nbOutputs-1.
    //!
    //! TensorRT invokes this method to ask if the input/output indexed by pos supports the format/datatype specified
    //! by inOut[pos].format and inOut[pos].type.  The override should return true if that format/datatype at inOut[pos]
    //! are supported by the plugin.  If support is conditional on other input/output formats/datatypes, the plugin can
    //! make its result conditional on the formats/datatypes in inOut[0..pos-1], which will be set to values
    //! that the plugin supports.  The override should not inspect inOut[pos+1..nbInputs+nbOutputs-1],
    //! which will have invalid values.  In other words, the decision for pos must be based on inOut[0..pos] only.
    //!
    //! Some examples:
    //!
    //! * A definition for a plugin that supports only FP16 NCHW:
    //!
    //!         return inOut.format[pos] == TensorFormat::kLINEAR && inOut.type[pos] == DataType::kHALF;
    //!
    //! * A definition for a plugin that supports only FP16 NCHW for its two inputs,
    //!   and FP32 NCHW for its single output:
    //!
    //!         return inOut.format[pos] == TensorFormat::kLINEAR && (inOut.type[pos] == pos < 2 ?  DataType::kHALF :
    //!         DataType::kFLOAT);
    //!
    //! * A definition for a "polymorphic" plugin with two inputs and one output that supports
    //!   any format or type, but the inputs and output must have the same format and type:
    //!
    //!         return pos == 0 || (inOut.format[pos] == inOut.format[0] && inOut.type[pos] == inOut.type[0]);
    //!
    //! Warning: TensorRT will stop asking for formats once it finds kFORMAT_COMBINATION_LIMIT on combinations.
    //!
    virtual bool supportsFormatCombination(
        int32_t pos, const PluginTensorDesc* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
        = 0;

    //!
    //! \brief Configure the plugin.
    //!
    //! configurePlugin() can be called multiple times in both the build and execution phases. The build phase happens
    //! before initialize() is called and only occurs during creation of an engine by IBuilder. The execution phase
    //! happens after initialize() is called and occurs during both creation of an engine by IBuilder and execution
    //! of an engine by IExecutionContext.
    //!
    //! Build phase:
    //! IPluginV2DynamicExt->configurePlugin is called when a plugin is being prepared for profiling but not for any
    //! specific input size. This provides an opportunity for the plugin to make algorithmic choices on the basis of
    //! input and output formats, along with the bound of possible dimensions. The min and max value of the
    //! DynamicPluginTensorDesc correspond to the kMIN and kMAX value of the current profile that the plugin is being
    //! profiled for, with the desc.dims field corresponding to the dimensions of plugin specified at network creation.
    //! Wildcard dimensions will exist during this phase in the desc.dims field.
    //!
    //! Execution phase:
    //! IPluginV2DynamicExt->configurePlugin is called when a plugin is being prepared for executing the plugin for a
    //! specific dimensions. This provides an opportunity for the plugin to change algorithmic choices based on the
    //! explicit input dimensions stored in desc.dims field.
    //!  * IBuilder will call this function once per profile, with desc.dims resolved to the values specified by the
    //!  kOPT
    //!    field of the current profile. Wildcard dimensions will not exist during this phase.
    //!  * IExecutionContext will call this during the next subsequent instance enqueue[V2]() or execute[V2]() if:
    //!    - The batch size is changed from previous call of execute()/enqueue() if hasImplicitBatchDimension() returns
    //!    true.
    //!    - The optimization profile is changed via setOptimizationProfile() or setOptimizationProfileAsync().
    //!    - An input shape binding is changed via setInputShapeBinding().
    //!    - An input execution binding is changed via setBindingDimensions().
    //! \warning The execution phase is timing critical during IExecutionContext but is not part of the timing loop when
    //! called from IBuilder. Performance bottlenecks of configurePlugin won't show up during engine building but will
    //! be visible during execution after calling functions that trigger layer resource updates.
    //!
    //! \param in The input tensors attributes that are used for configuration.
    //! \param nbInputs Number of input tensors.
    //! \param out The output tensors attributes that are used for configuration.
    //! \param nbOutputs Number of output tensors.
    //!
    virtual void configurePlugin(const DynamicPluginTensorDesc* in, int32_t nbInputs,
        const DynamicPluginTensorDesc* out, int32_t nbOutputs) noexcept
        = 0;

    //!
    //! \brief Find the workspace size required by the layer.
    //!
    //! This function is called after the plugin is configured, and possibly during execution.
    //! The result should be a sufficient workspace size to deal with inputs and outputs of the given size
    //! or any smaller problem.
    //!
    //! \return The workspace size.
    //!
    virtual size_t getWorkspaceSize(const PluginTensorDesc* inputs, int32_t nbInputs, const PluginTensorDesc* outputs,
        int32_t nbOutputs) const noexcept
        = 0;

    //!
    //! \brief Execute the layer.
    //!
    //! \param inputDesc how to interpret the memory for the input tensors.
    //! \param outputDesc how to interpret the memory for the output tensors.
    //! \param inputs The memory for the input tensors.
    //! \param outputs The memory for the output tensors.
    //! \param workspace Workspace for execution.
    //! \param stream The stream in which to execute the kernels.
    //!
    //! \return 0 for success, else non-zero (which will cause engine termination).
    //!
    virtual int32_t enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc,
        const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
        = 0;

protected:
    //!
    //! \brief Return the API version with which this plugin was built. The
    //!  upper byte reserved by TensorRT and is used to differentiate this from IPluginV2.
    //!
    //! Do not override this method as it is used by the TensorRT library to maintain backwards-compatibility with
    //! plugins.
    //!
    int32_t getTensorRTVersion() const noexcept override
    {
        return (static_cast<int32_t>(PluginVersion::kV2_DYNAMICEXT) << 24 | (NV_TENSORRT_VERSION & 0xFFFFFF));
    }

    virtual ~IPluginV2DynamicExt() noexcept {}

private:
    // Following are obsolete base class methods, and must not be implemented or used.

    void configurePlugin(Dims const*, int32_t, Dims const*, int32_t, DataType const*, DataType const*, bool const*,
        bool const*, PluginFormat, int32_t) noexcept override final
    {
    }

    bool supportsFormat(DataType, PluginFormat) const noexcept override final
    {
        return false;
    }

    Dims getOutputDimensions(int32_t, Dims const*, int32_t) noexcept override final
    {
        return Dims{-1, {}};
    }

    bool isOutputBroadcastAcrossBatch(int32_t, bool const*, int32_t) const noexcept override final
    {
        return false;
    }

    bool canBroadcastInputAcrossBatch(int32_t) const noexcept override final
    {
        return true;
    }

    size_t getWorkspaceSize(int32_t) const noexcept override final
    {
        return 0;
    }

    int32_t enqueue(int32_t, const void* const*, void* const*, void*, cudaStream_t) noexcept override final
    {
        return 1;
    }
};

//!
//! \class IProfiler
//!
//! \brief Application-implemented interface for profiling.
//!
//! When this class is added to an execution context, the profiler will be called once per layer for each invocation of execute().
//! Note that enqueue() does not currently support profiling.
//!
//! The profiler will only be called after execution is complete. It has a small impact on execution time.
//!
class IProfiler
{
public:
    //!
    //! \brief Layer time reporting callback.
    //!
    //! \param layerName The name of the layer, set when constructing the network definition.
    //! \param ms The time in milliseconds to execute the layer.
    //!
    virtual void reportLayerTime(const char* layerName, float ms) noexcept = 0;

    virtual ~IProfiler() noexcept {}
};

//!
//! \enum WeightsRole
//! \brief How a layer uses particular Weights.
//!
//! The power weights of an IScaleLayer are omitted.  Refitting those is not supported.
//!
enum class WeightsRole : int32_t
{
    kKERNEL = 0,   //!< kernel for IConvolutionLayer, IDeconvolutionLayer, or IFullyConnectedLayer
    kBIAS = 1,     //!< bias for IConvolutionLayer, IDeconvolutionLayer, or IFullyConnectedLayer
    kSHIFT = 2,    //!< shift part of IScaleLayer
    kSCALE = 3,    //!< scale part of IScaleLayer
    kCONSTANT = 4, //!< weights for IConstantLayer
    kANY = 5,      //!< Any other weights role
};

//! Maximum number of elements in WeightsRole enum. \see WeightsRole
template <>
constexpr inline int32_t EnumMax<WeightsRole>() noexcept
{
    return 6;
}

//!
//! \enum DeviceType
//! \brief The device that this layer/network will execute on.
//!
//!
enum class DeviceType : int32_t
{
    kGPU, //!< GPU Device
    kDLA, //!< DLA Core
};

//! Maximum number of elements in DeviceType enum. \see DeviceType
template <>
constexpr inline int32_t EnumMax<DeviceType>() noexcept
{
    return 2;
}

//!
//! \class IRuntime
//!
//! \brief Allows a serialized functionally unsafe engine to be deserialized.
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
class IRuntime : public INoCopy
{
public:
    virtual ~IRuntime() noexcept = default;

    //!
    //! \brief Deserialize an engine from a stream.
    //!
    //! If an error recorder has been set for the runtime, it will also be passed to the engine.
    //!
    //! \param blob The memory that holds the serialized engine.
    //! \param size The size of the memory in bytes.
    //! \param pluginFactory The plugin factory, if any plugins are used by the network, otherwise nullptr.
    //!
    //! \return The engine, or nullptr if it could not be deserialized.
    //!
    //! \deprecated Deprecated interface will be removed in TensorRT 10.0.
    //!
    //! \warning IPluginFactory is no longer supported, therefore pluginFactory must be a nullptr.
    //!
    TRT_DEPRECATED nvinfer1::ICudaEngine* deserializeCudaEngine(
        const void* blob, std::size_t size, IPluginFactory* pluginFactory) noexcept
    {
        return mImpl->deserializeCudaEngine(blob, size, nullptr);
    }

    //!
    //! \brief Set the DLA core that the deserialized engine must execute on.
    //! \param dlaCore The DLA core to execute the engine on (0 to N-1, where N is the maximum number of DLA's present
    //! on the device). Default value is 0. \see getDLACore()
    //!
    //! \warning Starting with TensorRT 8, the default value will be -1 if the DLA is not specified or unused.
    //!
    void setDLACore(int32_t dlaCore) noexcept
    {
        mImpl->setDLACore(dlaCore);
    }

    //!
    //! \brief Get the DLA core that the engine executes on.
    //! \return If setDLACore is called, returns DLA core from 0 to N-1, else returns 0.
    //!
    //! \warning Starting with TensorRT 8, the default value will be -1 if the DLA is not specified or unused.
    //!
    int32_t getDLACore() const noexcept
    {
        return mImpl->getDLACore();
    }

    //!
    //! \brief Returns number of DLA hardware cores accessible.
    //!
    int32_t getNbDLACores() const noexcept
    {
        return mImpl->getNbDLACores();
    }

    //!
    //! \brief Destroy this object.
    //!
    //! \deprecated Deprecated interface will be removed in TensorRT 10.0.
    //!
    //! \warning Calling destroy on a managed pointer will result in a double-free error.
    //!
    TRT_DEPRECATED void destroy() noexcept
    {
        delete this;
    }

    //!
    //! \brief Set the GPU allocator.
    //! \param allocator Set the GPU allocator to be used by the runtime. All GPU memory acquired will use this
    //! allocator. If NULL is passed, the default allocator will be used.
    //!
    //! Default: uses cudaMalloc/cudaFree.
    //!
    //! If nullptr is passed, the default allocator will be used.
    //!
    void setGpuAllocator(IGpuAllocator* allocator) noexcept
    {
        mImpl->setGpuAllocator(allocator);
    }

    //!
    //! \brief Set the ErrorRecorder for this interface
    //!
    //! Assigns the ErrorRecorder to this interface. The ErrorRecorder will track all errors during execution.
    //! This function will call incRefCount of the registered ErrorRecorder at least once. Setting
    //! recorder to nullptr unregisters the recorder with the interface, resulting in a call to decRefCount if
    //! a recorder has been registered.
    //!
    //! If an error recorder is not set, messages will be sent to the global log stream.
    //!
    //! \param recorder The error recorder to register with this interface.
    //
    //! \see getErrorRecorder()
    //!
    void setErrorRecorder(IErrorRecorder* recorder) noexcept
    {
        mImpl->setErrorRecorder(recorder);
    }

    //!
    //! \brief get the ErrorRecorder assigned to this interface.
    //!
    //! Retrieves the assigned error recorder object for the given class. A nullptr will be returned if
    //! an error handler has not been set.
    //!
    //! \return A pointer to the IErrorRecorder object that has been registered.
    //!
    //! \see setErrorRecorder()
    //!
    IErrorRecorder* getErrorRecorder() const noexcept
    {
        return mImpl->getErrorRecorder();
    }

    //!
    //! \brief Deserialize an engine from a stream.
    //!
    //! If an error recorder has been set for the runtime, it will also be passed to the engine.
    //!
    //! \param blob The memory that holds the serialized engine.
    //! \param size The size of the memory.
    //!
    //! \return The engine, or nullptr if it could not be deserialized.
    //!
    ICudaEngine* deserializeCudaEngine(const void* blob, std::size_t size) noexcept
    {
        return mImpl->deserializeCudaEngine(blob, size, nullptr);
    }

    //!
    //! \brief get the logger with which the runtime was created
    //!
    //! \return the logger
    //!
    ILogger* getLogger() const noexcept
    {
        return mImpl->getLogger();
    }

protected:
    apiv::VRuntime* mImpl;
};

//!
//! \class IRefitter
//!
//! \brief Updates weights in an engine.
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
class IRefitter : public INoCopy
{
public:
    virtual ~IRefitter() noexcept = default;

    //!
    //! \brief Specify new weights for a layer of given name.
    //! Returns true on success, or false if new weights are rejected.
    //! Possible reasons for rejection are:
    //!
    //! * There is no such layer by that name.
    //! * The layer does not have weights with the specified role.
    //! * The number of weights is inconsistent with the layer’s original specification.
    //!
    //! Modifying the weights before method refit() completes will result in undefined behavior.
    bool setWeights(const char* layerName, WeightsRole role, Weights weights) noexcept
    {
        return mImpl->setWeights(layerName, role, weights);
    }

    //!
    //! \brief Updates associated engine.  Return true if successful.
    //!
    //! Failure occurs if getMissing() != 0 before the call.
    //!
    //! The behavior is undefined if the engine has pending enqueued work.
    //!
    //! Extant IExecutionContexts associated with the engine should not be used afterwards.
    //! Instead, create new IExecutionContexts after refitting.
    //!
    bool refitCudaEngine() noexcept
    {
        return mImpl->refitCudaEngine();
    }

    //!
    //! \brief Get description of missing weights.
    //!
    //! For example, if some Weights have been set, but the engine was optimized
    //! in a way that combines weights, any unsupplied Weights in the combination
    //! are considered missing.
    //!
    //! \param size The number of items that can be safely written to a non-null layerNames or roles.
    //! \param layerNames Where to write the layer names.
    //! \param roles Where to write the weights roles.
    //!
    //! \return The number of missing Weights.
    //!
    //! If layerNames!=nullptr, each written pointer points to a string owned by
    //! the engine being refit, and becomes invalid when the engine is destroyed.
    //!
    int32_t getMissing(int32_t size, const char** layerNames, WeightsRole* roles) noexcept
    {
        return mImpl->getMissing(size, layerNames, roles);
    }

    //!
    //! \brief Get description of all weights that could be refit.
    //!
    //! \param size The number of items that can be safely written to a non-null layerNames or roles.
    //! \param layerNames Where to write the layer names.
    //! \param roles Where to write the weights roles.
    //!
    //! \return The number of Weights that could be refit.
    //!
    //! If layerNames!=nullptr, each written pointer points to a string owned by
    //! the engine being refit, and becomes invalid when the engine is destroyed.
    //!
    int32_t getAll(int32_t size, const char** layerNames, WeightsRole* roles) noexcept
    {
        return mImpl->getAll(size, layerNames, roles);
    }

    //!
    //! \deprecated Deprecated interface will be removed in TensorRT 10.0.
    //!
    //! \warning Calling destroy on a managed pointer will result in a double-free error.
    //!
    TRT_DEPRECATED void destroy() noexcept
    {
        delete this;
    }

    //!
    //! Update dynamic range for a tensor.
    //!
    //! \param tensorName The name of an ITensor in the network.
    //! \param min The minimum of the dynamic range for the tensor.
    //! \param max The maximum of the dynamic range for the tensor.
    //!
    //! \return True if successful; false otherwise.
    //!
    //! Returns false if there is no Int8 engine tensor derived from
    //! a network tensor of that name.  If successful, then getMissing
    //! may report that some weights need to be supplied.
    bool setDynamicRange(const char* tensorName, float min, float max) noexcept
    {
        return mImpl->setDynamicRange(tensorName, min, max);
    }

    //!
    //! \brief Get minimum of dynamic range.
    //!
    //! \return Minimum of dynamic range.
    //!
    //! If the dynamic range was never set, returns the minimum computed during calibration.
    //!
    float getDynamicRangeMin(const char* tensorName) const noexcept
    {
        return mImpl->getDynamicRangeMin(tensorName);
    }

    //!
    //! \brief Get maximum of dynamic range.
    //!
    //! \return Maximum of dynamic range.
    //!
    //! If the dynamic range was never set, returns the maximum computed during calibration.
    //!
    float getDynamicRangeMax(const char* tensorName) const noexcept
    {
        return mImpl->getDynamicRangeMax(tensorName);
    }

    //!
    //! \brief Get names of all tensors that have refittable dynamic ranges.
    //!
    //! \param size The number of items that can be safely written to a non-null tensorNames.
    //! \param tensorNames Where to write the layer names.
    //!
    //! \return The number of Weights that could be refit.
    //!
    //! If tensorNames!=nullptr, each written pointer points to a string owned by
    //! the engine being refit, and becomes invalid when the engine is destroyed.
    //!
    int32_t getTensorsWithDynamicRange(int32_t size, const char** tensorNames) const noexcept
    {
        return mImpl->getTensorsWithDynamicRange(size, tensorNames);
    }

    //!
    //! \brief Set the ErrorRecorder for this interface
    //!
    //! Assigns the ErrorRecorder to this interface. The ErrorRecorder will track all errors during execution.
    //! This function will call incRefCount of the registered ErrorRecorder at least once. Setting
    //! recorder to nullptr unregisters the recorder with the interface, resulting in a call to decRefCount if
    //! a recorder has been registered.
    //!
    //! If an error recorder is not set, messages will be sent to the global log stream.
    //!
    //! \param recorder The error recorder to register with this interface.
    //
    //! \see getErrorRecorder()
    //!
    void setErrorRecorder(IErrorRecorder* recorder) noexcept
    {
        mImpl->setErrorRecorder(recorder);
    }

    //!
    //! \brief Get the ErrorRecorder assigned to this interface.
    //!
    //! Retrieves the assigned error recorder object for the given class. A nullptr will be returned if
    //! an error handler has not been set.
    //!
    //! \return A pointer to the IErrorRecorder object that has been registered.
    //!
    //! \see setErrorRecorder()
    //!
    IErrorRecorder* getErrorRecorder() const noexcept
    {
        return mImpl->getErrorRecorder();
    }

    //!
    //! \brief Specify new weights of given name.
    //!
    //! \param name The name of the weights to be refit.
    //! \param weights The new weights to associate with the name.
    //!
    //! Returns true on success, or false if new weights are rejected.
    //! Possible reasons for rejection are:
    //!
    //! * The name of weights is nullptr or does not correspond to any refittable weights.
    //! * The number of weights is inconsistent with the original specification.
    //!
    //! Modifying the weights before method refitCudaEngine() completes will result in undefined behavior.
    bool setNamedWeights(const char* name, Weights weights) noexcept
    {
        return mImpl->setNamedWeights(name, weights);
    }

    //!
    //! \brief Get names of missing weights.
    //!
    //! For example, if some Weights have been set, but the engine was optimized
    //! in a way that combines weights, any unsupplied Weights in the combination
    //! are considered missing.
    //!
    //! \param size The number of weights names that can be safely written to.
    //! \param weightsNames The names of the weights to be updated, or nullptr for unnamed weights.
    //!
    //! \return The number of missing Weights.
    //!
    //! If layerNames!=nullptr, each written pointer points to a string owned by
    //! the engine being refit, and becomes invalid when the engine is destroyed.
    //!
    int32_t getMissingWeights(int32_t size, const char** weightsNames) noexcept
    {
        return mImpl->getMissingWeights(size, weightsNames);
    }

    //!
    //! \brief Get names of all weights that could be refit.
    //!
    //! \param size The number of weights names that can be safely written to.
    //! \param weightsNames The names of the weights to be updated, or nullptr for unnamed weights.
    //!
    //! \return The number of Weights that could be refit.
    //!
    //! If layerNames!=nullptr, each written pointer points to a string owned by
    //! the engine being refit, and becomes invalid when the engine is destroyed.
    //!
    int32_t getAllWeights(int32_t size, const char** weightsNames) noexcept
    {
        return mImpl->getAllWeights(size, weightsNames);
    }

    //!
    //! \brief get the logger with which the refitter was created
    //!
    //! \return the logger
    //!
    ILogger* getLogger() const noexcept
    {
        return mImpl->getLogger();
    }

protected:
    apiv::VRefitter* mImpl;
};

//!
//! \enum OptProfileSelector
//!
//! \brief When setting or querying optimization profile parameters (such as shape tensor inputs or dynamic dimensions),
//!        select whether we are interested in the minimum, optimum, or maximum values for these parameters.
//!        The minimum and maximum specify the permitted range that is supported at runtime, while the optimum value
//!        is used for the kernel selection. This should be the "typical" value that is expected to occur at runtime.
//!
//! \see IOptimizationProfile::setDimensions(), IOptimizationProfile::setShapeValues()
//!
enum class OptProfileSelector : int32_t
{
    kMIN = 0, //!< This is used to set or get the minimum permitted value for dynamic dimensions etc.
    kOPT = 1, //!< This is used to set or get the value that is used in the optimization (kernel selection).
    kMAX = 2  //!< This is used to set or get the maximum permitted value for dynamic dimensions etc.
};

//!< Number of different values of OptProfileSelector enum. \see OptProfileSelector
template <>
constexpr inline int32_t EnumMax<OptProfileSelector>() noexcept
{
    return 3;
}

//!
//! \class IOptimizationProfile
//! \brief Optimization profile for dynamic input dimensions and shape tensors.
//!
//! When building an ICudaEngine from an INetworkDefinition that has dynamically resizable inputs (at least
//! one input tensor has one or more of its dimensions specified as -1) or shape input tensors, users need to specify
//! at least one optimization profile. Optimization profiles are numbered 0, 1, ...
//! The first optimization profile that has been defined (with index 0) will be used by the ICudaEngine whenever no
//! optimization profile has been selected explicitly. If none of the inputs are dynamic, the default optimization
//! profile will be generated automatically unless it is explicitly provided by the user (this is possible but not
//! required in this case). If more than a single optimization profile is defined, users may set a target how
//! much additional weight space should be maximally allocated to each additional profile (as a fraction of the
//! maximum, unconstrained memory).
//!
//! Users set optimum input tensor dimensions, as well as minimum and maximum input tensor dimensions. The builder
//! selects the kernels that result in the lowest runtime for the optimum input tensor dimensions, and are valid for
//! all input tensor sizes in the valid range between minimum and maximum dimensions. A runtime error will be raised
//! if the input tensor dimensions fall outside the valid range for this profile. Likewise, users provide minimum,
//! optimum, and maximum values for all shape tensor input values.
//!
//! \see IBuilderConfig::addOptimizationProfile()
//!
class IOptimizationProfile : public INoCopy
{
public:
    //!
    //! \brief Set the minimum / optimum / maximum dimensions for a dynamic input tensor.
    //!
    //! This function must be called three times (for the minimum, optimum, and maximum) for any network input tensor
    //! that has dynamic dimensions. If minDims, optDims, and maxDims are the minimum, optimum, and maximum dimensions,
    //! and networkDims are the dimensions for this input tensor that are provided to the INetworkDefinition object,
    //! then the following conditions must all hold:
    //!
    //! (1) minDims.nbDims == optDims.nbDims == maxDims.nbDims == networkDims.nbDims
    //! (2) 0 <= minDims.d[i] <= optDims.d[i] <= maxDims.d[i] for i = 0, ..., networkDims.nbDims-1
    //! (3) if networkDims.d[i] != -1, then minDims.d[i] == optDims.d[i] == maxDims.d[i] == networkDims.d[i]
    //!
    //! This function may (but need not be) called for an input tensor that does not have dynamic dimensions. In this
    //! case, the third argument must always equal networkDims.
    //!
    //! \param inputName The input tensor name
    //! \param select Whether to set the minimum, optimum, or maximum dimensions
    //! \param dims The minimum, optimum, or maximum dimensions for this input tensor
    //!
    //! \return false if an inconsistency was detected (e.g. the rank does not match another dimension that was
    //!         previously set for the same input), true if no inconsistency was detected. Note that inputs can be
    //!         validated only partially; a full validation is performed at engine build time.
    //!
    //! \warning If run on DLA, minimum, optimum, and maximum dimensions must to be the same.
    //!
    bool setDimensions(const char* inputName, OptProfileSelector select, Dims dims) noexcept
    {
        return mImpl->setDimensions(inputName, select, dims);
    }

    //!
    //! \brief Get the minimum / optimum / maximum dimensions for a dynamic input tensor.
    //!
    //! If the dimensions have not been previously set via setDimensions(), return an invalid Dims with nbDims == -1.
    //!
    Dims getDimensions(const char* inputName, OptProfileSelector select) const noexcept
    {
        return mImpl->getDimensions(inputName, select);
    }

    //!
    //! \brief Set the minimum / optimum / maximum values for an input shape tensor.
    //!
    //! This function must be called three times for every input tensor t that is a shape tensor (t.isShape() == true).
    //! This implies that the datatype of t is DataType::kINT32, the rank is either 0 or 1, and the dimensions of t
    //! are fixed at network definition time. This function must not be called for any input tensor that is not a
    //! shape tensor.
    //!
    //! Each time this function is called for the same input tensor, the same nbValues must be supplied (either 1
    //! if the tensor rank is 0, or dims.d[0] if the rank is 1). Furthermore, if minVals, optVals, maxVals are the
    //! minimum, optimum, and maximum values, it must be true that minVals[i] <= optVals[i] <= maxVals[i] for
    //! i = 0, ..., nbValues - 1. Execution of the network must be valid for the optVals.
    //!
    //! Shape tensors are tensors that contribute to shape calculations in some way, and can contain
    //! any int32_t values appropriate for the network. Examples:
    //!
    //! * A shape tensor used as the second input to IShuffleLayer can contain a -1 wildcard.
    //!   The corresponding minVal[i] should be -1.
    //!
    //! * A shape tensor used as the stride input to ISliceLayer can contain any valid strides.
    //!   The values could be positive, negative, or zero.
    //!
    //! * A shape tensor subtracted from zero to compute the size input of an ISliceLayer can
    //!   contain any non-positive values that yield a valid slice operation.
    //!
    //! Tightening the minVals and maxVals bounds to cover only values that are necessary may help optimization.
    //!
    //! \param inputName The input tensor name
    //! \param select Whether to set the minimum, optimum, or maximum input values.
    //! \param values An array of length nbValues containing the minimum, optimum, or maximum shape tensor elements.
    //! \param nbValues The length of the value array, which must equal the number of shape tensor elements (>= 1)
    //!
    //! \return false if an inconsistency was detected (e.g. nbValues does not match a previous call for the same
    //!         tensor), else true. As for setDimensions(), a full validation can only be performed at engine build
    //!         time.
    //!
    //! \warning If run on DLA, minimum, optimum, and maximum shape values must to be the same.
    //!
    bool setShapeValues(
        const char* inputName, OptProfileSelector select, const int32_t* values, int32_t nbValues) noexcept
    {
        return mImpl->setShapeValues(inputName, select, values, nbValues);
    }

    //!
    //! \brief Get the number of values for an input shape tensor.
    //!
    //! This will return the number of shape values if setShapeValues() has been called before for this input tensor.
    //! Otherwise, return -1.
    //!
    int32_t getNbShapeValues(const char* inputName) const noexcept
    {
        return mImpl->getNbShapeValues(inputName);
    }

    //!
    //! \brief Get the minimum / optimum / maximum values for an input shape tensor.
    //!
    //! If the shape values have not been set previously with setShapeValues(), this returns nullptr.
    //!
    int32_t const* getShapeValues(const char* inputName, OptProfileSelector select) const noexcept
    {
        return mImpl->getShapeValues(inputName, select);
    }

    //!
    //! \brief Set a target for extra GPU memory that may be used by this profile.
    //!
    //! \param target Additional memory that the builder should aim to maximally allocate for this profile, as a
    //!        fraction of the memory it would use if the user did not impose any constraints on memory. This
    //!        unconstrained case is the default; it corresponds to target == 1.0. If target == 0.0, the builder
    //!        aims to create the new optimization profile without allocating any additional weight memory.
    //!        Valid inputs lie between 0.0 and 1.0. This parameter is only a hint, and TensorRT does not guarantee
    //!        that the target will be reached. This parameter is ignored for the first (default) optimization profile
    //!        that is defined.
    //!
    //! \return true if the input is in the valid range (between 0 and 1 inclusive), else false
    //!
    bool setExtraMemoryTarget(float target) noexcept
    {
        return mImpl->setExtraMemoryTarget(target);
    }

    //!
    //! \brief Get the extra memory target that has been defined for this profile.
    //!
    float getExtraMemoryTarget() const noexcept
    {
        return mImpl->getExtraMemoryTarget();
    }

    //!
    //! \brief Check whether the optimization profile can be passed to an IBuilderConfig object.
    //!
    //! This function performs partial validation, by e.g. checking that whenever one of the minimum, optimum, or
    //! maximum dimensions of a tensor have been set, the others have also been set and have the same rank, as
    //! well as checking that the optimum dimensions are always as least as large as the minimum dimensions, and
    //! that the maximum dimensions are at least as large as the optimum dimensions. Some validation steps require
    //! knowledge of the network definition and are deferred to engine build time.
    //!
    //! \return true if the optimization profile is valid and may be passed to an IBuilderConfig, else false
    //!
    bool isValid() const noexcept
    {
        return mImpl->isValid();
    }

protected:
    apiv::VOptimizationProfile* mImpl;
    virtual ~IOptimizationProfile() noexcept = default;
};

//!
//! \enum TacticSource
//!
//! \brief List of tactic sources for TensorRT.
//!
//! \see TacticSources, IBuilderConfig::setTacticSources(), IBuilderConfig::getTacticSources()
//!
enum class TacticSource : int32_t
{
    //! \note Disabling kCUBLAS will cause the cublas handle passed to plugins in attachToContext to be null.
    kCUBLAS = 0,    //!< cuBLAS tactics.
    kCUBLAS_LT = 1, //!< cuBLAS LT tactics
    kCUDNN = 2      //!< cuDNN tactics
};

template <>
constexpr inline int32_t EnumMax<TacticSource>() noexcept
{
    return 3;
} //!< Maximum number of tactic sources in TacticSource enum. \see TacticSource

//!
//! \brief Represents a collection of one or more TacticSource values
//! combine using bitwise-OR operations.
//!
//! \see IBuilderConfig::setTacticSources(), IBuilderConfig::getTacticSources()
//!
using TacticSources = uint32_t;

//!
//! \enum ProfilingVerbosity
//!
//! \brief List of verbosity levels of layer information exposed in NVTX annotations and in IEngineInspector.
//!
//! \see IBuilderConfig::setProfilingVerbosity(),
//!      IBuilderConfig::getProfilingVerbosity(),
//!      IEngineInspector
//!
enum class ProfilingVerbosity : int32_t
{
    kLAYER_NAMES_ONLY = 0, //!< Print only the layer names. This is the default setting.
    kNONE = 1,             //!< Do not print any layer information.
    kDETAILED = 2,         //!< Print detailed layer information including layer names and layer parameters.
    kDEFAULT TRT_DEPRECATED_ENUM = kLAYER_NAMES_ONLY,
    kVERBOSE TRT_DEPRECATED_ENUM = kDETAILED
};

//! Maximum number of profile verbosity levels in ProfilingVerbosity enum. \see ProfilingVerbosity
template <>
constexpr inline int32_t EnumMax<ProfilingVerbosity>() noexcept
{
    return 3;
}

//!
//! \class ICudaEngine
//!
//! \brief An engine for executing inference on a built network, with functionally unsafe features.
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
class ICudaEngine : public INoCopy
{
public:
    virtual ~ICudaEngine() noexcept = default;

    //!
    //! \brief Get the number of binding indices.
    //!
    //! There are separate binding indices for each optimization profile.
    //! This method returns the total over all profiles.
    //! If the engine has been built for K profiles, the first getNbBindings() / K bindings are used by profile
    //! number 0, the following getNbBindings() / K bindings are used by profile number 1 etc.
    //!
    //! \see getBindingIndex();
    //!
    int32_t getNbBindings() const noexcept
    {
        return mImpl->getNbBindings();
    }

    //!
    //! \brief Retrieve the binding index for a named tensor.
    //!
    //! IExecutionContext::enqueue() and IExecutionContext::execute() require an array of buffers.
    //!
    //! Engine bindings map from tensor names to indices in this array.
    //! Binding indices are assigned at engine build time, and take values in the range [0 ... n-1] where n is the total
    //! number of inputs and outputs.
    //!
    //! To get the binding index of the name in an optimization profile with index k > 0,
    //! mangle the name by appending " [profile k]", as described for method getBindingName().
    //!
    //! \param name The tensor name.
    //! \return The binding index for the named tensor, or -1 if the name is not found.
    //!
    //! \see getNbBindings() getBindingName()
    //!
    int32_t getBindingIndex(const char* name) const noexcept
    {
        return mImpl->getBindingIndex(name);
    }

    //!
    //! \brief Retrieve the name corresponding to a binding index.
    //!
    //! This is the reverse mapping to that provided by getBindingIndex().
    //!
    //! For optimization profiles with an index k > 0, the name is mangled by appending
    //! " [profile k]", with k written in decimal.  For example, if the tensor in the
    //! INetworkDefinition had the name "foo", and bindingIndex refers to that tensor in the
    //! optimization profile with index 3, getBindingName returns "foo [profile 3]".
    //!
    //! \param bindingIndex The binding index.
    //! \return The name corresponding to the index, or nullptr if the index is out of range.
    //!
    //! \see getBindingIndex()
    //!
    const char* getBindingName(int32_t bindingIndex) const noexcept
    {
        return mImpl->getBindingName(bindingIndex);
    }

    //!
    //! \brief Determine whether a binding is an input binding.
    //!
    //! \param bindingIndex The binding index.
    //! \return True if the index corresponds to an input binding and the index is in range.
    //!
    //! \see getBindingIndex()
    //!
    bool bindingIsInput(int32_t bindingIndex) const noexcept
    {
        return mImpl->bindingIsInput(bindingIndex);
    }

    //!
    //! \brief Get the dimensions of a binding.
    //!
    //! \param bindingIndex The binding index.
    //! \return The dimensions of the binding if the index is in range, otherwise Dims().
    //!         Has -1 for any dimension that varies within the optimization profile.
    //!
    //! For example, suppose an INetworkDefinition has an input with shape [-1,-1]
    //! that becomes a binding b in the engine.  If the associated optimization profile
    //! specifies that b has minimum dimensions as [6,9] and maximum dimensions [7,9],
    //! getBindingDimensions(b) returns [-1,9], despite the second dimension being
    //! dynamic in the INetworkDefinition.
    //!
    //! Because each optimization profile has separate bindings, the returned value can
    //! differ across profiles. Consider another binding b' for the same network input,
    //! but for another optimization profile.  If that other profile specifies minimum
    //! dimensions [5,8] and maximum dimensions [5,9], getBindingDimensions(b') returns [5,-1].
    //!
    //! \see getBindingIndex()
    //!
    Dims getBindingDimensions(int32_t bindingIndex) const noexcept
    {
        return mImpl->getBindingDimensions(bindingIndex);
    }

    //!
    //! \brief Determine the required data type for a buffer from its binding index.
    //!
    //! \param bindingIndex The binding index.
    //! \return The type of the data in the buffer.
    //!
    //! \see getBindingIndex()
    //!
    DataType getBindingDataType(int32_t bindingIndex) const noexcept
    {
        return mImpl->getBindingDataType(bindingIndex);
    }

    //!
    //! \brief Get the maximum batch size which can be used for inference.
    //!
    //! For an engine built from an INetworkDefinition without an implicit batch dimension, this will always return 1.
    //!
    //! \return The maximum batch size for this engine.
    //!
    int32_t getMaxBatchSize() const noexcept
    {
        return mImpl->getMaxBatchSize();
    }

    //!
    //! \brief Get the number of layers in the network.
    //!
    //! The number of layers in the network is not necessarily the number in the original network definition, as layers
    //! may be combined or eliminated as the engine is optimized. This value can be useful when building per-layer
    //! tables, such as when aggregating profiling data over a number of executions.
    //!
    //! \return The number of layers in the network.
    //!
    int32_t getNbLayers() const noexcept
    {
        return mImpl->getNbLayers();
    }

    //!
    //! \brief Serialize the network to a stream.
    //!
    //! \return A IHostMemory object that contains the serialized engine.
    //!
    //! The network may be deserialized with IRuntime::deserializeCudaEngine().
    //!
    //! \see IRuntime::deserializeCudaEngine()
    //!
    IHostMemory* serialize() const noexcept
    {
        return mImpl->serialize();
    }

    //!
    //! \brief Create an execution context.
    //!
    //! If the engine supports dynamic shapes, each execution context in concurrent use must use a separate optimization
    //! profile. The first execution context created will call setOptimizationProfile(0) implicitly. For other execution
    //! contexts, setOptimizationProfile() must be called with unique profile index before calling execute or enqueue.
    //! If an error recorder has been set for the engine, it will also be passed to the execution context.
    //!
    //! \see IExecutionContext.
    //! \see IExecutionContext::setOptimizationProfile()
    //!
    IExecutionContext* createExecutionContext() noexcept
    {
        return mImpl->createExecutionContext();
    }

    //!
    //! \brief Destroy this object;
    //!
    //! \deprecated Deprecated interface will be removed in TensorRT 10.0.
    //!
    //! \warning Calling destroy on a managed pointer will result in a double-free error.
    //!
    TRT_DEPRECATED void destroy() noexcept
    {
        delete this;
    }

    //!
    //! \brief Get location of binding
    //!
    //! This lets you know whether the binding should be a pointer to device or host memory.
    //!
    //! \see ITensor::setLocation() ITensor::getLocation()
    //!
    //! \param bindingIndex The binding index.
    //! \return The location of the bound tensor with given index.
    //!
    TensorLocation getLocation(int32_t bindingIndex) const noexcept
    {
        return mImpl->getLocation(bindingIndex);
    }

    //! \brief create an execution context without any device memory allocated
    //!
    //! The memory for execution of this device context must be supplied by the application.
    //!
    IExecutionContext* createExecutionContextWithoutDeviceMemory() noexcept
    {
        return mImpl->createExecutionContextWithoutDeviceMemory();
    }

    //!
    //! \brief Return the amount of device memory required by an execution context.
    //!
    //! \see IExecutionContext::setDeviceMemory()
    //!
    size_t getDeviceMemorySize() const noexcept
    {
        return mImpl->getDeviceMemorySize();
    }

    //!
    //! \brief Return true if an engine can be refit.
    //!
    //! \see nvinfer1::createInferRefitter()
    //!
    bool isRefittable() const noexcept
    {
        return mImpl->isRefittable();
    }

    //!
    //! \brief Return the number of bytes per component of an element.
    //!
    //! The vector component size is returned if getBindingVectorizedDim() != -1.
    //!
    //! \param bindingIndex The binding Index.
    //!
    //! \see ICudaEngine::getBindingVectorizedDim()
    //!
    int32_t getBindingBytesPerComponent(int32_t bindingIndex) const noexcept
    {
        return mImpl->getBindingBytesPerComponent(bindingIndex);
    }

    //!
    //! \brief Return the number of components included in one element.
    //!
    //! The number of elements in the vectors is returned if getBindingVectorizedDim() != -1.
    //!
    //! \param bindingIndex The binding Index.
    //!
    //! \see ICudaEngine::getBindingVectorizedDim()
    //!
    int32_t getBindingComponentsPerElement(int32_t bindingIndex) const noexcept
    {
        return mImpl->getBindingComponentsPerElement(bindingIndex);
    }

    //!
    //! \brief Return the binding format.
    //!
    //! \param bindingIndex The binding Index.
    //!
    TensorFormat getBindingFormat(int32_t bindingIndex) const noexcept
    {
        return mImpl->getBindingFormat(bindingIndex);
    }

    //!
    //! \brief Return the human readable description of the tensor format.
    //!
    //! The description includes the order, vectorization, data type, strides,
    //! and etc. Examples are shown as follows:
    //!   Example 1: kCHW + FP32
    //!     "Row major linear FP32 format"
    //!   Example 2: kCHW2 + FP16
    //!     "Two wide channel vectorized row major FP16 format"
    //!   Example 3: kHWC8 + FP16 + Line Stride = 32
    //!     "Channel major FP16 format where C % 8 == 0 and H Stride % 32 == 0"
    //!
    //! \param bindingIndex The binding Index.
    //!
    const char* getBindingFormatDesc(int32_t bindingIndex) const noexcept
    {
        return mImpl->getBindingFormatDesc(bindingIndex);
    }

    //!
    //! \brief Return the dimension index that the buffer is vectorized.
    //!
    //! Specifically -1 is returned if scalars per vector is 1.
    //!
    //! \param bindingIndex The binding Index.
    //!
    int32_t getBindingVectorizedDim(int32_t bindingIndex) const noexcept
    {
        return mImpl->getBindingVectorizedDim(bindingIndex);
    }

    //!
    //! \brief Returns the name of the network associated with the engine.
    //!
    //! The name is set during network creation and is retrieved after
    //! building or deserialization.
    //!
    //! \see INetworkDefinition::setName(), INetworkDefinition::getName()
    //!
    //! \return A null-terminated C-style string representing the name of the network.
    //!
    const char* getName() const noexcept
    {
        return mImpl->getName();
    }

    //!
    //! \brief Get the number of optimization profiles defined for this engine.
    //!
    //! \return Number of optimization profiles. It is always at least 1.
    //!
    //! \see IExecutionContext::setOptimizationProfile()
    int32_t getNbOptimizationProfiles() const noexcept
    {
        return mImpl->getNbOptimizationProfiles();
    }

    //!
    //! \brief Get the minimum / optimum / maximum dimensions for a particular binding under an optimization profile.
    //!
    //! \param bindingIndex The binding index, which must belong to the given profile,
    //!        or be between 0 and bindingsPerProfile-1 as described below.
    //!
    //! \param profileIndex The profile index, which must be between 0 and getNbOptimizationProfiles()-1.
    //!
    //! \param select Whether to query the minimum, optimum, or maximum dimensions for this binding.
    //!
    //! \return The minimum / optimum / maximum dimensions for this binding in this profile.
    //!         If the profileIndex or bindingIndex are invalid, return Dims with nbDims=-1.
    //!
    //! For backwards compatibility with earlier versions of TensorRT, if the bindingIndex
    //! does not belong to the current optimization profile, but is between 0 and bindingsPerProfile-1,
    //! where bindingsPerProfile = getNbBindings()/getNbOptimizationProfiles,
    //! then a corrected bindingIndex is used instead, computed by:
    //!
    //!     profileIndex * bindingsPerProfile + bindingIndex % bindingsPerProfile
    //!
    //! Otherwise the bindingIndex is considered invalid.
    //!
    Dims getProfileDimensions(int32_t bindingIndex, int32_t profileIndex, OptProfileSelector select) const noexcept
    {
        return mImpl->getProfileDimensions(bindingIndex, profileIndex, select);
    }

    //!
    //! \brief Get minimum / optimum / maximum values for an input shape binding under an optimization profile.
    //!
    //! \param profileIndex The profile index (must be between 0 and getNbOptimizationProfiles()-1)
    //!
    //! \param inputIndex The input index (must be between 0 and getNbBindings() - 1)
    //!
    //! \param select Whether to query the minimum, optimum, or maximum shape values for this binding.
    //!
    //! \return If the binding is an input shape binding, return a pointer to an array that has
    //!         the same number of elements as the corresponding tensor, i.e. 1 if dims.nbDims == 0, or dims.d[0]
    //!         if dims.nbDims == 1, where dims = getBindingDimensions(inputIndex). The array contains
    //!         the elementwise minimum / optimum / maximum values for this shape binding under the profile.
    //!         If either of the indices is out of range, or if the binding is not an input shape binding, return
    //!         nullptr.
    //!
    //! For backwards compatibility with earlier versions of TensorRT, a bindingIndex that does not belong
    //! to the profile is corrected as described for getProfileDimensions.
    //!
    //! \see ICudaEngine::getProfileDimensions
    //!
    const int32_t* getProfileShapeValues(int32_t profileIndex, int32_t inputIndex, OptProfileSelector select) const
        noexcept
    {
        return mImpl->getProfileShapeValues(profileIndex, inputIndex, select);
    }

    //!
    //! \brief True if tensor is required as input for shape calculations or output from them.
    //!
    //! TensorRT evaluates a network in two phases:
    //!
    //! 1. Compute shape information required to determine memory allocation requirements
    //!    and validate that runtime sizes make sense.
    //!
    //! 2. Process tensors on the device.
    //!
    //! Some tensors are required in phase 1.  These tensors are called "shape tensors", and always
    //! have type Int32 and no more than one dimension.  These tensors are not always shapes
    //! themselves, but might be used to calculate tensor shapes for phase 2.
    //!
    //! isShapeBinding(i) returns true if the tensor is a required input or an output computed in phase 1.
    //! isExecutionBinding(i) returns true if the tensor is a required input or an output computed in phase 2.
    //!
    //! For example, if a network uses an input tensor with binding i as an addend
    //! to an IElementWiseLayer that computes the "reshape dimensions" for IShuffleLayer,
    //! then isShapeBinding(i) == true.
    //!
    //! It's possible to have a tensor be required by both phases.  For instance, a tensor
    //! can be used for the "reshape dimensions" and as the indices for an IGatherLayer
    //! collecting floating-point data.
    //!
    //! It's also possible to have a tensor be required by neither phase, but nonetheless
    //! shows up in the engine's inputs.  For example, if an input tensor is used only
    //! as an input to IShapeLayer, only its shape matters and its values are irrelevant.
    //!
    //! \see isExecutionBinding()
    //!
    bool isShapeBinding(int32_t bindingIndex) const noexcept
    {
        return mImpl->isShapeBinding(bindingIndex);
    }

    //!
    //! \brief True if pointer to tensor data is required for execution phase, false if nullptr can be supplied.
    //!
    //! For example, if a network uses an input tensor with binding i ONLY as the "reshape dimensions"
    //! input of IShuffleLayer, then isExecutionBinding(i) is false, and a nullptr can be
    //! supplied for it when calling IExecutionContext::execute or IExecutionContext::enqueue.
    //!
    //! \see isShapeBinding()
    //!
    bool isExecutionBinding(int32_t bindingIndex) const noexcept
    {
        return mImpl->isExecutionBinding(bindingIndex);
    }

    //!
    //! \brief Determine what execution capability this engine has.
    //!
    //! If the engine has EngineCapability::kSTANDARD, then all engine functionality is valid.
    //! If the engine has EngineCapability::kSAFETY, then only the functionality in safe engine is valid.
    //! If the engine has EngineCapability::kDLA_STANDALONE, then only serialize, destroy, and const-accessor functions are
    //! valid.
    //!
    //! \return The EngineCapability flag that the engine was built for.
    //!
    EngineCapability getEngineCapability() const noexcept
    {
        return mImpl->getEngineCapability();
    }

    //! \brief Set the ErrorRecorder for this interface
    //!
    //! Assigns the ErrorRecorder to this interface. The ErrorRecorder will track all errors during execution.
    //! This function will call incRefCount of the registered ErrorRecorder at least once. Setting
    //! recorder to nullptr unregisters the recorder with the interface, resulting in a call to decRefCount if
    //! a recorder has been registered.
    //!
    //! If an error recorder is not set, messages will be sent to the global log stream.
    //!
    //! \param recorder The error recorder to register with this interface.
    //
    //! \see getErrorRecorder()
    //!
    void setErrorRecorder(IErrorRecorder* recorder) noexcept
    {
        return mImpl->setErrorRecorder(recorder);
    }

    //!
    //! \brief Get the ErrorRecorder assigned to this interface.
    //!
    //! Retrieves the assigned error recorder object for the given class. A nullptr will be returned if
    //! an error handler has not been set.
    //!
    //! \return A pointer to the IErrorRecorder object that has been registered.
    //!
    //! \see setErrorRecorder()
    //!
    IErrorRecorder* getErrorRecorder() const noexcept
    {
        return mImpl->getErrorRecorder();
    }

    //!
    //! \brief Query whether the engine was built with an implicit batch dimension.
    //!
    //! \return True if tensors have implicit batch dimension, false otherwise.
    //!
    //! This is an engine-wide property.  Either all tensors in the engine
    //! have an implicit batch dimension or none of them do.
    //!
    //! hasImplicitBatchDimension() is true if and only if the INetworkDefinition
    //! from which this engine was built was created with createNetwork() or
    //! createNetworkV2() without NetworkDefinitionCreationFlag::kEXPLICIT_BATCH flag.
    //!
    //! \see createNetworkV2
    //!
    bool hasImplicitBatchDimension() const noexcept
    {
        return mImpl->hasImplicitBatchDimension();
    }

    //! \brief return the tactic sources required by this engine
    //!
    //! \see IBuilderConfig::setTacticSources()
    //!
    TacticSources getTacticSources() const noexcept
    {
        return mImpl->getTacticSources();
    }

    //! \brief Return the \ref ProfilingVerbosity the builder config was set to when the engine was built.
    //!
    //! \return the profiling verbosity the builder config was set to when the engine was built.
    //!
    //! \see IBuilderConfig::setProfilingVerbosity()
    //!
    ProfilingVerbosity getProfilingVerbosity() const noexcept
    {
        return mImpl->getProfilingVerbosity();
    }

    //!
    //! \brief Create a new engine inspector which prints the layer information in an engine or an execution context.
    //!
    //! \see IEngineInspector.
    //!
    IEngineInspector* createEngineInspector() const noexcept
    {
        return mImpl->createEngineInspector();
    }

protected:
    apiv::VCudaEngine* mImpl;
};

//!
//! \class IExecutionContext
//!
//! \brief Context for executing inference using an engine, with functionally unsafe features.
//!
//! Multiple execution contexts may exist for one ICudaEngine instance, allowing the same
//! engine to be used for the execution of multiple batches simultaneously. If the engine supports
//! dynamic shapes, each execution context in concurrent use must use a separate optimization profile.
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
class IExecutionContext : public INoCopy
{
public:
    virtual ~IExecutionContext() noexcept = default;

    //!
    //! \brief Synchronously execute inference on a batch.
    //!
    //! This method requires an array of input and output buffers. The mapping from tensor names to indices
    //! can be queried using ICudaEngine::getBindingIndex()
    //!
    //! \param batchSize The batch size. This is at most the value supplied when the engine was built.
    //! \param bindings An array of pointers to input and output buffers for the network.
    //!
    //! \return True if execution succeeded.
    //!
    //! \warning This function will trigger layer resource updates if hasImplicitBatchDimension()
    //!          returns true and batchSize changes between subsequent calls, possibly resulting
    //!          in performance bottlenecks.
    //!
    //! \see ICudaEngine::getBindingIndex() ICudaEngine::getMaxBatchSize()
    //!
    bool execute(int32_t batchSize, void* const* bindings) noexcept
    {
        return mImpl->execute(batchSize, bindings);
    }

    //!
    //! \brief Asynchronously execute inference on a batch.
    //!
    //! This method requires an array of input and output buffers. The mapping from tensor names to indices can be
    //! queried using ICudaEngine::getBindingIndex() \param batchSize The batch size. This is at most the value supplied
    //! when the engine was built.
    //!
    //! \param bindings An array of pointers to input and output buffers for the network.
    //! \param stream A cuda stream on which the inference kernels will be enqueued.
    //! \param inputConsumed An optional event which will be signaled when the input buffers can be refilled with new
    //! data.
    //!
    //! \return True if the kernels were enqueued successfully.
    //!
    //! \see ICudaEngine::getBindingIndex() ICudaEngine::getMaxBatchSize()
    //!
    //! \warning Calling enqueue() in from the same IExecutionContext object with different CUDA streams concurrently
    //!          results in undefined behavior. To perform inference concurrently in multiple streams, use one execution
    //!          context per stream.
    //!
    //! \warning This function will trigger layer resource updates if hasImplicitBatchDimension()
    //!          returns true and batchSize changes between subsequent calls, possibly resulting in performance
    //!          bottlenecks.
    //!
    bool enqueue(int32_t batchSize, void* const* bindings, cudaStream_t stream, cudaEvent_t* inputConsumed) noexcept
    {
        return mImpl->enqueue(batchSize, bindings, stream, inputConsumed);
    }

    //!
    //! \brief Set the debug sync flag.
    //!
    //! If this flag is set to true, the engine will log the successful execution for each kernel during execute(). It
    //! has no effect when using enqueue().
    //!
    //! \see getDebugSync()
    //!
    void setDebugSync(bool sync) noexcept
    {
        mImpl->setDebugSync(sync);
    }

    //!
    //! \brief Get the debug sync flag.
    //!
    //! \see setDebugSync()
    //!
    bool getDebugSync() const noexcept
    {
        return mImpl->getDebugSync();
    }

    //!
    //! \brief Set the profiler.
    //!
    //! \see IProfiler getProfiler()
    //!
    void setProfiler(IProfiler* profiler) noexcept
    {
        mImpl->setProfiler(profiler);
    }

    //!
    //! \brief Get the profiler.
    //!
    //! \see IProfiler setProfiler()
    //!
    IProfiler* getProfiler() const noexcept
    {
        return mImpl->getProfiler();
    }

    //!
    //! \brief Get the associated engine.
    //!
    //! \see ICudaEngine
    //!
    const ICudaEngine& getEngine() const noexcept
    {
        return mImpl->getEngine();
    }

    //!
    //! \brief Destroy this object.
    //!
    //! \deprecated Deprecated interface will be removed in TensorRT 10.0.
    //!
    //! \warning Calling destroy on a managed pointer will result in a double-free error.
    //!
    TRT_DEPRECATED void destroy() noexcept
    {
        delete this;
    }

    //!
    //! \brief Set the name of the execution context.
    //!
    //! This method copies the name string.
    //!
    //! \see getName()
    //!
    void setName(const char* name) noexcept
    {
        mImpl->setName(name);
    }

    //!
    //! \brief Return the name of the execution context.
    //!
    //! \see setName()
    //!
    const char* getName() const noexcept
    {
        return mImpl->getName();
    }

    //!
    //! \brief Set the device memory for use by this execution context.
    //!
    //! The memory must be aligned with cuda memory alignment property (using cudaGetDeviceProperties()), and its size
    //! must be at least that returned by getDeviceMemorySize(). Setting memory to nullptr is acceptable if
    //! getDeviceMemorySize() returns 0. If using enqueue() to run the network, the memory is in use from the invocation
    //! of enqueue() until network execution is complete. If using execute(), it is in use until execute() returns.
    //! Releasing or otherwise using the memory for other purposes during this time will result in undefined behavior.
    //!
    //! \see ICudaEngine::getDeviceMemorySize() ICudaEngine::createExecutionContextWithoutDeviceMemory()
    //!
    void setDeviceMemory(void* memory) noexcept
    {
        mImpl->setDeviceMemory(memory);
    }

    //!
    //! \brief Return the strides of the buffer for the given binding.
    //!
    //! The strides are in units of elements, not components or bytes.
    //! For example, for TensorFormat::kHWC8, a stride of one spans 8 scalars.
    //!
    //! Note that strides can be different for different execution contexts
    //! with dynamic shapes.
    //!
    //! If the bindingIndex is invalid or there are dynamic dimensions that have not been
    //! set yet, returns Dims with Dims::nbDims = -1.
    //!
    //! \param bindingIndex The binding index.
    //!
    //! \see getBindingComponentsPerElement
    //!
    Dims getStrides(int32_t bindingIndex) const noexcept
    {
        return mImpl->getStrides(bindingIndex);
    }

public:
    //!
    //! \brief Select an optimization profile for the current context.
    //!
    //! \param profileIndex Index of the profile. It must lie between 0 and
    //!        getEngine().getNbOptimizationProfiles() - 1
    //!
    //! The selected profile will be used in subsequent calls to execute() or enqueue().
    //!
    //! When an optimization profile is switched via this API, TensorRT may
    //! enqueue GPU memory copy operations required to set up the new profile during the subsequent enqueue()
    //! operations. To avoid these calls during enqueue(), use setOptimizationProfileAsync() instead.
    //!
    //! If the associated CUDA engine has dynamic inputs, this method must be called at least once
    //! with a unique profileIndex before calling execute or enqueue (i.e. the profile index
    //! may not be in use by another execution context that has not been destroyed yet).
    //! For the first execution context that is created for an engine, setOptimizationProfile(0)
    //! is called implicitly.
    //!
    //! If the associated CUDA engine does not have inputs with dynamic shapes, this method need not be
    //! called, in which case the default profile index of 0 will be used (this is particularly
    //! the case for all safe engines).
    //!
    //! setOptimizationProfile() must be called before calling setBindingDimensions() and
    //! setInputShapeBinding() for all dynamic input tensors or input shape tensors, which in
    //! turn must be called before either execute() or enqueue().
    //!
    //! \warning This function will trigger layer resource updates on the next
    //!          call of enqueue[V2]()/execute[V2](), possibly resulting in performance bottlenecks.
    //!
    //! \return true if the call succeeded, else false (e.g. input out of range)
    //!
    //! \deprecated This API is superseded by setOptimizationProfileAsync and will be removed in TensorRT 9.0.
    //!
    //! \see ICudaEngine::getNbOptimizationProfiles() IExecutionContext::setOptimizationProfileAsync()
    //!
    TRT_DEPRECATED
    bool setOptimizationProfile(int32_t profileIndex) noexcept
    {
        return mImpl->setOptimizationProfile(profileIndex);
    }

    //!
    //! \brief Get the index of the currently selected optimization profile.
    //!
    //! If the profile index has not been set yet (implicitly to 0 for the first execution context
    //! to be created, or explicitly for all subsequent contexts), an invalid value of -1 will be returned
    //! and all calls to enqueue() or execute() will fail until a valid profile index has been set.
    //!
    int32_t getOptimizationProfile() const noexcept
    {
        return mImpl->getOptimizationProfile();
    }

    //!
    //! \brief Set the dynamic dimensions of a binding
    //!
    //! \param bindingIndex index of an input tensor whose dimensions must be compatible with
    //!        the network definition (i.e. only the wildcard dimension -1 can be replaced with a
    //!        new dimension >= 0).
    //!
    //! \param dimensions specifies the dimensions of the input tensor. It must be in the valid
    //!        range for the currently selected optimization profile, and the corresponding engine must
    //!        not be safety-certified.
    //!
    //! This method requires the engine to be built without an implicit batch dimension.
    //! This method will fail unless a valid optimization profile is defined for the current
    //! execution context (getOptimizationProfile() must not be -1).
    //!
    //! For all dynamic non-output bindings (which have at least one wildcard dimension of -1),
    //! this method needs to be called before either enqueue() or execute() may be called.
    //! This can be checked using the method allInputDimensionsSpecified().
    //!
    //! \warning This function will trigger layer resource updates on the next
    //!          call of enqueue[V2]()/execute[V2](), possibly resulting in performance bottlenecks,
    //!          if the dimensions are different than the previous set dimensions.
    //!
    //! \return false if an error occurs (e.g. bindingIndex is out of range for the currently selected
    //!         optimization profile or binding dimension is inconsistent with min-max range of the
    //!         optimization profile), else true. Note that the network can still be invalid for certain
    //!         combinations of input shapes that lead to invalid output shapes. To confirm the correctness
    //!         of the network input shapes, check whether the output binding has valid
    //!         dimensions using getBindingDimensions() on the output bindingIndex.
    //!
    //! \see ICudaEngine::getBindingIndex
    //!
    bool setBindingDimensions(int32_t bindingIndex, Dims dimensions) noexcept
    {
        return mImpl->setBindingDimensions(bindingIndex, dimensions);
    }

    //!
    //! \brief Get the dynamic dimensions of a binding
    //!
    //! If the engine was built with an implicit batch dimension, same as ICudaEngine::getBindingDimensions.
    //!
    //! If setBindingDimensions() has been called on this binding (or if there are no
    //! dynamic dimensions), all dimensions will be positive. Otherwise, it is necessary to
    //! call setBindingDimensions() before enqueue() or execute() may be called.
    //!
    //! If the bindingIndex is out of range, an invalid Dims with nbDims == -1 is returned.
    //! The same invalid Dims will be returned if the engine was not built with an implicit
    //! batch dimension and if the execution context is not currently associated with a valid
    //! optimization profile (i.e. if getOptimizationProfile() returns -1).
    //!
    //! If ICudaEngine::bindingIsInput(bindingIndex) is false, then both
    //! allInputDimensionsSpecified() and allInputShapesSpecified() must be true
    //! before calling this method.
    //!
    //! \return Currently selected binding dimensions
    //!
    //! For backwards compatibility with earlier versions of TensorRT, a bindingIndex that does not belong
    //! to the current profile is corrected as described for ICudaEngine::getProfileDimensions.
    //!
    //! \see ICudaEngine::getProfileDimensions
    //!
    Dims getBindingDimensions(int32_t bindingIndex) const noexcept
    {
        return mImpl->getBindingDimensions(bindingIndex);
    }

    //!
    //! \brief Set values of input tensor required by shape calculations.
    //!
    //! \param bindingIndex index of an input tensor for which
    //!        ICudaEngine::isShapeBinding(bindingIndex) and ICudaEngine::bindingIsInput(bindingIndex)
    //!        are both true.
    //!
    //! \param data pointer to values of the input tensor.  The number of values should be
    //!         the product of the dimensions returned by getBindingDimensions(bindingIndex).
    //!
    //! If ICudaEngine::isShapeBinding(bindingIndex) and ICudaEngine::bindingIsInput(bindingIndex)
    //! are both true, this method must be called before enqueue() or execute() may be called.
    //! This method will fail unless a valid optimization profile is defined for the current
    //! execution context (getOptimizationProfile() must not be -1).
    //!
    //! \warning This function will trigger layer resource updates on the next call of
    //!          enqueue[V2]()/execute[V2](), possibly resulting in performance bottlenecks, if the
    //!          shapes are different than the previous set shapes.
    //!
    //! \return false if an error occurs (e.g. bindingIndex is out of range for the currently selected
    //!         optimization profile or shape data is inconsistent with min-max range of the
    //!         optimization profile), else true. Note that the network can still be invalid for certain
    //!         combinations of input shapes that lead to invalid output shapes. To confirm the correctness
    //!         of the network input shapes, check whether the output binding has valid
    //!         dimensions using getBindingDimensions() on the output bindingIndex.
    bool setInputShapeBinding(int32_t bindingIndex, int32_t const* data) noexcept
    {
        return mImpl->setInputShapeBinding(bindingIndex, data);
    }

    //!
    //! \brief Get values of an input tensor required for shape calculations or an output tensor produced by shape
    //! calculations.
    //!
    //! \param bindingIndex index of an input or output tensor for which
    //!        ICudaEngine::isShapeBinding(bindingIndex) is true.
    //!
    //! \param data pointer to where values will be written.  The number of values written is
    //!        the product of the dimensions returned by getBindingDimensions(bindingIndex).
    //!
    //! If ICudaEngine::bindingIsInput(bindingIndex) is false, then both
    //! allInputDimensionsSpecified() and allInputShapesSpecified() must be true
    //! before calling this method. The method will also fail if no valid optimization profile
    //! has been set for the current execution context, i.e. if getOptimizationProfile() returns -1.
    //!
    //! \see isShapeBinding(bindingIndex)
    //!
    bool getShapeBinding(int32_t bindingIndex, int32_t* data) const noexcept
    {
        return mImpl->getShapeBinding(bindingIndex, data);
    }

    //!
    //! \brief Whether all dynamic dimensions of input tensors have been specified
    //!
    //! \return True if all dynamic dimensions of input tensors have been specified
    //!         by calling setBindingDimensions().
    //!
    //! Trivially true if network has no dynamically shaped input tensors.
    //!
    //! \see setBindingDimensions(bindingIndex,dimensions)
    //!
    bool allInputDimensionsSpecified() const noexcept
    {
        return mImpl->allInputDimensionsSpecified();
    }

    //!
    //! \brief Whether all input shape bindings have been specified
    //!
    //! \return True if all input shape bindings have been specified by setInputShapeBinding().
    //!
    //! Trivially true if network has no input shape bindings.
    //!
    //! \see isShapeBinding(bindingIndex)
    //!
    bool allInputShapesSpecified() const noexcept

    {
        return mImpl->allInputShapesSpecified();
    }

    //!
    //! \brief Set the ErrorRecorder for this interface
    //!
    //! Assigns the ErrorRecorder to this interface. The ErrorRecorder will track all errors during execution.
    //! This function will call incRefCount of the registered ErrorRecorder at least once. Setting
    //! recorder to nullptr unregisters the recorder with the interface, resulting in a call to decRefCount if
    //! a recorder has been registered.
    //!
    //! If an error recorder is not set, messages will be sent to the global log stream.
    //!
    //! \param recorder The error recorder to register with this interface.
    //
    //! \see getErrorRecorder()
    //!
    void setErrorRecorder(IErrorRecorder* recorder) noexcept
    {
        mImpl->setErrorRecorder(recorder);
    }

    //!
    //! \brief Get the ErrorRecorder assigned to this interface.
    //!
    //! Retrieves the assigned error recorder object for the given class. A nullptr will be returned if
    //! an error handler has not been set.
    //!
    //! \return A pointer to the IErrorRecorder object that has been registered.
    //!
    //! \see setErrorRecorder()
    //!
    IErrorRecorder* getErrorRecorder() const noexcept
    {
        return mImpl->getErrorRecorder();
    }

    //!
    //! \brief Synchronously execute inference a network.
    //!
    //! This method requires an array of input and output buffers. The mapping from tensor names to indices can be
    //! queried using ICudaEngine::getBindingIndex().
    //! This method only works for execution contexts built with full dimension networks.
    //! \param bindings An array of pointers to input and output buffers for the network.
    //!
    //! \return True if execution succeeded.
    //!
    //! \see ICudaEngine::getBindingIndex() ICudaEngine::getMaxBatchSize()
    //!
    bool executeV2(void* const* bindings) noexcept
    {
        return mImpl->executeV2(bindings);
    }

    //!
    //! \brief Asynchronously execute inference.
    //!
    //! This method requires an array of input and output buffers. The mapping from tensor names to indices can be
    //! queried using ICudaEngine::getBindingIndex().
    //! This method only works for execution contexts built with full dimension networks.
    //! \param bindings An array of pointers to input and output buffers for the network.
    //! \param stream A cuda stream on which the inference kernels will be enqueued
    //! \param inputConsumed An optional event which will be signaled when the input buffers can be refilled with new
    //! data
    //!
    //! \return True if the kernels were enqueued successfully.
    //!
    //! \see ICudaEngine::getBindingIndex() ICudaEngine::getMaxBatchSize()
    //!
    //! \note Calling enqueueV2() with a stream in CUDA graph capture mode has a known issue. If dynamic shapes are
    //!       used, the first enqueueV2() call after a setInputShapeBinding() call will cause failure in stream capture
    //!       due to resource allocation. Please call enqueueV2() once before capturing the graph.
    //!
    //! \warning Calling enqueueV2() in from the same IExecutionContext object with different CUDA streams concurrently
    //!          results in undefined behavior. To perform inference concurrently in multiple streams, use one execution
    //!          context per stream.
    //!
    bool enqueueV2(void* const* bindings, cudaStream_t stream, cudaEvent_t* inputConsumed) noexcept
    {
        return mImpl->enqueueV2(bindings, stream, inputConsumed);
    }

    //!
    //! \brief Select an optimization profile for the current context with async
    //! semantics.
    //!
    //! \param profileIndex Index of the profile. The value must lie between 0 and
    //!        getEngine().getNbOptimizationProfiles() - 1
    //!
    //! \param stream A cuda stream on which the cudaMemcpyAsyncs may be
    //! enqueued
    //!
    //! When an optimization profile is switched via this API, TensorRT may
    //! require that data is copied via cudaMemcpyAsync. It is the
    //! application’s responsibility to guarantee that synchronization between
    //! the profile sync stream and the enqueue stream occurs.
    //!
    //! The selected profile will be used in subsequent calls to execute() or
    //! enqueue().
    //! If the associated CUDA engine has inputs with dynamic shapes, the
    //! optimization profile must be set with a unique profileIndex before
    //! calling execute or enqueue.
    //! For the first execution context that is created for an engine,
    //! setOptimizationProfile(0) is called implicitly.
    //!
    //! If the associated CUDA engine does not have inputs with dynamic shapes,
    //! this method need not be called, in which case the default profile index
    //! of 0 will be used.
    //!
    //! setOptimizationProfileAsync() must be called before calling
    //! setBindingDimensions() and setInputShapeBinding() for all dynamic input
    //! tensors or input shape tensors, which in turn must be called before
    //! either execute() or enqueue().
    //!
    //! \warning This function will trigger layer resource updates on the next call of
    //!          enqueue[V2]()/execute[V2](), possibly resulting in performance bottlenecks.
    //!
    //! \warning Not synchronizing the stream used at enqueue with the stream
    //! used to set optimization profile asynchronously using this API will
    //! result in undefined behavior.
    //!
    //! \return true if the call succeeded, else false (e.g. input out of range)
    //!
    //! \see ICudaEngine::getNbOptimizationProfiles()
    //! \see IExecutionContext::setOptimizationProfile()
    bool setOptimizationProfileAsync(int32_t profileIndex, cudaStream_t stream) noexcept
    {
        return mImpl->setOptimizationProfileAsync(profileIndex, stream);
    }

    //!
    //! \brief Set whether enqueue emits layer timing to the profiler
    //!
    //! If set to true (default), enqueue is synchronous and does layer timing profiling implicitly if
    //! there is a profiler attached.
    //! If set to false, enqueue will be asynchronous if there is a profiler attached. An extra method
    //! reportToProfiler() needs to be called to obtain the profiling data and report to the profiler attached.
    //!
    //! \see IExecutionContext::getEnqueueEmitsProfile()
    //! \see IExecutionContext::reportToProfiler()
    void setEnqueueEmitsProfile(bool enqueueEmitsProfile) noexcept
    {
        mImpl->setEnqueueEmitsProfile(enqueueEmitsProfile);
    }

    //!
    //! \brief Get the enqueueEmitsProfile state.
    //!
    //! \return The enqueueEmitsProfile state.
    //!
    //! \see IExecutionContext::setEnqueueEmitsProfile()
    bool getEnqueueEmitsProfile() const noexcept
    {
        return mImpl->getEnqueueEmitsProfile();
    }

    //!
    //! \brief Calculate layer timing info for the current optimization profile in IExecutionContext
    //! and update the profiler after one iteration of inference launch.
    //!
    //! If IExecutionContext::getEnqueueEmitsProfile() returns true, the enqueue function will calculate layer timing
    //! implicitly if a profiler is provided. There is no need to call this function.
    //!
    //! If IExecutionContext::getEnqueueEmitsProfile() returns false, the enqueue function will record the CUDA event
    //! timers if a profiler is provided. But it will not perform the layer timing calculation.
    //! IExecutionContext::reportToProfiler() needs to be called explicitly to calculate layer timing for the previous
    //! inference launch.
    //!
    //! In the CUDA graph launch scenario, it will record the same set of CUDA events
    //! as in regular enqueue functions if the graph is captured from an IExecutionContext with profiler enabled.
    //! This function needs to be called after graph launch to report the layer timing info to the profiler.
    //!
    //! \warning profiling CUDA graphs is only available from CUDA 11.1 onwards.
    //!
    //! \return true if the call succeeded, else false (e.g. profiler not provided, in CUDA graph capture mode, etc.)
    //!
    //! \see IExecutionContext::setEnqueueEmitsProfile()
    //! \see IExecutionContext::getEnqueueEmitsProfile()
    bool reportToProfiler() const noexcept
    {
        return mImpl->reportToProfiler();
    }

protected:
    apiv::VExecutionContext* mImpl;
}; // class IExecutionContext

//!
//! \enum LayerInformationFormat
//!
//! \brief The format in which the IEngineInspector prints the layer information.
//!
//! \see IEngineInspector::getLayerInformation(), IEngineInspector::getEngineInformation()
//!
enum class LayerInformationFormat : int32_t
{
    kONELINE = 0, //!< Print layer information in one line per layer.
    kJSON = 1,    //!< Print layer information in JSON format.
};

//! Maximum number of layer information formats in LayerInformationFormat enum.
//! \see LayerInformationFormat
template <>
constexpr inline int32_t EnumMax<LayerInformationFormat>() noexcept
{
    return 2;
}

//!
//! \class IEngineInspector
//!
//! \brief An engine inspector which prints out the layer information of an engine or an execution context.
//!
//! The amount of printed information depends on the profiling verbosity setting of the builder config when the engine
//! is built:
//! - ProfilingVerbosity::kLAYER_NAMES_ONLY: only layer names will be printed.
//! - ProfilingVerbosity::kNONE: no layer information will be printed.
//! - ProfilingVerbosity::kDETAILED: layer names and layer parameters will be printed.
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
//! \see ProfilingVerbosity, IEngineInspector
//!
class IEngineInspector : public INoCopy
{
public:
    virtual ~IEngineInspector() noexcept = default;

    //!
    //! \brief Set an execution context as the inspection source.
    //!
    //! Setting the execution context and specifying all the input shapes allows the inspector
    //! to calculate concrete dimensions for any dynamic shapes and display their format information.
    //! Otherwise, values dependent on input shapes will be displayed as -1 and format information
    //! will not be shown.
    //!
    //! Passing nullptr will remove any association with an execution context.
    //!
    //! \return Whether the action succeeds.
    //!
    bool setExecutionContext(IExecutionContext const* context) noexcept
    {
        return mImpl->setExecutionContext(context);
    }

    //!
    //! \brief Get the context currently being inspected.
    //!
    //! \return The pointer to the context currently being inspected.
    //!
    //! \see setExecutionContext()
    //!
    IExecutionContext const* getExecutionContext() const noexcept
    {
        return mImpl->getExecutionContext();
    }

    //!
    //! \brief Get a string describing the information about a specific layer in the current engine or the execution
    //!        context.
    //!
    //! \param layerIndex the index of the layer. It must lie in range [0, engine.getNbLayers()).
    //!
    //! \param format the format the layer information should be printed in.
    //!
    //! \return A null-terminated C-style string describing the information about a specific layer in the current
    //!         engine or the execution context.
    //!
    //! \warning The content of the returned string may change when another execution context has
    //!          been set, or when another getLayerInformation() or getEngineInformation() has been called.
    //!
    //! \warning In a multi-threaded environment, this function must be protected from other threads changing the
    //!          inspection source. If the inspection source changes, the data that is being pointed to can change.
    //!          Copy the string to another buffer before releasing the lock in order to guarantee consistency.
    //!
    //! \see LayerInformationFormat
    //!
    AsciiChar const* getLayerInformation(int32_t layerIndex, LayerInformationFormat format) const noexcept
    {
        return mImpl->getLayerInformation(layerIndex, format);
    }

    //!
    //! \brief Get a string describing the information about all the layers in the current engine or the execution
    //!        context.
    //!
    //! \param layerIndex the index of the layer. It must lie in range [0, engine.getNbLayers()).
    //!
    //! \param format the format the layer information should be printed in.
    //!
    //! \return A null-terminated C-style string describing the information about all the layers in the current
    //!         engine or the execution context.
    //!
    //! \warning The content of the returned string may change when another execution context has
    //!          been set, or when another getLayerInformation() or getEngineInformation() has been called.
    //!
    //! \warning In a multi-threaded environment, this function must be protected from other threads changing the
    //!          inspection source. If the inspection source changes, the data that is being pointed to can change.
    //!          Copy the string to another buffer before releasing the lock in order to guarantee consistency.
    //!
    //! \see LayerInformationFormat
    //!
    AsciiChar const* getEngineInformation(LayerInformationFormat format) const noexcept
    {
        return mImpl->getEngineInformation(format);
    }

    //!
    //! \brief Set the ErrorRecorder for this interface
    //!
    //! Assigns the ErrorRecorder to this interface. The ErrorRecorder will track all errors during execution.
    //! This function will call incRefCount of the registered ErrorRecorder at least once. Setting
    //! recorder to nullptr unregisters the recorder with the interface, resulting in a call to decRefCount if
    //! a recorder has been registered.
    //!
    //! If an error recorder is not set, messages will be sent to the global log stream.
    //!
    //! \param recorder The error recorder to register with this interface.
    //
    //! \see getErrorRecorder()
    //!
    void setErrorRecorder(IErrorRecorder* recorder) noexcept
    {
        mImpl->setErrorRecorder(recorder);
    }

    //!
    //! \brief Get the ErrorRecorder assigned to this interface.
    //!
    //! Retrieves the assigned error recorder object for the given class. A nullptr will be returned if
    //! an error handler has not been set.
    //!
    //! \return A pointer to the IErrorRecorder object that has been registered.
    //!
    //! \see setErrorRecorder()
    //!
    IErrorRecorder* getErrorRecorder() const noexcept
    {
        return mImpl->getErrorRecorder();
    }

protected:
    apiv::VEngineInspector* mImpl;
}; // class IEngineInspector

} // namespace nvinfer1

//!
//! Internal C entry point for creating IRuntime.
//! @private
//!
extern "C" TENSORRTAPI void* createInferRuntime_INTERNAL(void* logger, int32_t version) noexcept;

//!
//! Internal C entry point for creating IRefitter.
//! @private
//!
extern "C" TENSORRTAPI void* createInferRefitter_INTERNAL(void* engine, void* logger, int32_t version) noexcept;

//!
//! \brief Return the plugin registry
//!
extern "C" TENSORRTAPI nvinfer1::IPluginRegistry* getPluginRegistry() noexcept;

//!
//! \brief Return the logger object.
//! \note the global logger is used only by standalone functions which have no associated builder, runtime
//! or refitter.
//!
extern "C" TENSORRTAPI nvinfer1::ILogger* getLogger() noexcept;

namespace nvinfer1
{
namespace // unnamed namespace avoids linkage surprises when linking objects built with different versions of this
          // header.
{
//!
//! \brief Create an instance of an IRuntime class.
//!
//! \param logger The logging class for the runtime.
//!
inline IRuntime* createInferRuntime(ILogger& logger) noexcept
{
    return static_cast<IRuntime*>(createInferRuntime_INTERNAL(&logger, NV_TENSORRT_VERSION));
}

//!
//! \brief Create an instance of an IRefitter class.
//!
//! \param logger The logging class for the refitter.
//!
inline IRefitter* createInferRefitter(ICudaEngine& engine, ILogger& logger) noexcept
{
    return static_cast<IRefitter*>(createInferRefitter_INTERNAL(&engine, &logger, NV_TENSORRT_VERSION));
}

} // namespace

//!
//! \brief Register the plugin creator to the registry
//! The static registry object will be instantiated when the plugin library is
//! loaded. This static object will register all creators available in the
//! library to the registry.
//!
//! \warning Statically registering plugins should be avoided in the automotive
//!  safety context as the application developer should first register an error recorder
//!  with the plugin registry via IPluginRegistry::setErrorRecorder() before using
//!  IPluginRegistry::registerCreator() or other methods.
//!
template <typename T>
class PluginRegistrar
{
public:
    PluginRegistrar()
    {
        getPluginRegistry()->registerCreator(instance, "");
    }

private:
    //! Plugin instance.
    T instance{};
};

} // namespace nvinfer1

#define REGISTER_TENSORRT_PLUGIN(name)                                                                                 \
    static nvinfer1::PluginRegistrar<name> pluginRegistrar##name {}
#endif // NV_INFER_RUNTIME_H
