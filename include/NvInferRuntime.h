/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef NV_INFER_RUNTIME_H
#define NV_INFER_RUNTIME_H

//!
//! \file NvInferRuntime.h
//!
//! This is the top-level API file for TensorRT extended runtime library.
//!

#include "NvInferImpl.h"
#define NV_INFER_INTERNAL_INCLUDE 1
#include "NvInferPluginBase.h"
#undef NV_INFER_INTERNAL_INCLUDE
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
    INoCopy(INoCopy const& other) = delete;
    INoCopy& operator=(INoCopy const& other) = delete;
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
//! DLA runtime APIs. See sampleCudla for an example of integrating cuDLA APIs with TensorRT APIs.
//!
enum class EngineCapability : int32_t
{
    //!
    //! Standard: TensorRT flow without targeting the safety runtime.
    //! This flow supports both DeviceType::kGPU and DeviceType::kDLA.
    //!
    kSTANDARD = 0,

    //!
    //! Safety: TensorRT flow with restrictions targeting the safety runtime.
    //! See safety documentation for list of supported layers and formats.
    //! This flow supports only DeviceType::kGPU.
    //!
    //! This flag is only supported in NVIDIA Drive(R) products.
    kSAFETY = 1,

    //!
    //! DLA Standalone: TensorRT flow with restrictions targeting external, to TensorRT, DLA runtimes.
    //! See DLA documentation for list of supported layers and formats.
    //! This flow supports only DeviceType::kDLA.
    //!
    kDLA_STANDALONE = 2,
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
    void const* values; //!< The weight values, in a contiguous array.
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
//!
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
//! \brief An IDimensionExpr represents an integer expression constructed from constants,
//! input dimensions, and binary operations.  These expressions are can be used
//! in overrides of IPluginV2DynamicExt::getOutputDimensions or IPluginV3OneBuild::getOutputShapes() to define output
//! dimensions in terms of input dimensions.
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
//! \see DimensionOperation, IPluginV2DynamicExt::getOutputDimensions, IPluginV3OneBuild::getOutputShapes()
//!
class IDimensionExpr : public INoCopy
{
public:
    //!
    //! \brief Return true if expression is a build-time constant.
    //!
    bool isConstant() const noexcept
    {
        return mImpl->isConstant();
    }

    //!
    //! \brief Get the value of the constant.
    //!
    //! If isConstant(), returns value of the constant.
    //! If !isConstant(), return std::numeric_limits<int64_t>::min().
    //!
    int64_t getConstantValue() const noexcept
    {
        return mImpl->getConstantValue();
    }

protected:
    apiv::VDimensionExpr* mImpl;
    virtual ~IDimensionExpr() noexcept = default;

public:
    //!
    //! \brief Return true if this denotes the value of a size tensor.
    //!
    //! \return True if this was created with method IExprBuilder::declareSizeTensor, false otherwise
    //!
    bool isSizeTensor() const noexcept
    {
        return mImpl->isSizeTensor();
    }
};

//!
//! \class IExprBuilder
//!
//! \brief Object for constructing IDimensionExpr.
//!
//! There is no public way to construct an IExprBuilder.  It appears as an argument to
//! method IPluginV2DynamicExt::getOutputDimensions() and IPluginV3OneBuild::getOutputShapes().  Overrides of that
//! method can use that IExprBuilder argument to construct expressions that define output dimensions in terms of input
//! dimensions.
//!
//! Clients should assume that any values constructed by the IExprBuilder are destroyed
//! after IPluginV2DynamicExt::getOutputDimensions() or IPluginV3OneBuild::getOutputShapes() returns.
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
//! \see IDimensionExpr
//!
class IExprBuilder : public INoCopy
{
public:
    //!
    //! \brief Return pointer to IDimensionExp for given value.
    //!
    IDimensionExpr const* constant(int64_t value) noexcept
    {
        return mImpl->constant(value);
    }

    //!
    //! \brief Get the operation.
    //!
    //! Return pointer to IDimensionExp that represents the given operation applied to first and second.
    //! Returns nullptr if op is not a valid DimensionOperation.
    //!
    IDimensionExpr const* operation(
        DimensionOperation op, IDimensionExpr const& first, IDimensionExpr const& second) noexcept
    {
        return mImpl->operation(op, first, second);
    }

protected:
    apiv::VExprBuilder* mImpl;
    virtual ~IExprBuilder() noexcept = default;

public:
    //!
    //! \brief Declare a size tensor at the given output index, with the specified auto-tuning formula and upper bound.
    //!
    //! A size tensor allows a plugin to have output dimensions that cannot be computed solely from input dimensions.
    //! For example, suppose a plugin implements the equivalent of INonZeroLayer for 2D input. The plugin can
    //! have one output for the indices of non-zero elements, and a second output containing the number of non-zero
    //! elements. Suppose the input has size [M,N] and has K non-zero elements. The plugin can write K to the second
    //! output. When telling TensorRT that the first output has shape [2,K], plugin uses IExprBuilder::constant() and
    //! IExprBuilder::declareSizeTensor(1,...) to create the IDimensionExpr that respectively denote 2 and K.
    //!
    //! TensorRT also needs to know the value of K to use for auto-tuning and an upper bound on K so that it can
    //! allocate memory for the output tensor. In the example, supposed typically half of the plugin's input elements
    //! are non-zero, and all the elements might be nonzero. then using M*N/2 might be a good expression for the opt
    //! parameter, and M*N for the upper bound. IDimensionsExpr for these expressions can be constructed from
    //! IDimensionsExpr for the input dimensions.
    //!
    //! \param outputIndex index of a plugin output that is a size tensor.
    //! \param opt formula for computing auto-tuning value. Must not depend on a size tensor.
    //! \param upper Upper bound on the size tensor.
    //!
    //! \return IDimensionExpr denoting the value of the size tensor.
    //!
    //! \see IPluginV3OneBuild::getOutputShapes()
    //!
    IDimensionExpr const* declareSizeTensor(int32_t outputIndex, IDimensionExpr const& opt, IDimensionExpr const& upper)
    {
        return mImpl->declareSizeTensor(outputIndex, opt, upper);
    }
};

//!
//! \class DimsExprs
//!
//! \brief Analog of class Dims with expressions instead of constants for the dimensions.
//!
class DimsExprs
{
public:
    int32_t nbDims;                          //!< The number of dimensions.
    IDimensionExpr const* d[Dims::MAX_DIMS]; //!< The extent of each dimension.
};

//!
//! \struct DynamicPluginTensorDesc
//!
//! \brief Summarizes tensors that a plugin might see for an input or output.
//!
struct DynamicPluginTensorDesc
{
    //! Information required to interpret a pointer to tensor data, except that desc.dims has -1 in place of any runtime dimension.
    PluginTensorDesc desc;

    //! Lower bounds on tensor’s dimensions
    Dims min;

    //! Upper bounds on tensor’s dimensions
    Dims max;

    //! Optimum value of tensor’s dimensions specified for auto-tuning
    Dims opt;
};

//!
//! \class IPluginV2DynamicExt
//!
//! \brief Similar to IPluginV2Ext, but with support for dynamic shapes.
//!
//! Clients should override the public methods, including the following inherited methods:
//!
//! * virtual int32_t getNbOutputs() const noexcept = 0;
//!
//! * virtual DataType getOutputDataType(int32_t index, DataType const* inputTypes,
//!                                      int32_t nbInputs) const noexcept = 0;
//!
//! * virtual size_t getSerializationSize() const noexcept = 0;
//!
//! * virtual void serialize(void* buffer) const noexcept = 0;
//!
//! * virtual void destroy() noexcept = 0;
//!
//! * virtual void setPluginNamespace(char const* pluginNamespace) noexcept = 0;
//!
//! * virtual char const* getPluginNamespace() const noexcept = 0;
//!
//! For weakly typed networks, the inputTypes will always be DataType::kFLOAT or DataType::kINT32,
//! and the returned type is canonicalized to DataType::kFLOAT if it is DataType::kHALF or DataType:kINT8.
//! For strongly typed networks, inputTypes are inferred from previous operations, and getOutputDataType
//! specifies the returned type based on the inputTypes.
//! Details about the floating-point precision are elicited later by method supportsFormatCombination.
//!
//! \deprecated Deprecated in TensorRT 10.0. Please implement IPluginV3 instead.
//!
class TRT_DEPRECATED IPluginV2DynamicExt : public nvinfer1::IPluginV2Ext
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
        int32_t outputIndex, DimsExprs const* inputs, int32_t nbInputs, IExprBuilder& exprBuilder) noexcept = 0;

    //!
    //! \brief Limit on number of format combinations accepted.
    //!
    static constexpr int32_t kFORMAT_COMBINATION_LIMIT = 100;

    //!
    //! \brief Return true if plugin supports the format and datatype for the input/output indexed by pos.
    //!
    //! For this method inputs are numbered 0..(nbInputs-1) and outputs are numbered nbInputs..(nbInputs+nbOutputs-1).
    //! Using this numbering, pos is an index into InOut, where 0 <= pos < nbInputs+nbOutputs.
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
    //!         return inOut[pos].format == TensorFormat::kLINEAR && inOut[pos].type == DataType::kHALF;
    //!
    //! * A definition for a plugin that supports only FP16 NCHW for its two inputs,
    //!   and FP32 NCHW for its single output:
    //!
    //!         return inOut[pos].format == TensorFormat::kLINEAR && (inOut[pos].type == (pos < 2 ? DataType::kHALF :
    //!         DataType::kFLOAT));
    //!
    //! * A definition for a "polymorphic" plugin with two inputs and one output that supports
    //!   any format or type, but the inputs and output must have the same format and type:
    //!
    //!         return pos == 0 || (inOut[pos].format == inOut.format[0] && inOut[pos].type == inOut[0].type);
    //!
    //! Warning: TensorRT will stop asking for formats once it finds kFORMAT_COMBINATION_LIMIT on combinations.
    //!
    virtual bool supportsFormatCombination(
        int32_t pos, PluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept = 0;

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
    //!    - The optimization profile is changed via setOptimizationProfileAsync().
    //!    - An input execution binding is changed via setInputShape().
    //! \warning The execution phase is timing critical during IExecutionContext but is not part of the timing loop when
    //! called from IBuilder. Performance bottlenecks of configurePlugin won't show up during engine building but will
    //! be visible during execution after calling functions that trigger layer resource updates.
    //!
    //! \param in The input tensors attributes that are used for configuration.
    //! \param nbInputs Number of input tensors.
    //! \param out The output tensors attributes that are used for configuration.
    //! \param nbOutputs Number of output tensors.
    //!
    virtual void configurePlugin(DynamicPluginTensorDesc const* in, int32_t nbInputs,
        DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept = 0;

    //!
    //! \brief Find the workspace size required by the layer.
    //!
    //! This function is called after the plugin is configured, and possibly during execution.
    //! The result should be a sufficient workspace size to deal with inputs and outputs of the given size
    //! or any smaller problem.
    //!
    //! \return The workspace size.
    //!
    virtual size_t getWorkspaceSize(PluginTensorDesc const* inputs, int32_t nbInputs, PluginTensorDesc const* outputs,
        int32_t nbOutputs) const noexcept = 0;

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
    virtual int32_t enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept = 0;

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

    //!
    //! \brief Set plugin configuration
    //!
    void configurePlugin(Dims const*, int32_t, Dims const*, int32_t, DataType const*, DataType const*, bool const*,
        bool const*, PluginFormat, int32_t) noexcept override final
    {
    }

    //!
    //! \brief Check if provided data type is supported
    //!
    bool supportsFormat(DataType, PluginFormat) const noexcept override final
    {
        return false;
    }

    //!
    //! \brief Get output dimensions.
    //!
    Dims getOutputDimensions(int32_t, Dims const*, int32_t) noexcept override final
    {
        return Dims{-1, {}};
    }

    //!
    //! \brief Is output broadcasted across batch.
    //!
    //! \warning Expected to return false as implicit batch support was removed in TensorRT 10.0.
    //!
    //! \deprecated Deprecated in TensorRT 10.0. Implicit batch support is removed in TensorRT 10.0.
    //!
    TRT_DEPRECATED bool isOutputBroadcastAcrossBatch(int32_t, bool const*, int32_t) const noexcept override final
    {
        return false;
    }

    //!
    //! \brief Can output broadcasted across batch.
    //!
    //! \warning Expected to return false as implicit batch support was removed in TensorRT 10.0.
    //!
    //! \deprecated Deprecated in TensorRT 10.0. Implicit batch support is removed in TensorRT 10.0.
    //!
    TRT_DEPRECATED bool canBroadcastInputAcrossBatch(int32_t) const noexcept override final
    {
        return true;
    }

    //!
    //! \brief Get required workspace size in bytes.
    //!
    size_t getWorkspaceSize(int32_t) const noexcept override final
    {
        return 0;
    }

    //!
    //! \brief Run inference.
    //!
    int32_t enqueue(int32_t, void const* const*, void* const*, void*, cudaStream_t) noexcept override final
    {
        return 1;
    }
};

namespace v_1_0
{
class IStreamReader : public IVersionedInterface
{
public:
    //!
    //! TensorRT never calls the destructor for an IStreamReader defined by the
    //! application.
    //!
    ~IStreamReader() override = default;
    IStreamReader() = default;

    //!
    //! \brief Return version information associated with this interface. Applications must not override this method.
    //!
    InterfaceInfo getInterfaceInfo() const noexcept override
    {
        return InterfaceInfo{"IStreamReader", 1, 0};
    }

    //!
    //! \brief Read the next number of bytes in the stream.
    //!
    //! \param destination The memory to write to
    //! \param nbBytes The number of bytes to read
    //!
    //! \returns The number of bytes read. Negative values will be considered an automatic error.
    //!
    virtual int64_t read(void* destination, int64_t nbBytes) = 0;

protected:
    IStreamReader(IStreamReader const&) = default;
    IStreamReader(IStreamReader&&) = default;
    IStreamReader& operator=(IStreamReader const&) & = default;
    IStreamReader& operator=(IStreamReader&&) & = default;
};
} // namespace v_1_0

//!
//! \class IStreamReader
//!
//! \brief Application-implemented class for reading data in a stream-based manner.
//!
//! \note To ensure compatibility of source code with future versions of TensorRT, use IStreamReader, not
//!       v_1_0::IStreamReader
//!
using IStreamReader = v_1_0::IStreamReader;

//!
//! \enum SeekPosition
//! \brief Controls the seek mode of IStreamReaderV2.
//!
enum class SeekPosition : int32_t
{
    //! From the beginning of the file.
    kSET = 0,

    //! From the current position of the file.
    kCUR = 1,

    //! From the tail of the file.
    kEND = 2,
};

namespace v_1_0
{
class IStreamReaderV2 : public IVersionedInterface
{
public:
    //!
    //! TensorRT never calls the destructor for an IStreamReaderV2 defined by the
    //! application.
    //!
    ~IStreamReaderV2() override = default;
    IStreamReaderV2() = default;

    //!
    //! \brief Return version information associated with this interface. Applications must not override this method.
    //!
    InterfaceInfo getInterfaceInfo() const noexcept override
    {
        return InterfaceInfo{"IStreamReaderV2", 1, 0};
    }

    //!
    //! \brief Read the next number of bytes in the stream asynchronously.
    //!
    //! \param destination The memory to write to, call cudaPointerGetAttributes to get the memory location
    //! \param nbBytes The number of bytes to read
    //! \param stream The CUDA stream used to do the copy
    //!
    //! \returns The number of bytes read. Negative values indicate an unrecoverable error.
    //! A zero indicates that the end of the stream has been reached.
    //!
    virtual int64_t read(void* destination, int64_t nbBytes, cudaStream_t stream) noexcept = 0;

    //!
    //! \brief Sets the position of the stream to the given offset.
    //!
    //! \param offset The number of bytes to offset from where.
    //! \param where The position from where the offset is added. \see SeekPosition
    //!
    //! \returns True if the position is updated successfully.
    //!
    virtual bool seek(int64_t offset, SeekPosition where) noexcept = 0;

protected:
    IStreamReaderV2(IStreamReaderV2 const&) = default;
    IStreamReaderV2(IStreamReaderV2&&) = default;
    IStreamReaderV2& operator=(IStreamReaderV2 const&) & = default;
    IStreamReaderV2& operator=(IStreamReaderV2&&) & = default;
};
} // namespace v_1_0

//!
//! \class IStreamReaderV2
//!
//! \brief Application-implemented class for reading data in a stream-based manner asynchronously. Intended for use with
//! the GDS API for optimizing load times.
//!
//! \note To ensure compatibility of source code with future versions of TensorRT, use IStreamReaderV2, not
//!       v_1_0::IStreamReaderV2
//!
using IStreamReaderV2 = v_1_0::IStreamReaderV2;

//!
//! \class IPluginResourceContext
//!
//! \brief Interface for plugins to access per context resources provided by TensorRT
//!
//! There is no public way to construct an IPluginResourceContext. It appears as an argument to
//! IPluginV3OneRuntime::attachToContext(). Overrides of that method can use the IPluginResourceContext object to access
//! any available per context resources.
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
//! \see IPluginV3OneRuntime::attachToContext()
//!
class IPluginResourceContext
{
public:
    //! \brief Get the GPU allocator associated with the resource context
    //!
    //! \see IPluginV3OneRuntime::attachToContext()
    //!
    virtual IGpuAllocator* getGpuAllocator() const noexcept = 0;

    //! \brief Get the error recorder associated with the resource context
    //!
    //! \see IPluginV3OneRuntime::attachToContext()
    //!
    virtual IErrorRecorder* getErrorRecorder() const noexcept = 0;
    virtual ~IPluginResourceContext() noexcept = default;

protected:
    IPluginResourceContext() = default;
    IPluginResourceContext(IPluginResourceContext const&) = default;
    IPluginResourceContext(IPluginResourceContext&&) = default;
    IPluginResourceContext& operator=(IPluginResourceContext const&) & = default;
    IPluginResourceContext& operator=(IPluginResourceContext&&) & = default;
};

namespace v_1_0
{
class IPluginV3OneCore : public IPluginCapability
{
public:
    //!
    //! \brief Return version information associated with this interface. Applications must not override this method.
    //!
    InterfaceInfo getInterfaceInfo() const noexcept override
    {
        return InterfaceInfo{"PLUGIN_V3ONE_CORE", 1, 0};
    }

    //!
    //! \brief Return the plugin name. Should match the plugin name returned by the corresponding plugin creator.
    //!
    //! \see IPluginCreatorV3One::getPluginName()
    //!
    //! \warning The string returned must be NULL-terminated and have a length of 1024 bytes or less including the
    //! NULL terminator.
    //!
    virtual AsciiChar const* getPluginName() const noexcept = 0;

    //!
    //! \brief Return the plugin version. Should match the plugin version returned by the corresponding plugin creator.
    //!
    //! \see IPluginCreatorV3One::getPluginVersion()
    //!
    //! \warning The string returned must be NULL-terminated and have a length of 1024 bytes or less including the
    //! NULL terminator.
    //!
    virtual AsciiChar const* getPluginVersion() const noexcept = 0;

    //!
    //! \brief Return the namespace of the plugin object. Should match the plugin namespace returned by the
    //! corresponding plugin creator.
    //!
    //! \see IPluginCreatorV3One::getPluginNamespace()
    //!
    //! \warning The string returned must be NULL-terminated and have a length of 1024 bytes or less including the
    //! NULL terminator.
    //!
    virtual AsciiChar const* getPluginNamespace() const noexcept = 0;
};

class IPluginV3OneBuild : public IPluginCapability
{
public:
    //!
    //! \brief The default maximum number of format combinations that will be timed by TensorRT during the build phase
    //!
    //! \see getFormatCombinationLimit
    //!
    static constexpr int32_t kDEFAULT_FORMAT_COMBINATION_LIMIT = 100;

    //!
    //! \brief Return version information associated with this interface. Applications must not override this method.
    //!
    InterfaceInfo getInterfaceInfo() const noexcept override
    {
        return InterfaceInfo{"PLUGIN_V3ONE_BUILD", 1, 0};
    }

    //!
    //! \brief Configure the plugin.
    //!
    //! configurePlugin() can be called multiple times in the build phase during creation of an engine by IBuilder.
    //!
    //! configurePlugin() is called when a plugin is being prepared for profiling but not for any
    //! specific input size. This provides an opportunity for the plugin to make algorithmic choices on the basis of
    //! input and output formats, along with the bound of possible dimensions. The min, opt and max value of the
    //! DynamicPluginTensorDesc correspond to the kMIN, kOPT and kMAX value of the current profile that the plugin is
    //! being profiled for, with the desc.dims field corresponding to the dimensions of plugin specified at network
    //! creation. Wildcard dimensions may exist during this phase in the desc.dims field.
    //!
    //! \param in The input tensors attributes that are used for configuration.
    //! \param nbInputs Number of input tensors.
    //! \param out The output tensors attributes that are used for configuration.
    //! \param nbOutputs Number of output tensors.
    //!
    //! \return 0 for success, else non-zero (which will cause engine termination, if invoked by TensorRT).
    //!
    virtual int32_t configurePlugin(DynamicPluginTensorDesc const* in, int32_t nbInputs,
        DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept = 0;

    //!
    //! \brief Provide the data types of the plugin outputs if the input tensors have the data types provided.
    //!
    //! \param outputTypes Pre-allocated array to which the output data types should be written.
    //! \param nbOutputs The number of output tensors. This matches the value returned from getNbOutputs().
    //! \param inputTypes The input data types.
    //! \param nbInputs The number of input tensors.
    //!
    //! \return 0 for success, else non-zero (which will cause engine termination). The returned code will be reported
    //! through the error recorder.
    //!
    //! \note Provide `DataType::kFLOAT`s if the layer has no inputs. The data type for any size tensor outputs must be
    //! `DataType::kINT32`. The returned data types must each have a format that is supported by the plugin.
    //!
    //! \warning DataType:kBOOL and DataType::kUINT8 are not supported.
    //!
    virtual int32_t getOutputDataTypes(
        DataType* outputTypes, int32_t nbOutputs, const DataType* inputTypes, int32_t nbInputs) const noexcept = 0;

    //!
    //! \brief Provide expressions for computing dimensions of the output tensors from dimensions of the input tensors.
    //!
    //! \param inputs Expressions for dimensions of the input tensors
    //! \param nbInputs The number of input tensors
    //! \param shapeInputs Expressions for values of the shape tensor inputs
    //! \param nbShapeInputs The number of shape tensor inputs
    //! \param outputs Pre-allocated array to which the output dimensions must be written
    //! \param exprBuilder Object for generating new dimension expressions
    //!
    //! \note Any size tensor outputs must be declared to be 0D.
    //!
    //! \note The declaration of shapeInputs as DimsExprs is slightly abusive, because the "dimensions"
    //!       are actually the values of the shape tensor. For example, if the input shape tensor
    //!       is a 2x3 matrix, the DimsExprs will have six "dimensions": the three values from the first
    //!       row of the matrix followed by the three values from the second row of the matrix.
    //!
    //! \return 0 for success, else non-zero (which will cause engine termination). Returned code will be reported
    //! through the error recorder.
    //!
    virtual int32_t getOutputShapes(DimsExprs const* inputs, int32_t nbInputs, DimsExprs const* shapeInputs,
        int32_t nbShapeInputs, DimsExprs* outputs, int32_t nbOutputs, IExprBuilder& exprBuilder) noexcept = 0;

    //!
    //! \brief Return true if plugin supports the format and datatype for the input/output indexed by pos.
    //!
    //! For this method inputs are numbered 0.. (nbInputs - 1) and outputs are numbered nbInputs.. (nbInputs + nbOutputs
    //! - 1). Using this numbering, pos is an index into InOut, where 0 <= pos < nbInputs + nbOutputs - 1.
    //!
    //! TensorRT invokes this method to ask if the input/output indexed by pos supports the format/datatype specified
    //! by inOut[pos].format and inOut[pos].type.  The override should return true if that format/datatype at inOut[pos]
    //! are supported by the plugin.  If support is conditional on other input/output formats/datatypes, the plugin can
    //! make its result conditional on the formats/datatypes in inOut[0.. pos - 1], which will be set to values
    //! that the plugin supports.  The override should not inspect inOut[pos1.. nbInputs + nbOutputs - 1],
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
    //! \warning TensorRT will stop querying once it finds getFormatCombinationLimit() of combinations.
    //!
    //! \see getFormatCombinationLimit
    //!
    virtual bool supportsFormatCombination(
        int32_t pos, DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept = 0;

    //!
    //! \brief Get the number of outputs from the plugin.
    //!
    //! \return The number of outputs, which must be a positive integer.
    //!
    virtual int32_t getNbOutputs() const noexcept = 0;

    //!
    //! \brief Find the workspace size required by the layer.
    //!
    //! This function is called after the plugin is configured, and possibly during execution.
    //! The result should be a sufficient workspace size to deal with inputs and outputs of the given size
    //! or any smaller problem.
    //!
    //! \return The workspace size.
    //!
    virtual size_t getWorkspaceSize(DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
        DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept
    {
        return 0;
    }

    //!
    //! \brief Query for any custom tactics that the plugin intends to use
    //!
    //! This method queries for the set of tactics T(f) supported by the plugin for the format combination f indicated
    //! by the immediately preceding call to configurePlugin(). It is guaranteed to be called after configurePlugin().
    //!
    //! For each format combination provided through configurePlugin(), up to a maximum of getFormatCombinationLimit(),
    //! the plugin will be timed for each tactic advertised through this method for that format combination. i.e. The
    //! plugin will be timed \f$N = sum_{i=0}^{i<getFormatCombinationLimit()} (T(f[i]))\f$ times. If \f$N = 1\f$, the
    //! plugin may not be timed. In peudocode, the timing protocol appears as the following:
    //!
    //! counter = 0
    //! for each supported format combination
    //!     ++counter
    //!     if counter > getFormatCombinationLimit()
    //!         goto done
    //!     configurePlugin(...)
    //!     for each tactic in getValidTactics(...)
    //!         time tactic
    //! done:
    //!
    //!
    //! \param tactics Pre-allocated buffer to which the tactic values should be written
    //! \param nbTactics The number of tactics advertised through getNbTactics()
    //!
    //! \note The provided tactic values must be unique and non-zero. The tactic value 0 is reserved for the default
    //! tactic attached to each format combination.
    //!
    //! \return 0 for success, else non-zero (which will cause engine termination). The returned code will be reported
    //! through the error recorder.
    //!
    virtual int32_t getValidTactics(int32_t* tactics, int32_t nbTactics) noexcept
    {
        return 0;
    }

    //!
    //! \brief Query for the number of custom tactics the plugin intends to use
    //!
    virtual int32_t getNbTactics() noexcept
    {
        return 0;
    }

    //!
    //! \brief Called to query the suffix to use for the timing cache ID. May be called anytime after plugin creation.
    //!
    //! \return Suffix to use for timing cache ID, considering only the creation state of the plugin.
    //!         Returning nullptr will disable timing caching for the plugin altogether.
    //!
    //! \note If timing caching is enabled for the plugin (by returning non-null), the I/O shape and format information
    //! will be automatically considered to form the prefix of the timing cache ID. Therefore, only other factors
    //! determining the creation state of the plugin, such as its attribute values, should be considered to compose the
    //! return value.
    //!
    virtual char const* getTimingCacheID() noexcept
    {
        return nullptr;
    }

    //!
    //! \brief Return the maximum number of format combinations that will be timed by TensorRT during the build phase
    //!
    virtual int32_t getFormatCombinationLimit() noexcept
    {
        return kDEFAULT_FORMAT_COMBINATION_LIMIT;
    }

    //!
    //! \brief Query for a string representing the configuration of the plugin. May be called anytime after
    //! plugin creation.
    //!
    //! \return A string representing the plugin's creation state, especially with regard to its attribute values.
    //!
    virtual char const* getMetadataString() noexcept
    {
        return nullptr;
    }
};

class IPluginV3OneRuntime : public IPluginCapability
{
public:
    //!
    //! \brief Return version information associated with this interface. Applications must not override this method.
    //!
    InterfaceInfo getInterfaceInfo() const noexcept override
    {
        return InterfaceInfo{"PLUGIN_V3ONE_RUNTIME", 1, 0};
    }

    //!
    //! \brief Set the tactic to be used in the subsequent call to enqueue(). If no custom tactics were advertised, this
    //! will have a value of 0, which is designated as the default tactic.
    //!
    //! \return 0 for success, else non-zero (which will cause engine termination). The returned code will be reported
    //! through the error recorder.
    //!
    virtual int32_t setTactic(int32_t tactic) noexcept
    {
        return 0;
    }

    //!
    //! \brief Called when a plugin is being prepared for execution for specific dimensions. This could
    //! happen multiple times in the execution phase, both during creation of an engine by IBuilder and execution of an
    //! engine by IExecutionContext.
    //!  * IBuilder will call this function once per profile, with `in` resolved to the values specified by the
    //!  kOPT field of the current profile.
    //!  * IExecutionContext will call this during the next subsequent instance of enqueueV3() or executeV2() if:
    //!    - The optimization profile is changed via setOptimizationProfile() or setOptimizationProfileAsync().
    //!    - An input binding is changed via setInputTensorAddress() or setTensorAddress() or setInputShape().
    //! \warning The execution phase is timing critical during IExecutionContext but is not part of the timing loop when
    //! called from IBuilder. Performance bottlenecks of onShapeChange() will not show up during engine building but
    //! will be visible during execution if any triggering functions are called.
    //!
    //! \param in The input tensors attributes that are used for configuration.
    //! \param nbInputs Number of input tensors.
    //! \param out The output tensors attributes that are used for configuration.
    //! \param nbOutputs Number of output tensors.
    //!
    virtual int32_t onShapeChange(
        PluginTensorDesc const* in, int32_t nbInputs, PluginTensorDesc const* out, int32_t nbOutputs) noexcept = 0;

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
    //! \return 0 for success, else non-zero (which will cause engine termination). The returned code will be reported
    //! through the error recorder.
    //!
    virtual int32_t enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept = 0;

    //!
    //! \brief Clone the plugin, attach the cloned plugin object to a execution context and grant the cloned plugin
    //! access to some context resources.
    //!
    //! This function is called automatically for each plugin when a new execution context is created. The plugin may
    //! use resources provided by the IPluginResourceContext until the plugin is deleted by TensorRT.
    //!
    //! If the plugin needs per-context resources, it can be allocated here.
    //!
    //! \param context A resource context that exposes methods to get access to execution context specific resources.
    //!                A different resource context is guaranteed for each different execution context to which the
    //!                plugin is attached.
    //! \see IPluginResourceContext
    //!
    //! \note This method should clone the entire IPluginV3 object, not just the runtime interface
    //!
    //! \return A clone of the IPluginV3 object whose runtime interface on which this method is invoked, which has
    //! attached to the provided resource context.
    //!
    virtual IPluginV3* attachToContext(IPluginResourceContext* context) noexcept = 0;

    //!
    //! \brief Get the plugin fields which should be serialized.
    //!
    //! \note The set of plugin fields returned does not necessarily need to match that advertised through
    //! getFieldNames() of the corresponding plugin creator.

    //! \note To serialize arbitrary plugin data, use a PluginField of
    //! PluginFieldType::kUNKNOWN, with the length of the PluginField set to the correct number of bytes.
    //!
    virtual PluginFieldCollection const* getFieldsToSerialize() noexcept = 0;
};
} // namespace v_1_0

namespace v_2_0
{

class IPluginV3OneBuild : public v_1_0::IPluginV3OneBuild
{
public:
    InterfaceInfo getInterfaceInfo() const noexcept override
    {
        return InterfaceInfo{"PLUGIN_V3ONE_BUILD", 2, 0};
    }

    //!
    //! \brief Communicates to TensorRT that the output at the specified output index is aliased to the input at the
    //! returned index
    //!
    //! Enables read-modify-write behavior in plugins. TensorRT may insert copies to facilitate this capability.
    //!
    //! \return An integer denoting the index of the input which is aliased to the output at outputIndex.
    //!         Returning -1 indicates that the output is not aliased to any input. Otherwise, the valid range for
    //!         return value is [0, nbInputs - 1].
    //!
    //! \note A given plugin input can only be aliased to a single plugin output.
    //!
    //! \note This API will only be called and have an effect when PreviewFeature::kALIASED_PLUGIN_IO_10_03 is turned
    //! on.
    //!
    //! \warning If an input is not shallow copyable, a copy inserted by TensorRT may not work as intended. Therefore,
    //!          using this feature with tensors requiring deep copies is not supported.
    //!
    //! \warning If a given tensor is requested to be aliased by two different plugins, this may result in divergent
    //! copies of the tensor after writes from each plugin. e.g. In the below example, t1 and t2 could be divergent.
    //!
    //!        +-----+            +--------+
    //!     +->|Copy +--> t* ---->|Plugin0 +--> t1
    //!     |  +-----+            +--------+
    //!     t
    //!     |  +-----+            +--------+
    //!     +->|Copy +--> t** --->|Plugin1 +--> t2
    //!        +-----+            +--------+
    //!
    virtual int32_t getAliasedInput(int32_t outputIndex) noexcept
    {
        return -1;
    }
};

} // namespace v_2_0

//!
//! \class IPluginV3OneCore
//!
//! \brief A plugin capability interface that enables the core capability (PluginCapabilityType::kCORE).
//!
//! \see IPluginCapability
//! \see PluginCapabilityType
//! \see IPluginV3::getCapabilityInterface()
//!
using IPluginV3OneCore = v_1_0::IPluginV3OneCore;

//!
//! \class IPluginV3OneBuild
//!
//! \brief A plugin capability interface that enables the build capability (PluginCapabilityType::kBUILD). Exposes
//! methods that allow the expression of the build time properties and behavior of a plugin.
//!
//! \see IPluginCapability
//! \see PluginCapabilityType
//! \see IPluginV3::getCapabilityInterface()
//!
using IPluginV3OneBuild = v_1_0::IPluginV3OneBuild;

//!
//! \class IPluginV3OneRuntime
//!
//! \brief A plugin capability interface that enables the runtime capability (PluginCapabilityType::kRUNTIME). Exposes
//! methods that allow the expression of the runtime properties and behavior of a plugin.
//!
//! \see IPluginCapability
//! \see PluginCapabilityType
//! \see IPluginV3::getCapabilityInterface()
//!
using IPluginV3OneRuntime = v_1_0::IPluginV3OneRuntime;

//!
//! \class IPluginV3OneBuildV2
//!
//! \brief A plugin capability interface that extends IPluginV3OneBuild by providing I/O aliasing functionality.
//!
//! \see IPluginV3OneBuild
//!
using IPluginV3OneBuildV2 = v_2_0::IPluginV3OneBuild;

namespace v_1_0
{
class IProfiler
{
public:
    //!
    //! \brief Layer time reporting callback.
    //!
    //! \param layerName The name of the layer, set when constructing the network definition. If the engine is built
    //!                  with profiling verbosity set to kNONE, the layerName is the decimal index of the layer.
    //! \param ms The time in milliseconds to execute the layer.
    //!
    virtual void reportLayerTime(char const* layerName, float ms) noexcept = 0;

    virtual ~IProfiler() noexcept {}
};
} // namespace v_1_0

//!
//! \class IProfiler
//!
//! \brief Application-implemented interface for profiling.
//!
//! When this class is added to an execution context, the profiler will be called once per layer for each invocation of
//! executeV2()/enqueueV3().
//!
//! It is not recommended to run inference with profiler enabled when the inference execution time is critical since the
//! profiler may affect execution time negatively.
//!
using IProfiler = v_1_0::IProfiler;

//!
//! \enum WeightsRole
//!
//! \brief How a layer uses particular Weights.
//!
//! The power weights of an IScaleLayer are omitted.  Refitting those is not supported.
//!
enum class WeightsRole : int32_t
{
    kKERNEL = 0,   //!< kernel for IConvolutionLayer or IDeconvolutionLayer
    kBIAS = 1,     //!< bias for IConvolutionLayer or IDeconvolutionLayer
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
    kGPU = 0, //!< GPU Device
    kDLA = 1, //!< DLA Core
};

//! Maximum number of elements in DeviceType enum. \see DeviceType
template <>
constexpr inline int32_t EnumMax<DeviceType>() noexcept
{
    return 2;
}

//!
//! \enum TempfileControlFlag
//!
//! \brief Flags used to control TensorRT's behavior when creating executable temporary files.
//!
//! On some platforms the TensorRT runtime may need to create files in a temporary directory or use platform-specific
//! APIs to create files in-memory to load temporary DLLs that implement runtime code. These flags allow the
//! application to explicitly control TensorRT's use of these files. This will preclude the use of certain TensorRT
//! APIs for deserializing and loading lean runtimes.
//!
enum class TempfileControlFlag : int32_t
{
    //! Allow creating and loading files in-memory (or unnamed files).
    kALLOW_IN_MEMORY_FILES = 0,

    //! Allow creating and loading named files in a temporary directory on the filesystem.
    //!
    //! \see IRuntime::setTemporaryDirectory()
    kALLOW_TEMPORARY_FILES = 1,
};

//! Maximum number of elements in TempfileControlFlag enum. \see TempfileControlFlag
template <>
constexpr inline int32_t EnumMax<TempfileControlFlag>() noexcept
{
    return 2;
}

//!
//! \brief Represents a collection of one or more TempfileControlFlag values combined using bitwise-OR operations.
//!
//! \see TempfileControlFlag,
//!      IRuntime::setTempfileControlFlags(),
//!      IRuntime::getTempfileControlFlags()
using TempfileControlFlags = uint32_t;

//!
//! \enum TensorFormat
//!
//! \brief Format of the input/output tensors.
//!
//! This enum is used by both plugins and network I/O tensors.
//!
//! \see IPluginV2::supportsFormat(), safe::ICudaEngine::getBindingFormat()
//!
//! Many of the formats are **vector-major** or **vector-minor**. These formats specify
//! a <em>vector dimension</em> and <em>scalars per vector</em>.
//! For example, suppose that the tensor has has dimensions [M,N,C,H,W],
//! the vector dimension is C and there are V scalars per vector.
//!
//! * A **vector-major** format splits the vectorized dimension into two axes in the
//!   memory layout. The vectorized dimension is replaced by an axis of length ceil(C/V)
//!   and a new dimension of length V is appended. For the example tensor, the memory layout
//!   is equivalent to an array with dimensions [M][N][ceil(C/V)][H][W][V].
//!   Tensor coordinate (m,n,c,h,w) maps to array location [m][n][c/V][h][w][c\%V].
//!
//! * A **vector-minor** format moves the vectorized dimension to become the last axis
//!   in the memory layout. For the example tensor, the memory layout is equivalent to an
//!   array with dimensions [M][N][H][W][ceil(C/V)*V]. Tensor coordinate (m,n,c,h,w) maps
//!   array location subscript [m][n][h][w][c].
//!
//! In interfaces that refer to "components per element", that's the value of V above.
//!
//! For more information about data formats, see the topic "Data Format Description" located in the
//! TensorRT Developer Guide. https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#data-format-desc
//!
enum class TensorFormat : int32_t
{
    //! Memory layout is similar to an array in C or C++.
    //! The stride of each dimension is the product of the dimensions after it.
    //! The last dimension has unit stride.
    //!
    //! For DLA usage, the tensor sizes are limited to C,H,W in the range [1,8192].
    kLINEAR = 0,

    //! Vector-major format with two scalars per vector.
    //! Vector dimension is third to last.
    //!
    //! This format requires FP16 and at least three dimensions.
    kCHW2 = 1,

    //! Vector-minor format with eight scalars per vector.
    //! Vector dimension is third to last.
    //! This format requires FP16 or BF16 and at least three dimensions.
    kHWC8 = 2,

    //! Vector-major format with four scalars per vector.
    //! Vector dimension is third to last.
    //!
    //! This format requires INT8 or FP16 and at least three dimensions.
    //! For INT8, the length of the vector dimension must be a build-time constant.
    //!
    //! Deprecated usage:
    //!
    //! If running on the DLA, this format can be used for acceleration
    //! with the caveat that C must be less than or equal to 4.
    //! If used as DLA input and the build option kGPU_FALLBACK is not specified,
    //! it needs to meet line stride requirement of DLA format. Column stride in
    //! bytes must be a multiple of 64 on Orin.
    kCHW4 = 3,

    //! Vector-major format with 16 scalars per vector.
    //! Vector dimension is third to last.
    //!
    //! This format requires FP16 and at least three dimensions.
    //!
    //! For DLA usage, this format maps to the native feature format for FP16,
    //! and the tensor sizes are limited to C,H,W in the range [1,8192].
    kCHW16 = 4,

    //! Vector-major format with 32 scalars per vector.
    //! Vector dimension is third to last.
    //!
    //! This format requires at least three dimensions.
    //!
    //! For DLA usage, this format maps to the native feature format for INT8,
    //! and the tensor sizes are limited to C,H,W in the range [1,8192].
    kCHW32 = 5,

    //! Vector-minor format with eight scalars per vector.
    //! Vector dimension is fourth to last.
    //!
    //! This format requires FP16 or BF16 and at least four dimensions.
    kDHWC8 = 6,

    //! Vector-major format with 32 scalars per vector.
    //! Vector dimension is fourth to last.
    //!
    //! This format requires FP16 or INT8 and at least four dimensions.
    kCDHW32 = 7,

    //! Vector-minor format where channel dimension is third to last and unpadded.
    //!
    //! This format requires either FP32, FP16, UINT8, INT64 or BF16 and at least three dimensions.
    kHWC = 8,

    //! DLA planar format. For a tensor with dimension {N, C, H, W}, the W axis
    //! always has unit stride. The stride for stepping along the H axis is
    //! rounded up to 64 bytes.
    //!
    //! The memory layout is equivalent to a C array with dimensions
    //! [N][C][H][roundUp(W, 64/elementSize)] where elementSize is
    //! 2 for FP16 and 1 for Int8, with the tensor coordinates (n, c, h, w)
    //! mapping to array subscript [n][c][h][w].
    kDLA_LINEAR = 9,

    //! DLA image format. For a tensor with dimension {N, C, H, W} the C axis
    //! always has unit stride. The stride for stepping along the H axis is rounded up
    //! to 64 bytes on Orin. C can only be 1, 3 or 4.
    //! If C == 1, it will map to grayscale format.
    //! If C == 3 or C == 4, it will map to color image format. And if C == 3,
    //! the stride for stepping along the W axis needs to be padded to 4 in elements.
    //!
    //! When C is {1, 3, 4}, then C' is {1, 4, 4} respectively,
    //! the memory layout is equivalent to a C array with dimensions
    //! [N][H][roundUp(W, 64/C'/elementSize)][C'] on Orin
    //! where elementSize is 2 for FP16
    //! and 1 for Int8. The tensor coordinates (n, c, h, w) mapping to array
    //! subscript [n][h][w][c].
    kDLA_HWC4 = 10,

    //! Vector-minor format with 16 scalars per vector.
    //! Vector dimension is third to last.
    //!
    //! This requires FP16 or INT8 and at least three dimensions.
    kHWC16 = 11,

    //! Vector-minor format with one scalar per vector.
    //! Vector dimension is fourth to last.
    //!
    //! This format requires FP32 and at least four dimensions.
    kDHWC = 12
};

namespace impl
{
//! Maximum number of elements in TensorFormat enum. \see TensorFormat
template <>
struct EnumMaxImpl<TensorFormat>
{
    //! Declaration of kVALUE that represents the maximum number of elements in the TensorFormat enum.
    static constexpr int32_t kVALUE = 13;
};
} // namespace impl

//!
//! \enum AllocatorFlag
//!
//! \brief Allowed type of memory allocation.
//!
enum class AllocatorFlag : int32_t
{
    //! TensorRT may call realloc() on this allocation.
    kRESIZABLE = 0,
};

namespace impl
{
//! Maximum number of elements in AllocatorFlag enum. \see AllocatorFlag
template <>
struct EnumMaxImpl<AllocatorFlag>
{
    //! Declaration of kVALUE that represents the maximum number of elements in the AllocatorFlag enum.
    static constexpr int32_t kVALUE = 1;
};
} // namespace impl

using AllocatorFlags = uint32_t;

//! DO NOT REFER TO namespace v_1_0 IN CODE. ALWAYS USE nvinfer1 INSTEAD.
//! The name v_1_0 may change in future versions of TensoRT.

//!
//! \class ILogger
//!
//! \brief Application-implemented logging interface for the builder, refitter and runtime.
//!
//! The logger used to create an instance of IBuilder, IRuntime or IRefitter is used for all objects created through
//! that interface. The logger must be valid until all objects created are released.
//!
//! The Logger object implementation must be thread safe. All locking and synchronization is pushed to the
//! interface implementation and TensorRT does not hold any synchronization primitives when calling the interface
//! functions.
//!
class ILogger
{
public:
    //!
    //! \enum Severity
    //!
    //! \brief The severity corresponding to a log message.
    //!
    enum class Severity : int32_t
    {
        //! An internal error has occurred. Execution is unrecoverable.
        kINTERNAL_ERROR = 0,
        //! An application error has occurred.
        kERROR = 1,
        //! An application error has been discovered, but TensorRT has recovered or fallen back to a default.
        kWARNING = 2,
        //!  Informational messages with instructional information.
        kINFO = 3,
        //!  Verbose messages with debugging information.
        kVERBOSE = 4,
    };

    //!
    //! \brief A callback implemented by the application to handle logging messages;
    //!
    //! \param severity The severity of the message.
    //! \param msg A null-terminated log message.
    //!
    //! \warning Loggers used in the safety certified runtime must set a maximum message length and truncate
    //!          messages exceeding this length. It is up to the implementer of the derived class to define
    //!          a suitable limit that will prevent buffer overruns, resource exhaustion, and other security
    //!          vulnerabilities in their implementation. The TensorRT safety certified runtime will never
    //!          emit messages longer than 1024 bytes.
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: Yes, this method is required to be thread-safe and may be called from multiple threads
    //!                  when multiple execution contexts are used during runtime, or if the same logger is used
    //!                  for multiple runtimes, builders, or refitters.
    //!
    virtual void log(Severity severity, AsciiChar const* msg) noexcept = 0;

    ILogger() = default;
    virtual ~ILogger() = default;

protected:
    // @cond SuppressDoxyWarnings
    ILogger(ILogger const&) = default;
    ILogger(ILogger&&) = default;
    ILogger& operator=(ILogger const&) & = default;
    ILogger& operator=(ILogger&&) & = default;
    // @endcond
};

namespace impl
{
//! Maximum number of elements in ILogger::Severity enum. \see ILogger::Severity
template <>
struct EnumMaxImpl<ILogger::Severity>
{
    //! Declaration of kVALUE that represents the maximum number of elements in the ILogger::Severity enum.
    static constexpr int32_t kVALUE = 5;
};
} // namespace impl

namespace v_1_0
{

class IGpuAllocator : public IVersionedInterface
{
public:
    //!
    //! \brief A thread-safe callback implemented by the application to handle acquisition of GPU memory.
    //!
    //! \param size The size of the memory block required (in bytes).
    //! \param alignment The required alignment of memory. Alignment will be zero
    //!        or a power of 2 not exceeding the alignment guaranteed by cudaMalloc.
    //!        Thus this allocator can be safely implemented with cudaMalloc/cudaFree.
    //!        An alignment value of zero indicates any alignment is acceptable.
    //! \param flags Reserved for future use. In the current release, 0 will be passed.
    //!
    //! \return If the allocation was successful, the start address of a device memory block of the requested size.
    //! If an allocation request of size 0 is made, nullptr must be returned.
    //! If an allocation request cannot be satisfied, nullptr must be returned.
    //! If a non-null address is returned, it is guaranteed to have the specified alignment.
    //!
    //! \note The implementation must guarantee thread safety for concurrent allocate/reallocate/deallocate
    //! requests.
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: Yes, this method is required to be thread-safe and may be called from multiple threads.
    //!
    //! \deprecated Deprecated in TensorRT 10.0. Superseded by allocateAsync
    //!
    TRT_DEPRECATED virtual void* allocate(
        uint64_t const size, uint64_t const alignment, AllocatorFlags const flags) noexcept = 0;

    ~IGpuAllocator() override = default;
    IGpuAllocator() = default;

    //!
    //! \brief A thread-safe callback implemented by the application to resize an existing allocation.
    //!
    //! Only allocations which were allocated with AllocatorFlag::kRESIZABLE will be resized.
    //!
    //! Options are one of:
    //! * resize in place leaving min(oldSize, newSize) bytes unchanged and return the original address
    //! * move min(oldSize, newSize) bytes to a new location of sufficient size and return its address
    //! * return nullptr, to indicate that the request could not be fulfilled.
    //!
    //! If nullptr is returned, TensorRT will assume that resize() is not implemented, and that the
    //! allocation at baseAddr is still valid.
    //!
    //! This method is made available for use cases where delegating the resize
    //! strategy to the application provides an opportunity to improve memory management.
    //! One possible implementation is to allocate a large virtual device buffer and
    //! progressively commit physical memory with cuMemMap. CU_MEM_ALLOC_GRANULARITY_RECOMMENDED
    //! is suggested in this case.
    //!
    //! TensorRT may call realloc to increase the buffer by relatively small amounts.
    //!
    //! \param baseAddr the address of the original allocation, which will have been returned by previously calling
    //!        allocate() or reallocate() on the same object.
    //! \param alignment The alignment used by the original allocation. This will be the same value that was previously
    //!        passed to the allocate() or reallocate() call that returned baseAddr.
    //! \param newSize The new memory size required (in bytes).
    //!
    //! \return The address of the reallocated memory, or nullptr. If a non-null address is returned, it is
    //!         guaranteed to have the specified alignment.
    //!
    //! \note The implementation must guarantee thread safety for concurrent allocate/reallocate/deallocate
    //! requests.
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: Yes, this method is required to be thread-safe and may be called from multiple threads.
    //!
    virtual void* reallocate(void* const /*baseAddr*/, uint64_t /*alignment*/, uint64_t /*newSize*/) noexcept
    {
        return nullptr;
    }

    //!
    //! \brief A thread-safe callback implemented by the application to handle release of GPU memory.
    //!
    //! TensorRT may pass a nullptr to this function if it was previously returned by allocate().
    //!
    //! \param memory A memory address that was previously returned by an allocate() or reallocate() call of the same
    //! allocator object.
    //!
    //! \return True if the acquired memory is released successfully.
    //!
    //! \note The implementation must guarantee thread safety for concurrent allocate/reallocate/deallocate
    //! requests.
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: Yes, this method is required to be thread-safe and may be called from multiple threads.
    //! \deprecated Deprecated in TensorRT 10.0. Superseded by deallocateAsync
    //!
    TRT_DEPRECATED virtual bool deallocate(void* const memory) noexcept = 0;

    //!
    //! \brief A thread-safe callback implemented by the application to handle stream-ordered acquisition of GPU memory.
    //!
    //! The default behavior is to call method allocate(), which is synchronous and thus loses
    //! any performance benefits of asynchronous allocation. If you want the benefits of asynchronous
    //! allocation, see discussion of IGpuAsyncAllocator vs. IGpuAllocator in the documentation
    //! for nvinfer1::IGpuAllocator.
    //!
    //! \param size The size of the memory block required (in bytes).
    //! \param alignment The required alignment of memory. Alignment will be zero
    //!        or a power of 2 not exceeding the alignment guaranteed by cudaMalloc.
    //!        Thus this allocator can be safely implemented with cudaMalloc/cudaFree.
    //!        An alignment value of zero indicates any alignment is acceptable.
    //! \param flags Reserved for future use. In the current release, 0 will be passed.
    //! \param stream specifies the cudaStream for asynchronous usage.
    //!
    //! \return If the allocation was successful, the start address of a device memory block of the requested size.
    //! If an allocation request of size 0 is made, nullptr must be returned.
    //! If an allocation request cannot be satisfied, nullptr must be returned.
    //! If a non-null address is returned, it is guaranteed to have the specified alignment.
    //!
    //! \note The implementation must guarantee thread safety for concurrent allocate/reallocate/deallocate
    //! requests.
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: Yes, this method is required to be thread-safe and may be called from multiple threads.
    //!
    virtual void* allocateAsync(
        uint64_t const size, uint64_t const alignment, AllocatorFlags const flags, cudaStream_t /*stream*/) noexcept
    {
        return allocate(size, alignment, flags);
    }
    //!
    //! \brief A thread-safe callback implemented by the application to handle stream-ordered release of GPU memory.
    //!
    //! The default behavior is to call method deallocate(), which is synchronous and thus loses
    //! any performance benefits of asynchronous deallocation. If you want the benefits of asynchronous
    //! deallocation, see discussion of IGpuAsyncAllocator vs. IGpuAllocator in the documentation
    //! for nvinfer1::IGpuAllocator.
    //!
    //! TensorRT may pass a nullptr to this function if it was previously returned by allocate().
    //!
    //! \param memory A memory address that was previously returned by an allocate() or reallocate() call of the same
    //! allocator object.
    //! \param stream specifies the cudaStream for asynchronous usage.
    //!
    //! \return True if the acquired memory is released successfully.
    //!
    //! \note The implementation must guarantee thread safety for concurrent allocate/reallocate/deallocate
    //! requests.
    //!
    //! \note The implementation is not required to be asynchronous. It is permitted to synchronize,
    //! albeit doing so will lose the performance advantage of asynchronous deallocation.
    //! Either way, it is critical that it not actually free the memory until the current
    //! stream position is reached.
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: Yes, this method is required to be thread-safe and may be called from multiple threads.
    //!
    virtual bool deallocateAsync(void* const memory, cudaStream_t /*stream*/) noexcept
    {
        return deallocate(memory);
    }

    //!
    //! \brief Return version information associated with this interface. Applications must not override this method.
    //!
    InterfaceInfo getInterfaceInfo() const noexcept override
    {
        return {"IGpuAllocator", 1, 0};
    }

protected:
    // @cond SuppressDoxyWarnings
    IGpuAllocator(IGpuAllocator const&) = default;
    IGpuAllocator(IGpuAllocator&&) = default;
    IGpuAllocator& operator=(IGpuAllocator const&) & = default;
    IGpuAllocator& operator=(IGpuAllocator&&) & = default;
    // @endcond
};

} // namespace v_1_0

//!
//! \class IGpuAllocator
//!
//! \brief Application-implemented class for controlling allocation on the GPU.
//!
//! \warning The lifetime of an IGpuAllocator object must exceed that of all objects that use it.
//!
//! This class is intended as a base class for allocators that implement synchronous allocation.
//! If you want the benefits of asynchronous allocation, you can do either of:
//!
//! * Derive your class from IGpuAllocator and override all four of its virtual methods
//!   for allocation/deallocation, including the two deprecated methods.
//!
//! * Derive your class from IGpuAsyncAllocator and override its two pure virtual
//!   methods for allocation/deallocation.
//!
//! The latter style is preferred because it does not tie code to deprecated methods.
//!
//! \see IGpuAsyncAllocator.
//!
using IGpuAllocator = v_1_0::IGpuAllocator;

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
    //! \brief Sets the DLA core used by the network. Defaults to -1.
    //!
    //! \param dlaCore The DLA core to execute the engine on, in the range [0,getNbDlaCores()).
    //!
    //! This function is used to specify which DLA core to use via indexing, if multiple DLA cores are available.
    //!
    //! \warning if getNbDLACores() returns 0, then this function does nothing.
    //!
    //! \see getDLACore()
    //!
    void setDLACore(int32_t dlaCore) noexcept
    {
        mImpl->setDLACore(dlaCore);
    }

    //!
    //! \brief Get the DLA core that the engine executes on.
    //!
    //! \return assigned DLA core or -1 for DLA not present or unset.
    //!
    int32_t getDLACore() const noexcept
    {
        return mImpl->getDLACore();
    }

    //!
    //! \brief Returns number of DLA hardware cores accessible or 0 if DLA is unavailable.
    //!
    int32_t getNbDLACores() const noexcept
    {
        return mImpl->getNbDLACores();
    }

    //!
    //! \brief Set the GPU allocator.
    //!
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
    //! \brief Deserialize an engine from host memory.
    //!
    //! If an error recorder has been set for the runtime, it will also be passed to the engine.
    //!
    //! \param blob The memory that holds the serialized engine.
    //! \param size The size of the memory.
    //!
    //! \return The engine, or nullptr if it could not be deserialized.
    //!
    ICudaEngine* deserializeCudaEngine(void const* blob, std::size_t size) noexcept
    {
        return mImpl->deserializeCudaEngine(blob, size);
    }

    //!
    //! \brief Deserialize an engine from a stream.
    //!
    //! If an error recorder has been set for the runtime, it will also be passed to the
    //! engine.
    //!
    //! This deserialization path will reduce host memory usage when weight streaming is enabled.
    //!
    //! \param streamReader a read-only stream from which TensorRT will deserialize a
    //!        previously serialized engine.
    //!
    //! \return The engine, or nullptr if it could not be deserialized.
    //!
    //! \deprecated Deprecated in TensorRT 10.7. Superseded by deserializeCudaEngine that takes an IStreamReaderV2
    //! instead of IStreamReader.
    //!
    TRT_DEPRECATED ICudaEngine* deserializeCudaEngine(IStreamReader& streamReader)
    {
        return mImpl->deserializeCudaEngine(streamReader);
    }

    //!
    //! \brief Deserialize an engine from a stream. IStreamReaderV2 is expected to support reading to both host and
    //! device pointers.
    //!
    //! If an error recorder has been set for the runtime, it will also be passed to the
    //! engine.
    //!
    //! This deserialization path will reduce engine load time when applied with GDS (GPU Direct storage), or when
    //! weight streaming is enabled.
    //!
    //! \param streamReader a read-only stream from which TensorRT will deserialize a previously serialized engine.
    //! \param stream The CUDA stream used when performing asynchronous I/O.
    //!
    //! \return The engine, or nullptr if it could not be deserialized. The pointer may not be valid immediately after
    //! the function returns.
    //!
    ICudaEngine* deserializeCudaEngine(IStreamReaderV2& streamReader)
    {
        return mImpl->deserializeCudaEngineV2(streamReader);
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

    //!
    //! \brief Set the maximum number of threads.
    //!
    //! \param maxThreads The maximum number of threads that can be used by the runtime.
    //! \return True if successful, false otherwise.
    //!
    //! The default value is 1 and includes the current thread.
    //! A value greater than 1 permits TensorRT to use multi-threaded algorithms.
    //! A value less than 1 triggers a kINVALID_ARGUMENT error.
    //!
    bool setMaxThreads(int32_t maxThreads) noexcept
    {
        return mImpl->setMaxThreads(maxThreads);
    }

    //!
    //! \brief Get the maximum number of threads that can be used by the runtime.
    //!
    //! Retrieves the maximum number of threads that can be used by the runtime.
    //!
    //! \return The maximum number of threads that can be used by the runtime.
    //!
    //! \see setMaxThreads()
    //!
    int32_t getMaxThreads() const noexcept
    {
        return mImpl->getMaxThreads();
    }

    //!
    //! \brief Set the directory that will be used by this runtime for temporary files.
    //!
    //! On some platforms the TensorRT runtime may need to create and use temporary files
    //! with read/write/execute permissions to implement runtime functionality.
    //!
    //! \param path Path to the temporary directory for use, or nullptr.
    //!
    //! If path is nullptr, then TensorRT will use platform-specific heuristics to pick
    //! a default temporary directory if required:
    //!
    //! - On UNIX/Linux platforms, TensorRT will first try the TMPDIR environment variable, then fall back to /tmp
    //! - On Windows, TensorRT will try the TEMP environment variable.
    //!
    //! See the TensorRT Developer Guide for more information.
    //!
    //! The default value is nullptr.
    //!
    //! \warning If path is not nullptr, it must be a non-empty string representing a relative
    //! or absolute path in the format expected by the host operating system.
    //!
    //! \warning The string path must be null-terminated, and be at most 4096 bytes including the
    //! terminator. Note that the operating system may have stricter path length requirements.
    //!
    //! \warning The process using TensorRT must have rwx permissions for the temporary directory,
    //! and the directory shall be configured to disallow other users from modifying created files
    //! (e.g. on Linux, if the directory is shared with other users, the sticky bit must be set).
    //!
    //! \see getTemporaryDirectory()
    //!
    void setTemporaryDirectory(char const* path) noexcept
    {
        return mImpl->setTemporaryDirectory(path);
    }

    //!
    //! \brief Get the directory that will be used by this runtime for temporary files.
    //!
    //! \returns A path to the temporary directory in use, or nullptr if no path is specified.
    //!
    //! \see setTemporaryDirectory()
    char const* getTemporaryDirectory() const noexcept
    {
        return mImpl->getTemporaryDirectory();
    }

    //!
    //! \brief Set the tempfile control flags for this runtime.
    //!
    //! \param flags The flags to set.
    //!
    //! The default value is all flags set, i.e.
    //!
    //! (1U << static_cast<uint32_t>(kALLOW_IN_MEMORY_FILES)) | (1U << static_cast<uint32_t>(kALLOW_TEMPORARY_FILES))
    //!
    //! \see TempfileControlFlag, TempfileControlFlags, getTempfileControlFlags()
    //!
    void setTempfileControlFlags(TempfileControlFlags flags) noexcept
    {
        return mImpl->setTempfileControlFlags(flags);
    }

    //!
    //! \brief Get the tempfile control flags for this runtime.
    //!
    //! \return The flags currently set.
    //!
    //! \see TempfileControlFlag, TempfileControlFlags, setTempfileControlFlags()
    //!
    TempfileControlFlags getTempfileControlFlags() const noexcept
    {
        return mImpl->getTempfileControlFlags();
    }

    //!
    //! \brief Get the local plugin registry that can be used by the runtime.
    //!
    //! \return The local plugin registry that can be used by the runtime.
    //!
    IPluginRegistry& getPluginRegistry() noexcept
    {
        return mImpl->getPluginRegistry();
    }

    //!
    //! \brief Load IRuntime from the file.
    //!
    //! This method loads a runtime library from a shared library file. The runtime can then be used to execute
    //! a plan file built with BuilderFlag::kVERSION_COMPATIBLE and BuilderFlag::kEXCLUDE_LEAN_RUNTIME both set
    //! and built with the same version of TensorRT as the loaded runtime library.
    //!
    //! \param path Path to the runtime lean library.
    //!
    //! \return the runtime library, or nullptr if it could not be loaded
    //!
    //! \warning The path string must be null-terminated, and be at most 4096 bytes including the terminator.
    //!
    IRuntime* loadRuntime(char const* path) noexcept
    {
        return mImpl->loadRuntime(path);
    }

    //!
    //! \brief Set whether the runtime is allowed to deserialize engines with host executable code.
    //!
    //! \param allowed Whether the runtime is allowed to deserialize engines with host executable code.
    //!
    //! The default value is false.
    //!
    void setEngineHostCodeAllowed(bool allowed) noexcept
    {
        return mImpl->setEngineHostCodeAllowed(allowed);
    }

    //!
    //! \brief Get whether the runtime is allowed to deserialize engines with host executable code.
    //!
    //! \return Whether the runtime is allowed to deserialize engines with host executable code.
    //!
    bool getEngineHostCodeAllowed() const noexcept
    {
        return mImpl->getEngineHostCodeAllowed();
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
    //! * The count of weights is inconsistent with the layer’s original specification.
    //! * The type of weights is inconsistent with the layer’s original specification.
    //!
    //! Modifying the weights before method refitCudaEngine or refitCudaEngineAsync returns will result in undefined
    //! behavior.
    //!
    //! \warning The string layerName must be null-terminated, and be at most 4096 bytes including the terminator.
    //!
    bool setWeights(char const* layerName, WeightsRole role, Weights weights) noexcept
    {
        return mImpl->setWeights(layerName, role, weights);
    }

    //!
    //! \brief Refits associated engine.
    //!
    //! \return True on success, or false if new weights validation fails or getMissingWeights() != 0 before the call.
    //! If false is returned, a subset of weights may have been refitted.
    //!
    //! The behavior is undefined if the engine has pending enqueued work.
    //! Provided weights on CPU or GPU can be unset and released, or updated after refitCudaEngine returns.
    //!
    //! IExecutionContexts associated with the engine remain valid for use afterwards. There is no need to set the same
    //! weights repeatedly for multiple refit calls as the weights memory can be updated directly instead.
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
    int32_t getMissing(int32_t size, char const** layerNames, WeightsRole* roles) noexcept
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
    int32_t getAll(int32_t size, char const** layerNames, WeightsRole* roles) noexcept
    {
        return mImpl->getAll(size, layerNames, roles);
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
    //!
    //! \warning The string tensorName must be null-terminated, and be at most 4096 bytes including the terminator.
    //!
    //! \deprecated Deprecated in TensorRT 10.1. Superseded by explicit quantization.
    //!
    TRT_DEPRECATED bool setDynamicRange(char const* tensorName, float min, float max) noexcept
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
    //! \warning The string tensorName must be null-terminated, and be at most 4096 bytes including the terminator.
    //!
    //! \deprecated Deprecated in TensorRT 10.1. Superseded by explicit quantization.
    //!
    TRT_DEPRECATED float getDynamicRangeMin(char const* tensorName) const noexcept
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
    //! \warning The string tensorName must be null-terminated, and be at most 4096 bytes including the terminator.
    //!
    //! \deprecated Deprecated in TensorRT 10.1. Superseded by explicit quantization.
    //!
    TRT_DEPRECATED float getDynamicRangeMax(char const* tensorName) const noexcept
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
    //! \deprecated Deprecated in TensorRT 10.1. Superseded by explicit quantization.
    //!
    TRT_DEPRECATED int32_t getTensorsWithDynamicRange(int32_t size, char const** tensorNames) const noexcept
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
    //! * The count of the weights is inconsistent with the count returned from calling getWeightsPrototype() with the
    //! same name.
    //! * The type of the weights is inconsistent with the type returned from calling getWeightsPrototype() with the
    //! same name.
    //!
    //! Modifying the weights before method refitCudaEngine or refitCudaEngineAsync returns will result in undefined
    //! behavior.
    //!
    //! \warning The string name must be null-terminated, and be at most 4096 bytes including the terminator.
    //!
    bool setNamedWeights(char const* name, Weights weights) noexcept
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
    int32_t getMissingWeights(int32_t size, char const** weightsNames) noexcept
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
    int32_t getAllWeights(int32_t size, char const** weightsNames) noexcept
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

    //!
    //! \brief Set the maximum number of threads.
    //!
    //! \param maxThreads The maximum number of threads that can be used by the refitter.
    //!
    //! \return True if successful, false otherwise.
    //!
    //! The default value is 1 and includes the current thread.
    //! A value greater than 1 permits TensorRT to use multi-threaded algorithms.
    //! A value less than 1 triggers a kINVALID_ARGUMENT error.
    //!
    bool setMaxThreads(int32_t maxThreads) noexcept
    {
        return mImpl->setMaxThreads(maxThreads);
    }

    //!
    //! \brief get the maximum number of threads that can be used by the refitter.
    //!
    //! Retrieves the maximum number of threads that can be used by the refitter.
    //!
    //! \return The maximum number of threads that can be used by the refitter.
    //!
    //! \see setMaxThreads()
    //!
    int32_t getMaxThreads() const noexcept
    {
        return mImpl->getMaxThreads();
    }

    //!
    //! \brief Specify new weights on a specified device of given name.
    //!
    //! \param name The name of the weights to be refitted.
    //! \param weights The new weights on the specified device.
    //! \param location The location (host vs. device) of the new weights.
    //!
    //! \return True on success, or false if new weights are rejected.
    //! Possible reasons for rejection are:
    //!
    //! * The name of the weights is nullptr or does not correspond to any refittable weights.
    //! * The count of the weights is inconsistent with the count returned from calling getWeightsPrototype() with the
    //! same name.
    //! * The type of the weights is inconsistent with the type returned from calling getWeightsPrototype() with the
    //! same name.
    //!
    //! It is allowed to provide some weights on CPU and others on GPU.
    //! Modifying the weights before the method refitCudaEngine() or refitCudaEngineAsync() completes will result in
    //! undefined behavior.
    //!
    //! \warning The string name must be null-terminated, and be at most 4096 bytes including the terminator.
    //!
    bool setNamedWeights(char const* name, Weights weights, TensorLocation location) noexcept
    {
        return mImpl->setNamedWeightsWithLocation(name, weights, location);
    }

    //!
    //! \brief Get weights associated with the given name.
    //!
    //! \param weightsName The name of the weights to be refitted.
    //!
    //! \return Weights associated with the given name.
    //!
    //! If the weights were never set, returns null weights and reports an error to the refitter errorRecorder.
    //!
    //! \warning The string weightsName must be null-terminated, and be at most 4096 bytes including the terminator.
    //!
    Weights getNamedWeights(char const* weightsName) const noexcept
    {
        return mImpl->getNamedWeights(weightsName);
    }

    //!
    //! \brief Get location for the weights associated with the given name.
    //!
    //! \param weightsName The name of the weights to be refitted.
    //!
    //! \return Location for the weights associated with the given name.
    //!
    //! If the weights were never set, returns TensorLocation::kHOST and reports an error to the refitter errorRecorder.
    //!
    //! \warning The string weightsName must be null-terminated, and be at most 4096 bytes including the terminator.
    //!
    TensorLocation getWeightsLocation(char const* weightsName) const noexcept
    {
        return mImpl->getWeightsLocation(weightsName);
    }

    //!
    //! \brief Unset weights associated with the given name.
    //!
    //! \param weightsName The name of the weights to be refitted.
    //!
    //! \return False if the weights were never set, returns true otherwise.
    //!
    //! Unset weights before releasing them.
    //!
    //! \warning The string weightsName must be null-terminated, and be at most 4096 bytes including the terminator.
    //!
    bool unsetNamedWeights(char const* weightsName) noexcept
    {
        return mImpl->unsetNamedWeights(weightsName);
    }

    //!
    //! \brief Set whether to validate weights during refitting.
    //!
    //! \param weightsValidation Indicate whether to validate weights during refitting.
    //!
    //! When set to true, TensorRT will validate weights during FP32 to FP16/BF16 weights conversions or
    //! sparsifying weights in the refit call. If provided weights are not proper for some weights transformations,
    //! TensorRT will issue a warning and continue the transformation for minor issues (such as overflow during
    //! narrowing conversion), or issue an error and stop the refitting process for severe issues (such as sparsifying
    //! dense weights). By default the flag is true. Set the flag to false for faster refitting performance.
    //!
    void setWeightsValidation(bool weightsValidation) noexcept
    {
        return mImpl->setWeightsValidation(weightsValidation);
    }

    //!
    //! \brief Get whether to validate weights values during refitting.
    //!
    bool getWeightsValidation() const noexcept
    {
        return mImpl->getWeightsValidation();
    }

    //!
    //! \brief Enqueue weights refitting of the associated engine on the given stream.
    //!
    //! \param stream The stream to enqueue the weights updating task.
    //!
    //! \return True on success, or false if new weights validation fails or getMissingWeights() != 0 before the call.
    //! If false is returned, a subset of weights may have been refitted.
    //!
    //! The behavior is undefined if the engine has pending enqueued work on a different stream from the provided one.
    //! Provided weights on CPU can be unset and released, or updated after refitCudaEngineAsync returns.
    //! Freeing or updating of the provided weights on GPU can be enqueued on the same stream after refitCudaEngineAsync
    //! returns.
    //!
    //! IExecutionContexts associated with the engine remain valid for use afterwards. There is no need to set the same
    //! weights repeatedly for multiple refit calls as the weights memory can be updated directly instead. The weights
    //! updating task should use the same stream as the one used for the refit call.
    //!
    bool refitCudaEngineAsync(cudaStream_t stream) noexcept
    {
        return mImpl->refitCudaEngineAsync(stream);
    }

    //!
    //! \brief Get the Weights prototype associated with the given name.
    //!
    //! \param weightsName The name of the weights to be refitted.
    //!
    //! \return Weights prototype associated with the given name.
    //!
    //! The type and count of weights prototype is the same as weights used for engine building. The values property
    //! is nullptr for weights prototypes. The count of the weights prototype is -1 when the name of the weights is
    //! nullptr or does not correspond to any refittable weights.
    //!
    //! \warning The string weightsName must be null-terminated, and be at most 4096 bytes including the terminator.
    //!
    Weights getWeightsPrototype(char const* weightsName) const noexcept
    {
        return mImpl->getWeightsPrototype(weightsName);
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

//!
//! \brief Number of different values of OptProfileSelector enum.
//!
//! \see OptProfileSelector
//!
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
    //! \warning The string inputName must be null-terminated, and be at most 4096 bytes including the terminator.
    //!
    bool setDimensions(char const* inputName, OptProfileSelector select, Dims const& dims) noexcept
    {
        return mImpl->setDimensions(inputName, select, dims);
    }

    //!
    //! \brief Get the minimum / optimum / maximum dimensions for a dynamic input tensor.
    //!
    //! If the dimensions have not been previously set via setDimensions(), return an invalid Dims with nbDims == -1.
    //!
    //! \warning The string inputName must be null-terminated, and be at most 4096 bytes including the terminator.
    //!
    Dims getDimensions(char const* inputName, OptProfileSelector select) const noexcept
    {
        return mImpl->getDimensions(inputName, select);
    }

    //!
    //! \brief Set the minimum / optimum / maximum values for an input shape tensor.
    //!
    //! This function must be called three times for every input tensor t that is a shape tensor (t.isShape() == true).
    //! This implies that the dimensions of t are fixed at network definition time and the volume does not exceed 64.
    //! This function must not be called for any input tensor that is not a shape tensor.
    //!
    //! Each time this function is called for the same input tensor, the same nbValues must be supplied (either 1
    //! if the tensor rank is 0, or dims.d[0] if the rank is 1). Furthermore, if minVals, optVals, maxVals are the
    //! minimum, optimum, and maximum values, it must be true that minVals[i] <= optVals[i] <= maxVals[i] for
    //! i = 0, ..., nbValues - 1. Execution of the network must be valid for the optVals.
    //!
    //! Shape tensors are tensors that contribute to shape calculations in some way. While input shape tensors can be
    //! type kINT32 or kINT64, the values used to set the minimum, optimium, and maximum values must fit in int32_t.
    //!
    //! Examples:
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
    //!               For multidimensional tensors, the array is in row-major order.
    //! \param nbValues The length of the value array, which must equal the number of shape tensor elements (>= 1)
    //!
    //! \return false if an inconsistency was detected (e.g. nbValues does not match a previous call for the same
    //!         tensor), else true. As for setDimensions(), a full validation can only be performed at engine build
    //!         time.
    //!
    //! \warning If run on DLA, minimum, optimum, and maximum shape values must to be the same.
    //!
    //! \warning The string inputName must be null-terminated, and be at most 4096 bytes including the terminator.
    //!
    bool setShapeValues(
        char const* inputName, OptProfileSelector select, int32_t const* values, int32_t nbValues) noexcept
    {
        return mImpl->setShapeValues(inputName, select, values, nbValues);
    }

    //!
    //! \brief Get the number of values for an input shape tensor.
    //!
    //! This will return the number of shape values if setShapeValues() has been called before for this input tensor.
    //! Otherwise, return -1.
    //!
    //! \warning The string inputName must be null-terminated, and be at most 4096 bytes including the terminator.
    //!
    int32_t getNbShapeValues(char const* inputName) const noexcept
    {
        return mImpl->getNbShapeValues(inputName);
    }

    //!
    //! \brief Get the minimum / optimum / maximum values for an input shape tensor.
    //!
    //! If the shape values have not been set previously with setShapeValues(), this returns nullptr.
    //!
    //! \warning The string inputName must be null-terminated, and be at most 4096 bytes including the terminator.
    //!
    int32_t const* getShapeValues(char const* inputName, OptProfileSelector select) const noexcept
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
    //! \return true if the input is in the valid range (between 0 and 1 inclusive), else false.
    //!
    bool setExtraMemoryTarget(float target) noexcept
    {
        return mImpl->setExtraMemoryTarget(target);
    }

    //!
    //! \brief Get the extra memory target that has been defined for this profile.
    //!
    //! This defaults to 1.0F.
    //!
    //! \return the valid value set by setExtraMemoryTarget or 1.0F.
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
    //!
    //! \return true if the optimization profile is valid and may be passed to an IBuilderConfig, else false.
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
    //! cuBLAS tactics. Disabled by default.
    //! \note Disabling kCUBLAS will cause the cuBLAS handle passed to plugins in attachToContext to be null.
    //! \deprecated Deprecated in TensorRT 10.0.
    kCUBLAS TRT_DEPRECATED_ENUM = 0,

    //! cuBLAS LT tactics. Disabled by default.
    //! \deprecated Deprecated in TensorRT 9.0.
    kCUBLAS_LT TRT_DEPRECATED_ENUM = 1,

    //! cuDNN tactics. Disabled by default.
    //! \note Disabling kCUDNN will cause the cuDNN handle passed to plugins in attachToContext to be null.
    //! \deprecated Deprecated in TensorRT 10.0.
    kCUDNN TRT_DEPRECATED_ENUM = 2,

    //! Enables convolution tactics implemented with edge mask tables. These tactics tradeoff memory for performance by
    //! consuming additional memory space proportional to the input size.
    //! Enabled by default.
    kEDGE_MASK_CONVOLUTIONS = 3,

    //! Enables convolution tactics implemented with source-code JIT fusion. The engine building time may increase
    //! when this is enabled. Enabled by default.
    kJIT_CONVOLUTIONS = 4,
};

template <>
constexpr inline int32_t EnumMax<TacticSource>() noexcept
{
    return 5;
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
};

//! Maximum number of profile verbosity levels in ProfilingVerbosity enum. \see ProfilingVerbosity
template <>
constexpr inline int32_t EnumMax<ProfilingVerbosity>() noexcept
{
    return 3;
}

//!
//! \brief Represents one or more SerializationFlag values using binary OR
//! operations, e.g., 1U << SerializationFlag::kEXCLUDE_LEAN_RUNTIME
//!
//! \see ISerializationConfig::setFlags(), ISerializationConfig::getFlags()
//!
using SerializationFlags = uint32_t;

//!
//! \enum SerializationFlag
//!
//! \brief List of valid flags that the engine can enable when serializing the bytes.
//!
//! \see ISerializationConfig::setFlags(), ISerializationConfig::getFlags()
//!
enum class SerializationFlag : int32_t
{
    kEXCLUDE_WEIGHTS = 0,      //!< Exclude the weights that can be refitted.
    kEXCLUDE_LEAN_RUNTIME = 1, //!< Exclude the lean runtime.
};

//! Maximum number of serialization flags in SerializationFlag enum. \see SerializationFlag
template <>
constexpr inline int32_t EnumMax<SerializationFlag>() noexcept
{
    return 2;
}

//!
//! \class ISerializationConfig
//!
//! \brief Holds properties for configuring an engine to serialize the binary.
//!
//! \see SerializationFlag
//!
class ISerializationConfig : public INoCopy
{
public:
    virtual ~ISerializationConfig() noexcept = default;

    //!
    //! \brief Set the serialization flags to turn on for this config.
    //!
    //! The flags are listed in the SerializationFlag enum.
    //!
    //! \param serializationFlags The serialization flags for an engine.
    //!
    //! \note This function will override the previous set flags, rather than bitwise ORing the new flag.
    //!
    //! \see getFlags()
    //!
    bool setFlags(SerializationFlags serializationFlags) noexcept
    {
        return mImpl->setFlags(serializationFlags);
    }

    //!
    //! \brief Get the serialization flags for this config.
    //!
    //! \return The serialization flags as a bitmask.
    //!
    //! \see setFlags()
    //!
    SerializationFlags getFlags() const noexcept
    {
        return mImpl->getFlags();
    }

    //!
    //! \brief clear a serialization flag.
    //!
    //! clears the serialization flag from the config.
    //!
    //! \see setFlags()
    //!
    bool clearFlag(SerializationFlag serializationFlag) noexcept
    {
        return mImpl->clearFlag(serializationFlag);
    }

    //!
    //! \brief Set a serialization flag.
    //!
    //! Add the input serialization flag to the already enabled flags.
    //!
    //! \see setFlags()
    //!
    bool setFlag(SerializationFlag serializationFlag) noexcept
    {
        return mImpl->setFlag(serializationFlag);
    }

    //!
    //! \brief Returns true if the serialization flag is set
    //!
    //! \see getFlags()
    //!
    //! \return True if flag is set, false if unset.
    //!
    bool getFlag(SerializationFlag serializationFlag) const noexcept
    {
        return mImpl->getFlag(serializationFlag);
    }

protected:
    apiv::VSerializationConfig* mImpl;
};

//!
//! \enum ExecutionContextAllocationStrategy
//!
//! \brief Different memory allocation behaviors for IExecutionContext.
//!
//! IExecutionContext requires a block of device memory for internal activation tensors during inference. The user can
//! either let the execution context manage the memory in various ways or allocate the memory themselves.
//!
//! \see ICudaEngine::createExecutionContext()
//! \see IExecutionContext::setDeviceMemory()
//!
enum class ExecutionContextAllocationStrategy : int32_t
{
    kSTATIC = 0,            //!< Default static allocation with the maximum size across all profiles.
    kON_PROFILE_CHANGE = 1, //!< Reallocate for a profile when it's selected.
    kUSER_MANAGED = 2,      //!< The user supplies custom allocation to the execution context.
};

//!
//! \brief Maximum number of memory allocation strategies in ExecutionContextAllocationStrategy enum.
//!
//! \see ExecutionContextAllocationStrategy
//!
template <>
constexpr inline int32_t EnumMax<ExecutionContextAllocationStrategy>() noexcept
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
    //! \brief Get shape of an input or output tensor.
    //!
    //! \param tensorName The name of an input or output tensor.
    //!
    //! \return shape of the tensor, with -1 in place of each dynamic runtime dimension,
    //!         or Dims{-1, {}} if the provided name does not map to an input or output tensor.
    //!
    //! \warning The string tensorName must be null-terminated, and be at most 4096 bytes including the terminator.
    //!
    Dims getTensorShape(char const* tensorName) const noexcept
    {
        return mImpl->getTensorShape(tensorName);
    }

    //!
    //! \brief Determine the required data type for a buffer from its tensor name.
    //!
    //! \param tensorName The name of an input or output tensor.
    //!
    //! \return The type of the data in the buffer, or DataType::kFLOAT if the provided name does not map to an input or
    //! output tensor.
    //!
    //! \warning The string tensorName must be null-terminated, and be at most 4096 bytes including the terminator.
    //!
    DataType getTensorDataType(char const* tensorName) const noexcept
    {
        return mImpl->getTensorDataType(tensorName);
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
    //! \brief Create an execution context and specify the strategy for allocating internal activation memory.
    //!
    //! The default value for the allocation strategy is ExecutionContextAllocationStrategy::kSTATIC, which means the
    //! context will pre-allocate a block of device memory that is sufficient for all profiles. The newly created
    //! execution context will be assigned optimization profile 0. If an error recorder has been set for the engine, it
    //! will also be passed to the execution context.
    //!
    //! \see IExecutionContext
    //! \see IExecutionContext::setOptimizationProfileAsync()
    //! \see ExecutionContextAllocationStrategy
    //!
    IExecutionContext* createExecutionContext(
        ExecutionContextAllocationStrategy strategy = ExecutionContextAllocationStrategy::kSTATIC) noexcept
    {
        return mImpl->createExecutionContext(strategy);
    }

    //!
    //! \brief Get whether an input or output tensor must be on GPU or CPU.
    //!
    //! \param tensorName The name of an input or output tensor.
    //!
    //! \return TensorLocation::kDEVICE if tensorName must be on GPU, or TensorLocation::kHOST if on CPU, or
    //! TensorLocation::kDEVICE if the provided name does not map to an input or output tensor.
    //!
    //! The location is established at build time. E.g. shape tensors inputs are typically required to be on the CPU.
    //!
    //! \warning The string tensorName must be null-terminated, and be at most 4096 bytes including the terminator.
    //!
    TensorLocation getTensorLocation(char const* tensorName) const noexcept
    {
        return mImpl->getTensorLocation(tensorName);
    }

    //!
    //! \brief True if tensor is required as input for shape calculations or is output from shape calculations.
    //!
    //! Return true for either of the following conditions:
    //!
    //! * The tensor is a network input, and its value is required for IExecutionContext::getTensorShape()
    //!   to return the shape of a network output.
    //!
    //! * The tensor is a network output, and inferShape() will compute its values.
    //!
    //! For example, if a network uses an input tensor "foo" as an addend to an IElementWiseLayer
    //! that computes the "reshape dimensions" for IShuffleLayer, then isShapeInferenceIO("foo") == true.
    //! If the network copies said input tensor "foo" to an output "bar", then
    //! isShapeInferenceIO("bar") == true and IExecutionContext::inferShapes() will write to "bar".
    //!
    bool isShapeInferenceIO(char const* tensorName) const noexcept
    {
        return mImpl->isShapeInferenceIO(tensorName);
    }

    //!
    //! \brief Determine whether a tensor is an input or output tensor.
    //!
    //! \param tensorName The name of an input or output tensor.
    //!
    //! \return kINPUT if tensorName is an input, kOUTPUT if tensorName is an output, or kNONE if neither.
    //!
    //! \warning The string tensorName must be null-terminated, and be at most 4096 bytes including the terminator.
    //!
    TensorIOMode getTensorIOMode(char const* tensorName) const noexcept
    {
        return mImpl->getTensorIOMode(tensorName);
    }

    //!
    //! \brief create an execution context without any device memory allocated
    //!
    //! The memory for execution of this device context must be supplied by the application.
    //!
    //! \deprecated Deprecated in TensorRT 10.0. Superseded by createExecutionContext() with parameter.
    //!
    TRT_DEPRECATED IExecutionContext* createExecutionContextWithoutDeviceMemory() noexcept
    {
        return mImpl->createExecutionContextWithoutDeviceMemory();
    }

    //!
    //! \brief Return the maximum device memory required by the context over all profiles.
    //!
    //! \deprecated Deprecated in TensorRT 10.1. Superseded by getDeviceMemorySizeV2().
    //!
    //! \see IExecutionContext::setDeviceMemory()
    //!
    TRT_DEPRECATED size_t getDeviceMemorySize() const noexcept
    {
        return mImpl->getDeviceMemorySize();
    }

    //!
    //! \brief Return the maximum device memory required by the context for a profile.
    //!
    //! \deprecated Deprecated in TensorRT 10.1. Superseded by getDeviceMemorySizeForProfileV2(int32_t).
    //!
    //! \see IExecutionContext::setDeviceMemoryV2()
    //!
    TRT_DEPRECATED size_t getDeviceMemorySizeForProfile(int32_t profileIndex) const noexcept
    {
        return mImpl->getDeviceMemorySizeForProfile(profileIndex);
    }

    //!
    //! \brief Return the maximum device memory required by the context over all profiles.
    //!
    //! This API is stateful, so its call returns different values based on the following calls:
    //! * setWeightStreamingBudget()
    //! * setWeightStreamingBudgetV2()
    //!
    //! \see IExecutionContext::setDeviceMemoryV2()
    //! \see setWeightStreamingBudget()
    //! \see setWeightStreamingBudgetV2()
    //!
    int64_t getDeviceMemorySizeV2() const noexcept
    {
        return mImpl->getDeviceMemorySizeV2();
    }

    //!
    //! \brief Return the maximum device memory required by the context for a profile.
    //!
    //! This API is stateful, so its call returns different values based on the following calls:
    //! * setWeightStreamingBudget()
    //! * setWeightStreamingBudgetV2()
    //!
    //! \see IExecutionContext::setDeviceMemoryV2()
    //! \see setWeightStreamingBudget()
    //! \see setWeightStreamingBudgetV2()
    //!
    int64_t getDeviceMemorySizeForProfileV2(int32_t profileIndex) const noexcept
    {
        return mImpl->getDeviceMemorySizeForProfileV2(profileIndex);
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
    //! \brief Return the number of bytes per component of an element, or -1 if the
    //! tensor is not vectorized or provided name does not map to an input or output tensor.
    //!
    //! The vector component size is returned if getTensorVectorizedDim(tensorName) != -1.
    //!
    //! \param tensorName The name of an input or output tensor.
    //!
    //! \warning The string tensorName must be null-terminated, and be at most 4096 bytes including the terminator.
    //! \warning The function can only return the result of profile 0, and issues a warning message when there are
    //! multiple profiles in the engine, use getTensorBytesPerComponent with profileIndex when there are multiple
    //! profiles.
    //!
    //! \see getTensorVectorizedDim()
    //! \see getTensorBytesPerComponent(tensorName, profileIndex)
    //!
    int32_t getTensorBytesPerComponent(char const* tensorName) const noexcept
    {
        return mImpl->getTensorBytesPerComponent(tensorName);
    }

    //!
    //! \brief Return the number of bytes per component of an element given of given profile, or -1 if the tensor is not
    //! vectorized or provided name does not map to an input or output tensor.
    //!
    //! The vector component size is returned if getTensorVectorizedDim(tensorName, profileIndex) != -1.
    //!
    //! \param tensorName The name of an input or output tensor.
    //! \param profileIndex The profile index to query
    //!
    //! \warning The string tensorName must be null-terminated, and be at most 4096 bytes including the terminator.
    //!
    //! \see getTensorVectorizedDim(tensorName, profileIndex)
    //!
    int32_t getTensorBytesPerComponent(char const* tensorName, int32_t profileIndex) const noexcept
    {
        return mImpl->getTensorBytesPerComponentV2(tensorName, profileIndex);
    }

    //!
    //! \brief Return the number of components included in one element, or -1 if tensor is
    //! not vectorized or if the provided name does not map to an input or output tensor.
    //!
    //! The number of elements in the vectors is returned if getTensorVectorizedDim(tensorName) != -1.
    //!
    //! \param tensorName The name of an input or output tensor.
    //!
    //! \warning The string tensorName must be null-terminated, and be at most 4096 bytes including the terminator.
    //! \warning The function can only return the result of profile 0, and issues a warning message when there
    //! are multiple profiles in the engine, use getTensorComponentsPerElement with profileIndex when there are
    //! multiple profiles.
    //!
    //! \see getTensorVectorizedDim()
    //! \see getTensorComponentsPerElement(tensorName, profileIndex)
    //!
    int32_t getTensorComponentsPerElement(char const* tensorName) const noexcept
    {
        return mImpl->getTensorComponentsPerElement(tensorName);
    }

    //!
    //! \brief Return the number of components included in one element of given profile, or -1 if tensor is not
    //! vectorized or the provided name does not map to an input or output tensor.
    //!
    //! The number of elements in the vectors is returned if getTensorVectorizedDim(tensorName, profileIndex) != -1.
    //!
    //! \param tensorName The name of an input or output tensor.
    //! \param profileIndex The profile index to query
    //!
    //! \warning The string tensorName must be null-terminated, and be at most 4096 bytes including the terminator.
    //!
    //! \see getTensorVectorizedDim(tensorName, profileIndex)
    //!
    int32_t getTensorComponentsPerElement(char const* tensorName, int32_t profileIndex) const noexcept
    {
        return mImpl->getTensorComponentsPerElementV2(tensorName, profileIndex);
    }

    //!
    //! \brief Return the tensor format, or TensorFormat::kLINEAR if the provided name does not map to an input or
    //! output tensor.
    //!
    //! \warning The string tensorName must be null-terminated, and be at most 4096 bytes including the terminator.
    //! \warning This API can only return the tensor format of profile 0, and issues a warning message when there are
    //! multiple profiles in the engine, use getTensorFormat with profileIndex when there are multiple profiles.
    //!
    //! \see getTensorFormat(tensorName, profileIndex)
    //!
    TensorFormat getTensorFormat(char const* tensorName) const noexcept
    {
        return mImpl->getTensorFormat(tensorName);
    }

    //!
    //! \brief Return the tensor format of given profile, or TensorFormat::kLINEAR if the provided name does not map to
    //! an input or output tensor.
    //!
    //! \param tensorName The name of an input or output tensor.
    //! \param profileIndex The profile index to query the format for.
    //!
    //! \warning The string tensorName must be null-terminated, and be at most 4096 bytes including the terminator.
    //!
    TensorFormat getTensorFormat(char const* tensorName, int32_t profileIndex) const noexcept
    {
        return mImpl->getTensorFormatV2(tensorName, profileIndex);
    }

    //!
    //! \brief Return the human readable description of the tensor format, or empty string if the provided name does not
    //! map to an input or output tensor.
    //!
    //! The description includes the order, vectorization, data type, and strides.
    //! Examples are shown as follows:
    //!   Example 1: kCHW + FP32
    //!     "Row-major linear FP32 format"
    //!   Example 2: kCHW2 + FP16
    //!     "Two-wide channel vectorized row-major FP16 format"
    //!   Example 3: kHWC8 + FP16 + Line Stride = 32
    //!     "Channel major FP16 format where C % 8 == 0 and H Stride % 32 == 0"
    //!
    //! \param tensorName The name of an input or output tensor.
    //!
    //! \warning The string tensorName must be null-terminated, and be at most 4096 bytes including the terminator.
    //! \warning The function can only return the result of profile 0, and issues a warning message when there are
    //! multiple profiles in the engine, use getTensorFormatDesc with profileIndex when there are multiple profiles.
    //!
    char const* getTensorFormatDesc(char const* tensorName) const noexcept
    {
        return mImpl->getTensorFormatDesc(tensorName);
    }

    //!
    //! \brief Return the human readable description of the tensor format of given profile, or empty string if the
    //! provided name does not map to an input or output tensor.
    //!
    //! The description includes the order, vectorization, data type, and strides.
    //! Examples are shown as follows:
    //!   Example 1: kCHW + FP32
    //!     "Row-major linear FP32 format"
    //!   Example 2: kCHW2 + FP16
    //!     "Two-wide channel vectorized row-major FP16 format"
    //!   Example 3: kHWC8 + FP16 + Line Stride = 32
    //!     "Channel major FP16 format where C % 8 == 0 and H Stride % 32 == 0"
    //!
    //! \param tensorName The name of an input or output tensor.
    //! \param profileIndex The profile index to query the format for.
    //!
    //! \warning The string tensorName must be null-terminated, and be at most 4096 bytes including the terminator.
    //!
    char const* getTensorFormatDesc(char const* tensorName, int32_t profileIndex) const noexcept
    {
        return mImpl->getTensorFormatDescV2(tensorName, profileIndex);
    }

    //!
    //! \brief Return the dimension index that the buffer is vectorized, or -1 if the provided name does not
    //! map to an input or output tensor.
    //!
    //! Specifically -1 is returned if scalars per vector is 1.
    //!
    //! \param tensorName The name of an input or output tensor.
    //!
    //! \warning The string tensorName must be null-terminated, and be at most 4096 bytes including the terminator.
    //! \warning The function can only return the result of profile 0, and issues a warning message when there are
    //!  multiple profiles in the engine, use getTensorVectorizedDim with profileIndex when there are multiple profiles.
    //!
    int32_t getTensorVectorizedDim(char const* tensorName) const noexcept
    {
        return mImpl->getTensorVectorizedDim(tensorName);
    }

    //!
    //! \brief Return the dimension index that the buffer is vectorized of given profile, or -1 if the provided name
    //! does not map to an input or output tensor.
    //!
    //! Specifically -1 is returned if scalars per vector is 1.
    //!
    //! \param tensorName The name of an input.
    //! \param profileIndex The profile index to query the format for.
    //!
    //! \warning The string tensorName must be null-terminated, and be at most 4096 bytes including the terminator.
    //!
    int32_t getTensorVectorizedDim(char const* tensorName, int32_t profileIndex) const noexcept
    {
        return mImpl->getTensorVectorizedDimV2(tensorName, profileIndex);
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
    char const* getName() const noexcept
    {
        return mImpl->getName();
    }

    //!
    //! \brief Get the number of optimization profiles defined for this engine.
    //!
    //! \return Number of optimization profiles. It is always at least 1.
    //!
    //! \see IExecutionContext::setOptimizationProfileAsync()
    int32_t getNbOptimizationProfiles() const noexcept
    {
        return mImpl->getNbOptimizationProfiles();
    }

    //!
    //! \brief Get the minimum / optimum / maximum dimensions for an input tensor given its name under an optimization
    //! profile.
    //!
    //! \param tensorName The name of an input tensor.
    //!
    //! \param profileIndex The profile index, which must be between 0 and getNbOptimizationProfiles()-1.
    //!
    //! \param select Whether to query the minimum, optimum, or maximum dimensions for this input tensor.
    //!
    //! \return The minimum / optimum / maximum dimensions for an input tensor in this profile.
    //!         If the profileIndex is invalid or provided name does not map to an input tensor, return Dims{-1, {}}
    //!
    //! \warning The string tensorName must be null-terminated, and be at most 4096 bytes including the terminator.
    //!
    Dims getProfileShape(char const* tensorName, int32_t profileIndex, OptProfileSelector select) const noexcept
    {
        return mImpl->getProfileShape(tensorName, profileIndex, select);
    }

    //!
    //! \brief Get the minimum / optimum / maximum values (not dimensions) for an input tensor given
    //! its name under an optimization profile. These correspond to the values set using
    //! IOptimizationProfile::setShapeValues when the engine was built.
    //!
    //! \param tensorName The name of an input tensor.
    //!
    //! \param profileIndex The profile index, which must be between 0 and getNbOptimizationProfiles()-1.
    //!
    //! \param select Whether to query the minimum, optimum, or maximum values for this input tensor.
    //!
    //! \return The minimum / optimum / maximum values for an input tensor in this profile. If the profileIndex is
    //! invalid or the provided name does not map to an input tensor, or the tensor is not a shape binding, return
    //! nullptr.
    //!
    //! \warning The string tensorName must be null-terminated, and be at most 4096 bytes including the terminator.
    //!
    int32_t const* getProfileTensorValues(char const* tensorName, int32_t profileIndex, OptProfileSelector select) const
        noexcept
    {
        return mImpl->getProfileTensorValues(tensorName, profileIndex, select);
    }

    //!
    //! \brief Determine what execution capability this engine has.
    //!
    //! If the engine has EngineCapability::kSTANDARD, then all engine functionality is valid.
    //! If the engine has EngineCapability::kSAFETY, then only the functionality in safe engine is valid.
    //! If the engine has EngineCapability::kDLA_STANDALONE, then only serialize, destroy, and const-accessor functions
    //! are valid.
    //!
    //! \return The EngineCapability flag that the engine was built for.
    //!
    EngineCapability getEngineCapability() const noexcept
    {
        return mImpl->getEngineCapability();
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
    //!
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
    //! \return Always false since TensorRT 10.0 does not support an implicit batch dimension.
    //!
    //! \see createNetworkV2
    //!
    //! \deprecated Deprecated in TensorRT 10.0. Implicit batch is no supported since TensorRT 10.0.
    //!
    TRT_DEPRECATED bool hasImplicitBatchDimension() const noexcept
    {
        return mImpl->hasImplicitBatchDimension();
    }

    //!
    //! \brief return the tactic sources required by this engine.
    //!
    //! The value returned is equal to zero or more tactics sources set
    //! at build time via setTacticSources() in IBuilderConfig. Sources
    //! set by the latter but not returned by \ref ICudaEngine::getTacticSources
    //! do not reduce overall engine execution time, and can be removed from
    //! future builds to reduce build time.
    //!
    //! \see IBuilderConfig::setTacticSources()
    //!
    TacticSources getTacticSources() const noexcept
    {
        return mImpl->getTacticSources();
    }

    //!
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

    //!
    //! \brief Return number of IO tensors.
    //!
    //! It is the number of input and output tensors for the network from which the engine was built.
    //! The names of the IO tensors can be discovered by calling getIOTensorName(i) for i in 0 to getNbIOTensors()-1.
    //!
    //! \see getIOTensorName()
    //!
    int32_t getNbIOTensors() const noexcept
    {
        return mImpl->getNbIOTensors();
    }

    //!
    //! \brief Return name of an IO tensor.
    //!
    //! \param index value between 0 and getNbIOTensors()-1
    //!
    //! \see getNbIOTensors()
    //!
    char const* getIOTensorName(int32_t index) const noexcept
    {
        return mImpl->getIOTensorName(index);
    }

    //!
    //! \brief Return the hardware compatibility level of this engine.
    //!
    //! \return hardwareCompatibilityLevel The level of hardware
    //!        compatibility.
    //!
    //! This is only supported for Ampere and newer architectures.
    //!
    HardwareCompatibilityLevel getHardwareCompatibilityLevel() const noexcept
    {
        return mImpl->getHardwareCompatibilityLevel();
    }

    //!
    //! \brief Return the number of auxiliary streams used by this engine.
    //!
    //! This number will be less than or equal to the maximum allowed number of auxiliary streams set by
    //! IBuilderConfig::setMaxAuxStreams() API call when the engine was built.
    //!
    //! \return the number of auxiliary streams used by this engine.
    //!
    //! \see IBuilderConfig::setMaxAuxStreams(), IExecutionContext::setAuxStreams()
    //!
    int32_t getNbAuxStreams() const noexcept
    {
        return mImpl->getNbAuxStreams();
    }

    //!
    //! \brief Create a serialization configuration object.
    //!
    //! \see ISerializationConfig
    //!
    ISerializationConfig* createSerializationConfig() noexcept
    {
        return mImpl->createSerializationConfig();
    }

    //!
    //! \brief Serialize the network to a stream with the provided SerializationConfig.
    //!
    //! \return An IHostMemory object that contains the serialized engine.
    //!
    //! The network may be deserialized with IRuntime::deserializeCudaEngine().
    //! Serializing plan file with SerializationFlag::kEXCLUDE_WEIGHTS requires building the engine with kREFIT,
    //! kREFIT_IDENTICAL or kREFIT_INDIVIDUAL.
    //!
    //! \see IRuntime::deserializeCudaEngine()
    //!
    IHostMemory* serializeWithConfig(ISerializationConfig& config) const noexcept
    {
        return mImpl->serializeWithConfig(config);
    }

    //!
    //! \brief Limit the maximum amount of GPU memory usable for network weights
    //! in bytes.
    //!
    //! \param gpuMemoryBudget  This parameter may take on 3 types of values:
    //!  -1: Allows TensorRT to choose the budget according to the streamable weights size.
    //!      Free CUDA memory will be queried at createExecutionContext() and accordingly:
    //!       * If streamable weights all fit: weight streaming is not required and disabled.
    //!       * Otherwise: Budget is set to getMinimumWeightStreamingBudget
    //!   0: (default) Disables weight streaming. The execution may fail if the network is too large for GPU memory.
    //!  >0: The maximum bytes of GPU memory that weights can occupy. It must be bounded by
    //!      [getMinimumWeightStreamingBudget, free GPU memory)].
    //!
    //! By setting a weight limit, users can expect a GPU memory usage reduction
    //! of (total bytes for network weights) - gpuMemoryBudget bytes. Maximum memory savings occur
    //! when gpuMemoryBudget is set to getMinimumWeightStreamingBudget(). Creating additional
    //! IExecutionContexts will increase memory usage by O(getMinimumStreamingBudget()).
    //!
    //! Streaming larger amounts of memory will likely result in lower performance
    //! except in some boundary cases where streaming weights allows the user to
    //! run larger batch sizes. The higher throughput offsets the increased
    //! latency in these cases. Tuning the value of the memory limit is
    //! recommended for best performance.
    //!
    //! \warning GPU memory for the weights is allocated in this call and will be deallocated by enabling weight
    //!          streaming or destroying the ICudaEngine.
    //!
    //! \warning BuilderFlag::kWEIGHT_STREAMING must be set during engine building.
    //!
    //! \warning The weights streaming budget cannot be modified while there are active IExecutionContexts.
    //!
    //! \return true if the memory limit is valid and the call was successful, false otherwise.
    //!
    //! \deprecated Deprecated in TensorRT 10.1. Superceded by setWeightStreamingBudgetV2().
    //!
    //! \see BuilderFlag::kWEIGHT_STREAMING
    //! \see getWeightStreamingBudget()
    //! \see getMinimumWeightStreamingBudget()
    //! \see getStreamableWeightsSize()
    //!
    TRT_DEPRECATED bool setWeightStreamingBudget(int64_t gpuMemoryBudget) noexcept
    {
        return mImpl->setWeightStreamingBudget(gpuMemoryBudget);
    }

    //!
    //! \brief Returns the current weight streaming device memory budget in bytes.
    //!
    //! \warning BuilderFlag::kWEIGHT_STREAMING must be set during engine building.
    //!
    //! \returns The weight streaming budget in bytes. Please see setWeightStreamingBudget() for the possible
    //!          values.
    //!
    //! \deprecated Deprecated in TensorRT 10.1. Superceded by getWeightStreamingBudgetV2().
    //!
    //! \see BuilderFlag::kWEIGHT_STREAMING,
    //! \see setWeightStreamingBudget()
    //! \see getMinimumWeightStreamingBudget()
    //! \see getStreamableWeightsSize()
    //!
    TRT_DEPRECATED int64_t getWeightStreamingBudget() const noexcept
    {
        return mImpl->getWeightStreamingBudget();
    }

    //!
    //! \brief The minimum number of bytes of GPU memory required by network
    //! weights for successful weight streaming.
    //!
    //! This is a positive integer for engines with streamable weights because a
    //! staging buffer on the GPU is required to temporarily hold the streamed
    //! weights. The size of the staging buffer is determined by TensorRT and must
    //! be at least as large as the size of the largest streamable weight in the
    //! network.
    //!
    //! \warning BuilderFlag::kWEIGHT_STREAMING must be set during engine building.
    //!
    //! \returns The minimum number of bytes of GPU memory required for streaming.
    //!
    //! \deprecated Deprecated in TensorRT 10.1. The minimum budget is 0 in the V2 APIs.
    //!
    //! \see setWeightStreamingBudget()
    //!
    TRT_DEPRECATED int64_t getMinimumWeightStreamingBudget() const noexcept
    {
        return mImpl->getMinimumWeightStreamingBudget();
    }

    //!
    //! \brief Get the total size in bytes of all streamable weights.
    //!
    //! The set of streamable weights is a subset of all network weights. The
    //! total size may exceed free GPU memory.
    //!
    //! \returns The total size in bytes of all streamable weights.
    //!          Returns 0 if BuilderFlag::kWEIGHT_STREAMING is unset during engine building.
    //!
    //! \see setWeightStreamingBudget()
    //!
    int64_t getStreamableWeightsSize() const noexcept
    {
        return mImpl->getStreamableWeightsSize();
    }

    //!
    //! \brief Limit the maximum amount of GPU memory usable for network weights in bytes.
    //!
    //! \param gpuMemoryBudget This parameter must be a non-negative value.
    //!   0: Only small amounts of scratch memory will required to run the model.
    //!  >= getStreamableWeightsSize (default): Disables weight streaming.
    //!       The execution may fail if the network is too large for GPU memory.
    //!
    //! By setting a weight limit, users can expect a GPU memory usage reduction on the order
    //! of (total bytes for network weights) - gpuMemoryBudget bytes. Maximum memory savings occur
    //! when gpuMemoryBudget is set to 0. Each IExecutionContext will require getWeightStreamingScratchMemorySize()
    //! bytes of additional device memory if the engine is streaming its weights (budget < getStreamableWeightsSize()).
    //!
    //! Streaming larger amounts of memory will likely result in lower performance
    //! except in some boundary cases where streaming weights allows the user to
    //! run larger batch sizes. The higher throughput offsets the increased
    //! latency in these cases. Tuning the value of the memory limit is
    //! recommended for best performance.
    //!
    //! \warning GPU memory for the weights is allocated in this call and will be deallocated by enabling weight
    //! streaming or destroying the ICudaEngine.
    //!
    //! \warning BuilderFlag::kWEIGHT_STREAMING must be set during engine building.
    //!
    //! \warning The weights streaming budget cannot be modified while there are active IExecutionContexts.
    //!
    //! \warning Using the V2 weight streaming APIs with V1 APIs (setWeightStreamingBudget(),
    //!          getWeightStreamingBudget(), getWeightStreamingMinimumBudget()) leads to undefined behavior.
    //!
    //! \return true if the memory limit is valid and the call was successful, false otherwise.
    //!
    //! \see BuilderFlag::kWEIGHT_STREAMING
    //! \see getWeightStreamingBudgetV2()
    //! \see getWeightStreamingScratchMemorySize()
    //! \see getWeightStreamingAutomaticBudget()
    //! \see getStreamableWeightsSize()
    //!
    bool setWeightStreamingBudgetV2(int64_t gpuMemoryBudget) noexcept
    {
        return mImpl->setWeightStreamingBudgetV2(gpuMemoryBudget);
    }

    //!
    //! \brief Returns the current weight streaming device memory budget in bytes.
    //!
    //! \warning BuilderFlag::kWEIGHT_STREAMING must be set during engine building.
    //!
    //! \returns The weight streaming budget in bytes. Please see setWeightStreamingBudgetV2() for the possible
    //!          return values. Returns getStreamableWeightsSize() if weight streaming is disabled.
    //!
    //! \see BuilderFlag::kWEIGHT_STREAMING
    //! \see setWeightStreamingBudget()
    //! \see getMinimumWeightStreamingBudget()
    //! \see getStreamableWeightsSize()
    //!
    int64_t getWeightStreamingBudgetV2() const noexcept
    {
        return mImpl->getWeightStreamingBudgetV2();
    }

    //!
    //! \brief TensorRT automatically determines a device memory budget for the model to run. The budget is close to the
    //! current free memory size, leaving some space for other memory needs in the user's application. If the budget
    //! exceeds the size obtained from getStreamableWeightsSize(), it is capped to that size, effectively disabling
    //! weight streaming. Since TensorRT lacks information about the user's allocations, the remaining memory size might
    //! be larger than required, leading to wasted memory, or smaller than required, causing an out-of-memory error. For
    //! optimal memory allocation, it is recommended to manually calculate and set the budget.
    //!
    //! \warning BuilderFlag::kWEIGHT_STREAMING must be set during engine building.
    //!
    //! \warning The return value may change between TensorRT minor versions.
    //!
    //! \warning Setting the returned budget with V1 APIs (setWeightStreamingBudget()) will lead to undefined behavior.
    //! Please use V2 APIs.
    //!
    //! \returns The weight streaming budget in bytes. Please set with setWeightStreamingBudgetV2().
    //!
    //! \see BuilderFlag::kWEIGHT_STREAMING
    //! \see setWeightStreamingBudgetV2()
    //!
    int64_t getWeightStreamingAutomaticBudget() const noexcept
    {
        return mImpl->getWeightStreamingAutomaticBudget();
    }

    //!
    //! \brief Returns the size of the scratch memory required by the current weight streaming budget.
    //!
    //! Weight streaming requires small amounts of scratch memory on the GPU to stage CPU weights right before
    //! execution. This value is typically much smaller than the total streamable weights size. Each IExecutionContext
    //! will then allocate this additional memory or the user can provide the additional memory through
    //! getDeviceMemorySizeV2() and IExecutionContext::setDeviceMemoryV2().
    //!
    //! The return value of this call depends on
    //!    1. setWeightStreamingBudget()
    //!    2. setWeightStreamingBudgetV2()
    //!
    //! \warning BuilderFlag::kWEIGHT_STREAMING must be set during engine building.
    //!
    //! \returns The weight streaming scratch memory in bytes. Returns 0 if weight streaming is disabled.
    //!
    //! \see BuilderFlag::kWEIGHT_STREAMING
    //! \see setWeightStreamingBudgetV2()
    //! \see getStreamableWeightsSize()
    //! \see getDeviceMemorySizeV2()
    //! \see getDeviceMemorySizeForProfileV2()
    //! \see IExecutionContext::setDeviceMemoryV2()
    //!
    int64_t getWeightStreamingScratchMemorySize() const noexcept
    {
        return mImpl->getWeightStreamingScratchMemorySize();
    }

    //!
    //! \brief Check if a tensor is marked as a debug tensor.
    //!
    //! Determine whether the given name corresponds to a debug tensor.
    //!
    //! \returns True if tensor is a debug tensor, false otherwise.
    //!
    //! \see INetworkDefinition::markDebug
    //!
    bool isDebugTensor(char const* name) const noexcept
    {
        return mImpl->isDebugTensor(name);
    }

protected:
    apiv::VCudaEngine* mImpl;
};

namespace v_1_0
{
class IOutputAllocator : public IVersionedInterface
{
public:
    //!
    //! \brief Return version information associated with this interface. Applications must not override this method.
    //!
    InterfaceInfo getInterfaceInfo() const noexcept override
    {
        return {"IOutputAllocator", 1, 0};
    }

    //!
    //! \brief Return a pointer to memory for an output tensor, or nullptr if memory cannot be allocated.
    //!        If the requested memory size exceeds the currentMemory size, the currentMemory can be freed as well.
    //!        If currentMemory is known to be big enough, one option is to return currentMemory.
    //!
    //! \param tensorName name of the output tensor.
    //! \param currentMemory points to the address set by IExectionContext::setTensorAddress.
    //! \param size number of bytes required. Always positive, even for an empty tensor.
    //! \param alignment required alignment of the allocation.
    //!
    //! \return A pointer to memory to use for the output tensor or nullptr.
    //!
    //!
    //! To preallocate memory and have the engine fail if the preallocation is not big enough,
    //! use IExecutionContext::setTensorAddress to set a pointer to the preallocated memory,
    //! and have reallocateOutput return nullptr if that memory is not big enough.
    //!
    //! \deprecated Deprecated in TensorRT 10.0. Superseded by reallocateOutputAsync with cudaStream_t argument
    //!
    TRT_DEPRECATED virtual void* reallocateOutput(
        char const* tensorName, void* currentMemory, uint64_t size, uint64_t alignment) noexcept
    {
        return nullptr;
    }

    //!
    //! \brief Return a pointer to memory for an output tensor, or nullptr if memory cannot be allocated.
    //!        If the requested memory size exceeds the currentMemory size, the currentMemory can be freed as well.
    //!        If currentMemory is known to be big enough, one option is to return currentMemory.
    //!
    //! \param tensorName name of the output tensor.
    //! \param currentMemory points to the address set by IExectionContext::setTensorAddress.
    //! \param size number of bytes required. Always positive, even for an empty tensor.
    //! \param alignment required alignment of the allocation.
    //! \param stream The stream in which to execute the kernels.
    //!
    //! \return A pointer to memory to use for the output tensor or nullptr.
    //!
    //! To preallocate memory and have the engine fail if the preallocation is not big enough,
    //! use IExecutionContext::setTensorAddress to set a pointer to the preallocated memory,
    //! and have reallocateOutputAsync return nullptr if that memory is not big enough.
    //!
    //! The default definition exists for sake of backward compatibility with earlier versions of TensorRT.
    //! Eventually this method will become a pure virtual method that requires an override, and method
    //! reallocateOutput() will disappear. Code moving away from TensorRT 9.x should override method
    //! reallocateOutputAsync() and NOT override method reallocateOutput().
    //!
    virtual void* reallocateOutputAsync(
        char const* tensorName, void* currentMemory, uint64_t size, uint64_t alignment, cudaStream_t /*stream*/)
    {
        return reallocateOutput(tensorName, currentMemory, size, alignment);
    }

    //!
    //! \brief Called by TensorRT when the shape of the output tensor is known.
    //!
    //! Called by TensorRT sometime between when it calls reallocateOutput and enqueueV3 returns.
    //!
    //! \param dims dimensions of the output
    //! \param tensorName name of the tensor
    //!
    virtual void notifyShape(char const* tensorName, Dims const& dims) noexcept = 0;
};
} // namespace v_1_0

//!
//! \class IOutputAllocator
//!
//! \brief Callback from ExecutionContext::enqueueV3()
//!
//! \see IExecutionContext::enqueueV3()
//!
using IOutputAllocator = v_1_0::IOutputAllocator;

namespace v_1_0
{
class IDebugListener : public IVersionedInterface
{
public:
    //!
    //! \brief Return version information associated with this interface. Applications must not override this method.
    //!
    InterfaceInfo getInterfaceInfo() const noexcept override
    {
        return {"IDebugListener", 1, 0};
    }

    //!
    //! \brief Callback function that is called when a debug tensor’s value is updated and the debug state of the tensor
    //! is set to true. Content in the given address is only guaranteed to be valid for the duration of the callback.
    //!
    //! \param location TensorLocation of the tensor.
    //! \param addr pointer to buffer.
    //! \param type data Type of the tensor.
    //! \param shape shape of the tensor.
    //! \param name name of the tensor.
    //! \param stream CUDA stream object.
    //!
    //! \return True on success, false otherwise.
    //!
    virtual bool processDebugTensor(void const* addr, TensorLocation location, DataType type, Dims const& shape,
        char const* name, cudaStream_t stream)
        = 0;

    ~IDebugListener() override = default;
};
} // namespace v_1_0

//!
//! \class IDebugListener
//!
//! \brief User-implemented callback for notification when value of a debug tensor is updated.
//!
using IDebugListener = v_1_0::IDebugListener;

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
    //! \brief Set the debug sync flag.
    //!
    //! If this flag is set to true, the engine will log the successful execution for each kernel during executeV2(). It
    //! has no effect when using enqueueV3().
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
    ICudaEngine const& getEngine() const noexcept
    {
        return mImpl->getEngine();
    }

    //!
    //! \brief Set the name of the execution context.
    //!
    //! This method copies the name string.
    //!
    //! \warning The string name must be null-terminated, and be at most 4096 bytes including the terminator.
    //!
    //! \see getName()
    //!
    void setName(char const* name) noexcept
    {
        mImpl->setName(name);
    }

    //!
    //! \brief Return the name of the execution context.
    //!
    //! \see setName()
    //!
    char const* getName() const noexcept
    {
        return mImpl->getName();
    }

    //!
    //! \brief Set the device memory for use by this execution context.
    //!
    //! The memory must be aligned with CUDA memory alignment property (using cudaGetDeviceProperties()), and its size
    //! must be large enough for performing inference with the given network inputs. getDeviceMemorySize() and
    //! getDeviceMemorySizeForProfile() report upper bounds of the size. Setting memory to nullptr is acceptable if the
    //! reported size is 0. If using enqueueV3() to run the network, the memory is in use from the invocation of
    //! enqueueV3() until network execution is complete. If using executeV2(), it is in use until executeV2() returns.
    //! Releasing or otherwise using the memory for other purposes, including using it in another execution context
    //! running in parallel, during this time will result in undefined behavior.
    //!
    //! \deprecated Deprecated in TensorRT 10.1. Superceded by setDeviceMemoryV2().
    //!
    //! \warning Weight streaming related scratch memory will be allocated by TensorRT if the memory is set by this API.
    //!          Please use setDeviceMemoryV2() instead.
    //!
    //! \see ICudaEngine::getDeviceMemorySize()
    //! \see ICudaEngine::getDeviceMemorySizeForProfile()
    //! \see ExecutionContextAllocationStrategy
    //! \see ICudaEngine::createExecutionContext()
    //! \see ICudaEngine::createExecutionContextWithoutDeviceMemory()
    //!
    void setDeviceMemory(void* memory) noexcept
    {
        mImpl->setDeviceMemory(memory);
    }

    //!
    //! \brief Set the device memory and its corresponding size for use by this execution context.
    //!
    //! The memory must be aligned with CUDA memory alignment property (using cudaGetDeviceProperties()), and its size
    //! must be large enough for performing inference with the given network inputs. getDeviceMemorySize() and
    //! getDeviceMemorySizeForProfile() report upper bounds of the size. Setting memory to nullptr is acceptable if the
    //! reported size is 0. If using enqueueV3() to run the network, the memory is in use from the invocation of
    //! enqueueV3() until network execution is complete. If using executeV2(), it is in use until executeV2() returns.
    //! Releasing or otherwise using the memory for other purposes, including using it in another execution context
    //! running in parallel, during this time will result in undefined behavior.
    //!
    //! \see ICudaEngine::getDeviceMemorySizeV2()
    //! \see ICudaEngine::getDeviceMemorySizeForProfileV2()
    //! \see ExecutionContextAllocationStrategy
    //! \see ICudaEngine::createExecutionContext()
    //! \see ICudaEngine::createExecutionContextWithoutDeviceMemory()
    //!
    void setDeviceMemoryV2(void* memory, int64_t size) noexcept
    {
        return mImpl->setDeviceMemoryV2(memory, size);
    }

    //!
    //! \brief Return the strides of the buffer for the given tensor name.
    //!
    //! The strides are in units of elements, not components or bytes.
    //! For example, for TensorFormat::kHWC8, a stride of one spans 8 scalars.
    //!
    //! Note that strides can be different for different execution contexts
    //! with dynamic shapes.
    //!
    //! If the provided name does not map to an input or output tensor, or there are dynamic dimensions that have not
    //! been set yet, return Dims{-1, {}}
    //!
    //! \param tensorName The name of an input or output tensor.
    //!
    //! \warning The string tensorName must be null-terminated, and be at most 4096 bytes including the terminator.
    //!
    Dims getTensorStrides(char const* tensorName) const noexcept
    {
        return mImpl->getTensorStrides(tensorName);
    }

public:
    //!
    //! \brief Get the index of the currently selected optimization profile.
    //!
    //! If the profile index has not been set yet (implicitly to 0 if no other execution context has been set to
    //! profile 0, or explicitly for all subsequent contexts), an invalid value of -1 will be returned
    //! and all calls to enqueueV3()/executeV2() will fail until a valid profile index has been set.
    //! This behavior is deprecated in TensorRT 8.6, all profiles will default to optimization
    //! profile 0 and -1 will no longer be returned.
    //!
    int32_t getOptimizationProfile() const noexcept
    {
        return mImpl->getOptimizationProfile();
    }

    //!
    //! \brief Set shape of given input.
    //!
    //! \param tensorName The name of an input tensor.
    //! \param dims The shape of an input tensor.
    //!
    //! \return True on success, false if the provided name does not map to an input tensor, or if some other error
    //! occurred.
    //!
    //! Each dimension must agree with the network dimension unless the latter was -1.
    //!
    //! \warning The string tensorName must be null-terminated, and be at most 4096 bytes including the terminator.
    //!
    bool setInputShape(char const* tensorName, Dims const& dims) noexcept
    {
        return mImpl->setInputShape(tensorName, dims);
    }

    //!
    //! \brief Return the shape of the given input or output.
    //!
    //! \param tensorName The name of an input or output tensor.
    //!
    //! Return Dims{-1, {}} if the provided name does not map to an input or output tensor.
    //! Otherwise return the shape of the input or output tensor.
    //!
    //! A dimension in an input tensor will have a -1 wildcard value if all the following are true:
    //!  * setInputShape() has not yet been called for this tensor
    //!  * The dimension is a runtime dimension that is not implicitly constrained to be a single value.
    //!
    //! A dimension in an output tensor will have a -1 wildcard value if the dimension depends
    //! on values of execution tensors OR if all the following are true:
    //!  * It is a runtime dimension.
    //!  * setInputShape() has NOT been called for some input tensor(s) with a runtime shape.
    //!  * setTensorAddress() has NOT been called for some input tensor(s) with isShapeInferenceIO() = true.
    //!
    //! An output tensor may also have -1 wildcard dimensions if its shape depends on values of tensors supplied to
    //! enqueueV3().
    //!
    //! If the request is for the shape of an output tensor with runtime dimensions,
    //! all input tensors with isShapeInferenceIO() = true should have their value already set,
    //! since these values might be needed to compute the output shape.
    //!
    //! Examples of an input dimension that is implicitly constrained to a single value:
    //! * The optimization profile specifies equal min and max values.
    //! * The dimension is named and only one value meets the optimization profile requirements
    //!   for dimensions with that name.
    //!
    //! \warning The string tensorName must be null-terminated, and be at most 4096 bytes including the terminator.
    //!
    Dims getTensorShape(char const* tensorName) const noexcept
    {
        return mImpl->getTensorShape(tensorName);
    }

    //!
    //! \brief Whether all dynamic dimensions of input tensors have been specified
    //!
    //! \return True if all dynamic dimensions of input tensors have been specified
    //!         by calling setInputShape().
    //!
    //! Trivially true if network has no dynamically shaped input tensors.
    //!
    //! Does not work with name-base interfaces eg. IExecutionContext::setInputShape(). Use
    //! IExecutionContext::inferShapes() instead.
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
    //! Does not work with name-base interfaces eg. IExecutionContext::setInputShape(). Use
    //! IExecutionContext::inferShapes() instead.
    //!
    //! \deprecated Deprecated in TensorRT 10.0. setInputShapeBinding() is removed since TensorRT 10.0.
    //!
    TRT_DEPRECATED bool allInputShapesSpecified() const noexcept
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
    //!
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
    //! \brief Synchronously execute a network.
    //!
    //! This method requires an array of input and output buffers. The mapping
    //! from indices to tensor names can be queried using ICudaEngine::getIOTensorName().
    //!
    //! \param bindings An array of pointers to input and output buffers for the network.
    //!
    //! \return True if execution succeeded.
    //!
    //! \see ICudaEngine::getIOTensorName()
    //!
    bool executeV2(void* const* bindings) noexcept
    {
        return mImpl->executeV2(bindings);
    }

    //!
    //! \brief Select an optimization profile for the current context with async
    //! semantics.
    //!
    //! \param profileIndex Index of the profile. The value must lie between 0 and
    //!        getEngine().getNbOptimizationProfiles() - 1
    //!
    //! \param stream A CUDA stream on which the cudaMemcpyAsyncs may be
    //! enqueued
    //!
    //! When an optimization profile is switched via this API, TensorRT may
    //! require that data is copied via cudaMemcpyAsync. It is the
    //! application’s responsibility to guarantee that synchronization between
    //! the profile sync stream and the enqueue stream occurs.
    //!
    //! The selected profile will be used in subsequent calls to executeV2()/enqueueV3().
    //! If the associated CUDA engine has inputs with dynamic shapes, the optimization profile must
    //! be set with its corresponding profileIndex before calling execute or enqueue. The newly created execution
    //! context will be assigned optimization profile 0.
    //!
    //! If the associated CUDA engine does not have inputs with dynamic shapes,
    //! this method need not be called, in which case the default profile index
    //! of 0 will be used.
    //!
    //! setOptimizationProfileAsync() must be called before calling
    //! setInputShape() for all dynamic input
    //! tensors or input shape tensors, which in turn must be called before
    //! executeV2()/enqueueV3().
    //!
    //! \warning This function will trigger layer resource updates on the next call of
    //!          executeV2()/enqueueV3(), possibly resulting in performance bottlenecks.
    //!
    //! \warning Not synchronizing the stream used at enqueue with the stream
    //! used to set optimization profile asynchronously using this API will
    //! result in undefined behavior.
    //!
    //! \return true if the call succeeded, else false (e.g. input out of range)
    //!
    //! \see ICudaEngine::getNbOptimizationProfiles()
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
    //!
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
    //!
    bool getEnqueueEmitsProfile() const noexcept
    {
        return mImpl->getEnqueueEmitsProfile();
    }

    //!
    //! \brief Calculate layer timing info for the current optimization profile in IExecutionContext
    //! and update the profiler after one iteration of inference launch.
    //!
    //! If IExecutionContext::getEnqueueEmitsProfile() returns true, the enqueue function will calculate layer timing
    //! implicitly if a profiler is provided. This function returns true and does nothing.
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
    //! \warning reportToProfiler uses the stream of the previous enqueue call, so the stream must be live otherwise
    //! behavior is undefined.
    //!
    //! \return true if the call succeeded, else false (e.g. profiler not provided, in CUDA graph capture mode, etc.)
    //!
    //! \see IExecutionContext::setEnqueueEmitsProfile()
    //! \see IExecutionContext::getEnqueueEmitsProfile()
    //!
    bool reportToProfiler() const noexcept
    {
        return mImpl->reportToProfiler();
    }

    //!
    //! \brief Set memory address for given input or output tensor.
    //!
    //! \param tensorName The name of an input or output tensor.
    //! \param data The pointer (void*) to the data owned by the user.
    //!
    //! \return True on success, false if error occurred.
    //!
    //! An address defaults to nullptr.
    //! Pass data=nullptr to reset to the default state.
    //!
    //! Return false if the provided name does not map to an input or output tensor.
    //!
    //! If an input pointer has type (void const*), use setInputTensorAddress() instead.
    //!
    //! Before calling enqueueV3(), each input must have a non-null address and
    //! each output must have a non-null address or an IOutputAllocator to set it later.
    //!
    //! If the TensorLocation of the tensor is kHOST:
    //! - The pointer must point to a host buffer of sufficient size.
    //! - Data representing shape values is not copied until enqueueV3 is invoked.
    //!
    //! If the TensorLocation of the tensor is kDEVICE:
    //! - The pointer must point to a device buffer of sufficient size and alignment, or
    //! - Be nullptr if the tensor is an output tensor that will be allocated by IOutputAllocator.
    //!
    //! If getTensorShape(name) reports a -1 for any dimension of an output after all
    //! input shapes have been set, use setOutputAllocator() to associate an IOutputAllocator
    //! to which the dimensions will be reported when known.
    //!
    //! Calling both setTensorAddress and setOutputAllocator() for the same output is allowed,
    //! and can be useful for preallocating memory, and then reallocating if it's not big enough.
    //!
    //! The pointer must have at least 256-byte alignment.
    //!
    //! \warning The string tensorName must be null-terminated, and be at most 4096 bytes including the terminator.
    //!
    //! \see setInputTensorAddress() setOutputTensorAddress() getTensorShape() setOutputAllocator() IOutputAllocator
    //!
    bool setTensorAddress(char const* tensorName, void* data) noexcept
    {
        return mImpl->setTensorAddress(tensorName, data);
    }

    //!
    //! \brief Get memory address bound to given input or output tensor, or nullptr if the provided name does not map to
    //! an input or output tensor.
    //!
    //! \param tensorName The name of an input or output tensor.
    //!
    //! Use method getOutputTensorAddress() if a non-const pointer for an output tensor is required.
    //!
    //! \warning The string tensorName must be null-terminated, and be at most 4096 bytes including the terminator.
    //!
    //! \see getOutputTensorAddress()
    //!
    void const* getTensorAddress(char const* tensorName) const noexcept
    {
        return mImpl->getTensorAddress(tensorName);
    }

    //!
    //! \brief Set the memory address for a given output tensor.
    //!
    //! \param tensorName The name of an output tensor.
    //! \param data The pointer to the buffer to which to write the output.
    //!
    //! \return True on success, false if the provided name does not map to an output tensor, does not meet alignment
    //! requirements, or some other error occurred.
    //!
    //! Output addresses can also be set using method setTensorAddress. This method is provided for applications which
    //! prefer to use different methods for setting input and output tensors.
    //!
    //! See setTensorAddress() for alignment and data type constraints.
    //!
    //! \warning The string tensorName must be null-terminated, and be at most 4096 bytes including the terminator.
    //!
    //! \see setTensorAddress()
    //!
    bool setOutputTensorAddress(char const* tensorName, void* data) noexcept
    {
        return mImpl->setOutputTensorAddress(tensorName, data);
    }

    //!
    //! \brief Set memory address for given input.
    //!
    //! \param tensorName The name of an input tensor.
    //! \param data The pointer (void const*) to the const data owned by the user.
    //!
    //! \return True on success, false if the provided name does not map to an input tensor, does not meet alignment
    //! requirements, or some other error occurred.
    //!
    //! Input addresses can also be set using method setTensorAddress, which requires a (void*).
    //!
    //! See description of method setTensorAddress() for alignment and data type constraints.
    //!
    //! \warning The string tensorName must be null-terminated, and be at most 4096 bytes including the terminator.
    //!
    //! \see setTensorAddress()
    //!
    bool setInputTensorAddress(char const* tensorName, void const* data) noexcept
    {
        return mImpl->setInputTensorAddress(tensorName, data);
    }

    //!
    //! \brief Get memory address for given output.
    //!
    //! \param tensorName The name of an output tensor.
    //!
    //! \return Raw output data pointer (void*) for given output tensor, or nullptr if the provided name does not map to
    //! an output tensor.
    //!
    //! If only a (void const*) pointer is needed, an alternative is to call method getTensorAddress().
    //!
    //! \warning The string tensorName must be null-terminated, and be at most 4096 bytes including the terminator.
    //!
    //! \see getTensorAddress()
    //!
    void* getOutputTensorAddress(char const* tensorName) const noexcept
    {
        return mImpl->getOutputTensorAddress(tensorName);
    }

    //!
    //! \brief Run shape calculations.
    //!
    //! \param nbMaxNames Maximum number of names to write to tensorNames.
    //!        When the return value is a positive value n and tensorNames != nullptr,
    //!        the names of min(n,nbMaxNames) insufficiently specified input tensors are
    //!        written to tensorNames.
    //!
    //! \param tensorNames Buffer in which to place names of insufficiently specified input tensors.
    //!
    //! \return 0 on success.
    //!         Positive value n if n input tensors were not sufficiently specified.
    //!         -1 for other errors.
    //!
    //! An input tensor is insufficiently specified if either of the following is true:
    //!
    //! * It has dynamic dimensions and its runtime dimensions have not yet
    //!   been specified via IExecutionContext::setInputShape.
    //!
    //! * isShapeInferenceIO(t)=true and the tensor's address has not yet been set.
    //!
    //! If an output tensor has isShapeInferenceIO(t)=true and its address has been specified,
    //! then its value is written.
    //!
    //! Returns -1 if tensorNames == nullptr and nbMaxNames != 0.
    //! Returns -1 if nbMaxNames < 0.
    //! Returns -1 if a tensor's dimensions are invalid, e.g. a tensor ends up with a negative dimension.
    //!
    int32_t inferShapes(int32_t nbMaxNames, char const** tensorNames) noexcept
    {
        return mImpl->inferShapes(nbMaxNames, tensorNames);
    }

    //!
    //! \brief Recompute the internal activation buffer sizes based on the current input shapes, and return the total
    //! amount of memory required.
    //!
    //! Users can allocate the device memory based on the size returned and provided the memory to TRT with
    //! IExecutionContext::setDeviceMemory(). Must specify all input shapes and the optimization profile to use before
    //! calling this function, otherwise the partition will be invalidated.
    //!
    //! \return Total amount of memory required on success, 0 if error occurred.
    //!
    //! \see IExecutionContext::setDeviceMemory()
    //!
    size_t updateDeviceMemorySizeForShapes() noexcept
    {
        return mImpl->updateDeviceMemorySizeForShapes();
    }

    //!
    //! \brief Mark input as consumed.
    //!
    //! \param event The CUDA event that is triggered after all input tensors have been consumed.
    //!
    //! \warning The set event must be valid during the inferece.
    //!
    //! \return True on success, false if error occurred.
    //!
    //! Passing event==nullptr removes whatever event was set, if any.
    //!
    bool setInputConsumedEvent(cudaEvent_t event) noexcept
    {
        return mImpl->setInputConsumedEvent(event);
    }

    //!
    //! \brief The event associated with consuming the input.
    //!
    //! \return The CUDA event. Nullptr will be returned if the event is not set yet.
    //!
    cudaEvent_t getInputConsumedEvent() const noexcept
    {
        return mImpl->getInputConsumedEvent();
    }

    //!
    //! \brief Set output allocator to use for output tensor of given name.
    //! Pass nullptr to outputAllocator to unset.
    //! The allocator is called by enqueueV3().
    //!
    //! \param tensorName The name of an output tensor.
    //! \param outputAllocator IOutputAllocator for the tensors.
    //!
    //! \return True if success, false if the provided name does not map to an output or, if some other error occurred.
    //!
    //! \warning The string tensorName must be null-terminated, and be at most 4096 bytes including the terminator.
    //!
    //! \see enqueueV3() IOutputAllocator
    //!
    bool setOutputAllocator(char const* tensorName, IOutputAllocator* outputAllocator) noexcept
    {
        return mImpl->setOutputAllocator(tensorName, outputAllocator);
    }

    //!
    //! \brief Get output allocator associated with output tensor of given name, or nullptr if the provided name does
    //! not map to an output tensor.
    //!
    //! \warning The string tensorName must be null-terminated, and be at most 4096 bytes including the terminator.
    //!
    //! \see IOutputAllocator
    //!
    IOutputAllocator* getOutputAllocator(char const* tensorName) const noexcept
    {
        return mImpl->getOutputAllocator(tensorName);
    }

    //!
    //! \brief Get upper bound on an output tensor's size, in bytes, based on
    //! the current optimization profile and input dimensions.
    //!
    //! If the profile or input dimensions are not yet set, or the provided name
    //! does not map to an output, returns -1.
    //!
    //! \param tensorName The name of an output tensor.
    //!
    //! \return Upper bound in bytes.
    //!
    //! \warning The string tensorName must be null-terminated, and be at most 4096 bytes including the terminator.
    //!
    int64_t getMaxOutputSize(char const* tensorName) const noexcept
    {
        return mImpl->getMaxOutputSize(tensorName);
    }

    //!
    //! \brief Specify allocator to use for internal temporary storage.
    //!
    //! This allocator is used only by enqueueV3() for temporary storage whose size cannot be
    //! predicted ahead of enqueueV3(). It is not used for output tensors, because memory
    //! allocation for those is allocated by the allocator set by setOutputAllocator().
    //! All memory allocated is freed by the time enqueueV3() returns.
    //!
    //! \param allocator pointer to allocator to use. Pass nullptr to revert to using TensorRT's
    //!        default allocator.
    //!
    //! \return True on success, false if error occurred.
    //!
    //! \see enqueueV3() setOutputAllocator()
    //!
    bool setTemporaryStorageAllocator(IGpuAllocator* allocator) noexcept
    {
        return mImpl->setTemporaryStorageAllocator(allocator);
    }

    //!
    //! \brief Get allocator set by setTemporaryStorageAllocator.
    //!
    //! Returns a nullptr if a nullptr was passed with setTemporaryStorageAllocator().
    //!
    IGpuAllocator* getTemporaryStorageAllocator() const noexcept
    {
        return mImpl->getTemporaryStorageAllocator();
    }

    //!
    //! \brief Enqueue inference on a stream.
    //!
    //! \param stream A CUDA stream on which the inference kernels will be enqueued.
    //!
    //! \return True if the kernels were enqueued successfully, false otherwise.
    //!
    //! Modifying or releasing memory that has been registered for the tensors before stream
    //! synchronization or the event passed to setInputConsumedEvent has been being triggered results in undefined
    //! behavior.
    //! Input tensor can be released after the setInputConsumedEvent whereas output tensors require stream
    //! synchronization.
    //!
    //! \warning Using default stream may lead to performance issues due to additional cudaDeviceSynchronize() calls by
    //!          TensorRT to ensure correct synchronizations. Please use non-default stream instead.
    //!
    //! \warning If the Engine is streaming weights, enqueueV3 will become synchronous, and
    //!          the graph will not be capturable.
    //!
    bool enqueueV3(cudaStream_t stream) noexcept
    {
        return mImpl->enqueueV3(stream);
    }

    //!
    //! \brief Set the maximum size for persistent cache usage.
    //!
    //! This function sets the maximum persistent L2 cache that this execution context may use for activation caching.
    //! Activation caching is not supported on all architectures - see "How TensorRT uses Memory" in the developer guide
    //! for details
    //!
    //! \param size the size of persistent cache limitation in bytes.
    //! The default is 0 Bytes.
    //!
    //! \see getPersistentCacheLimit
    void setPersistentCacheLimit(size_t size) noexcept
    {
        mImpl->setPersistentCacheLimit(size);
    }

    //!
    //! \brief Get the maximum size for persistent cache usage.
    //!
    //! \returns The size of the persistent cache limit
    //!
    //! \see setPersistentCacheLimit
    size_t getPersistentCacheLimit() const noexcept
    {
        return mImpl->getPersistentCacheLimit();
    }

    //!
    //! \brief Set the verbosity of the NVTX markers in the execution context.
    //!
    //! Building with kDETAILED verbosity will generally increase latency in enqueueV3(). Call this method
    //! to select NVTX verbosity in this execution context at runtime.
    //!
    //! The default is the verbosity with which the engine was built, and the verbosity may not be raised above that
    //! level.
    //!
    //! This function does not affect how IEngineInspector interacts with the engine.
    //!
    //! \param verbosity The verbosity of the NVTX markers.
    //!
    //! \return True if the NVTX verbosity is set successfully. False if the provided verbosity level is higher than the
    //! profiling verbosity of the corresponding engine.
    //!
    //! \see getNvtxVerbosity()
    //! \see ICudaEngine::getProfilingVerbosity()
    //!
    bool setNvtxVerbosity(ProfilingVerbosity verbosity) noexcept
    {
        return mImpl->setNvtxVerbosity(verbosity);
    }

    //!
    //! \brief Get the NVTX verbosity of the execution context.
    //!
    //! \return The current NVTX verbosity of the execution context.
    //!
    //! \see setNvtxVerbosity()
    //!
    ProfilingVerbosity getNvtxVerbosity() const noexcept
    {
        return mImpl->getNvtxVerbosity();
    }

    //!
    //! \brief Set the auxiliary streams that TensorRT should launch kernels on in the next enqueueV3() call.
    //!
    //! If set, TensorRT will launch the kernels that are supposed to run on the auxiliary streams using the streams
    //! provided by the user with this API. If this API is not called before the enqueueV3() call, then TensorRT will
    //! use the auxiliary streams created by TensorRT internally.
    //!
    //! TensorRT will always insert event synchronizations between the main stream provided via enqueueV3() call and the
    //! auxiliary streams:
    //!  - At the beginning of the enqueueV3() call, TensorRT will make sure that all the auxiliary streams wait on
    //!    the activities on the main stream.
    //!  - At the end of the enqueueV3() call, TensorRT will make sure that the main stream wait on the activities on
    //!    all the auxiliary streams.
    //!
    //! \param auxStreams The pointer to an array of cudaStream_t with the array length equal to nbStreams.
    //! \param nbStreams The number of auxiliary streams provided. If nbStreams is greater than
    //!        `engine->getNbAuxStreams()`, then only the first `engine->getNbAuxStreams()` streams will be used. If
    //!        `nbStreams` is less than `engine->getNbAuxStreams()`, such as setting `nbStreams` to 0, then TensorRT
    //!        will use the provided streams for the first `nbStreams` auxiliary streams, and will create additional
    //!        streams internally for the rest of the auxiliary streams.
    //!
    //! \note The provided auxiliary streams must not be the default stream and must all be different to avoid
    //!       deadlocks.
    //!
    //! \see enqueueV3(), IBuilderConfig::setMaxAuxStreams(), ICudaEngine::getNbAuxStreams()
    //!
    void setAuxStreams(cudaStream_t* auxStreams, int32_t nbStreams) noexcept
    {
        mImpl->setAuxStreams(auxStreams, nbStreams);
    }

    //!
    //! \brief Set DebugListener for this execution context.
    //!
    //! \param listener DebugListener for this execution context.
    //!
    //! \return true if succeed, false if failure.
    //!
    bool setDebugListener(IDebugListener* listener) noexcept
    {
        return mImpl->setDebugListener(listener);
    }

    //!
    //! \brief Get the DebugListener of this execution context.
    //!
    //! \return DebugListener of this execution context.
    //!
    IDebugListener* getDebugListener() noexcept
    {
        return mImpl->getDebugListener();
    }

    //!
    //! \brief Set debug state of tensor given the tensor name.
    //!
    //! Turn the debug state of a tensor on or off.
    //! A tensor with the parameter tensor name must exist in the network, and the tensor must have
    //! been marked as a debug tensor during build time. Otherwise, an error is thrown.
    //!
    //! \param name Name of target tensor.
    //!
    //! \param flag True if turning on debug state, false if turning off debug state of tensor
    //! The default is off.
    //!
    //! \return True if successful, false otherwise.
    //!
    bool setTensorDebugState(char const* name, bool flag) noexcept
    {
        return mImpl->setTensorDebugState(name, flag);
    }

    //!
    //! Turn the debug state of all debug tensors on or off.
    //!
    //! \param flag true if turning on debug state, false if turning off debug state.
    //!
    //! \return true if successful, false otherwise.
    //!
    //! The default is off.
    bool setAllTensorsDebugState(bool flag) noexcept
    {
        return mImpl->setAllTensorsDebugState(flag);
    }

    //!
    //! Get the debug state.
    //!
    //! \return true if there is a debug tensor with the given name and it has debug state turned on.
    //!
    bool getDebugState(char const* name) const noexcept
    {
        return mImpl->getDebugState(name);
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
    char const* getLayerInformation(int32_t layerIndex, LayerInformationFormat format) const noexcept
    {
        return mImpl->getLayerInformation(layerIndex, format);
    }

    //!
    //! \brief Get a string describing the information about all the layers in the current engine or the execution
    //!        context.
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
    char const* getEngineInformation(LayerInformationFormat format) const noexcept
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
    //!
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

namespace nvinfer1
{
//!
//! \class ILoggerFinder
//!
//! \brief A virtual base class to find a logger.
//! Allows a plugin to find an instance of a logger if it needs to emit a log message.
//! A pointer to an instance of this class is passed to a plugin shared library on initialization when that plugin
//! is serialized as part of a version-compatible plan. See the plugin chapter in the developer guide for details.
//!
class ILoggerFinder
{
public:
    //!
    //! \brief Get the logger used by the engine or execution context which called the plugin method.
    //!
    //! \warning Must be called from the thread in which the plugin method was called.
    //!
    //! \return A pointer to the logger.
    //!
    virtual ILogger* findLogger() = 0;

protected:
    virtual ~ILoggerFinder() = default;
};

//! DO NOT REFER TO namespace v_1_0 IN CODE. ALWAYS USE nvinfer1 INSTEAD.
//! The name v_1_0 may change in future versions of TensoRT.
namespace v_1_0
{

class IGpuAsyncAllocator : public IGpuAllocator
{
public:
    IGpuAsyncAllocator() = default;
    ~IGpuAsyncAllocator() override = default;

    //!
    //! \brief A thread-safe callback implemented by the application to handle stream-ordered asynchronous
    //!        acquisition of GPU memory.
    //!
    //! \param size The size of the memory block required (in bytes).
    //! \param alignment The required alignment of memory. Alignment will be zero
    //!        or a power of 2 not exceeding the alignment guaranteed by cudaMalloc.
    //!        Thus this allocator can be safely implemented with cudaMalloc/cudaFree.
    //!        An alignment value of zero indicates any alignment is acceptable.
    //! \param flags Reserved for future use. In the current release, 0 will be passed.
    //!
    //! \param stream Specifies the cudastream for the asynchronous allocation. If nullptr or 0 is
    //!        passed, the default stream will be used.
    //!
    //! \return If the allocation was successful, the start address of a device memory block of the requested size.
    //!         If an allocation request of size 0 is made, nullptr must be returned.
    //!         If an allocation request cannot be satisfied, nullptr must be returned.
    //!         If a non-null address is returned, it is guaranteed to have the specified alignment.
    //!
    //! \note The implementation must guarantee thread safety for concurrent allocateAsync/deallocateAsync
    //! requests.
    //!
    //! \note The implementation is not required to be asynchronous. It is permitted to synchronize,
    //! albeit doing so will lose the performance advantage of asynchronous allocation.
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: Yes, this method is required to be thread-safe and may be called from multiple threads.
    //!
    void* allocateAsync(uint64_t const size, uint64_t const alignment, AllocatorFlags const flags,
        cudaStream_t /*stream*/) noexcept override = 0;

    //!
    //! \brief A thread-safe callback implemented by the application to handle stream-ordered asynchronous
    //! release of GPU memory.
    //!
    //! TensorRT may pass a nullptr to this function if it was previously returned by allocate().
    //!
    //! \param memory A memory address that was previously returned by an allocate() or reallocate() call of the same
    //! allocator object.
    //!
    //! \param stream Specifies the cudastream for the asynchronous deallocation. If nullptr or 0 is
    //!        passed, the default stream will be used.
    //!
    //! \return True if the acquired memory is released successfully.
    //!
    //! \note The implementation must guarantee thread safety for concurrent allocateAsync/deallocateAsync
    //! requests.
    //!
    //! \note The implementation is not required to be asynchronous. It is permitted to synchronize,
    //! albeit doing so will lose the performance advantage of asynchronous deallocation.
    //! Either way, it is critical that it not actually free the memory until the current
    //! stream position is reached.
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: Yes, this method is required to be thread-safe and may be called from multiple threads.
    bool deallocateAsync(void* const memory, cudaStream_t /*stream*/) noexcept override = 0;

    //!
    //! \brief A thread-safe callback implemented by the application to handle acquisition of GPU memory.
    //!
    //! \param size The size of the memory block required (in bytes).
    //! \param alignment The required alignment of memory. Alignment will be zero
    //!        or a power of 2 not exceeding the alignment guaranteed by cudaMalloc.
    //!        Thus this allocator can be safely implemented with cudaMalloc/cudaFree.
    //!        An alignment value of zero indicates any alignment is acceptable.
    //! \param flags Reserved for future use. In the current release, 0 will be passed.
    //!
    //! \return If the allocation was successful, the start address of a device memory block of the requested size.
    //!         If an allocation request of size 0 is made, nullptr must be returned.
    //!         If an allocation request cannot be satisfied, nullptr must be returned.
    //!         If a non-null address is returned, it is guaranteed to have the specified alignment.
    //!
    //! \note The implementation must guarantee thread safety for concurrent allocateAsync/deallocateAsync/reallocate requests.
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: Yes, this method is required to be thread-safe and may be called from multiple threads.
    //! \deprecated Deprecated in TensorRT 10.0. Superseded by allocateAsync
    //!
    TRT_DEPRECATED void* allocate(
        uint64_t const size, uint64_t const alignment, AllocatorFlags const flags) noexcept override
    {
        return allocateAsync(size, alignment, flags, nullptr);
    }

    //!
    //! \brief A thread-safe callback implemented by the application to handle release of GPU memory.
    //!
    //! TensorRT may pass a nullptr to this function if it was previously returned by allocate().
    //!
    //! \param memory A memory address that was previously returned by an allocate() or reallocate() call of the same
    //! allocator object.
    //!
    //! \return True if the acquired memory is released successfully.
    //!
    //! \note The implementation must guarantee thread safety for concurrent allocate/reallocate/deallocate
    //! requests.
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: Yes, this method is required to be thread-safe and may be called from multiple threads.
    //! \deprecated Deprecated in TensorRT 10.0. Superseded by deallocateAsync
    //!
    TRT_DEPRECATED bool deallocate(void* const memory) noexcept override
    {
        return deallocateAsync(memory, nullptr);
    }

    //!
    //! \brief Return version information associated with this interface. Applications must not override this method.
    //!
    InterfaceInfo getInterfaceInfo() const noexcept override
    {
        return {"IGpuAllocator", 1, 0};
    }
};

class IPluginCreatorV3One : public IPluginCreatorInterface
{
public:
    //!
    //! \brief Return version information associated with this interface. Applications must not override this method.
    //!
    InterfaceInfo getInterfaceInfo() const noexcept override
    {
        return InterfaceInfo{"PLUGIN CREATOR_V3ONE", 1, 0};
    }

    //!
    //! \brief Return a plugin object. Return nullptr in case of error.
    //!
    //! \param name A NULL-terminated name string of length 1024 or less, including the NULL terminator.
    //! \param fc A pointer to a collection of fields needed for constructing the plugin.
    //! \param phase The TensorRT phase in which the plugin is being created
    //!
    //! When the phase is TensorRTPhase::kRUNTIME, the PluginFieldCollection provided for serialization by the plugin's
    //! runtime interface will be passed as fc.
    //!
    //! \note The returned plugin object must be in an initialized state
    //!
    //! \note If invoked by the user (e.g. with TensorRTPhase::kBUILD, to add to the network defintion with
    //! addPluginV3()), it is the user's responsibility to delete the plugin object. If invoked by TensorRT (e.g. during
    //! engine deserialization), TensorRT will delete any objects it creates.
    //!
    virtual IPluginV3* createPlugin(
        AsciiChar const* name, PluginFieldCollection const* fc, TensorRTPhase phase) noexcept = 0;

    //!
    //! \brief Return a list of fields that need to be passed to createPlugin() when creating a plugin for use in the
    //! TensorRT build phase.
    //!
    //! \see PluginFieldCollection
    //!
    virtual PluginFieldCollection const* getFieldNames() noexcept = 0;

    //!
    //! \brief Return the plugin name.
    //!
    //! \warning The string returned must be NULL-terminated and have a length of 1024 bytes or less including
    //! the NULL terminator.
    //!
    virtual AsciiChar const* getPluginName() const noexcept = 0;

    //!
    //! \brief Return the plugin version.
    //!
    //! \warning The string returned must be NULL-terminated and have a length of 1024 bytes or less including
    //! the NULL terminator.
    //!
    virtual AsciiChar const* getPluginVersion() const noexcept = 0;

    //!
    //! \brief Return the plugin namespace.
    //!
    //! \warning The string returned must be NULL-terminated and have a length of 1024 bytes or less including
    //! the NULL terminator.
    //!
    virtual AsciiChar const* getPluginNamespace() const noexcept = 0;

    IPluginCreatorV3One() = default;
    virtual ~IPluginCreatorV3One() = default;

protected:
    IPluginCreatorV3One(IPluginCreatorV3One const&) = default;
    IPluginCreatorV3One(IPluginCreatorV3One&&) = default;
    IPluginCreatorV3One& operator=(IPluginCreatorV3One const&) & = default;
    IPluginCreatorV3One& operator=(IPluginCreatorV3One&&) & = default;
};

} // namespace v_1_0

//!
//! \class IGpuAsyncAllocator
//!
//! \brief Application-implemented class for controlling asynchronous (stream ordered) memory allocation on the GPU.
//!
//! \warning The lifetime of an IGpuAsyncAllocator object must exceed that of all objects that use it.
//!
//! The advantage of deriving from IGpuAsyncAllocator instead of IGpuAllocator is that you only have
//! to override two methods: allocateAsync() and deallocateAsync() to implement an allocator with
//! asynchronous capability, whereas deriving from IGpuAllocator requires overriding four methods,
//! including two deprecated methods.
//!
//! \see IGpuAllocator
using IGpuAsyncAllocator = v_1_0::IGpuAsyncAllocator;

//!
//! \class IPluginCreatorV3One
//!
//! \brief A plugin creator class capable of producing IPluginV3 objects
//!
//! \see IPluginV3
//! \see IPluginRegistry
//!
using IPluginCreatorV3One = v_1_0::IPluginCreatorV3One;

} // namespace nvinfer1

//!
//! \brief Return the library major version number.
//!
extern "C" TENSORRTAPI int32_t getInferLibMajorVersion() noexcept;
//!
//! \brief Return the library minor version number.
//!
extern "C" TENSORRTAPI int32_t getInferLibMinorVersion() noexcept;
//!
//! \brief Return the library patch version number.
//!
extern "C" TENSORRTAPI int32_t getInferLibPatchVersion() noexcept;
//!
//! \brief Return the library build version number.
//!
extern "C" TENSORRTAPI int32_t getInferLibBuildVersion() noexcept;

#endif // NV_INFER_RUNTIME_H
