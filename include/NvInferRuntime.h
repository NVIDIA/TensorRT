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

#include "NvInferRuntimeCommon.h"

namespace nvinfer1
{

class IExecutionContext; //!< Forward declaration of IExecutionContext for use by other interfaces.
class ICudaEngine; //!< Forward declaration of ICudaENgine for use by other interfaces.
class IPluginFactory; //!< Forward declaration of IPluginFactory for use by other interfaces.

//!
//! \enum EngineCapability
//!
//! \brief List of supported engine capability flows.
//!
//! The EngineCapability determines the restrictions of a network during build time for what can be executed
//! at runtime. EngineCapability::kDEFAULT does not provide any restrictions on functionality and the
//! resulting serialized engine can be executed with TensorRT's standard runtime APIs in the nvinfer1 namespace.
//! EngineCapabiltiy::kSAFE_GPU provides a restricted subset of network operations that are safety certified and
//! the resulting serialized engine can be executed with TensorRT's safe runtime APIs in the nvinfer1::safe namespace.
//! EngineCapability::kSAFE_DLA provides a restricted subset of network operations that are DLA compatible and
//! the resulting serialized engine can be executed using NvMediaDLA's runtime APIs. See sampleNvmedia for an
//! example of integrating NvMediaDLA APIs with TensorRT APIs.
//!
enum class EngineCapability : int32_t
{
    kDEFAULT = 0,  //!< Full capability, TensorRT mode without any restrictions using TensorRT nvinfer1 APIs.
    kSAFE_GPU = 1, //!< Safety restricted capability, TensorRT flow that can only run on GPU devices via TensorRT
                   //!< nvinfer1::safe APIs.
    kSAFE_DLA = 2, //!< Safety restricted capability, TensorRT flow that can only run on DLA devices via
                   //!< NvMediaDLA APIs.
};

//! Maximum number of elements in EngineCapability enum. \see EngineCapability
template <>
constexpr inline int32_t EnumMax<EngineCapability>()
{
    return 3;
}

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
class IHostMemory
{
public:
    virtual void* data() const noexcept = 0;       //!< A pointer to the raw data that is owned by the library.
    virtual std::size_t size() const noexcept = 0; //!< The size in bytes of the data that was allocated.
    virtual DataType type() const noexcept = 0;    //!< The type of the memory that was allocated.
    virtual void destroy() noexcept = 0;           //!< Destroy the allocated memory.
protected:
    virtual ~IHostMemory() {}
};

//! \class IPlugin
//!
//! \brief Plugin class for user-implemented layers.
//!
//! Plugins are a mechanism for applications to implement custom layers. Each plugin is owned by the application, and its lifetime
//! must span any use of it by TensorRT
//!
class IPlugin
{
public:
    //!
    //! \brief Get the number of outputs from the layer.
    //!
    //! \return The number of outputs.
    //!
    //! This function is called by the implementations of INetworkDefinition and IBuilder. In particular, it is called
    //! prior to any call to initialize().
    //!
    virtual int32_t getNbOutputs() const TRTNOEXCEPT = 0;

    //!
    //! \brief Get the dimension of an output tensor.
    //!
    //! \param index The index of the output tensor.
    //! \param inputs The input tensors.
    //! \param nbInputDims The number of input tensors.
    //!
    //! This function is called by the implementations of INetworkDefinition and IBuilder. In particular, it is called
    //! prior to any call to initialize().
    //!
    virtual Dims getOutputDimensions(int32_t index, const Dims* inputs, int32_t nbInputDims) TRTNOEXCEPT = 0;

    //!
    //! \brief Configure the layer.
    //!
    //! This function is called by the builder prior to initialize(). It provides an opportunity for the layer to make
    //! algorithm choices on the basis of its weights, dimensions, and maximum batch size. The type is assumed to be
    //! FP32 and format NCHW.
    //!
    //! \param inputDims The input tensor dimensions.
    //! \param nbInputs The number of inputs.
    //! \param outputDims The output tensor dimensions.
    //! \param nbOutputs The number of outputs.
    //! \param maxBatchSize The maximum batch size.
    //!
    //! The dimensions passed here do not include the outermost batch size (i.e. for 2-D image networks, they will be
    //! 3-dimensional CHW dimensions).
    //!
    //! This method is not called for PluginExt classes, configureWithFormat is called instead.
    //!
    virtual void configure(const Dims* inputDims, int32_t nbInputs, const Dims* outputDims, int32_t nbOutputs,
        int32_t maxBatchSize) TRTNOEXCEPT = 0;

    //!
    //! \brief Initialize the layer for execution. This is called when the engine is created.
    //!
    //! \return 0 for success, else non-zero (which will cause engine termination).
    //!
    virtual int32_t initialize() TRTNOEXCEPT = 0;

    //!
    //! \brief Release resources acquired during plugin layer initialization. This is called when the engine is
    //! destroyed. \see initialize()
    //!
    virtual void terminate() TRTNOEXCEPT = 0;

    //!
    //! \brief Find the workspace size required by the layer.
    //!
    //! This function is called during engine startup, after initialize(). The workspace size returned should be
    //! sufficient for any batch size up to the maximum.
    //!
    //! \return The workspace size.
    //!
    virtual size_t getWorkspaceSize(int32_t maxBatchSize) const TRTNOEXCEPT = 0;

    //!
    //! \brief Execute the layer.
    //!
    //! \param batchSize The number of inputs in the batch.
    //! \param inputs The memory for the input tensors.
    //! \param outputs The memory for the output tensors.
    //! \param workspace Workspace for execution.
    //! \param stream The stream in which to execute the kernels.
    //!
    //! \return 0 for success, else non-zero (which will cause engine termination).
    //!
    virtual int32_t enqueue(int32_t batchSize, const void* const* inputs, void** outputs, void* workspace,
        cudaStream_t stream) TRTNOEXCEPT = 0;

    //!
    //! \brief Find the size of the serialization buffer required.
    //!
    //! \return The size of the serialization buffer.
    //!
    virtual size_t getSerializationSize() TRTNOEXCEPT = 0;

    //!
    //! \brief Serialize the layer.
    //!
    //! \param buffer A pointer to a buffer of size at least that returned by getSerializationSize().
    //!
    //! \see getSerializationSize()
    //!
    virtual void serialize(void* buffer) TRTNOEXCEPT = 0;

    virtual ~IPlugin() {}
};

//!
//! \class IPluginExt
//!
//! \brief Plugin class for user-implemented layers.
//!
//! Plugins are a mechanism for applications to implement custom layers. Each plugin is owned by the application, and its lifetime
//! must span any use of it by TensorRT.
//!
class IPluginExt : public IPlugin
{
public:
    //!
    //! \brief Return the API version with which this plugin was built.
    //!
    //! Do not override this method as it is used by the TensorRT library to maintain backwards-compatibility with
    //! plugins.
    //!
    virtual int32_t getTensorRTVersion() const TRTNOEXCEPT
    {
        return NV_TENSORRT_VERSION;
    }

    //!
    //! \brief Check format support.
    //!
    //! \param type DataType requested.
    //! \param format PluginFormat requested.
    //! \return true if the plugin supports the type-format combination.
    //!
    //! This function is called by the implementations of INetworkDefinition, IBuilder, and ICudaEngine.
    //! In particular, it is called when creating an engine and when deserializing an engine.
    //!
    //! \warning DataType:kBOOL not supported.
    //!
    virtual bool supportsFormat(DataType type, PluginFormat format) const TRTNOEXCEPT = 0;

    //!
    //! \brief Configure the layer.
    //!
    //! This function is called by the builder prior to initialize(). It provides an opportunity for the layer to make
    //! algorithm choices on the basis of its weights, dimensions, and maximum batch size.
    //!
    //! \param inputDims The input tensor dimensions.
    //! \param nbInputs The number of inputs.
    //! \param outputDims The output tensor dimensions.
    //! \param nbOutputs The number of outputs.
    //! \param type The data type selected for the engine.
    //! \param format The format selected for the engine.
    //! \param maxBatchSize The maximum batch size.
    //!
    //! The dimensions passed here do not include the outermost batch size (i.e. for 2-D image networks, they will be
    //! 3-dimensional CHW dimensions).
    //!
    //! \warning DataType:kBOOL not supported.
    //!
    virtual void configureWithFormat(const Dims* inputDims, int32_t nbInputs, const Dims* outputDims, int32_t nbOutputs,
        DataType type, PluginFormat format, int32_t maxBatchSize) TRTNOEXCEPT = 0;

    virtual ~IPluginExt() {}

protected:
    //!
    //! \brief Derived classes should not implement this. In a C++11 API it would be override final.
    //!
    void configure(const Dims* /*inputDims*/, int32_t /*nbInputs*/, const Dims* /*outputDims*/, int32_t /*nbOutputs*/,
        int32_t /*maxBatchSize*/) _TENSORRT_FINAL TRTNOEXCEPT
    {
    }
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
constexpr inline int32_t EnumMax<DimensionOperation>()
{
    return 9;
}

//!
//! \class IDimensionExpr
//!
//! An IDimensionExpr represents an integer expression constructed from constants,
//! input dimensions, and binary operations.  These expressions are can be used
//! in overrides of IPluginV2DynamicExt::getOutputDimensions to define output
//! dimensions in terms of input dimensions.
//!
//! \see DimensionOperation, IPluginV2DynamicExt::getOutputDimensions
//!
class IDimensionExpr
{
public:
    //! Return true if expression is a build-time constant.
    virtual bool isConstant() const = 0;

    //! If isConstant(), returns value of the constant.
    //! If !isConstant(), return std::numeric_limits<int32_t>::min().
    virtual int32_t getConstantValue() const = 0;

protected:
    virtual ~IDimensionExpr() {}
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
class IExprBuilder
{
public:
    //! Return pointer to IDimensionExp for given value.
    virtual const IDimensionExpr* constant(int32_t value) = 0;

    //! Return pointer to IDimensionExp that represents the given operation applied to first and second.
    //! Returns nullptr if op is not a valid DimensionOperation.
    virtual const IDimensionExpr* operation(DimensionOperation op, const IDimensionExpr& first, const IDimensionExpr& second) = 0;

protected:
    virtual ~IExprBuilder() {}
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
//!     virtual int32_t getNbOutputs() const TRTNOEXCEPT = 0;
//!     virtual nvinfer1::DataType getOutputDataType(int32_t index, const nvinfer1::DataType* inputTypes, int32_t
//!     nbInputs) const TRTNOEXCEPT = 0; virtual size_t getSerializationSize() const TRTNOEXCEPT = 0; virtual void
//!     serialize(void* buffer) const TRTNOEXCEPT = 0; virtual void destroy() TRTNOEXCEPT = 0; virtual void
//!     setPluginNamespace(const char* pluginNamespace) TRTNOEXCEPT = 0; virtual const char* getPluginNamespace() const
//!     TRTNOEXCEPT = 0;
//!
//! For getOutputDataType, the inputTypes will always be DataType::kFLOAT or DataType::kINT32,
//! and the returned type is canonicalized to DataType::kFLOAT if it is DataType::kHALF or DataType:kINT8.
//! Details about the floating-point precision are elicited later by method supportsFormatCombination.
//!
class IPluginV2DynamicExt : public nvinfer1::IPluginV2Ext
{
public:
    IPluginV2DynamicExt* clone() const _TENSORRT_OVERRIDE TRTNOEXCEPT = 0;

    //!
    //! \brief Get expressions for computing dimensions of an output tensor from dimensions of the input tensors.
    //!
    //! \param outputIndex The index of the output tensor
    //! \param inputs Expressions for dimensions of the input tensors
    //! \param nbInputDims The number of input tensors
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
        int32_t outputIndex, const DimsExprs* inputs, int32_t nbInputs, IExprBuilder& exprBuilder)
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
        int32_t pos, const PluginTensorDesc* inOut, int32_t nbInputs, int32_t nbOutputs) TRTNOEXCEPT = 0;

    //!
    //! \brief Configure the layer.
    //!
    //! This function is called by the builder prior to initialize().  It provides an opportunity for the layer to make
    //! algorithm choices on the basis of bounds on the input and output tensors, and the target value.
    //!
    //! This function is also called once when the resource requirements are changed based on the optimization profiles.
    //!
    //! \param in The input tensors attributes that are used for configuration.
    //! \param nbInputs Number of input tensors.
    //! \param out The output tensors attributes that are used for configuration.
    //! \param nbOutputs Number of output tensors.
    //!
    virtual void configurePlugin(const DynamicPluginTensorDesc* in, int32_t nbInputs,
        const DynamicPluginTensorDesc* out, int32_t nbOutputs) TRTNOEXCEPT = 0;

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
        int32_t nbOutputs) const TRTNOEXCEPT = 0;

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
        const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) TRTNOEXCEPT = 0;

protected:
    int32_t getTensorRTVersion() const _TENSORRT_OVERRIDE TRTNOEXCEPT
    {
        return (static_cast<int32_t>(PluginVersion::kV2_DYNAMICEXT) << 24 | (NV_TENSORRT_VERSION & 0xFFFFFF));
    }

    virtual ~IPluginV2DynamicExt() {}

    // Rest of the methods below are obsolete inherited methods, and marked final when using a C++11 compiler.
    // Derived classes should not override them.

    //!
    //! \brief Derived classes should not implement this. In a C++11 API it would be override final.
    //!
    //! Instead, derived classes should override the overload of getOutputDimensions that returns DimsExprs.
    //!
    //! \deprecated Deprecated interface will be removed in TensorRT 8.0.
    //!
    TRT_DEPRECATED
    Dims getOutputDimensions(
        int32_t /*index*/, const Dims* /*inputs*/, int32_t /*nbInputDims*/) _TENSORRT_FINAL TRTNOEXCEPT
    {
        return Dims{-1, {}, {}};
    }

    //!
    //! \brief Derived classes should not implement this. In a C++11 API it would be override final.
    //!
    //! This method is not used because with dynamic shapes there is no implicit batch dimension to broadcast across.
    //!
    //! \deprecated Deprecated interface will be removed in TensorRT 8.0.
    //!
    TRT_DEPRECATED
    bool isOutputBroadcastAcrossBatch(int32_t /*outputIndex*/, const bool* /*inputIsBroadcasted*/,
        int32_t /*nbInputs*/) const _TENSORRT_FINAL TRTNOEXCEPT
    {
        return false;
    }

    //!
    //! \brief Derived classes should not implement this. In a C++11 API it would be override final.
    //!
    //! This method is not used because with dynamic shapes there is no implicit batch dimension to broadcast across.
    //!
    //! \deprecated Deprecated interface will be removed in TensorRT 8.0.
    //!
    TRT_DEPRECATED
    bool canBroadcastInputAcrossBatch(int32_t /*inputIndex*/) const _TENSORRT_FINAL TRTNOEXCEPT
    {
        return true;
    }

    //!
    //! \brief Derived classes should not implement this. In a C++11 API it would be override final.
    //!
    //! This method is not used because it does not allow a plugin to specify mixed formats.
    //!
    //! Instead, derived classes should override supportsFormatCombination, which allows plugins
    //! to express mixed formats.
    //!
    //! \deprecated Deprecated interface will be removed in TensorRT 8.0.
    //!
    TRT_DEPRECATED
    bool supportsFormat(DataType /*type*/, PluginFormat /*format*/) const _TENSORRT_FINAL TRTNOEXCEPT
    {
        return false;
    }

    //!
    //! \brief Derived classes should not implement this. In a C++11 API it would be override final.
    //!
    //! This method is not used because tensors with dynamic shapes do not have an implicit batch dimension,
    //! input dimensions might be variable, and outputs might have different floating-point formats.
    //!
    //! Instead, derived classes should override the overload of configurePlugin that takes poiners to
    //! DynamicPluginTensorDesc.
    //!
    //! \deprecated Deprecated interface will be removed in TensorRT 8.0.
    //!
    TRT_DEPRECATED
    void configurePlugin(const Dims* /*inputDims*/, int32_t /*nbInputs*/, const Dims* /*outputDims*/,
        int32_t /*nbOutputs*/, const DataType* /*inputTypes*/, const DataType* /*outputTypes*/,
        const bool* /*inputIsBroadcast*/, const bool* /*outputIsBroadcast*/, PluginFormat /*floatFormat*/,
        int32_t /*maxBatchSize*/) _TENSORRT_FINAL TRTNOEXCEPT
    {
    }

    //!
    //! \brief Derived classes should not implement this. In a C++11 API it would be override final.
    //!
    //! This method is not used because tensors with dynamic shapes do not have an implicit batch dimension,
    //! and the other dimensions might not be build-time constants.
    //!
    //! Instead, derived classes should override the overload of getWorkspaceSize that takes pointers to
    //! PluginTensorDesc. The arguments to that overload provide maximum bounds on all dimensions.
    //!
    //! \deprecated Deprecated interface will be removed in TensorRT 8.0.
    //!
    TRT_DEPRECATED
    size_t getWorkspaceSize(int32_t /*maxBatchSize*/) const _TENSORRT_FINAL TRTNOEXCEPT
    {
        return 0;
    }

    //!
    //! \brief Derived classes should not implement this. In a C++11 API it would be override final.
    //!
    //! This method is not used because tensors with dynamic shapes can have different sizes in different execution
    //! contexts.
    //!
    //! Instead, derived classes should override the overload of enqueue that takes pointers to PluginTensorDesc.
    //!
    //! \deprecated Deprecated interface will be removed in TensorRT 8.0.
    //!
    TRT_DEPRECATED
    int32_t enqueue(int32_t /*batchSize*/, const void* const* /*inputs*/, void** /*outputs*/, void* /*workspace*/,
        cudaStream_t /*stream*/) _TENSORRT_FINAL TRTNOEXCEPT
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
    virtual void reportLayerTime(const char* layerName, float ms) TRTNOEXCEPT = 0;

    virtual ~IProfiler() {}
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
};

//! Maximum number of elements in WeightsRole enum. \see WeightsRole
template <>
constexpr inline int32_t EnumMax<WeightsRole>()
{
    return 5;
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
constexpr inline int32_t EnumMax<DeviceType>()
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
class IRuntime
{
public:
    //!
    //! \brief Deserialize an engine from a stream.
    //!
    //! \param blob The memory that holds the serialized engine.
    //! \param size The size of the memory.
    //! \param pluginFactory The plugin factory, if any plugins are used by the network, otherwise nullptr.
    //!
    //! \return The engine, or nullptr if it could not be deserialized.
    //!
    virtual nvinfer1::ICudaEngine* deserializeCudaEngine(const void* blob, std::size_t size, IPluginFactory* pluginFactory) noexcept = 0;

    //!
    //! \brief Set the DLA core that the deserialized engine must execute on.
    //! \param dlaCore The DLA core to execute the engine on (0 to N-1, where N is the maximum number of DLA's present
    //! on the device). Default value is 0. \see getDLACore()
    //!
    //! \warning Starting with TensorRT 8, the default value will be -1 if the DLA is not specified or unused.
    //!
    virtual void setDLACore(int32_t dlaCore) noexcept = 0;

    //!
    //! \brief Get the DLA core that the engine executes on.
    //! \return If setDLACore is called, returns DLA core from 0 to N-1, else returns 0.
    //!
    //! \warning Starting with TensorRT 8, the default value will be -1 if the DLA is not specified or unused.
    //!
    virtual int32_t getDLACore() const noexcept = 0;

    //!
    //! \brief Returns number of DLA hardware cores accessible.
    //!
    virtual int32_t getNbDLACores() const noexcept = 0;

    //!
    //! \brief Destroy this object.
    //!
    virtual void destroy() noexcept = 0;

protected:
    virtual ~IRuntime() {}

public:
    //!
    //! \brief Set the GPU allocator.
    //! \param allocator Set the GPU allocator to be used by the runtime. All GPU memory acquired will use this allocator. If NULL is passed, the default allocator will be used.
    //!
    //! Default: uses cudaMalloc/cudaFree.
    //!
    //! If nullptr is passed, the default allocator will be used.
    //!
    virtual void setGpuAllocator(IGpuAllocator* allocator) noexcept = 0;

    //!
    //! \brief Set the ErrorRecorder for this interface
    //!
    //! Assigns the ErrorRecorder to this interface. The ErrorRecorder will track all errors during execution.
    //! This function will call incRefCount of the registered ErrorRecorder at least once. Setting
    //! recorder to nullptr unregisters the recorder with the interface, resulting in a call to decRefCount if
    //! a recorder has been registered.
    //!
    //! \param recorder The error recorder to register with this interface.
    //
    //! \see getErrorRecorder
    //!
    virtual void setErrorRecorder(IErrorRecorder* recorder) noexcept = 0;

    //!
    //! \brief get the ErrorRecorder assigned to this interface.
    //!
    //! Retrieves the assigned error recorder object for the given class. A default error recorder does not exist,
    //! so a nullptr will be returned if setErrorRecorder has not been called.
    //!
    //! \return A pointer to the IErrorRecorder object that has been registered.
    //!
    //! \see setErrorRecorder
    //!
    virtual IErrorRecorder* getErrorRecorder() const noexcept = 0;

    //!
    //! \brief Deserialize an engine from a stream when plugin factory is not used.
    //!
    //! \param blob The memory that holds the serialized engine.
    //! \param size The size of the memory.
    //!
    //! \return The engine, or nullptr if it could not be deserialized.
    //!
    nvinfer1::ICudaEngine* deserializeCudaEngine(const void* blob, std::size_t size) noexcept
    {
        return deserializeCudaEngine(blob, size, nullptr);
    }
};

//!
//! \class IRefitter
//!
//! \brief Updates weights in an engine.
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
class IRefitter
{
public:
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
    virtual bool setWeights(const char* layerName, WeightsRole role, Weights weights) TRTNOEXCEPT = 0;

    //!
    //! \brief Updates associated engine.  Return true if successful.
    //!
    //! Failure occurs if getMissing() != 0 before the call.
    //!
    virtual bool refitCudaEngine() TRTNOEXCEPT = 0;

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
    //! the engine being refitted, and becomes invalid when the engine is destroyed.
    //!
    virtual int32_t getMissing(int32_t size, const char** layerNames, WeightsRole* roles) TRTNOEXCEPT = 0;

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
    //! the engine being refitted, and becomes invalid when the engine is destroyed.
    //!
    virtual int32_t getAll(int32_t size, const char** layerNames, WeightsRole* roles) TRTNOEXCEPT = 0;

    virtual void destroy() TRTNOEXCEPT = 0;

protected:
    virtual ~IRefitter() {}

public:
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
    virtual bool setDynamicRange(const char* tensorName, float min, float max) TRTNOEXCEPT = 0;

    //!
    //! \brief Get minimum of dynamic range.
    //!
    //! \return Minimum of dynamic range.
    //!
    //! If the dynamic range was never set, returns the minimum computed during calibration.
    //!
    virtual float getDynamicRangeMin(const char* tensorName) const TRTNOEXCEPT = 0;

    //!
    //! \brief Get maximum of dynamic range.
    //!
    //! \return Maximum of dynamic range.
    //!
    //! If the dynamic range was never set, returns the maximum computed during calibration.
    //!
    virtual float getDynamicRangeMax(const char* tensorName) const TRTNOEXCEPT = 0;

    //!
    //! \brief Get names of all tensors that have refittable dynamic ranges.
    //!
    //! \param size The number of items that can be safely written to a non-null tensorNames.
    //! \param tensorNames Where to write the layer names.
    //!
    //! \return The number of Weights that could be refit.
    //!
    //! If tensorNames!=nullptr, each written pointer points to a string owned by
    //! the engine being refitted, and becomes invalid when the engine is destroyed.
    //!
    virtual int32_t getTensorsWithDynamicRange(int32_t size, const char** tensorNames) const TRTNOEXCEPT = 0;

    //!
    //! \brief Set the ErrorRecorder for this interface
    //!
    //! Assigns the ErrorRecorder to this interface. The ErrorRecorder will track all errors during execution.
    //! This function will call incRefCount of the registered ErrorRecorder at least once. Setting
    //! recorder to nullptr unregisters the recorder with the interface, resulting in a call to decRefCount if
    //! a recorder has been registered.
    //!
    //! \param recorder The error recorder to register with this interface.
    //
    //! \see getErrorRecorder
    //!
    virtual void setErrorRecorder(IErrorRecorder* recorder) TRTNOEXCEPT = 0;

    //!
    //! \brief get the ErrorRecorder assigned to this interface.
    //!
    //! Retrieves the assigned error recorder object for the given class. A default error recorder does not exist,
    //! so a nullptr will be returned if setErrorRecorder has not been called.
    //!
    //! \return A pointer to the IErrorRecorder object that has been registered.
    //!
    //! \see setErrorRecorder
    //!
    virtual IErrorRecorder* getErrorRecorder() const TRTNOEXCEPT = 0;
};

//!
//! \class IPluginFactory
//!
//! \brief Plugin factory for deserialization.
//!
//! This Interface is guaranteed not to change for the same major version of TensorRT.
class IPluginFactory
{
public:
    //!
    //! \brief Create a plugin from serialized data.
    //!
    //! Responsibility of destroying this plugin lies with the application.
    //! It can be done anytime after consumers of this plugin are destroyed.
    //!
    //! \param layerName The name of the layer.
    //! \param serialData The serialized data.
    //! \param serialLength The length of the serialized data.
    //!
    //! \return The plugin.
    //!
    //! \see IPlugin::serialize()
    //!
    virtual IPlugin* createPlugin(const char* layerName, const void* serialData, size_t serialLength) TRTNOEXCEPT = 0;

    virtual ~IPluginFactory() {}
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
constexpr inline int32_t EnumMax<OptProfileSelector>()
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
class IOptimizationProfile
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
    virtual bool setDimensions(const char* inputName, OptProfileSelector select, Dims dims) noexcept = 0;

    //!
    //! \brief Get the minimum / optimum / maximum dimensions for a dynamic input tensor.
    //!
    //! If the dimensions have not been previously set via setDimensions(), return an invalid Dims with nbDims == -1.
    //!
    virtual Dims getDimensions(const char* inputName, OptProfileSelector select) const noexcept = 0;

    //!
    //! \brief Set the minimum / optimum / maximum values for an input shape tensor.
    //!
    //! This function must be called three times for every input tensor t that is a shape tensor (t.isShape() == true).
    //! This implies that the datatype of t is DataType::kINT32, the rank is either 0 or 1, and the dimensions of t
    //! are fixed at network definition time. This function must not be called for any input tensor that is not a
    //! shape tensor.
    //! Each time this function is called for the same input tensor, the same nbValues must be supplied (either 1
    //! if the tensor rank is 0, or dims.d[0] if the rank is 1). Furthermore, if minVals, optVals, maxVals are the
    //! minimum, optimum, and maximum values, it must be true that minVals[i] <= optVals[i] <= maxVals[i] for
    //! i = 0, ..., nbValues - 1.
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
    virtual bool setShapeValues(
        const char* inputName, OptProfileSelector select, const int32_t* values, int32_t nbValues) noexcept
        = 0;

    //!
    //! \brief Get the number of values for an input shape tensor.
    //!
    //! This will return the number of shape values if setShapeValues() has been called before for this input tensor.
    //! Otherwise, return -1.
    //!
    virtual int32_t getNbShapeValues(const char* inputName) const noexcept = 0;

    //!
    //! \brief Get the minimum / optimum / maximum values for an input shape tensor.
    //!
    //! If the shape values have not been set previously with setShapeValues(), this returns nullptr.
    //!
    virtual const int32_t* getShapeValues(const char* inputName, OptProfileSelector select) const noexcept = 0;

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
    virtual bool setExtraMemoryTarget(float target) noexcept = 0;

    //!
    //! \brief Get the extra memory target that has been defined for this profile.
    //!
    virtual float getExtraMemoryTarget() const noexcept = 0;

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
    virtual bool isValid() const noexcept = 0;

protected:
    ~IOptimizationProfile() noexcept = default;
};

//!
//! \class ICudaEngine
//!
//! \brief An engine for executing inference on a built network, with functionally unsafe features.
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
class ICudaEngine
{
public:
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
    virtual int32_t getNbBindings() const noexcept = 0;

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
    virtual int32_t getBindingIndex(const char* name) const noexcept = 0;

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
    virtual const char* getBindingName(int32_t bindingIndex) const noexcept = 0;

    //!
    //! \brief Determine whether a binding is an input binding.
    //!
    //! \param bindingIndex The binding index.
    //! \return True if the index corresponds to an input binding and the index is in range.
    //!
    //! \see getBindingIndex()
    //!
    virtual bool bindingIsInput(int32_t bindingIndex) const noexcept = 0;

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
    virtual Dims getBindingDimensions(int32_t bindingIndex) const noexcept = 0;

    //!
    //! \brief Determine the required data type for a buffer from its binding index.
    //!
    //! \param bindingIndex The binding index.
    //! \return The type of the data in the buffer.
    //!
    //! \see getBindingIndex()
    //!
    virtual DataType getBindingDataType(int32_t bindingIndex) const noexcept = 0;

    //!
    //! \brief Get the maximum batch size which can be used for inference.
    //!
    //! For an engine built from an INetworkDefinition without an implicit batch dimension, this will always return 1.
    //!
    //! \return The maximum batch size for this engine.
    //!
    virtual int32_t getMaxBatchSize() const noexcept = 0;

    //!
    //! \brief Get the number of layers in the network.
    //!
    //! The number of layers in the network is not necessarily the number in the original network definition, as layers
    //! may be combined or eliminated as the engine is optimized. This value can be useful when building per-layer
    //! tables, such as when aggregating profiling data over a number of executions.
    //!
    //! \return The number of layers in the network.
    //!
    virtual int32_t getNbLayers() const noexcept = 0;

    //!
    //! \brief Get the amount of workspace the engine uses.
    //!
    //! The workspace size will be no greater than the value provided to the builder when the engine was built, and will
    //! typically be smaller. Workspace will be allocated for each execution context.
    //!
    //! This method is not used because getDeviceMemorySize returns the total amount of device memory required by an
    //! execution context.
    //!
    //! \deprecated Deprecated interface will be removed in TensorRT 8.0.
    //!
    TRT_DEPRECATED
    virtual std::size_t getWorkspaceSize() const noexcept = 0;

    //!
    //! \brief Serialize the network to a stream.
    //!
    //! \return A IHostMemory object that contains the serialized engine.
    //!
    //! The network may be deserialized with IRuntime::deserializeCudaEngine() and also safe::IRuntime::deserializeCudaEngine() if only functional-safe features are used in the engine.
    //!
    //! \see IRuntime::deserializeCudaEngine() safe::IRuntime::deserializeCudaEngine()
    //!
    virtual IHostMemory* serialize() const noexcept = 0;

    //!
    //! \brief Create an execution context.
    //!
    //! If the engine supports dynamic shapes, each execution context in concurrent use must use a separate optimization
    //! profile. The first execution context created will call setOptimizationProfile(0) implicitly. For other execution
    //! contexts, setOptimizationProfile() must be called with unique profile index before calling execute or enqueue.
    //!
    //! \see IExecutionContext.
    //! \see IExecutionContext::setOptimizationProfile()
    //!
    virtual IExecutionContext* createExecutionContext() noexcept = 0;

    //!
    //! \brief Destroy this object;
    //!
    virtual void destroy() noexcept = 0;

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
    virtual TensorLocation getLocation(int32_t bindingIndex) const noexcept = 0;

protected:
    virtual ~ICudaEngine() {}

public:
    //! \brief create an execution context without any device memory allocated
    //!
    //! The memory for execution of this device context must be supplied by the application.
    //!
    //! \see getDeviceMemorySize() IExecutionContext::setDeviceMemory()
    //!
    virtual IExecutionContext* createExecutionContextWithoutDeviceMemory() noexcept = 0;

    //!
    //! \brief Return the amount of device memory required by an execution context.
    //!
    //! \see IExecutionContext::setDeviceMemory()
    //!
    virtual size_t getDeviceMemorySize() const noexcept = 0;

    //!
    //! \brief Return true if engine can be refit.
    //!
    //! \see nvinfer1::createInferRefitter()
    //!
    virtual bool isRefittable() const noexcept = 0;

    //!
    //! \brief Return the number of bytes per component of an element.
    //!
    //! The vector component size is returned if getBindingVectorizedDim() != -1.
    //!
    //! \param bindingIndex The binding Index.
    //!
    //! \see ICudaEngine::getBindingVectorizedDim()
    //!
    virtual int32_t getBindingBytesPerComponent(int32_t bindingIndex) const noexcept = 0;

    //!
    //! \brief Return the number of components included in one element.
    //!
    //! The number of elements in the vectors is returned if getBindingVectorizedDim() != -1.
    //!
    //! \param bindingIndex The binding Index.
    //!
    //! \see ICudaEngine::getBindingVectorizedDim()
    //!
    virtual int32_t getBindingComponentsPerElement(int32_t bindingIndex) const noexcept = 0;

    //!
    //! \brief Return the binding format.
    //!
    //! \param bindingIndex The binding Index.
    //!
    virtual TensorFormat getBindingFormat(int32_t bindingIndex) const noexcept = 0;

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
    virtual const char* getBindingFormatDesc(int32_t bindingIndex) const noexcept = 0;

    //!
    //! \brief Return the dimension index that the buffer is vectorized.
    //!
    //! Specifically -1 is returned if scalars per vector is 1.
    //!
    //! \param bindingIndex The binding Index.
    //!
    virtual int32_t getBindingVectorizedDim(int32_t bindingIndex) const noexcept = 0;

    //!
    //! \brief Returns the name of the network associated with the engine.
    //!
    //! The name is set during network creation and is retrieved after
    //! building or deserialization.
    //!
    //! \see INetworkDefinition::setName(), INetworkDefinition::getName()
    //!
    //! \return A zero delimited C-style string representing the name of the network.
    //!
    virtual const char* getName() const noexcept = 0;

    //!
    //! \brief Get the number of optimization profiles defined for this engine.
    //!
    //! \return Number of optimization profiles. It is always at least 1.
    //!
    //! \see IExecutionContext::setOptimizationProfile()
    virtual int32_t getNbOptimizationProfiles() const noexcept = 0;

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
    virtual Dims getProfileDimensions(int32_t bindingIndex, int32_t profileIndex, OptProfileSelector select) const
        noexcept
        = 0;

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
    virtual const int32_t* getProfileShapeValues(
        int32_t profileIndex, int32_t inputIndex, OptProfileSelector select) const noexcept
        = 0;

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
    virtual bool isShapeBinding(int32_t bindingIndex) const noexcept = 0;

    //!
    //! \brief True if pointer to tensor data is required for execution phase, false if nullptr can be supplied.
    //!
    //! For example, if a network uses an input tensor with binding i ONLY as the "reshape dimensions"
    //! input of IShuffleLayer, then isExecutionBinding(i) is false, and a nullptr can be
    //! supplied for it when calling IExecutionContext::execute or IExecutionContext::enqueue.
    //!
    //! \see isShapeBinding()
    //!
    virtual bool isExecutionBinding(int32_t bindingIndex) const noexcept = 0;

    //!
    //! \brief determine that execution capability this engine has.
    //!
    //! If the engine has EngineCapability::kDEFAULT, then all engine functionality is valid..
    //! If the engine has EngineCapability::kSAFE_GPU, then only the functionality in safe::ICudaEngine is valid.
    //! If the engine has EngineCapability::kSAFE_DLA, then only serialize, destroy, and const-accessor functions are
    //! valid.
    //!
    //! \return The EngineCapability flag that the engine was built for.
    //!
    virtual EngineCapability getEngineCapability() const noexcept = 0;

    //! \brief Set the ErrorRecorder for this interface
    //!
    //! Assigns the ErrorRecorder to this interface. The ErrorRecorder will track all errors during execution.
    //! This function will call incRefCount of the registered ErrorRecorder at least once. Setting
    //! recorder to nullptr unregisters the recorder with the interface, resulting in a call to decRefCount if
    //! a recorder has been registered.
    //!
    //! \param recorder The error recorder to register with this interface.
    //
    //! \see getErrorRecorder
    //!
    virtual void setErrorRecorder(IErrorRecorder* recorder) noexcept = 0;

    //!
    //! \brief get the ErrorRecorder assigned to this interface.
    //!
    //! Retrieves the assigned error recorder object for the given class. A default error recorder does not exist,
    //! so a nullptr will be returned if setErrorRecorder has not been called.
    //!
    //! \return A pointer to the IErrorRecorder object that has been registered.
    //!
    //! \see setErrorRecorder
    //!
    virtual IErrorRecorder* getErrorRecorder() const noexcept = 0;

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
    virtual bool hasImplicitBatchDimension() const TRTNOEXCEPT = 0;
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
class IExecutionContext
{
public:
    //!
    //! \brief Synchronously execute inference on a batch.
    //!
    //! This method requires an array of input and output buffers. The mapping from tensor names to indices can be
    //! queried using ICudaEngine::getBindingIndex() \param batchSize The batch size. This is at most the value supplied
    //! when the engine was built. \param bindings An array of pointers to input and output buffers for the network.
    //!
    //! \return True if execution succeeded.
    //!
    //! \see ICudaEngine::getBindingIndex() ICudaEngine::getMaxBatchSize()
    //!
    virtual bool execute(int32_t batchSize, void** bindings) noexcept = 0;

    //!
    //! \brief Asynchronously execute inference on a batch.
    //!
    //! This method requires an array of input and output buffers. The mapping from tensor names to indices can be
    //! queried using ICudaEngine::getBindingIndex() \param batchSize The batch size. This is at most the value supplied
    //! when the engine was built. \param bindings An array of pointers to input and output buffers for the network.
    //! \param stream A cuda stream on which the inference kernels will be enqueued
    //! \param inputConsumed An optional event which will be signaled when the input buffers can be refilled with new
    //! data
    //!
    //! \return True if the kernels were enqueued successfully.
    //!
    //! \see ICudaEngine::getBindingIndex() ICudaEngine::getMaxBatchSize()
    //!
    virtual bool enqueue(int32_t batchSize, void** bindings, cudaStream_t stream, cudaEvent_t* inputConsumed) noexcept
        = 0;

    //!
    //! \brief Set the debug sync flag.
    //!
    //! If this flag is set to true, the engine will log the successful execution for each kernel during execute(). It
    //! has no effect when using enqueue().
    //!
    //! \see getDebugSync()
    //!
    virtual void setDebugSync(bool sync) noexcept = 0;

    //!
    //! \brief Get the debug sync flag.
    //!
    //! \see setDebugSync()
    //!
    virtual bool getDebugSync() const noexcept = 0;

    //!
    //! \brief Set the profiler.
    //!
    //! \see IProfiler getProfiler()
    //!
    virtual void setProfiler(IProfiler*) noexcept = 0;

    //!
    //! \brief Get the profiler.
    //!
    //! \see IProfiler setProfiler()
    //!
    virtual IProfiler* getProfiler() const noexcept = 0;

    //!
    //! \brief Get the associated engine.
    //!
    //! \see ICudaEngine
    //!
    virtual const ICudaEngine& getEngine() const noexcept = 0;

    //!
    //! \brief Destroy this object.
    //!
    virtual void destroy() noexcept = 0;

protected:
    virtual ~IExecutionContext() noexcept {}

public:
    //!
    //! \brief Set the name of the execution context.
    //!
    //! This method copies the name string.
    //!
    //! \see getName()
    //!
    virtual void setName(const char* name) noexcept = 0;

    //!
    //! \brief Return the name of the execution context.
    //!
    //! \see setName()
    //!
    virtual const char* getName() const noexcept = 0;

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
    virtual void setDeviceMemory(void* memory) noexcept = 0;

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
    virtual Dims getStrides(int32_t bindingIndex) const noexcept = 0;

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
    //! \return true if the call succeeded, else false (e.g. input out of range)
    //!
    //! \deprecated This API is superseded by setOptimizationProfileAsync and will be removed in TensorRT 9.0.
    //!
    //! \see ICudaEngine::getNbOptimizationProfiles() IExecutionContext::setOptimizationProfileAsync()
    //!
    TRT_DEPRECATED
    virtual bool setOptimizationProfile(int32_t profileIndex) noexcept = 0;

    //!
    //! \brief Get the index of the currently selected optimization profile.
    //!
    //! If the profile index has not been set yet (implicitly to 0 for the first execution context
    //! to be created, or explicitly for all subsequent contexts), an invalid value of -1 will be returned
    //! and all calls to enqueue() or execute() will fail until a valid profile index has been set.
    //!
    virtual int32_t getOptimizationProfile() const noexcept = 0;

    //!
    //! \brief Set the dynamic dimensions of a binding
    //!
    //! Requires the engine to be built without an implicit batch dimension.
    //! The binding must be an input tensor, and all dimensions must be compatible with
    //! the network definition (i.e. only the wildcard dimension -1 can be replaced with a
    //! new dimension > 0). Furthermore, the dimensions must be in the valid range for the
    //! currently selected optimization profile, and the corresponding engine must not be
    //! safety-certified.
    //!
    //! This method will fail unless a valid optimization profile is defined for the current
    //! execution context (getOptimizationProfile() must not be -1).
    //!
    //! For all dynamic non-output bindings (which have at least one wildcard dimension of -1),
    //! this method needs to be called before either enqueue() or execute() may be called.
    //! This can be checked using the method allInputDimensionsSpecified().
    //!
    //! \return false if an error occurs (e.g. index out of range), else true
    //!
    //! \see ICudaEngine::getBindingIndex
    //!
    virtual bool setBindingDimensions(int32_t bindingIndex, Dims dimensions) noexcept = 0;

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
    virtual Dims getBindingDimensions(int32_t bindingIndex) const noexcept = 0;

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
    virtual bool setInputShapeBinding(int32_t bindingIndex, const int32_t* data) noexcept = 0;

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
    virtual bool getShapeBinding(int32_t bindingIndex, int32_t* data) const noexcept = 0;

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
    virtual bool allInputDimensionsSpecified() const noexcept = 0;

    //!
    //! \brief Whether all input shape bindings have been specified
    //!
    //! \return True if all input shape bindings have been specified by setInputShapeBinding().
    //!
    //! Trivially true if network has no input shape bindings.
    //!
    //! \see isShapeBinding(bindingIndex)
    //!
    virtual bool allInputShapesSpecified() const noexcept = 0;

    //!
    //! \brief Set the ErrorRecorder for this interface
    //!
    //! Assigns the ErrorRecorder to this interface. The ErrorRecorder will track all errors during execution.
    //! This function will call incRefCount of the registered ErrorRecorder at least once. Setting
    //! recorder to nullptr unregisters the recorder with the interface, resulting in a call to decRefCount if
    //! a recorder has been registered.
    //!
    //! \param recorder The error recorder to register with this interface.
    //
    //! \see getErrorRecorder
    //!
    virtual void setErrorRecorder(IErrorRecorder* recorder) noexcept = 0;

    //!
    //! \brief get the ErrorRecorder assigned to this interface.
    //!
    //! Retrieves the assigned error recorder object for the given class. A default error recorder does not exist,
    //! so a nullptr will be returned if setErrorRecorder has not been called.
    //!
    //! \return A pointer to the IErrorRecorder object that has been registered.
    //!
    //! \see setErrorRecorder
    //!
    virtual IErrorRecorder* getErrorRecorder() const noexcept = 0;

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
    virtual bool executeV2(void** bindings) noexcept = 0;

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
    virtual bool enqueueV2(void** bindings, cudaStream_t stream, cudaEvent_t* inputConsumed) noexcept = 0;

    //!
    //! \brief Select an optimization profile for the current context with async
    //! semantics.
    //!
    //! \param profileIndex Index of the profile. It must lie between 0 and
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
    //! \warning Not synchronizing the stream used at enqueue with the stream
    //! used to set optimization profile asynchronously using this API will
    //! result in undefined behavior.
    //!
    //! \return true if the call succeeded, else false (e.g. input out of range)
    //!
    //! \see ICudaEngine::getNbOptimizationProfiles()
    //! IExecutionContext::setOptimizationProfile()
    virtual bool setOptimizationProfileAsync(int32_t profileIndex, cudaStream_t stream) noexcept = 0;
}; // class IExecutionContext
} // namespace nvinfer1

//!
//! Internal C entry point for creating IRuntime.
//! @private
//!
extern "C" TENSORRTAPI void* createInferRuntime_INTERNAL(void* logger, int32_t version);

//!
//! Internal C entry point for creating IRefitter.
//! @private
//!
extern "C" TENSORRTAPI void* createInferRefitter_INTERNAL(void* engine, void* logger, int32_t version);

namespace nvinfer1
{
namespace // unnamed namespace avoids linkage surprises when linking objects built with different versions of this header.
{
//!
//! \brief Create an instance of an IRuntime class.
//!
//! This class is the logging class for the runtime.
//!
inline IRuntime* createInferRuntime(ILogger& logger)
{
    return static_cast<IRuntime*>(createInferRuntime_INTERNAL(&logger, NV_TENSORRT_VERSION));
}

//!
//! \brief Create an instance of an IRefitter class.
//!
//! This class is the logging class for the refitter.
//!
inline IRefitter* createInferRefitter(ICudaEngine& engine, ILogger& logger)
{
    return static_cast<IRefitter*>(createInferRefitter_INTERNAL(&engine, &logger, NV_TENSORRT_VERSION));
}
}
}

#endif // NV_INFER_RUNTIME_H
