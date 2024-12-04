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

#ifndef NV_INFER_RUNTIME_PLUGIN_H
#define NV_INFER_RUNTIME_PLUGIN_H

#define NV_INFER_INTERNAL_INCLUDE 1
#include "NvInferPluginBase.h"
#undef NV_INFER_INTERNAL_INCLUDE

//!
//! \file NvInferRuntimePlugin.h
//!
//! This file contains common definitions, data structures and interfaces that relate to plugins and are shared
//! between the standard and safe runtime.
//!
//! \warning Do not directly include this file. Instead include NvInferRuntime.h
//!

//!
//! \namespace nvinfer1
//!
//! \brief The TensorRT API version 1 namespace.
//!
namespace nvinfer1
{

enum class TensorFormat : int32_t;
namespace v_1_0
{
class IGpuAllocator;
}
using IGpuAllocator = v_1_0::IGpuAllocator;

//!
//! \brief PluginFormat is reserved for backward compatibility.
//!
//! \see IPluginV2::supportsFormat()
//!
using PluginFormat = TensorFormat;

//!
//! \brief Bit at the plugin version to identify that it is a plugin.
//!
static constexpr int32_t kPLUGIN_VERSION_PYTHON_BIT = 0x40;

//!
//! \struct PluginTensorDesc
//!
//! \brief Fields that a plugin might see for an input or output.
//!
//! Scale is only valid when data type is DataType::kINT8. TensorRT will set
//! the value to -1.0F if it is invalid.
//!
//! \see IPluginV2IOExt::supportsFormatCombination
//! \see IPluginV2IOExt::configurePlugin
//!
struct PluginTensorDesc
{
    //! Dimensions.
    Dims dims;
    //! \warning DataType:kBOOL and DataType::kUINT8 are not supported.
    DataType type;
    //! Tensor format.
    TensorFormat format;
    //! Scale for INT8 data type.
    float scale;
};

//!
//! \struct PluginVersion
//!
//! \brief Definition of plugin versions.
//!
//! Tag for plug-in versions.  Used in upper byte of getTensorRTVersion().
//!
enum class PluginVersion : uint8_t
{
    //! IPluginV2
    kV2 = 0,
    //! IPluginV2Ext
    kV2_EXT = 1,
    //! IPluginV2IOExt
    kV2_IOEXT = 2,
    //! IPluginV2DynamicExt
    kV2_DYNAMICEXT = 3,
    //! IPluginV2DynamicExt-based Python plugins
    kV2_DYNAMICEXT_PYTHON = kPLUGIN_VERSION_PYTHON_BIT | 3
};

//!
//! \enum PluginCreatorVersion
//!
//! \brief Enum to identify version of the plugin creator.
//!
enum class PluginCreatorVersion : int32_t
{
    //! IPluginCreator
    kV1 = 0,
    //! IPluginCreator-based Python plugin creators
    kV1_PYTHON = kPLUGIN_VERSION_PYTHON_BIT
};

//!
//! \class IPluginV2
//!
//! \brief Plugin class for user-implemented layers.
//!
//! Plugins are a mechanism for applications to implement custom layers. When
//! combined with IPluginCreator it provides a mechanism to register plugins and
//! look up the Plugin Registry during de-serialization.
//!
//! \see IPluginCreator
//! \see IPluginRegistry
//!
//! \deprecated Deprecated in TensorRT 8.5. Implement IPluginV3 instead.
//!
class TRT_DEPRECATED IPluginV2
{
public:
    //!
    //! \brief Return the API version with which this plugin was built.
    //!
    //! Do not override this method as it is used by the TensorRT library to maintain backwards-compatibility with
    //! plugins.
    //!
    //! \return The TensorRT version in the format (major * 100 + minor) * 100 + patch.
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: Yes, the implementation provided here is safe to call from any thread.
    //!
    virtual int32_t getTensorRTVersion() const noexcept
    {
        return NV_TENSORRT_VERSION;
    }

    //!
    //! \brief Return the plugin type. Should match the plugin name returned by the corresponding plugin creator
    //!
    //! \see IPluginCreator::getPluginName()
    //!
    //! \warning The string returned must be NULL-terminated and have a length of 1024 bytes or less including the
    //! NULL terminator.
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: Yes, this method is required to be thread-safe and may be called from multiple threads
    //!                  when building networks on multiple devices sharing the same plugin.
    //!
    virtual AsciiChar const* getPluginType() const noexcept = 0;

    //!
    //! \brief Return the plugin version. Should match the plugin version returned by the corresponding plugin creator
    //!
    //! \see IPluginCreator::getPluginVersion()
    //!
    //! \warning The string returned must be NULL-terminated and have a length of 1024 bytes or less including the
    //! NULL terminator.
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: Yes, this method is required to be thread-safe and may be called from multiple threads
    //!                  when building networks on multiple devices sharing the same plugin.
    //!
    virtual AsciiChar const* getPluginVersion() const noexcept = 0;

    //!
    //! \brief Get the number of outputs from the layer.
    //!
    //! \return The number of outputs, which is a positive integer.
    //!
    //! This function is called by the implementations of INetworkDefinition and IBuilder. In particular, it is called
    //! prior to any call to initialize().
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: Yes, this method is required to be thread-safe and may be called from multiple threads
    //!                  when building networks on multiple devices sharing the same plugin.
    //!
    virtual int32_t getNbOutputs() const noexcept = 0;

    //!
    //! \brief Get the dimension of an output tensor.
    //!
    //! \param index The index of the output tensor. Will lie in the valid range (between 0 and getNbOutputs()-1
    //! inclusive).
    //! \param inputs The input tensor dimensions. Will be the start address of a Dims array of length nbInputDims.
    //! \param nbInputDims The number of input tensors. Will be a non-negative integer.
    //!
    //! \return The output tensor dimensions if the index is in the valid range.
    //!         An invalid value of Dims{-1, {}} must be returned if the index is not in the valid range.
    //!
    //! This function is called by the implementations of INetworkDefinition and IBuilder. In particular, it is called
    //! prior to any call to initialize().
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: Yes, this method is required to be thread-safe and may be called from multiple threads
    //!                  when building networks on multiple devices sharing the same plugin.
    //!
    //! \note In any non-IPluginV2DynamicExt plugin, batch size must not be included in the returned dimensions,
    //! even if the plugin is expected to be run in a network with explicit batch mode enabled.
    //! Please see the TensorRT Developer Guide for more details on how plugin inputs and outputs behave.
    //!
    virtual Dims getOutputDimensions(int32_t index, Dims const* inputs, int32_t nbInputDims) noexcept = 0;

    //!
    //! \brief Check format support.
    //!
    //! \param type DataType requested.
    //! \param format PluginFormat requested.
    //!
    //! \return true if the plugin supports the type-format combination.
    //!
    //! This function is called by the implementations of INetworkDefinition, IBuilder, and
    //! safe::ICudaEngine/ICudaEngine. In particular, it is called when creating an engine and when deserializing an
    //! engine.
    //!
    //! \warning for the format field, the values PluginFormat::kCHW4, PluginFormat::kCHW16, and PluginFormat::kCHW32
    //! will not be passed in, this is to keep backward compatibility with TensorRT 5.x series.  Use PluginV2IOExt
    //! or PluginV2DynamicExt for other PluginFormats.
    //!
    //! \warning DataType:kBOOL and DataType::kUINT8 are not supported.
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: Yes, this method is required to be thread-safe and may be called from multiple threads
    //!                  when building networks on multiple devices sharing the same plugin.
    //!
    virtual bool supportsFormat(DataType type, PluginFormat format) const noexcept = 0;

    //!
    //! \brief Configure the layer.
    //!
    //! This function is called by the builder prior to initialize(). It provides an opportunity for the layer to make
    //! algorithm choices on the basis of its weights, dimensions, and maximum batch size.
    //!
    //! \param inputDims The input tensor dimensions. Will be the start address of a Dims array of length nbInputs.
    //! \param nbInputs The number of inputs. Will be a non-negative integer.
    //! \param outputDims The output tensor dimensions. Will be the start address of a Dims array of length nbOutputs.
    //! \param nbOutputs The number of outputs. Will be a positive integer identical to the return value of
    //! getNbOutputs().
    //! \param type The data type selected for the engine.
    //! \param format The format selected for the engine.
    //! \param maxBatchSize The maximum batch size. Will be a positive integer.
    //!
    //! The dimensions passed here do not include the outermost batch size (i.e. for 2D image networks, they will be
    //! 3-dimensional CHW dimensions).
    //!
    //! \warning for the format field, the values PluginFormat::kCHW4, PluginFormat::kCHW16, and PluginFormat::kCHW32
    //! will not be passed in, this is to keep backward compatibility with TensorRT 5.x series.  Use PluginV2IOExt
    //! or PluginV2DynamicExt for other PluginFormats.
    //!
    //! \warning DataType:kBOOL and DataType::kUINT8 are not supported.
    //!
    //! \see clone()
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: Yes, this method is required to be thread-safe and may be called from multiple threads
    //!                  when building networks on multiple devices sharing the same plugin. However, TensorRT
    //!                  will not call this method from two threads simultaneously on a given clone of a plugin.
    //!
    virtual void configureWithFormat(Dims const* inputDims, int32_t nbInputs, Dims const* outputDims, int32_t nbOutputs,
        DataType type, PluginFormat format, int32_t maxBatchSize) noexcept
        = 0;

    //!
    //! \brief Initialize the layer for execution. This is called when the engine is created.
    //!
    //! \return 0 for success, else non-zero (which will cause engine termination).
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: Yes, this method is required to be thread-safe and may be called from multiple threads
    //!                  when building networks on multiple devices sharing the same plugin or when using multiple
    //!                  execution contexts using this plugin.
    //!
    virtual int32_t initialize() noexcept = 0;

    //!
    //! \brief Release resources acquired during plugin layer initialization. This is called when the engine is
    //! destroyed.
    //!
    //! \see initialize()
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: Yes, this method is required to be thread-safe and may be called from multiple threads
    //!                  when building networks on multiple devices sharing the same plugin or when using multiple
    //!                  execution contexts using this plugin. However, TensorRT will not call this method from
    //!                  two threads simultaneously on a given clone of a plugin.
    //!
    virtual void terminate() noexcept = 0;

    //!
    //! \brief Find the workspace size required by the layer.
    //!
    //! This function is called during engine startup, after initialize(). The workspace size returned must be
    //! sufficient for any batch size up to the maximum.
    //!
    //! \param maxBatchSize The maximum batch size, which will be a positive integer.
    //!
    //! \return The workspace size in bytes, i.e. the device memory size that the plugin requires for its internal
    //! computations.
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: Yes, this method is required to be thread-safe and may be called from multiple threads
    //!                  when building networks on multiple devices sharing the same plugin. However, TensorRT
    //!                  will not call this method from two threads simultaneously on a given clone of a plugin.
    //!
    virtual size_t getWorkspaceSize(int32_t maxBatchSize) const noexcept = 0;

    //!
    //! \brief Execute the layer.
    //!
    //! \param batchSize The number of inputs in the batch.
    //! \param inputs The memory for the input tensors. Will be an array of device addresses corresponding to input
    //!        tensors of length nbInputs, where nbInputs is the second parameter passed to configureWithFormat().
    //!        The i-th input tensor will have the dimensions inputDims[i], where inputDims is the first parameter
    //!        that was passed to configureWithFormat().
    //! \param outputs The memory for the output tensors. Will be an array of device addresses corresponding to output
    //!        tensors of length getNbOutputs().
    //! \param workspace Workspace for execution. Will be the start address of a device buffer whose length will be at
    //!        least getWorkspaceSize(batchSize).
    //! \param stream The stream in which to execute the kernels. This will be a valid CUDA stream.
    //!
    //! \return 0 for success, else non-zero (which will cause engine termination).
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: Yes, this method is required to be thread-safe and may be called from multiple threads
    //!                  when multiple execution contexts are used during runtime.
    //!
    virtual int32_t enqueue(int32_t batchSize, void const* const* inputs, void* const* outputs, void* workspace,
        cudaStream_t stream) noexcept
        = 0;

    //!
    //! \brief Find the size of the serialization buffer required to store the plugin configuration in a binary file.
    //!
    //! \return The size of the serialization buffer in bytes.
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: Yes, this method is required to be thread-safe and may be called from multiple threads
    //!                  when building networks on multiple devices sharing the same plugin.
    //!
    virtual size_t getSerializationSize() const noexcept = 0;

    //!
    //! \brief Serialize the layer.
    //!
    //! \param buffer A pointer to a host buffer to serialize data. Size of buffer will be at least as large as the
    //! value returned by getSerializationSize.
    //!
    //! \see getSerializationSize()
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: Yes, this method is required to be thread-safe and may be called from multiple threads
    //!                  when building networks on multiple devices sharing the same plugin.
    //!
    virtual void serialize(void* buffer) const noexcept = 0;

    //!
    //! \brief Destroy the plugin object. This will be called when the network, builder or engine is destroyed.
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: Yes, this method is required to be thread-safe and may be called from multiple threads
    //!                  when building networks on multiple devices sharing the same plugin.
    //!
    virtual void destroy() noexcept = 0;

    //!
    //! \brief Clone the plugin object. This copies over internal plugin parameters and returns a new plugin object with
    //! these parameters.
    //!
    //! The TensorRT runtime calls clone() to clone the plugin when an execution context is created for an engine,
    //! after the engine has been created.  The runtime does not call initialize() on the cloned plugin,
    //! so the cloned plugin must be created in an initialized state.
    //!
    //! \return A cloned plugin object in an initialized state with the same parameters as the current object.
    //!         nullptr must be returned if the cloning fails, e.g. because of resource exhaustion.
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: Yes, this method is required to be thread-safe and may be called from multiple threads
    //!                  when building networks on multiple devices sharing the same plugin or when creating multiple
    //!                  execution contexts.
    //!
    virtual IPluginV2* clone() const noexcept = 0;

    //!
    //! \brief Set the namespace that this plugin object belongs to. Ideally, all plugin
    //! objects from the same plugin library must have the same namespace.
    //!
    //! \param pluginNamespace The namespace for the plugin object.
    //!
    //! \warning The string pluginNamespace will be NULL-terminated and have a length of 1024 bytes or less including the
    //! NULL terminator.
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: Yes, this method is required to be thread-safe and may be called from multiple threads
    //!                  when building networks on multiple devices sharing the same plugin.
    //!
    virtual void setPluginNamespace(AsciiChar const* pluginNamespace) noexcept = 0;

    //!
    //! \brief Return the namespace of the plugin object.
    //!
    //! \return The namespace string that was passed to setPluginNamespace(), possibly after truncation to 1024 bytes
    //! if a longer string was passed. An empty string must be returned as default value.
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: Yes, this method is required to be thread-safe and may be called from multiple threads
    //!                  when building networks on multiple devices sharing the same plugin.
    //!
    virtual AsciiChar const* getPluginNamespace() const noexcept = 0;

    // @cond SuppressDoxyWarnings
    IPluginV2() = default;
    virtual ~IPluginV2() noexcept = default;
// @endcond

protected:
// @cond SuppressDoxyWarnings
    IPluginV2(IPluginV2 const&) = default;
    IPluginV2(IPluginV2&&) = default;
    IPluginV2& operator=(IPluginV2 const&) & = default;
    IPluginV2& operator=(IPluginV2&&) & = default;
// @endcond
};

//!
//! \class IPluginV2Ext
//!
//! \brief Plugin class for user-implemented layers.
//!
//! Plugins are a mechanism for applications to implement custom layers. This
//! interface provides additional capabilities to the IPluginV2 interface by
//! supporting different output data types and broadcast across batches.
//!
//! \see IPluginV2
//!
//! \deprecated Deprecated in TensorRT 8.5. Implement IPluginV3 instead.
//!
class TRT_DEPRECATED IPluginV2Ext : public IPluginV2
{
public:
    //!
    //! \brief Return the DataType of the plugin output at the requested index.
    //!
    //! \param index The output tensor index in the valid range between 0 and getNbOutputs()-1.
    //! \param inputTypes The data types of the input tensors, stored in an array of length nbInputs.
    //! \param nbInputs The number of input tensors. Will be a non-negative integer.
    //!
    //! \return The data type of the output tensor with the provided index if the input tensors have the data types
    //! provided in inputTypes, provided the output tensor index is in the valid range. DataType::kFLOAT must be
    //! returned if the index is not in the valid range.
    //!
    //! The default behavior must be to return the type of the first input, or DataType::kFLOAT if the layer has no
    //! inputs. The returned data type must have a format that is supported by the plugin.
    //!
    //! \see supportsFormat()
    //!
    //! \warning DataType:kBOOL and DataType::kUINT8 are not supported.
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: Yes, this method is required to be thread-safe and may be called from multiple threads
    //!                  when building networks on multiple devices sharing the same plugin.
    //!
    virtual nvinfer1::DataType getOutputDataType(
        int32_t index, nvinfer1::DataType const* inputTypes, int32_t nbInputs) const noexcept
        = 0;

    //!
    //! \brief Return true if the output tensor is broadcast across a batch.
    //!
    //! \param outputIndex The index of the output tensor, which will be in the valid range between 0 and
    //! nbOutputs()-1.
    //! \param inputIsBroadcasted A boolean array of length nbInputs. The i-th element will be true if and only if
    //! the tensor for the ith input is broadcast across a batch.
    //! \param nbInputs The number of inputs. Will be a non-negative integer.
    //!
    //! The values in inputIsBroadcasted refer to broadcasting at the semantic level,
    //! i.e. are unaffected by whether method canBroadcastInputAcrossBatch requests
    //! physical replication of the values.
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: Yes, this method is required to be thread-safe and may be called from multiple threads
    //!                  when building networks on multiple devices sharing the same plugin.
    //!
    //! \deprecated Deprecated in TensorRT 10.0. Implicit batch support is removed in TensorRT 10.0.
    //!
    TRT_DEPRECATED virtual bool isOutputBroadcastAcrossBatch(
        int32_t outputIndex, bool const* inputIsBroadcasted, int32_t nbInputs) const noexcept
        = 0;

    //!
    //! \brief Return true if the plugin can use an input tensor that is broadcast across batch without replication.
    //!
    //! \param inputIndex Index of input that could be broadcast. Will be in the valid range between 0 and
    //! nbInputs - 1 where nbInputs is the maximum number of input tensors supported by this plugin.
    //!
    //! \return true if the index is in the valid range and the plugin is able to broadcast a single copy of this
    //! input tensor across the batch. False otherwise.
    //!
    //! For each input whose tensor is semantically broadcast across a batch,
    //! TensorRT calls this method before calling configurePlugin.
    //! If canBroadcastInputAcrossBatch returns true, TensorRT will not replicate the input tensor;
    //! i.e., there will be a single copy that the plugin must share across the batch.
    //! If it returns false, TensorRT will replicate the input tensor
    //! so that it appears like a non-broadcasted tensor.
    //!
    //! This method is called only for inputs that can be broadcast.
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: Yes, this method is required to be thread-safe and may be called from multiple threads
    //!                  when building networks on multiple devices sharing the same plugin.
    //!
    //! \deprecated Deprecated in TensorRT 10.0. Implicit batch support is removed in TensorRT 10.0.
    //!
    TRT_DEPRECATED virtual bool canBroadcastInputAcrossBatch(int32_t inputIndex) const noexcept = 0;

    //!
    //! \brief Configure the layer with input and output data types.
    //!
    //! This function is called by the builder prior to initialize(). It provides an opportunity for the layer to make
    //! algorithm choices on the basis of its weights, dimensions, data types and maximum batch size.
    //!
    //! \param inputDims The input tensor dimensions. Will be an array of length nbInputs.
    //! \param nbInputs The number of inputs. Will be a non-negative integer.
    //! \param outputDims The output tensor dimensions. Will be an array of length nbOutputs.
    //! \param nbOutputs The number of outputs. Will be a positive integer.
    //! \param inputTypes The data types selected for the plugin inputs. Will be an array of length nbInputs.
    //! \param outputTypes The data types selected for the plugin outputs. Will be an array of length nbOutputs.
    //! \param inputIsBroadcast True for each input that the plugin must broadcast across the batch.
    //!                         Will be an array of length nbInputs.
    //! \param outputIsBroadcast True for each output that TensorRT will broadcast across the batch.
    //!                          Will be an array of length nbOutputs.
    //! \param floatFormat The format selected for the engine for the floating point inputs/outputs.
    //! \param maxBatchSize The maximum batch size. Will be a positive integer.
    //!
    //! The dimensions passed here do not include the outermost batch size (i.e. for 2D image networks, they will be
    //! 3-dimensional CHW dimensions). When inputIsBroadcast or outputIsBroadcast is true, the outermost batch size for
    //! that input or output must be treated as if it is one.
    //! Index 'i' of inputIsBroadcast is true only if the input is semantically broadcast across the batch and
    //! calling canBroadcastInputAcrossBatch with argument 'i' returns true.
    //! Index 'i' of outputIsBroadcast is true only if calling isOutputBroadcastAcrossBatch with argument 'i'
    //! returns true.
    //!
    //! \warning for the floatFormat field, the values PluginFormat::kCHW4, PluginFormat::kCHW16, and
    //! PluginFormat::kCHW32 will not be passed in, this is to keep backward compatibility with TensorRT 5.x series. Use
    //! PluginV2IOExt or PluginV2DynamicExt for other PluginFormats.
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: Yes, this method is required to be thread-safe and may be called from multiple threads
    //!                  when building networks on multiple devices sharing the same plugin. However, TensorRT
    //!                  will not call this method from two threads simultaneously on a given clone of a plugin.
    //!
    virtual void configurePlugin(Dims const* inputDims, int32_t nbInputs, Dims const* outputDims, int32_t nbOutputs,
        DataType const* inputTypes, DataType const* outputTypes, bool const* inputIsBroadcast,
        bool const* outputIsBroadcast, PluginFormat floatFormat, int32_t maxBatchSize) noexcept
        = 0;

    IPluginV2Ext() = default;
    ~IPluginV2Ext() override = default;

    //!
    //! \brief Attach the plugin object to an execution context and grant the plugin the access to some context
    //! resources.
    //!
    //! \param cudnn The cuDNN context handle of the execution context. Will be a valid cuDNN context handle, or
    //!              nullptr if TacticSource::kCUDNN is disabled.
    //! \param cublas The cuBLAS context handle of the execution context. Will be a valid cuBLAS context handle, or
    //!               nullptr if TacticSource::kCUBLAS is disabled.
    //! \param allocator The allocator used by the execution context
    //!
    //! This function is called automatically for each plugin when a new execution context is created. If the context
    //! was created without resources, this method is not called until the resources are assigned. It is also called if
    //! new resources are assigned to the context.
    //!
    //! If the plugin needs per-context resource, it can be allocated here.
    //! The plugin can also get context-owned cuDNN and cuBLAS context here.
    //!
    //! \note The TacticSource::kCUDNN and TacticSource::kCUBLAS flag is disabled by default.
    //! The allocator pointer is unique to each building or execution context instance having overlapping lifetimes.
    //! It can be used as a key to manage resources across plugin instances sharing the same context.
    //! Plugins attached to different contexts will have different handles as their execution will not overlap.
    //!
    //! \see TacticSources
    //! \see getPluginCudnnHandle(void* executionContextIdentifier)
    //! \see getPluginCublasHandle(void* excecutionContextIdentifier)
    //!
    //! \note In the automotive safety context, the cuDNN and cuBLAS parameters will be nullptr because cuDNN and cuBLAS
    //!       are not used by the safe runtime.
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: Yes, this method is required to be thread-safe and may be called from multiple threads
    //!                  when building networks on multiple devices sharing the same plugin.
    //!
    virtual void attachToContext(
        cudnnContext* /*cudnn*/, cublasContext* /*cublas*/, IGpuAllocator* /*allocator*/) noexcept
    {
    }

    //!
    //! \brief Detach the plugin object from its execution context.
    //!
    //! This function is called automatically for each plugin when an execution context is destroyed or the context
    //! resources are unassigned from the context.
    //!
    //! If the plugin owns per-context resource, it can be released here.
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: Yes, this method is required to be thread-safe and may be called from multiple threads
    //!                  when building networks on multiple devices sharing the same plugin.
    //!
    virtual void detachFromContext() noexcept {}

    //!
    //! \brief Clone the plugin object. This copies over internal plugin parameters as well and returns a new plugin
    //! object with these parameters. If the source plugin is pre-configured with configurePlugin(), the returned object
    //! must also be pre-configured. The returned object must allow attachToContext() with a new execution context.
    //! Cloned plugin objects can share the same per-engine immutable resource (e.g. weights) with the source object
    //! (e.g. via ref-counting) to avoid duplication.
    //!
    //! \return A pointer to a cloned plugin object if cloning was successful, otherwise nullptr.
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: Yes, this method is required to be thread-safe and may be called from multiple threads
    //!                  when building networks on multiple devices sharing the same plugin.
    //!
    IPluginV2Ext* clone() const noexcept override = 0;

protected:
    // @cond SuppressDoxyWarnings
    IPluginV2Ext(IPluginV2Ext const&) = default;
    IPluginV2Ext(IPluginV2Ext&&) = default;
    IPluginV2Ext& operator=(IPluginV2Ext const&) & = default;
    IPluginV2Ext& operator=(IPluginV2Ext&&) & = default;
// @endcond

    //!
    //! \brief Return the API version with which this plugin was built. The
    //!  upper byte reserved by TensorRT and is used to differentiate this from IPluginV2.
    //!
    //! \return In the lower three bytes, the TensorRT version in the format
    //!         (major * 100 + minor) * 100 + patch.
    //!         In the upper byte, the value 1.
    //!
    //! Do not override this method as it is used by the TensorRT library to maintain backwards-compatibility with
    //! plugins.
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: Yes, the implementation provided here is safe to call from any thread.
    //!
    int32_t getTensorRTVersion() const noexcept override
    {
        return static_cast<int32_t>((static_cast<uint32_t>(PluginVersion::kV2_EXT) << 24U)
            | (static_cast<uint32_t>(NV_TENSORRT_VERSION) & 0xFFFFFFU));
    }

    //!
    //! \brief Derived classes must not implement this. In a C++11 API it would be override final.
    //!
    //! IPluginV2Ext::configureWithFormat() is a NOP operation for all classes derived from IPluginV2Ext.
    //! These classes call configurePlugin() instead.
    //!
    void configureWithFormat(Dims const* /*inputDims*/, int32_t /*nbInputs*/, Dims const* /*outputDims*/,
        int32_t /*nbOutputs*/, DataType /*type*/, PluginFormat /*format*/, int32_t /*maxBatchSize*/) noexcept override
    {
    }
};

//!
//! \class IPluginV2IOExt
//!
//! \brief Plugin class for user-implemented layers.
//!
//! Plugins are a mechanism for applications to implement custom layers. This interface provides additional
//! capabilities to the IPluginV2Ext interface by extending different I/O data types and tensor formats.
//!
//! \see IPluginV2Ext
//!
//! \deprecated Deprecated in TensorRT 10.0. Implement IPluginV3 instead.
//!
class TRT_DEPRECATED IPluginV2IOExt : public IPluginV2Ext
{
public:
    //!
    //! \brief Configure the layer.
    //!
    //! This function is called by the builder prior to initialize(). It provides an opportunity for the layer to make
    //! algorithm choices on the basis of the provided I/O PluginTensorDesc.
    //!
    //! \param in The input tensors attributes that are used for configuration.
    //! \param nbInput Number of input tensors.
    //! \param out The output tensors attributes that are used for configuration.
    //! \param nbOutput Number of output tensors.
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: Yes, this method is required to be thread-safe and may be called from multiple threads
    //!                  when building networks on multiple devices sharing the same plugin. However, TensorRT
    //!                  will not call this method from two threads simultaneously on a given clone of a plugin.
    //!
    virtual void configurePlugin(
        PluginTensorDesc const* in, int32_t nbInput, PluginTensorDesc const* out, int32_t nbOutput) noexcept
        = 0;

    //!
    //! \brief Return true if plugin supports the format and datatype for the input/output indexed by pos.
    //!
    //! For this method inputs are numbered 0..(nbInputs-1) and outputs are numbered nbInputs..(nbInputs+nbOutputs-1).
    //! Using this numbering, pos is an index into InOut, where 0 <= pos < nbInputs+nbOutputs.
    //!
    //! TensorRT invokes this method to ask if the input/output indexed by pos supports the format/datatype specified
    //! by inOut[pos].format and inOut[pos].type. The override must return true if that format/datatype at inOut[pos]
    //! are supported by the plugin. If support is conditional on other input/output formats/datatypes, the plugin can
    //! make its result conditional on the formats/datatypes in inOut[0..pos-1], which will be set to values
    //! that the plugin supports. The override must not inspect inOut[pos+1..nbInputs+nbOutputs-1],
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
    //!         return inOut.format[pos] == TensorFormat::kLINEAR &&
    //!                (inOut.type[pos] == (pos < 2 ?  DataType::kHALF : DataType::kFLOAT));
    //!
    //! * A definition for a "polymorphic" plugin with two inputs and one output that supports
    //!   any format or type, but the inputs and output must have the same format and type:
    //!
    //!         return pos == 0 || (inOut.format[pos] == inOut.format[0] && inOut.type[pos] == inOut.type[0]);
    //!
    //! Warning: TensorRT will stop asking for formats once it finds kFORMAT_COMBINATION_LIMIT on combinations.
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: Yes, this method is required to be thread-safe and may be called from multiple threads
    //!                  when building networks on multiple devices sharing the same plugin.
    //!
    virtual bool supportsFormatCombination(
        int32_t pos, PluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) const noexcept
        = 0;

    // @cond SuppressDoxyWarnings
    IPluginV2IOExt() = default;
    ~IPluginV2IOExt() override = default;
// @endcond

protected:
// @cond SuppressDoxyWarnings
    IPluginV2IOExt(IPluginV2IOExt const&) = default;
    IPluginV2IOExt(IPluginV2IOExt&&) = default;
    IPluginV2IOExt& operator=(IPluginV2IOExt const&) & = default;
    IPluginV2IOExt& operator=(IPluginV2IOExt&&) & = default;
// @endcond

    //!
    //! \brief Return the API version with which this plugin was built. The upper byte is reserved by TensorRT and is
    //! used to differentiate this from IPluginV2 and IPluginV2Ext.
    //!
    //! Do not override this method as it is used by the TensorRT library to maintain backwards-compatibility with
    //! plugins.
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: Yes, the implementation provided here is safe to call from any thread.
    //!
    int32_t getTensorRTVersion() const noexcept override
    {
        return static_cast<int32_t>((static_cast<uint32_t>(PluginVersion::kV2_IOEXT) << 24U)
            | (static_cast<uint32_t>(NV_TENSORRT_VERSION) & 0xFFFFFFU));
    }

private:
    // Following are obsolete base class methods, and must not be implemented or used.

    //!
    //! \brief Set plugin configuration.
    //!
    void configurePlugin(Dims const*, int32_t, Dims const*, int32_t, DataType const*, DataType const*, bool const*,
        bool const*, PluginFormat, int32_t) noexcept final
    {
    }

    //!
    //! \brief Check if provided data type is supported.
    //!
    bool supportsFormat(DataType, PluginFormat) const noexcept final
    {
        return false;
    }
};

namespace v_1_0
{
class TRT_DEPRECATED IPluginCreator : public IPluginCreatorInterface
{
public:
    //!
    //! \brief Return the plugin name.
    //!
    //! \warning The string returned must be NULL-terminated and have a length of 1024 bytes or less including
    //! the NULL terminator.
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: Yes, this method is required to be thread-safe and may be called from multiple threads
    //!                  when building networks on multiple devices sharing the same plugin or when deserializing
    //!                  multiple engines concurrently sharing plugins.
    //!
    virtual AsciiChar const* getPluginName() const noexcept = 0;

    //!
    //! \brief Return the plugin version.
    //!
    //! \warning The string returned must be NULL-terminated and have a length of 1024 bytes or less including
    //! the NULL terminator.
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: Yes, this method is required to be thread-safe and may be called from multiple threads
    //!                  when building networks on multiple devices sharing the same plugin or when deserializing
    //!                  multiple engines concurrently sharing plugins.
    //!
    virtual AsciiChar const* getPluginVersion() const noexcept = 0;

    //!
    //! \brief Return a list of fields that need to be passed to createPlugin.
    //!
    //! \see PluginFieldCollection
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: Yes, this method is required to be thread-safe and may be called from multiple threads
    //!                  when building networks on multiple devices sharing the same plugin or when deserializing
    //!                  multiple engines concurrently sharing plugins.
    //!
    virtual PluginFieldCollection const* getFieldNames() noexcept = 0;

    //!
    //! \brief Return a plugin object. Return nullptr in case of error.
    //!
    //! \param name A NULL-terminated name string of length 1024 or less, including the NULL terminator.
    //! \param fc A pointer to a collection of fields needed for constructing the plugin.
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: Yes, this method is required to be thread-safe and may be called from multiple threads
    //!                  when building networks on multiple devices sharing the same plugin or when deserializing
    //!                  multiple engines concurrently sharing plugins.
    //!
    virtual IPluginV2* createPlugin(AsciiChar const* name, PluginFieldCollection const* fc) noexcept = 0;

    //!
    //! \brief Called during deserialization of plugin layer. Return a plugin object.
    //!
    //! \param name A NULL-terminated name string of length 1024 or less, including the NULL terminator.
    //! \param serialData The start address of a byte array with the serialized plugin representation.
    //! \param serialLength The length in bytes of the byte array with the serialized plugin representation.
    //!
    //! \return A deserialized plugin object
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: Yes, this method is required to be thread-safe and may be called from multiple threads
    //!                  when building networks on multiple devices sharing the same plugin or when deserializing
    //!                  multiple engines concurrently sharing plugins.
    //!
    virtual IPluginV2* deserializePlugin(AsciiChar const* name, void const* serialData, size_t serialLength) noexcept
        = 0;

    //!
    //! \brief Set the namespace of the plugin creator based on the plugin
    //! library it belongs to. This can be set while registering the plugin creator.
    //!
    //! \param pluginNamespace A NULL-terminated namespace string of length 1024 or less, including the NULL terminator
    //!
    //! \see IPluginRegistry::registerCreator()
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: Yes, this method is required to be thread-safe and may be called from multiple threads
    //!                  when building networks on multiple devices sharing the same plugin or when deserializing
    //!                  multiple engines concurrently sharing plugins.
    //!
    virtual void setPluginNamespace(AsciiChar const* pluginNamespace) noexcept = 0;

    //!
    //! \brief Return the namespace of the plugin creator object.
    //!
    //! \warning The string returned must be NULL-terminated and have a length of 1024 bytes or less including the
    //! NULL terminator.
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: Yes, this method is required to be thread-safe and may be called from multiple threads
    //!                  when building networks on multiple devices sharing the same plugin or when deserializing
    //!                  multiple engines concurrently sharing plugins.
    //!
    virtual AsciiChar const* getPluginNamespace() const noexcept = 0;

    IPluginCreator() = default;
    ~IPluginCreator() override = default;

protected:
    // @cond SuppressDoxyWarnings
    IPluginCreator(IPluginCreator const&) = default;
    IPluginCreator(IPluginCreator&&) = default;
    IPluginCreator& operator=(IPluginCreator const&) & = default;
    IPluginCreator& operator=(IPluginCreator&&) & = default;
    // @endcond
public:
    //!
    //! \brief Return version information associated with this interface. Applications must not override this method.
    //!
    InterfaceInfo getInterfaceInfo() const noexcept override
    {
        return InterfaceInfo{"PLUGIN CREATOR_V1", 1, 0};
    }
};
} // namespace v_1_0

//!
//! \class IPluginCreator
//!
//! \brief Plugin creator class for user implemented layers.
//!
//! \see IPlugin and IPluginFactory
//!
//! \deprecated Deprecated in TensorRT 10.0. Please implement IPluginCreatorV3One
//! along with IPluginV3 plugins instead.
//!
using IPluginCreator = v_1_0::IPluginCreator;

} // namespace nvinfer1

#endif // NV_INFER_RUNTIME_PLUGIN_H
