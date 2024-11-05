/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef TRT_PYTHON_IMPL_PLUGIN_H
#define TRT_PYTHON_IMPL_PLUGIN_H

#include "NvInfer.h"

//!
//! \file plugin.h
//!
//! This file contains definitions for supporting the `tensorrt.plugin` Python module
//!
//! \warning None of the defintions here are part of the TensorRT C++ API and may not follow semantic versioning rules.
//! TensorRT clients must not utilize them directly.
//!

namespace nvinfer1
{
namespace v_1_0
{

class IPluginV3QuickCore : public IPluginCapability
{
public:
    InterfaceInfo getInterfaceInfo() const noexcept override
    {
        return InterfaceInfo{"PLUGIN_V3QUICK_CORE", 1, 0};
    }

    virtual AsciiChar const* getPluginName() const noexcept = 0;

    virtual AsciiChar const* getPluginVersion() const noexcept = 0;

    virtual AsciiChar const* getPluginNamespace() const noexcept = 0;
};

class IPluginV3QuickBuild : public IPluginCapability
{
public:
    InterfaceInfo getInterfaceInfo() const noexcept override
    {
        return InterfaceInfo{"PLUGIN_V3QUICK_BUILD", 1, 0};
    }

    //!
    //! \brief Provide the data types of the plugin outputs if the input tensors have the data types provided.
    //!
    //! \param outputTypes Pre-allocated array to which the output data types should be written.
    //! \param nbOutputs The number of output tensors. This matches the value returned from getNbOutputs().
    //! \param inputTypes The input data types.
    //! \param inputRanks Ranks of the input tensors
    //! \param nbInputs The number of input tensors.
    //!
    //! \return 0 for success, else non-zero
    //!
    virtual int32_t getOutputDataTypes(DataType* outputTypes, int32_t nbOutputs, DataType const* inputTypes,
        int32_t const* inputRanks, int32_t nbInputs) const noexcept = 0;

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
    //! \return 0 for success, else non-zero
    //!
    virtual int32_t getOutputShapes(DimsExprs const* inputs, int32_t nbInputs, DimsExprs const* shapeInputs,
        int32_t nbShapeInputs, DimsExprs* outputs, int32_t nbOutputs, IExprBuilder& exprBuilder) noexcept = 0;

    //!
    //! \brief Configure the plugin. Behaves similarly to `IPluginV3OneBuild::configurePlugin()`
    //!
    //! \return 0 for success, else non-zero
    //!
    virtual int32_t configurePlugin(DynamicPluginTensorDesc const* in, int32_t nbInputs,
        DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept = 0;

    //!
    //! \brief Get number of format combinations supported by the plugin for the I/O characteristics indicated by
    //! `inOut`.
    //!
    virtual int32_t getNbSupportedFormatCombinations(
        DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept = 0;

    //!
    //! \brief Write all format combinations supported by the plugin for the I/O characteristics indicated by `inOut` to
    //! `supportedCombinations`. It is guaranteed to have sufficient memory allocated for (nbInputs + nbOutputs) *
    //! getNbSupportedFormatCombinations() `PluginTensorDesc`s.
    //!
    //! \return 0 for success, else non-zero
    //!
    virtual int32_t getSupportedFormatCombinations(DynamicPluginTensorDesc const* inOut, int32_t nbInputs,
        int32_t nbOutputs, PluginTensorDesc* supportedCombinations, int32_t nbFormatCombinations) noexcept = 0;

    //!
    //! \brief Get the number of outputs from the plugin.
    //!
    virtual int32_t getNbOutputs() const noexcept = 0;

    //!
    //! \brief Communicates to TensorRT that the output at the specified output index is aliased to the input at the
    //! returned index. Behaves similary to `v_2_0::IPluginV3OneBuild.getAliasedInput()`.
    //!
    virtual int32_t getAliasedInput(int32_t outputIndex) noexcept
    {
        return -1;
    }

    //!
    //! \brief Query for any custom tactics that the plugin intends to use specific to the I/O characteristics indicated
    //! by the immediately preceding call to `configurePlugin()`.
    //!
    //! \return 0 for success, else non-zero
    //!
    virtual int32_t getValidTactics(int32_t* tactics, int32_t nbTactics) noexcept
    {
        return 0;
    }

    //!
    //! \brief Query for number of custom tactics related to the `getValidTactics()` call.
    //!
    virtual int32_t getNbTactics() noexcept
    {
        return 0;
    }

    //!
    //! \brief Called to query the suffix to use for the timing cache ID. May be called anytime after plugin creation.
    //!
    virtual char const* getTimingCacheID() noexcept
    {
        return nullptr;
    }

    //!
    //! \brief Query for a string representing the configuration of the plugin. May be called anytime after
    //! plugin creation.
    //!
    virtual char const* getMetadataString() noexcept
    {
        return nullptr;
    }
};

class IPluginV3QuickRuntime : public IPluginCapability
{
public:
    InterfaceInfo getInterfaceInfo() const noexcept override
    {
        return InterfaceInfo{"PLUGIN_V3QUICK_RUNTIME", 1, 0};
    }

    //!
    //! \brief Set the tactic to be used in the subsequent call to enqueue(). Behaves similar to
    //! `IPluginV3OneRuntime::setTactic()`.
    //!
    //! \return 0 for success, else non-zero
    //!
    virtual int32_t setTactic(int32_t tactic) noexcept
    {
        return 0;
    }

    //!
    //! \brief Execute the plugin.
    //!
    //! \param inputDesc how to interpret the memory for the input tensors.
    //! \param outputDesc how to interpret the memory for the output tensors.
    //! \param inputs The memory for the input tensors.
    //! \param inputStrides Strides for input tensors.
    //! \param outputStrides Strides for output tensors.
    //! \param outputs The memory for the output tensors.
    //! \param nbInputs Number of input tensors.
    //! \param nbOutputs Number of output tensors.
    //! \param stream The stream in which to execute the kernels.
    //!
    //! \return 0 for success, else non-zero
    //!
    virtual int32_t enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs, Dims const* inputStrides, Dims const* outputStrides,
        int32_t nbInputs, int32_t nbOutputs, cudaStream_t stream) noexcept = 0;

    //!
    //! \brief Get the plugin fields which should be serialized.
    //!
    virtual PluginFieldCollection const* getFieldsToSerialize() noexcept = 0;
};

class IPluginCreatorV3Quick : public IPluginCreatorInterface
{
public:
    InterfaceInfo getInterfaceInfo() const noexcept override
    {
        return InterfaceInfo{"PLUGIN CREATOR_V3QUICK", 1, 0};
    }

    //!
    //! \brief Return a plugin object. Return nullptr in case of error.
    //!
    //! \param name A NULL-terminated name string of length 1024 or less, including the NULL terminator.
    //! \param namespace A NULL-terminated name string of length 1024 or less, including the NULL terminator.
    //! \param fc A pointer to a collection of fields needed for constructing the plugin.
    //! \param phase The TensorRT phase in which the plugin is being created
    //!
    virtual IPluginV3* createPlugin(AsciiChar const* name, AsciiChar const* nspace, PluginFieldCollection const* fc,
        TensorRTPhase phase) noexcept = 0;

    //!
    //! \brief Return a list of fields that need to be passed to createPlugin() when creating a plugin for use in the
    //! TensorRT build phase.
    //!
    virtual PluginFieldCollection const* getFieldNames() noexcept = 0;

    virtual AsciiChar const* getPluginName() const noexcept = 0;

    virtual AsciiChar const* getPluginVersion() const noexcept = 0;

    virtual AsciiChar const* getPluginNamespace() const noexcept = 0;

    IPluginCreatorV3Quick() = default;
    virtual ~IPluginCreatorV3Quick() = default;

protected:
    IPluginCreatorV3Quick(IPluginCreatorV3Quick const&) = default;
    IPluginCreatorV3Quick(IPluginCreatorV3Quick&&) = default;
    IPluginCreatorV3Quick& operator=(IPluginCreatorV3Quick const&) & = default;
    IPluginCreatorV3Quick& operator=(IPluginCreatorV3Quick&&) & = default;
};

} // namespace v_1_0

//!
//! \class IPluginV3QuickCore
//!
//! \brief Provides core capability (`IPluginCapability::kCORE` for quickly-deployable TRT plugins)
//!
//! \warning This class is strictly for the purpose of supporting quickly-deployable TRT Python plugins and is not part
//! of the public TensorRT C++ API. Users must not inherit from this class.
//!
using IPluginV3QuickCore = v_1_0::IPluginV3QuickCore;

//!
//! \class IPluginV3QuickBuild
//!
//! \brief Provides build capability (`IPluginCapability::kBUILD` for quickly-deployable TRT plugins)
//!
//! \warning This class is strictly for the purpose of supporting quickly-deployable TRT Python plugins and is not part
//! of the public TensorRT C++ API. Users must not inherit from this class.
//!
using IPluginV3QuickBuild = v_1_0::IPluginV3QuickBuild;

//!
//! \class IPluginV3QuickRuntime
//!
//! \warning This class is strictly for the purpose of supporting quickly-deployable TRT Python plugins and is not part
//! of the public TensorRT C++ API. Users must not inherit from this class.
//!
using IPluginV3QuickRuntime = v_1_0::IPluginV3QuickRuntime;

//!
//! \class IPluginCreatorV3Quick
//!
//! \warning This class is strictly for the purpose of supporting quickly-deployable TRT Python plugins and is not part
//! of the public TensorRT C++ API. Users must not inherit from this class.
//!
using IPluginCreatorV3Quick = v_1_0::IPluginCreatorV3Quick;

} // namespace nvinfer1

#endif // TRT_PYTHON_IMPL_PLUGIN_H
