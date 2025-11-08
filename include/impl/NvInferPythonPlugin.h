/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
//! \file NvInferPythonPlugin.h
//!
//! This file contains definitions for supporting the `tensorrt.plugin` Python module
//!
//! \warning None of the defintions here are part of the TensorRT C++ API and may not follow semantic versioning rules.
//! TensorRT clients must not utilize them directly.
//!

namespace nvinfer1
{

//! \enum PluginArgType
//! \brief Numeric type of an extra kernel input argument in an AOT Python plugin
enum class PluginArgType : int32_t
{
    //! Integer argument
    kINT = 0,
};

//! \enum PluginArgDataType
//! \brief Data type of an extra kernel input argument in an AOT Python plugin
enum class PluginArgDataType : int32_t
{
    //! 8-bit signed integer
    kINT8 = 0,
    //! 16-bit signed integer
    kINT16 = 1,
    //! 32-bit signed integer
    kINT32 = 2,
};
//! \class ISymExpr
//! \brief Generic interface for a scalar symbolic expression implementable by a Python plugin / TensorRT Python backend
class ISymExpr
{
public:
    //! \brief Get the type of the symbolic expression
    virtual PluginArgType getType() const noexcept = 0;
    //! \brief Get the data type of the symbolic expression
    virtual PluginArgDataType getDataType() const noexcept = 0;
    //! \brief Underlying symbolic expression
    virtual void* getExpr() noexcept = 0;
};

//! Impl class for ISymExprs
class ISymExprsImpl
{
public:
    virtual ISymExpr* getSymExpr(int32_t index) const noexcept = 0;
    virtual bool setSymExpr(int32_t index, ISymExpr* symExpr) noexcept = 0;
    virtual int32_t getNbSymExprs() const noexcept = 0;
    virtual bool setNbSymExprs(int32_t count) noexcept = 0;

    virtual ~ISymExprsImpl() noexcept = default;
};

//! \class ISymExprs
//! \brief Allows for a sequence of symbolic expressions to be communicated to the TensorRT backend
//! \note Clients must not implement this class.
//! \see ISymExpr
class ISymExprs
{
public:
    //! \brief Get the symbolic expression at the given index
    //! \return A pointer to the symbolic expression or nullptr if the index is out of range
    ISymExpr* getSymExpr(int32_t index) const noexcept
    {
        return mImpl->getSymExpr(index);
    }

    //! \brief Set the symbolic expression at the given index
    //! \return true if the index is in range and the symbolic expression was set successfully, false otherwise
    bool setSymExpr(int32_t index, ISymExpr* symExpr) noexcept
    {
        return mImpl->setSymExpr(index, symExpr);
    }

    //! \brief Get the number of symbolic expressions
    int32_t getNbSymExprs() const noexcept
    {
        return mImpl->getNbSymExprs();
    }

    //! \brief Set the number of symbolic expressions
    //! \return true if the number of symbolic expressions was set successfully, false otherwise
    bool setNbSymExprs(int32_t count) noexcept
    {
        return mImpl->setNbSymExprs(count);
    }

protected:
    ISymExprsImpl* mImpl{nullptr};
    virtual ~ISymExprs() noexcept = default;
};

//! \enum QuickPluginCreationRequest
//! \brief Communicates preference when a quickly deployable plugin is to be added to the network
enum class QuickPluginCreationRequest : int32_t
{
    //! No preference specified
    kUNKNOWN = 0,
    //! JIT plugin is preferred
    kPREFER_JIT = 1,
    //! AOT plugin is preferred
    kPREFER_AOT = 2,
    //! JIT plugin must be used. TensorRT should fail if a JIT implementation cannot be found.
    kSTRICT_JIT = 3,
    //! AOT plugin must be used. TensorRT should fail if an AOT implementation cannot be found.
    kSTRICT_AOT = 4,
};

//! Impl class for IKernelLaunchParams
class IKernelLaunchParamsImpl
{
public:
    virtual ISymExpr* getGridX() noexcept = 0;
    virtual bool setGridX(ISymExpr* gridX) noexcept = 0;

    virtual ISymExpr* getGridY() noexcept = 0;
    virtual bool setGridY(ISymExpr* gridY) noexcept = 0;

    virtual ISymExpr* getGridZ() noexcept = 0;
    virtual bool setGridZ(ISymExpr* gridZ) noexcept = 0;

    virtual ISymExpr* getBlockX() noexcept = 0;
    virtual bool setBlockX(ISymExpr* blockX) noexcept = 0;

    virtual ISymExpr* getBlockY() noexcept = 0;
    virtual bool setBlockY(ISymExpr* blockY) noexcept = 0;

    virtual ISymExpr* getBlockZ() noexcept = 0;
    virtual bool setBlockZ(ISymExpr* blockZ) noexcept = 0;

    virtual ISymExpr* getSharedMem() noexcept = 0;
    virtual bool setSharedMem(ISymExpr* sharedMem) noexcept = 0;

    virtual ~IKernelLaunchParamsImpl() noexcept = default;
};

//! \class IKernelLaunchParams
//! \brief Allows for kernel launch parameters to be communicated to the TensorRT backend
//! \note Clients must not implement this class.
class IKernelLaunchParams
{
public:
    //! Get the X dimension of the grid
    ISymExpr* getGridX() noexcept
    {
        return mImpl->getGridX();
    }

    //! \brief Set the X dimension of the grid
    //! \return true if the grid's X dimension was set successfully, false otherwise
    bool setGridX(ISymExpr* gridX) noexcept
    {
        return mImpl->setGridX(gridX);
    }

    //! Get the Y dimension of the grid
    ISymExpr* getGridY() noexcept
    {
        return mImpl->getGridY();
    }

    //! \brief Set the Y dimension of the grid
    //! \return true if the grid's Y dimension was set successfully, false otherwise
    bool setGridY(ISymExpr* gridY) noexcept
    {
        return mImpl->setGridY(gridY);
    }

    //! Get the Z dimension of the grid
    ISymExpr* getGridZ() noexcept
    {
        return mImpl->getGridZ();
    }

    //! \brief Set the Z dimension of the grid
    //! \return true if the grid's Z dimension was set successfully, false otherwise
    bool setGridZ(ISymExpr* gridZ) noexcept
    {
        return mImpl->setGridZ(gridZ);
    }

    //! \brief Get the X dimension of each thread block
    ISymExpr* getBlockX() noexcept
    {
        return mImpl->getBlockX();
    }

    //! \brief Set the X dimension of each thread block
    //! \return true if each thread block's X dimension was set successfully, false otherwise
    bool setBlockX(ISymExpr* blockX) noexcept
    {
        return mImpl->setBlockX(blockX);
    }

    //! \brief Get the Y dimension of each thread block
    ISymExpr* getBlockY() noexcept
    {
        return mImpl->getBlockY();
    }

    //! \brief Set the Y dimension of each thread block
    //! \return true if each thread block's Y dimension was set successfully, false otherwise
    bool setBlockY(ISymExpr* blockY) noexcept
    {
        return mImpl->setBlockY(blockY);
    }

    //! \brief Get the Z dimension of each thread block
    ISymExpr* getBlockZ() noexcept
    {
        return mImpl->getBlockZ();
    }

    //! \brief Set the Z dimension of each thread block
    //! \return true if each thread block's Z dimension was set successfully, false otherwise
    bool setBlockZ(ISymExpr* blockZ) noexcept
    {
        return mImpl->setBlockZ(blockZ);
    }

    //! \brief Get the dynamic shared-memory per thread block in bytes
    ISymExpr* getSharedMem() noexcept
    {
        return mImpl->getSharedMem();
    }

    //! \brief Set the dynamic shared-memory per thread block in bytes
    //! \return true if the dynamic shared-memory per thread block was set successfully, false otherwise
    bool setSharedMem(ISymExpr* sharedMem) noexcept
    {
        return mImpl->setSharedMem(sharedMem);
    }

protected:
    IKernelLaunchParamsImpl* mImpl{nullptr};
    virtual ~IKernelLaunchParams() noexcept = default;
};

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

class IPluginV3QuickAOTBuild : public IPluginV3QuickBuild
{
public:
    InterfaceInfo getInterfaceInfo() const noexcept override
    {
        return InterfaceInfo{"PLUGIN_V3QUICKAOT_BUILD", 1, 0};
    }

    //! \brief Get the launch parameters for the kernel to be used for the specified input and output types/formats and
    //! any corresponding custom tactics.
    //!        If custom tactics are being advertised by the plugin, the corresponding tactic is the one specified by
    //!        the immediately preceding call to setTactic().
    //!
    //! \param inputs Expressions for dimensions of the input tensors
    //! \param inOut The input and output tensors' attributes
    //! \param nbInputs The number of input tensors
    //! \param nbOutputs The number of output tensors
    //! \param launchParams Interface which allows the specification of kernel launch parameters as symbolic expressions
    //! of the input dimensions
    //! \param extraArgs Interface which allows the specification of any scalar arguments to be
    //! passed to the kernel, as symbolic expressions of the input dimensions
    //! \param exprBuilder Object for generating new symbolic expressions
    //!
    //! \return 0 for success, else non-zero
    //!
    virtual int32_t getLaunchParams(DimsExprs const* inputs, DynamicPluginTensorDesc const* inOut, int32_t nbInputs,
        int32_t nbOutputs, IKernelLaunchParams* launchParams, ISymExprs* extraArgs,
        IExprBuilder& exprBuilder) noexcept = 0;

    //!
    //! \brief Get the compiled form for the kernel to be used for the specified input and output types/formats and any
    //! corresponding custom tactics.
    //!        If custom tactics are being advertised by the plugin, the corresponding tactic is the one specified by
    //!        the immediately preceding call to setTactic().
    //!
    //! \param in The input tensors' attributes that are used for configuration.
    //! \param nbInputs Number of input tensors.
    //! \param out The output tensors' attributes that are used for configuration.
    //! \param nbOutputs Number of output tensors.
    //! \param kernelName The name for the kernel.
    //! \param compiledKernel Compiled form of the kernel.
    //! \param compiledKernelSize The size of the compiled kernel.
    //!
    //! \return 0 for success, else non-zero
    //!
    virtual int32_t getKernel(PluginTensorDesc const* in, int32_t nbInputs, PluginTensorDesc const* out,
        int32_t nbOutputs, const char** kernelName, char** compiledKernel, int32_t* compiledKernelSize) noexcept = 0;

    //!
    //! \brief Set the tactic to be used in the subsequent call to enqueue(). Behaves similar to
    //! IPluginV3OneRuntime::setTactic()
    //!
    //! \return 0 for success, else non-zero
    //!
    virtual int32_t setTactic(int32_t tactic) noexcept
    {
        return 0;
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
    //! \param quickPluginCreationRequest Whether a JIT or AOT plugin should be created
    //!
    virtual IPluginV3* createPlugin(AsciiChar const* name, AsciiChar const* nspace, PluginFieldCollection const* fc,
        TensorRTPhase phase, QuickPluginCreationRequest quickPluginCreationRequest) noexcept = 0;

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
//! \brief Provides core capability (`IPluginCapability::kCORE`) for quickly-deployable TRT plugins
//!
//! \warning This class is strictly for the purpose of supporting quickly-deployable TRT Python plugins and is not part
//! of the public TensorRT C++ API. Users must not inherit from this class.
//!
using IPluginV3QuickCore = v_1_0::IPluginV3QuickCore;

//!
//! \class IPluginV3QuickBuild
//!
//! \brief Provides build capability (`IPluginCapability::kBUILD`) for quickly-deployable TRT plugins
//!
//! \warning This class is strictly for the purpose of supporting quickly-deployable TRT Python plugins and is not part
//! of the public TensorRT C++ API. Users must not inherit from this class.
//!
using IPluginV3QuickBuild = v_1_0::IPluginV3QuickBuild;

//!
//! \class IPluginV3QuickAOTBuild
//!
//! \brief Provides additional build capabilities for AOT quickly-deployable TRT plugins. Descends from
//! IPluginV3QuickBuild.
//!
//! \warning This class is strictly for the purpose of supporting quickly-deployable TRT Python plugins and is not part
//! of the public TensorRT C++ API. Users must not inherit from this class.
//!
using IPluginV3QuickAOTBuild = v_1_0::IPluginV3QuickAOTBuild;

//!
//! \class IPluginV3QuickRuntime
//!
//! \brief Provides runtime capability (`IPluginCapability::kRUNTIME`) for JIT quickly-deployable TRT plugins
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
