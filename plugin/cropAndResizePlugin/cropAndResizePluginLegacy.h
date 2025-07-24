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

#ifndef TRT_CROP_AND_RESIZE_PLUGIN_LEGACY_H
#define TRT_CROP_AND_RESIZE_PLUGIN_LEGACY_H

#include "NvInferPlugin.h"
#include "common/kernels/kernel.h"
#include "common/plugin.h"
#include <string>
#include <vector>

namespace nvinfer1
{
namespace plugin
{

//!
//! \brief Legacy implementation of CropAndResizePlugin that implements IPluginV2Ext.
//! \deprecated This class is deprecated, use CropAndResizeDynamicPlugin (IPluginV3) instead.
//!
class TRT_DEPRECATED CropAndResizePlugin : public IPluginV2Ext
{
public:
    //!
    //! \brief Constructs CropAndResizePlugin with specified crop dimensions.
    //!
    //! \param cropWidth Width of the output crop
    //! \param cropHeight Height of the output crop
    //!
    CropAndResizePlugin(int32_t cropWidth, int32_t cropHeight);

    //!
    //! \brief Constructs CropAndResizePlugin with full parameters.
    //!
    //! \param cropWidth Width of the output crop
    //! \param cropHeight Height of the output crop
    //! \param depth Depth (channels) of the input tensor
    //! \param inputWidth Width of the input tensor
    //! \param inputHeight Height of the input tensor
    //! \param maxBoxNum Maximum number of boxes
    //!
    CropAndResizePlugin(int32_t cropWidth, int32_t cropHeight, int32_t depth, int32_t inputWidth, int32_t inputHeight,
        int32_t maxBoxNum);

    //!
    //! \brief Deserialize constructor
    //!
    //! \param serialBuf The buffer containing serialized plugin data
    //! \param serialSize Size of the serialized data in bytes
    //!
    CropAndResizePlugin(void const* serialBuf, size_t serialSize);

    // It doesn't make sense to make CropAndResizePlugin without arguments, so we delete default constructor.
    CropAndResizePlugin() = delete;

    ~CropAndResizePlugin() override;

    int32_t getNbOutputs() const noexcept override;

    Dims getOutputDimensions(int32_t index, Dims const* inputs, int32_t nbInputDims) noexcept override;

    int32_t initialize() noexcept override;

    void terminate() noexcept override;

    size_t getWorkspaceSize(int32_t /*maxBatchSize*/) const noexcept override;

    int32_t enqueue(int32_t batchSize, void const* const* inputs, void* const* outputs, void* workspace,
        cudaStream_t stream) noexcept override;

    size_t getSerializationSize() const noexcept override;

    void serialize(void* buffer) const noexcept override;

    bool supportsFormat(DataType type, PluginFormat format) const noexcept override;

    char const* getPluginType() const noexcept override;

    char const* getPluginVersion() const noexcept override;

    void destroy() noexcept override;

    nvinfer1::IPluginV2Ext* clone() const noexcept override;

    void setPluginNamespace(char const* pluginNamespace) noexcept override;

    char const* getPluginNamespace() const noexcept override;

    DataType getOutputDataType(
        int32_t index, nvinfer1::DataType const* inputTypes, int32_t nbInputs) const noexcept override;

    bool isOutputBroadcastAcrossBatch(
        int32_t outputIndex, bool const* inputIsBroadcasted, int32_t nbInputs) const noexcept override;

    bool canBroadcastInputAcrossBatch(int32_t inputIndex) const noexcept override;

    void attachToContext(
        cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) noexcept override;

    void configurePlugin(Dims const* inputDims, int32_t nbInputs, Dims const* outputDims, int32_t nbOutputs,
        DataType const* inputTypes, DataType const* outputTypes, bool const* inputIsBroadcast,
        bool const* outputIsBroadcast, PluginFormat floatFormat, int32_t maxBatchSize) noexcept override;

    void detachFromContext() noexcept override;

private:
    int32_t mCropWidth;
    int32_t mCropHeight;
    int32_t mDepth;
    int32_t mInputWidth;
    int32_t mInputHeight;
    int32_t mNumBoxes;
    std::string mNamespace;
};

//!
//! \brief Legacy dynamic implementation of CropAndResizePlugin that implements IPluginV2DynamicExt.
//! \deprecated This class is deprecated, use CropAndResizeDynamicPlugin (IPluginV3) instead.
class TRT_DEPRECATED CropAndResizeDynamicPluginLegacy : public IPluginV2DynamicExt
{
public:
    //!
    //! \brief Constructs CropAndResizeDynamicPluginLegacy with specified crop dimensions.
    //!
    //! \param cropWidth Width of the output crop
    //! \param cropHeight Height of the output crop
    //!
    CropAndResizeDynamicPluginLegacy(int32_t cropWidth, int32_t cropHeight);

    //!
    //! \brief Constructs CropAndResizeDynamicPluginLegacy with full parameters.
    //!
    //! \param cropWidth Width of the output crop
    //! \param cropHeight Height of the output crop
    //! \param depth Depth (channels) of the input tensor
    //! \param inputWidth Width of the input tensor
    //! \param inputHeight Height of the input tensor
    //! \param maxBoxNum Maximum number of boxes
    //!
    CropAndResizeDynamicPluginLegacy(int32_t cropWidth, int32_t cropHeight, int32_t depth, int32_t inputWidth,
        int32_t inputHeight, int32_t maxBoxNum);
    CropAndResizeDynamicPluginLegacy(void const* serialBuf, size_t serialSize);

    // It doesn't make sense to make CropAndResizeDynamicPlugin without arguments, so we delete default constructor.
    CropAndResizeDynamicPluginLegacy() noexcept = delete;

    ~CropAndResizeDynamicPluginLegacy() noexcept override;

    // IPluginV2 methods
    char const* getPluginType() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    int32_t getNbOutputs() const noexcept override;
    int32_t initialize() noexcept override;
    void terminate() noexcept override;
    size_t getSerializationSize() const noexcept override;
    void serialize(void* buffer) const noexcept override;
    void destroy() noexcept override;
    void setPluginNamespace(char const* libNamespace) noexcept override;
    char const* getPluginNamespace() const noexcept override;

    // IPluginV2Ext methods
    DataType getOutputDataType(
        int32_t index, nvinfer1::DataType const* inputType, int32_t nbInputs) const noexcept override;

    // IPluginV2DynamicExt methods
    IPluginV2DynamicExt* clone() const noexcept override;
    DimsExprs getOutputDimensions(
        int32_t outputIndex, DimsExprs const* inputs, int32_t nbInputs, IExprBuilder& exprBuilder) noexcept override;
    bool supportsFormatCombination(
        int32_t pos, PluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept override;
    void configurePlugin(DynamicPluginTensorDesc const* in, int32_t nbInputs, DynamicPluginTensorDesc const* out,
        int32_t nbOutputs) noexcept override;
    size_t getWorkspaceSize(PluginTensorDesc const* inputs, int32_t nbInputs, PluginTensorDesc const* outputs,
        int32_t nbOutputs) const noexcept override;
    int32_t enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc, void const* const* inputs,
        void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

private:
    int32_t mCropWidth;
    int32_t mCropHeight;
    int32_t mDepth;
    int32_t mInputWidth;
    int32_t mInputHeight;
    int32_t mNumBoxes;
    std::string mNamespace;
};

//!
//! \brief Base creator class for CropAndResize plugins
//!
class CropAndResizeBasePluginCreator : public nvinfer1::pluginInternal::BaseCreator
{
public:
    CropAndResizeBasePluginCreator();
    ~CropAndResizeBasePluginCreator() override = default;
    char const* getPluginName() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    PluginFieldCollection const* getFieldNames() noexcept override;

protected:
    PluginFieldCollection mFC;
    std::vector<PluginField> mPluginAttributes;
    std::string mPluginName;
    std::string mPluginVersion;
};

//!
//! \brief Creator class for the deprecated CropAndResizePlugin
//! \deprecated This class is deprecated, use CropAndResizeDynamicPluginCreator instead
//!
class TRT_DEPRECATED CropAndResizePluginCreator : public CropAndResizeBasePluginCreator
{
public:
    CropAndResizePluginCreator();
    ~CropAndResizePluginCreator() override = default;
    IPluginV2Ext* createPlugin(char const* name, PluginFieldCollection const* fc) noexcept override;
    IPluginV2Ext* deserializePlugin(char const* name, void const* serialData, size_t serialLength) noexcept override;
};

//!
//! \brief Creator class for CropAndResizeDynamicPluginLegacy
//! \deprecated This class is deprecated, use CropAndResizeDynamicPluginCreator (IPluginV3) instead
//!
class TRT_DEPRECATED CropAndResizeDynamicPluginLegacyCreator : public CropAndResizeBasePluginCreator
{
public:
    CropAndResizeDynamicPluginLegacyCreator();
    ~CropAndResizeDynamicPluginLegacyCreator() override = default;
    IPluginV2DynamicExt* createPlugin(char const* name, PluginFieldCollection const* fc) noexcept override;
    IPluginV2DynamicExt* deserializePlugin(
        char const* name, void const* serialData, size_t serialLength) noexcept override;
};

} // namespace plugin

} // namespace nvinfer1

#endif // TRT_CROP_AND_RESIZE_PLUGIN_LEGACY_H
