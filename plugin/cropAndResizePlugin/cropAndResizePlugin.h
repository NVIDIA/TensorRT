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

#ifndef TRT_CROP_AND_RESIZE_PLUGIN_H
#define TRT_CROP_AND_RESIZE_PLUGIN_H

#include "NvInferRuntime.h"
#include "common/kernels/kernel.h"
#include "common/plugin.h"
#include <string>
#include <vector>

namespace nvinfer1
{
namespace plugin
{

//!
//! \brief CropAndResizeDynamicPlugin implements a TensorRT plugin that performs
//! crop and resize operations on input tensors based on provided boxes.
//!
class CropAndResizeDynamicPlugin : public IPluginV3,
                                   public IPluginV3OneCore,
                                   public IPluginV3OneBuild,
                                   public IPluginV3OneRuntime
{
public:
    //!
    //! \brief Constructs a CropAndResizeDynamicPlugin with specified crop dimensions.
    //!
    //! \param cropWidth Width of the output crop
    //! \param cropHeight Height of the output crop
    //!
    CropAndResizeDynamicPlugin(int32_t cropWidth, int32_t cropHeight);

    //!
    //! \brief Constructs a CropAndResizeDynamicPlugin with full parameters.
    //!
    //! \param cropWidth Width of the output crop
    //! \param cropHeight Height of the output crop
    //! \param depth Depth (channels) of the input tensor
    //! \param inputWidth Width of the input tensor
    //! \param inputHeight Height of the input tensor
    //! \param maxBoxNum Maximum number of boxes
    //!
    CropAndResizeDynamicPlugin(int32_t cropWidth, int32_t cropHeight, int32_t depth, int32_t inputWidth,
        int32_t inputHeight, int32_t maxBoxNum);

    // It doesn't make sense to make CropAndResizeDynamicPlugin without arguments, so we delete default constructor.
    CropAndResizeDynamicPlugin() noexcept = delete;

    ~CropAndResizeDynamicPlugin() noexcept override;

    // IPluginV3 methods
    IPluginCapability* getCapabilityInterface(PluginCapabilityType type) noexcept override;
    IPluginV3* clone() noexcept override;
    char const* getPluginName() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    char const* getPluginNamespace() const noexcept override;
    void setPluginNamespace(char const* libNamespace) noexcept;

    // IPluginV3OneCore methods
    int32_t getNbOutputs() const noexcept override;
    int32_t getOutputDataTypes(
        DataType* outputTypes, int32_t nbOutputs, DataType const* inputTypes, int32_t nbInputs) const noexcept override;

    // IPluginV3OneBuild methods
    int32_t getOutputShapes(DimsExprs const* inputs, int32_t nbInputs, DimsExprs const* shapeInputs,
        int32_t nbShapeInputs, DimsExprs* outputs, int32_t nbOutputs, IExprBuilder& exprBuilder) noexcept override;
    bool supportsFormatCombination(
        int32_t pos, DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept override;
    int32_t configurePlugin(DynamicPluginTensorDesc const* in, int32_t nbInputs, DynamicPluginTensorDesc const* out,
        int32_t nbOutputs) noexcept override;
    int32_t onShapeChange(PluginTensorDesc const* inputs, int32_t nbInputs, PluginTensorDesc const* outputs,
        int32_t nbOutputs) noexcept override;
    size_t getWorkspaceSize(DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
        DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept override;

    // IPluginV3OneRuntime methods
    int32_t enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc, void const* const* inputs,
        void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;
    IPluginV3* attachToContext(IPluginResourceContext* context) noexcept override;
    PluginFieldCollection const* getFieldsToSerialize() noexcept override;

private:
    int32_t mCropWidth;
    int32_t mCropHeight;
    int32_t mDepth;
    int32_t mInputWidth;
    int32_t mInputHeight;
    int32_t mNumBoxes;
    std::string mNamespace;

    PluginFieldCollection mFCToSerialize;
    std::vector<PluginField> mDataToSerialize;
};

//!
//! \brief Creator class for CropAndResizeDynamicPlugin
//!
class CropAndResizeDynamicPluginCreator : public IPluginCreatorV3One
{
public:
    CropAndResizeDynamicPluginCreator();
    ~CropAndResizeDynamicPluginCreator() override = default;

    // IPluginCreatorV3One methods
    char const* getPluginName() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    PluginFieldCollection const* getFieldNames() noexcept override;
    void setPluginNamespace(char const* pluginNamespace) noexcept;
    char const* getPluginNamespace() const noexcept override;
    IPluginV3* createPlugin(char const* name, PluginFieldCollection const* fc, TensorRTPhase phase) noexcept override;

private:
    PluginFieldCollection mFC;
    std::vector<PluginField> mPluginAttributes;
    std::string mNamespace;
};

} // namespace plugin

} // namespace nvinfer1

#endif // TRT_CROP_AND_RESIZE_PLUGIN_H
