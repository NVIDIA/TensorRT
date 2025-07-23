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

#ifndef _DECODE_BBOX_3D_H_
#define _DECODE_BBOX_3D_H_

#include "NvInferPlugin.h"
#include <cuda_runtime_api.h>
#include <string>
#include <vector>

namespace nvinfer1
{
namespace plugin
{

class DecodeBbox3DPlugin : public nvinfer1::IPluginV2DynamicExt
{
public:
    DecodeBbox3DPlugin() = delete;
    DecodeBbox3DPlugin(float xMin, float xMax, float yMin, float yMax, float zMin, float zMax, int32_t numDirBins,
        float dirOffset, float dirLimitOffset, std::vector<float> const& anchorBottomHeight,
        std::vector<float> const& anchors, float scoreThresh);
    DecodeBbox3DPlugin(float xMin, float xMax, float yMin, float yMax, float zMin, float zMax, int32_t numDirBins,
        float dirOffset, float dirLimitOffset, std::vector<float> const& anchorBottomHeight,
        std::vector<float> const& anchors, float scoreThresh, int32_t featureH, int32_t featureW);
    DecodeBbox3DPlugin(void const* data, size_t length);
    // IPluginV2DynamicExt Methods
    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;
    nvinfer1::DimsExprs getOutputDimensions(int32_t outputIndex, nvinfer1::DimsExprs const* inputs, int32_t nbInputs,
        nvinfer1::IExprBuilder& exprBuilder) noexcept override;
    bool supportsFormatCombination(
        int32_t pos, nvinfer1::PluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept override;
    void configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int32_t nbInputs,
        nvinfer1::DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept override;
    size_t getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs, int32_t nbInputs,
        nvinfer1::PluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept override;
    int32_t enqueue(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;
    // IPluginV2Ext Methods
    nvinfer1::DataType getOutputDataType(
        int32_t index, nvinfer1::DataType const* inputTypes, int32_t nbInputs) const noexcept override;
    // IPluginV2 Methods
    char const* getPluginType() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    int32_t getNbOutputs() const noexcept override;
    int32_t initialize() noexcept override;
    void terminate() noexcept override;
    size_t getSerializationSize() const noexcept override;
    void serialize(void* buffer) const noexcept override;
    void destroy() noexcept override;
    void setPluginNamespace(char const* pluginNamespace) noexcept override;
    char const* getPluginNamespace() const noexcept override;

private:
    std::string mNamespace;
    float mMinXRange;
    float mMaxXRange;
    float mMinYRange;
    float mMaxYRange;
    float mMinZRange;
    float mMaxZRange;
    int32_t mNumDirBins;
    float mDirOffset;
    float mDirLimitOffset;
    int32_t mNumClasses;
    std::vector<float> mAnchorBottomHeight;
    std::vector<float> mAnchors;
    float mScoreThreashold;
    int32_t mFeatureH;
    int32_t mFeatureW;
};

class DecodeBbox3DPluginCreator : public nvinfer1::IPluginCreator
{
public:
    DecodeBbox3DPluginCreator();
    char const* getPluginName() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override;
    nvinfer1::IPluginV2* createPlugin(char const* name, nvinfer1::PluginFieldCollection const* fc) noexcept override;
    nvinfer1::IPluginV2* deserializePlugin(
        char const* name, void const* serialData, size_t serialLength) noexcept override;
    void setPluginNamespace(char const* pluginNamespace) noexcept override;
    char const* getPluginNamespace() const noexcept override;

private:
    nvinfer1::PluginFieldCollection mFC;
    std::vector<nvinfer1::PluginField> mPluginAttributes;
    std::string mNamespace;
};

} // namespace plugin
} // namespace nvinfer1

#endif
