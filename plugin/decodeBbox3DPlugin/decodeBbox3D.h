/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "common/bboxUtils.h"
#include "common/kernel.h"
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
    DecodeBbox3DPlugin(float x_min, float x_max, float y_min, float y_max,
              float z_min, float z_max,
              int num_dir_bins, float dir_offset, float dir_limit_offset,
              const std::vector<float>& anchor_bottom_height,
              const std::vector<float>& anchors,
              float score_thresh);
    DecodeBbox3DPlugin(float x_min, float x_max, float y_min, float y_max,
              float z_min, float z_max,
              int num_dir_bins, float dir_offset, float dir_limit_offset,
              const std::vector<float>& anchor_bottom_height,
              const std::vector<float>& anchors,
              float score_thresh,
              int feature_h, int feature_w);
    DecodeBbox3DPlugin(const void* data, size_t length);
    // IPluginV2DynamicExt Methods
    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;
    nvinfer1::DimsExprs getOutputDimensions(int outputIndex, 
        const nvinfer1::DimsExprs* inputs, int nbInputs,
        nvinfer1::IExprBuilder& exprBuilder) noexcept override;
    bool supportsFormatCombination(
        int pos, const nvinfer1::PluginTensorDesc* inOut, 
        int nbInputs, int nbOutputs) noexcept override;
    void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
        const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept override;
    size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
        const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept override;
    int enqueue(const nvinfer1::PluginTensorDesc* inputDesc, 
        const nvinfer1::PluginTensorDesc* outputDesc,
        const void* const* inputs, void* const* outputs, 
        void* workspace, cudaStream_t stream) noexcept override;
    // IPluginV2Ext Methods
    nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, 
        int nbInputs) const noexcept override;
    // IPluginV2 Methods
    const char* getPluginType() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    int getNbOutputs() const noexcept override;
    int initialize() noexcept override;
    void terminate() noexcept override;
    size_t getSerializationSize() const noexcept override;
    void serialize(void* buffer) const noexcept override;
    void destroy() noexcept override;
    void setPluginNamespace(const char* pluginNamespace) noexcept override;
    const char* getPluginNamespace() const noexcept override;

private:
    std::string mNamespace;
    float min_x_range_;
    float max_x_range_;
    float min_y_range_;
    float max_y_range_;
    float min_z_range_;
    float max_z_range_;
    int num_dir_bins_;
    float dir_offset_;
    float dir_limit_offset_;
    int num_classes_;
    std::vector<float> anchor_bottom_height_;
    std::vector<float> anchors_;
    float score_thresh_;
    int feature_h_;
    int feature_w_;
};

class DecodeBbox3DPluginCreator : public nvinfer1::IPluginCreator
{
public:
    DecodeBbox3DPluginCreator();
    const char* getPluginName() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override;
    nvinfer1::IPluginV2* createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc) noexcept override;
    nvinfer1::IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept override;
    void setPluginNamespace(const char* pluginNamespace) noexcept override;
    const char* getPluginNamespace() const noexcept override;

private:
    static nvinfer1::PluginFieldCollection mFC;
    static std::vector<nvinfer1::PluginField> mPluginAttributes;
    std::string mNamespace;
};

} // namespace plugin
} // namespace nvinfer1

#endif
