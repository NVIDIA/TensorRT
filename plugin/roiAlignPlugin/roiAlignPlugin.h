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
#ifndef TRT_ROIALIGN_PLUGIN_H
#define TRT_ROIALIGN_PLUGIN_H

#include "common/plugin.h"
#include <cuda_runtime_api.h>
#include <stdint.h>
#include <vector>

#include "NvInfer.h"
#include "NvInferPlugin.h"

namespace nvinfer1
{
namespace plugin
{

class ROIAlign : public IPluginV2DynamicExt
{
public:
    ROIAlign(
        int32_t outputHeight, int32_t outputWidth, int32_t samplingRatio, int32_t mode, float spatialScale, int32_t aligned);
    ROIAlign(void const* data, size_t length);
    ROIAlign() = default;
    ~ROIAlign() override = default;

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
    void setClipParam(bool clip) noexcept;
    void setScoreBits(int32_t scoreBits) noexcept;
    void setCaffeSemantics(bool caffeSemantics) noexcept;

    // IPluginV2Ext methods
    nvinfer1::DataType getOutputDataType(int32_t index, nvinfer1::DataType const* inputType, int32_t nbInputs) const
        noexcept override;

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
    void checkValidInputs(nvinfer1::DynamicPluginTensorDesc const* inputs, int32_t nbInputDims);
    void validateAttributes(int32_t outputHeight, int32_t outputWidth, int32_t samplingRatio, int32_t mode, float spatialScale, int32_t aligned);

    int32_t mOutputHeight{};
    int32_t mOutputWidth{};
    int32_t mSamplingRatio{};
    float mSpatialScale{};
    int32_t mMode{};
    int32_t mAligned{};

    int32_t mROICount{};
    int32_t mFeatureLength{}; // number of channels
    int32_t mHeight{};
    int32_t mWidth{};

    int32_t mMaxThreadsPerBlock{};

    std::string mNameSpace{};
};

class ROIAlignPluginCreator : public nvinfer1::pluginInternal::BaseCreator
{
public:
    ROIAlignPluginCreator();

    ~ROIAlignPluginCreator() override = default;

    char const* getPluginName() const noexcept override;

    char const* getPluginVersion() const noexcept override;

    PluginFieldCollection const* getFieldNames() noexcept override;

    IPluginV2DynamicExt* createPlugin(char const* name, nvinfer1::PluginFieldCollection const* fc) noexcept override;

    IPluginV2DynamicExt* deserializePlugin(
        char const* name, void const* serialData, size_t serialLength) noexcept override;

private:
    static PluginFieldCollection mFC;
    static std::vector<PluginField> mPluginAttributes;
};

} // namespace plugin
} // namespace nvinfer1
#endif // TRT_ROIALIGN_PLUGIN_H
