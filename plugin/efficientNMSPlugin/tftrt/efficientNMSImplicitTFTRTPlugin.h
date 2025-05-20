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
#ifndef TRT_EFFICIENT_NMS_IMPLICIT_TFTRT_PLUGIN_H
#define TRT_EFFICIENT_NMS_IMPLICIT_TFTRT_PLUGIN_H

#include <vector>

#include "common/plugin.h"
#include "efficientNMSPlugin/efficientNMSParameters.h"

// This plugin provides CombinedNMS op compatibility for TF-TRT in Implicit Batch
// mode for legacy back-compatibilty

namespace nvinfer1
{
namespace plugin
{

#if NV_TENSORRT_MAJOR >= 8
using EfficientNMSImplicitTFTRTOutputsDataType = void* const*;
#else
using EfficientNMSImplicitTFTRTOutputsDataType = void**;
#endif

// TF-TRT CombinedNMS Op Compatibility, for Legacy Implicit Batch Mode
class EfficientNMSImplicitTFTRTPlugin : public nvinfer1::IPluginV2IOExt
{
public:
    explicit EfficientNMSImplicitTFTRTPlugin(EfficientNMSParameters param);
    EfficientNMSImplicitTFTRTPlugin(const void* data, size_t length);
    ~EfficientNMSImplicitTFTRTPlugin() override = default;

    // IPluginV2 methods
    const char* getPluginType() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    int32_t getNbOutputs() const noexcept override;
    int32_t initialize() noexcept override;
    void terminate() noexcept override;
    size_t getSerializationSize() const noexcept override;
    void serialize(void* buffer) const noexcept override;
    void destroy() noexcept override;
    void setPluginNamespace(const char* libNamespace) noexcept override;
    const char* getPluginNamespace() const noexcept override;

    nvinfer1::Dims getOutputDimensions(
        int32_t outputIndex, const nvinfer1::Dims* inputs, int32_t nbInputs) noexcept override;
    size_t getWorkspaceSize(int32_t maxBatchSize) const noexcept override;
    int32_t enqueue(int32_t batchSize, void const* const* inputs, EfficientNMSImplicitTFTRTOutputsDataType outputs,
        void* workspace, cudaStream_t stream) noexcept override;

    // IPluginV2Ext methods
    bool canBroadcastInputAcrossBatch(int32_t inputIndex) const noexcept override;
    nvinfer1::DataType getOutputDataType(
        int32_t index, const nvinfer1::DataType* inputType, int32_t nbInputs) const noexcept override;
    nvinfer1::IPluginV2IOExt* clone() const noexcept override;
    bool isOutputBroadcastAcrossBatch(
        int32_t outputIndex, bool const* inputIsBroadcasted, int32_t nbInputs) const noexcept override;

    // IPluginV2IOExt methods
    bool supportsFormatCombination(int32_t pos, const nvinfer1::PluginTensorDesc* inOut, int32_t nbInputs,
        int32_t nbOutputs) const noexcept override;
    void configurePlugin(const nvinfer1::PluginTensorDesc* in, int32_t nbInputs, const nvinfer1::PluginTensorDesc* out,
        int32_t nbOutputs) noexcept override;

protected:
    void deserialize(int8_t const* data, size_t length);

    EfficientNMSParameters mParam{};
    std::string mNamespace;
};

class EfficientNMSImplicitTFTRTPluginCreator : public nvinfer1::pluginInternal::BaseCreator
{
public:
    EfficientNMSImplicitTFTRTPluginCreator();
    ~EfficientNMSImplicitTFTRTPluginCreator() override = default;

    const char* getPluginName() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override;

    nvinfer1::IPluginV2IOExt* createPlugin(
        const char* name, const nvinfer1::PluginFieldCollection* fc) noexcept override;
    nvinfer1::IPluginV2IOExt* deserializePlugin(
        const char* name, const void* serialData, size_t serialLength) noexcept override;

protected:
    nvinfer1::PluginFieldCollection mFC;
    EfficientNMSParameters mParam;
    std::vector<nvinfer1::PluginField> mPluginAttributes;
    std::string mPluginName;
};

} // namespace plugin
} // namespace nvinfer1

#endif // TRT_EFFICIENT_NMS_IMPLICIT_TFTRT_PLUGIN_H
