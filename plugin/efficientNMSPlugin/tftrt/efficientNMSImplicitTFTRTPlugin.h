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
    int getNbOutputs() const noexcept override;
    int initialize() noexcept override;
    void terminate() noexcept override;
    size_t getSerializationSize() const noexcept override;
    void serialize(void* buffer) const noexcept override;
    void destroy() noexcept override;
    void setPluginNamespace(const char* libNamespace) noexcept override;
    const char* getPluginNamespace() const noexcept override;

    nvinfer1::Dims getOutputDimensions(int outputIndex, const nvinfer1::Dims* inputs, int nbInputs) noexcept override;
    size_t getWorkspaceSize(int maxBatchSize) const noexcept override;
    int enqueue(int batchSize, void const* const* inputs, EfficientNMSImplicitTFTRTOutputsDataType outputs,
        void* workspace, cudaStream_t stream) noexcept override;

    // IPluginV2Ext methods
    bool canBroadcastInputAcrossBatch(int inputIndex) const noexcept override;
    nvinfer1::DataType getOutputDataType(
        int index, const nvinfer1::DataType* inputType, int nbInputs) const noexcept override;
    nvinfer1::IPluginV2IOExt* clone() const noexcept override;
    bool isOutputBroadcastAcrossBatch(
        int outputIndex, bool const* inputIsBroadcasted, int nbInputs) const noexcept override;

    // IPluginV2IOExt methods
    bool supportsFormatCombination(
        int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) const noexcept override;
    void configurePlugin(const nvinfer1::PluginTensorDesc* in, int nbInputs, const nvinfer1::PluginTensorDesc* out,
        int nbOutputs) noexcept override;

protected:
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
