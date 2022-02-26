/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef TRT_EFFICIENT_NMS_PLUGIN_H
#define TRT_EFFICIENT_NMS_PLUGIN_H

#include <vector>

#include "plugin.h"
#include "efficientNMSParameters.h"


using namespace nvinfer1::plugin;
namespace nvinfer1
{
namespace plugin
{

class EfficientNMSPlugin : public IPluginV2DynamicExt
{
public:
    explicit EfficientNMSPlugin(EfficientNMSParameters param);
    EfficientNMSPlugin(const void* data, size_t length);
    ~EfficientNMSPlugin() override = default;

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

    // IPluginV2Ext methods
    nvinfer1::DataType getOutputDataType(
        int index, const nvinfer1::DataType* inputType, int nbInputs) const noexcept override;

    // IPluginV2DynamicExt methods
    IPluginV2DynamicExt* clone() const noexcept override;
    DimsExprs getOutputDimensions(
        int outputIndex, const DimsExprs* inputs, int nbInputs, IExprBuilder& exprBuilder) noexcept override;
    bool supportsFormatCombination(
        int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept override;
    void configurePlugin(const DynamicPluginTensorDesc* in, int nbInputs, const DynamicPluginTensorDesc* out,
        int nbOutputs) noexcept override;
    size_t getWorkspaceSize(const PluginTensorDesc* inputs, int nbInputs, const PluginTensorDesc* outputs,
        int nbOutputs) const noexcept override;
    int enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc, const void* const* inputs,
        void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

private:
    EfficientNMSParameters mParam{};
    std::string mNamespace;
};

// Standard NMS Operation
class EfficientNMSPluginCreator : public BaseCreator
{
public:
    EfficientNMSPluginCreator();
    ~EfficientNMSPluginCreator() override = default;

    const char* getPluginName() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    const PluginFieldCollection* getFieldNames() noexcept override;

    IPluginV2DynamicExt* createPlugin(const char* name, const PluginFieldCollection* fc) noexcept override;
    IPluginV2DynamicExt* deserializePlugin(
        const char* name, const void* serialData, size_t serialLength) noexcept override;

protected:
    static PluginFieldCollection mFC;
    EfficientNMSParameters mParam;
    static std::vector<PluginField> mPluginAttributes;
    std::string mPluginName;
};

// ONNX NonMaxSuppression Op Support
class EfficientNMSONNXPluginCreator : public BaseCreator
{
public:
    EfficientNMSONNXPluginCreator();
    ~EfficientNMSONNXPluginCreator() override = default;

    const char* getPluginName() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    const PluginFieldCollection* getFieldNames() noexcept override;

    IPluginV2DynamicExt* createPlugin(const char* name, const PluginFieldCollection* fc) noexcept override;
    IPluginV2DynamicExt* deserializePlugin(
        const char* name, const void* serialData, size_t serialLength) noexcept override;

protected:
    static PluginFieldCollection mFC;
    EfficientNMSParameters mParam;
    static std::vector<PluginField> mPluginAttributes;
    std::string mPluginName;
};

} // namespace plugin
} // namespace nvinfer1

#endif // TRT_EFFICIENT_NMS_PLUGIN_H
