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

#ifndef TRT_GROUP_NORM_PLUGIN_H
#define TRT_GROUP_NORM_PLUGIN_H

#include "plugin.h"
#include "serialize.hpp"
#include <cudnn.h>
#include <iostream>
#include <string>
#include <vector>

// One of the preferred ways of making TensorRT to be able to see
// our custom layer requires extending IPluginV2 and IPluginCreator classes.
// For requirements for overriden functions, check TensorRT API docs.
namespace nvinfer1
{
namespace plugin
{

template <typename T>
void scaleShiftChannelsInplace(T* inOut, const int B, const int C, const int channelVolume, const float* beta,
    const float* gamma, cudaStream_t stream);

class GroupNormalizationPlugin final : public nvinfer1::IPluginV2DynamicExt
{
public:
    GroupNormalizationPlugin(float epsilon, const int nbGroups);

    GroupNormalizationPlugin(const void* data, size_t length);

    // It doesn't make sense to make GroupNormalizationPlugin without arguments, so we
    // delete default constructor.
    GroupNormalizationPlugin() = delete;

    int getNbOutputs() const override;

    // DynamicExt plugins returns DimsExprs class instead of Dims
    DimsExprs getOutputDimensions(
        int index, const nvinfer1::DimsExprs* inputs, int nbInputDims, nvinfer1::IExprBuilder& exprBuilder) override;

    int initialize() override;

    void terminate() override;

    size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
        const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const override;

    int enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
        const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) override;

    size_t getSerializationSize() const override;

    void serialize(void* buffer) const override;

    bool supportsFormatCombination(
        int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) override;

    const char* getPluginType() const override;

    const char* getPluginVersion() const override;

    nvinfer1::IPluginV2DynamicExt* clone() const override;

    void destroy() override;

    DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const override;

    void attachToContext(cudnnContext* cudnn, cublasContext* cublas, nvinfer1::IGpuAllocator* allocator) override;

    void detachFromContext() override;

    void setPluginNamespace(const char* pluginNamespace) override;

    const char* getPluginNamespace() const override;

    void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
        const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) override;

private:
    const char* mPluginNamespace;
    std::string mNamespace;

    float mEpsilon;
    int mNbGroups;
    int mChannelVolume;

    cudnnHandle_t _cudnn_handle;
    cudnnTensorDescriptor_t desc, bnDesc; // describes input and output
    // These are buffers initialized to 1 and 0 respectively
    void* bnScale;
    void* bnBias;

protected:
    // To prevent compiler warnings.
    using nvinfer1::IPluginV2DynamicExt::getOutputDimensions;
    using nvinfer1::IPluginV2DynamicExt::isOutputBroadcastAcrossBatch;
    using nvinfer1::IPluginV2DynamicExt::canBroadcastInputAcrossBatch;
    using nvinfer1::IPluginV2DynamicExt::supportsFormat;
    using nvinfer1::IPluginV2DynamicExt::configurePlugin;
    using nvinfer1::IPluginV2DynamicExt::getWorkspaceSize;
    using nvinfer1::IPluginV2DynamicExt::enqueue;
};

class GroupNormalizationPluginCreator : public IPluginCreator
{
public:
    GroupNormalizationPluginCreator();

    ~GroupNormalizationPluginCreator() override = default;

    const char* getPluginName() const override;

    const char* getPluginVersion() const override;

    const PluginFieldCollection* getFieldNames() override;

    IPluginV2DynamicExt* createPlugin(const char* name, const PluginFieldCollection* fc) override;

    IPluginV2DynamicExt* deserializePlugin(const char* name, const void* serialData, size_t serialLength) override;

    void setPluginNamespace(const char* pluginNamespace) override;

    const char* getPluginNamespace() const override;

private:
    static PluginFieldCollection mFC;
    static std::vector<PluginField> mPluginAttributes;
    std::string mNamespace;
};
} // namespace plugin
} // namespace nvinfer1

#endif // TRT_GROUP_NORM_PLUGIN_H
