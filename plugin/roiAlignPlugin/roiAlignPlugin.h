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
#ifndef TRT_ROIALIGN_PLUGIN_H
#define TRT_ROIALIGN_PLUGIN_H
#include <cublas_v2.h>

#include "plugin.h"
#include <memory>
#include <string>
#include <vector>

using namespace nvinfer1::plugin;

namespace nvinfer1
{
namespace plugin
{

class RoIAlignPluginDynamic : public nvinfer1::IPluginV2DynamicExt
{
public:
    RoIAlignPluginDynamic(int coordTransformMode, int poolingMode, int outputHeight, int outputWidth, int samplingRatio,
        float spatialScale);

    RoIAlignPluginDynamic(const void* data, size_t length);

    RoIAlignPluginDynamic() = delete;

    // IPluginV2DynamicExt Methods
    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;
    nvinfer1::DimsExprs getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs,
        nvinfer1::IExprBuilder& exprBuilder) noexcept override;
    bool supportsFormatCombination(
        int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept override;
    void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
        const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept override;
    size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
        const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept override;
    int enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
        const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

    // IPluginV2Ext Methods
    DataType getOutputDataType(int index, const DataType* inputTypes, int nbInputs) const noexcept override;

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

    enum PoolingMode
    {
        MAX = 0,
        AVG = 1
    };

    enum CoordinateTransformationMode
    {
        OUTPUT_HALF_PIXEL = 0,
        HALF_PIXEL = 1
    };

private:
    std::string mNamespace;
    int mCoordTransformMode = static_cast<int>(CoordinateTransformationMode::HALF_PIXEL);
    int mPoolingMode = static_cast<int>(PoolingMode::AVG);
    int mOutputHeight = 1;
    int mOutputWidth = 1;
    int mSamplingRatio = 0;
    float mSpatialScale = 1.0;
};

class RoIAlignPluginDynamicCreator : public nvinfer1::IPluginCreator
{
public:
    RoIAlignPluginDynamicCreator();

    const char* getPluginName() const noexcept override;

    const char* getPluginVersion() const noexcept override;

    const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override;

    IPluginV2* createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc) noexcept override;

    IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept override;

    void setPluginNamespace(const char* pluginNamespace) noexcept override;

    const char* getPluginNamespace() const noexcept override;

private:
    static nvinfer1::PluginFieldCollection mFC;
    static std::vector<nvinfer1::PluginField> mPluginAttributes;
    std::string mNamespace;
};

} // namespace plugin
} // namespace nvinfer1
#endif // TRT_ROIALIGN_PLUGIN_H
