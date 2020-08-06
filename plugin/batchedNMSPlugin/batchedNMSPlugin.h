/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
#ifndef TRT_BATCHED_NMS_PLUGIN_H
#define TRT_BATCHED_NMS_PLUGIN_H
#include "gatherNMSOutputs.h"
#include "kernel.h"
#include "nmsUtils.h"
#include "plugin.h"
#include <string>
#include <vector>

using namespace nvinfer1::plugin;
namespace nvinfer1
{
namespace plugin
{

class BatchedNMSPlugin : public IPluginV2Ext
{
public:
    BatchedNMSPlugin(NMSParameters param);
    BatchedNMSPlugin(const void* data, size_t length);
    ~BatchedNMSPlugin() override = default;

    // IPluginV2 methods
    const char* getPluginType() const override;
    const char* getPluginVersion() const override;
    int getNbOutputs() const override;
    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override;
    bool supportsFormat(DataType type, PluginFormat format) const override;
    size_t getWorkspaceSize(int maxBatchSize) const override;
    int enqueue(
        int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream) override;
    int initialize() override;
    void terminate() override;
    size_t getSerializationSize() const override;
    void serialize(void* buffer) const override;
    void destroy() override;
    void setPluginNamespace(const char* libNamespace) override;
    const char* getPluginNamespace() const override;
    void setClipParam(bool clip);

    // IPluginV2Ext methods
    nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType* inputType, int nbInputs) const override;
    bool isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const override;
    bool canBroadcastInputAcrossBatch(int inputIndex) const override;
    void configurePlugin(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,
        const DataType* inputTypes, const DataType* outputTypes, const bool* inputIsBroadcast,
        const bool* outputIsBroadcast, PluginFormat floatFormat, int maxBatchSize) override;
    IPluginV2Ext* clone() const override;

private:
    NMSParameters param{};
    int boxesSize{};
    int scoresSize{};
    int numPriors{};
    std::string mNamespace;
    bool mClipBoxes{};
};

class BatchedNMSDynamicPlugin : public IPluginV2DynamicExt
{
public:
    BatchedNMSDynamicPlugin(NMSParameters param);
    BatchedNMSDynamicPlugin(const void* data, size_t length);
    ~BatchedNMSDynamicPlugin() override = default;

    // IPluginV2 methods
    const char* getPluginType() const override;
    const char* getPluginVersion() const override;
    int getNbOutputs() const override;
    int initialize() override;
    void terminate() override;
    size_t getSerializationSize() const override;
    void serialize(void* buffer) const override;
    void destroy() override;
    void setPluginNamespace(const char* libNamespace) override;
    const char* getPluginNamespace() const override;
    void setClipParam(bool clip);

    // IPluginV2Ext methods
    nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType* inputType, int nbInputs) const override;

    // IPluginV2DynamicExt methods
    IPluginV2DynamicExt* clone() const override;
    DimsExprs getOutputDimensions(
        int outputIndex, const DimsExprs* inputs, int nbInputs, IExprBuilder& exprBuilder) override;
    bool supportsFormatCombination(int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) override;
    void configurePlugin(
        const DynamicPluginTensorDesc* in, int nbInputs, const DynamicPluginTensorDesc* out, int nbOutputs) override;
    size_t getWorkspaceSize(
        const PluginTensorDesc* inputs, int nbInputs, const PluginTensorDesc* outputs, int nbOutputs) const override;
    int enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc, const void* const* inputs,
        void* const* outputs, void* workspace, cudaStream_t stream) override;

private:
    NMSParameters param{};
    int boxesSize{};
    int scoresSize{};
    int numPriors{};
    std::string mNamespace;
    bool mClipBoxes{};
};

class BatchedNMSBasePluginCreator : public BaseCreator
{
public:
    BatchedNMSBasePluginCreator();
    ~BatchedNMSBasePluginCreator() override = default;

    const char* getPluginName() const override;
    const char* getPluginVersion() const override;
    const PluginFieldCollection* getFieldNames() override;

protected:
    static PluginFieldCollection mFC;
    NMSParameters params;
    static std::vector<PluginField> mPluginAttributes;
    bool mClipBoxes;
    std::string mPluginName;
};

class BatchedNMSPluginCreator : public BatchedNMSBasePluginCreator
{
public:
    BatchedNMSPluginCreator();
    ~BatchedNMSPluginCreator() override = default;

    IPluginV2Ext* createPlugin(const char* name, const PluginFieldCollection* fc) override;
    IPluginV2Ext* deserializePlugin(const char* name, const void* serialData, size_t serialLength) override;
};

class BatchedNMSDynamicPluginCreator : public BatchedNMSBasePluginCreator
{
public:
    BatchedNMSDynamicPluginCreator();
    ~BatchedNMSDynamicPluginCreator() override = default;

    IPluginV2DynamicExt* createPlugin(const char* name, const PluginFieldCollection* fc) override;
    IPluginV2DynamicExt* deserializePlugin(const char* name, const void* serialData, size_t serialLength) override;
};

} // namespace plugin
} // namespace nvinfer1

#endif // TRT_BATCHED_NMS_PLUGIN_H
