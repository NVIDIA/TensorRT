/*
 * Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
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

#ifndef TRT_QKV_TO_CONTEXT_PLUGIN_h
#define TRT_QKV_TO_CONTEXT_PLUGIN_h

#include "NvInferPlugin.h"
#include "cublas_v2.h"
#include <string>
#include <vector>

namespace bert
{

using namespace nvinfer1;

// One of the preferred ways of making TensorRT to be able to see
// our custom layer requires extending IPluginV2 and IPluginCreator classes.
// For requirements for overriden functions, check TensorRT API docs.

class QKVToContextPlugin : public IPluginV2Ext
{
public:
    QKVToContextPlugin(
        const std::string name, const int hidden_size, const int num_heads, const int S, bool has_imask = false);

    QKVToContextPlugin(const std::string name, const void* data, size_t length);

    // It doesn't make sense to make QKVToContextPlugin without arguments, so we
    // delete default constructor.
    QKVToContextPlugin() = delete;

    bool isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const
    {
        return false;
    }

    bool canBroadcastInputAcrossBatch(int inputIndex) const
    {
        return false;
    }

    int getNbOutputs() const override;

    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override;

    int initialize() override;

    void terminate() override;

    size_t getWorkspaceSize(int) const override;

    int enqueue(
        int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream) override;

    size_t getSerializationSize() const override;

    void serialize(void* buffer) const override;

    DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const override;

    void configurePlugin(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,
        const DataType* inputTypes, const DataType* outputTypes, const bool* inputIsBroadcast,
        const bool* outputIsBroadcast, PluginFormat floatFormat, int maxBatchSize) override;

    bool supportsFormat(DataType type, PluginFormat format) const override;

    const char* getPluginType() const override;

    const char* getPluginVersion() const override;

    void destroy() override;

    nvinfer1::IPluginV2Ext* clone() const override;

    void setPluginNamespace(const char* pluginNamespace) override;

    const char* getPluginNamespace() const override;

    void attachToContext(cudnnContext* cudnn, cublasContext* cublas, IGpuAllocator* alloc) override;

private:
    size_t scratchSize(int bs) const;
    float mRsqrtHeadSize;
    int mHeadSize;
    int mB;
    int mS;
    int mHiddenSize;
    int mNumHeads;
    bool mHasImask;
    const std::string mLayerName;
    std::string mNamespace;

    DataType mType;
    cublasHandle_t cublas;
};

class QKVToContextPluginCreator : public IPluginCreator
{
public:
    QKVToContextPluginCreator();

    const char* getPluginName() const override;

    const char* getPluginVersion() const override;

    const PluginFieldCollection* getFieldNames() override;

    IPluginV2* createPlugin(const char* name, const PluginFieldCollection* fc) override;

    IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) override;

    void setPluginNamespace(const char* pluginNamespace) override;

    const char* getPluginNamespace() const override;

private:
    static PluginFieldCollection mFC;
    static std::vector<PluginField> mPluginAttributes;
    std::string mNamespace;
};
}
#endif // TRT_QKV_TO_CONTEXT_PLUGIN_h
