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

#ifndef TRT_EMB_LAYER_NORM_PLUGIN_H
#define TRT_EMB_LAYER_NORM_PLUGIN_H

#include "NvInferPlugin.h"
#include "NvInferRuntime.h"
#include <string>
#include <vector>
namespace bert
{

namespace test
{

// One of the preferred ways of making TensorRT to be able to see
// our custom layer requires extending IPluginV2 and IPluginCreator classes.
// For requirements for overriden functions, check TensorRT API docs.

class EmbLayerNormPluginDynamic : public nvinfer1::IPluginV2DynamicExt
{
public:
    EmbLayerNormPluginDynamic(const std::string& name, const bool use_fp16, const Weights& beta, const Weights& gamma,
        const Weights& word_emb, const Weights& pos_emb, const Weights& tok_emb);

    EmbLayerNormPluginDynamic(const std::string& name, const void* data, size_t length);

    // It doesn't make sense to make EmbLayerNormPluginDynamic without arguments, so we
    // delete default constructor.
    EmbLayerNormPluginDynamic() = delete;

    // IPluginV2DynamicExt Methods
    nvinfer1::IPluginV2DynamicExt* clone() const override;
    nvinfer1::DimsExprs getOutputDimensions(
        int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) override;
    bool supportsFormatCombination(
        int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) override;
    void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
        const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) override;
    size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
        const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const override;
    int enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
        const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) override;

    // IPluginV2Ext Methods
    nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const override;

    // IPluginV2 Methods
    const char* getPluginType() const override;
    const char* getPluginVersion() const override;
    int getNbOutputs() const override;
    int initialize() override;
    void terminate() override;
    size_t getSerializationSize() const override;
    void serialize(void* buffer) const override;
    void destroy() override;
    void setPluginNamespace(const char* pluginNamespace) override;
    const char* getPluginNamespace() const override;

private:
    const std::string mLayerName;
    std::string mNamespace;

    float* mGammaDev;
    float* mBetaDev;
    float* mWordEmbDev;
    float* mTokEmbDev;
    float* mPosEmbDev;
    size_t mLd; // leading dim = hidden size
    size_t mB;  // batch size
    size_t mS;  // sequence length
    size_t mWordVocabSize;
    size_t mPosVocabSize;
    size_t mTokVocabSize;
    nvinfer1::Weights mBeta;
    nvinfer1::Weights mGamma;
    nvinfer1::Weights mWordEmb;
    nvinfer1::Weights mTokEmb;
    nvinfer1::Weights mPosEmb;
    nvinfer1::DataType mType;
};

class EmbLayerNormPluginDynamicCreator : public nvinfer1::IPluginCreator
{
public:
    EmbLayerNormPluginDynamicCreator();

    const char* getPluginName() const override;

    const char* getPluginVersion() const override;

    const nvinfer1::PluginFieldCollection* getFieldNames() override;

    nvinfer1::IPluginV2* createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc) override;

    nvinfer1::IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) override;

    void setPluginNamespace(const char* pluginNamespace) override;

    const char* getPluginNamespace() const override;

private:
    static nvinfer1::PluginFieldCollection mFC;
    static std::vector<nvinfer1::PluginField> mPluginAttributes;
    std::string mNamespace;
};
}
}
#endif // TRT_EMB_LAYER_NORM_PLUGIN_H
