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

#include <cuda.h>
#if CUDA_VERSION >= 10010

#include "NvInfer.h"
#include "common/serialize.hpp"
#include "skipLayerNormPlugin.h"

#include <cstring>
#include <vector>

using namespace nvinfer1;
using namespace nvinfer1::plugin;
using namespace nvinfer1::plugin::bert;



// Clip plugin specific constants
namespace
{
const char* SKIP_LAYER_NORM_VERSION{"1"};
const char* SKIP_LAYER_NORM_NAME{"CustomSkipLayerNormPluginDynamic"};
const char* SKIP_LAYER_NORM_VAR_SEQLEN_VERSION{"2"};
} // namespace

// Static class fields initialization
PluginFieldCollection SkipLayerNormPluginDynamicCreator::mFC{};
std::vector<PluginField> SkipLayerNormPluginDynamicCreator::mPluginAttributes;

PluginFieldCollection SkipLayerNormVarSeqlenPluginCreator::mFC{};
std::vector<PluginField> SkipLayerNormVarSeqlenPluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(SkipLayerNormPluginDynamicCreator);
REGISTER_TENSORRT_PLUGIN(SkipLayerNormVarSeqlenPluginCreator);

static inline DataType getParamWordType(DataType cfgType) noexcept
{
    if (cfgType == DataType::kINT8)
    {
        return DataType::kHALF;
    }

    return cfgType;
}

SkipLayerNormPluginDynamic::SkipLayerNormPluginDynamic(const std::string name, const DataType type, const int ld,
    const Weights& beta, const Weights& gamma, const Weights& bias)
    : mLayerName(name)
    , mGammaDev(nullptr)
    , mBetaDev(nullptr)
    , mLd(ld)
    , mType(type)
    , mBiasDev(nullptr)
{
    PLUGIN_VALIDATE(mType == nvinfer1::DataType::kFLOAT || mType == nvinfer1::DataType::kHALF
        || mType == nvinfer1::DataType::kINT8);
    // mCfgType is the dataType for beta, gamma bias weights, always fp16 or fp32
    // mType is the plugin IO datatype, can be int8
    mCfgType = mType == DataType::kINT8 ? DataType::kHALF :  mType;
    mParamWordsize = getElementSize(mCfgType);

    mBeta.convertAndCopy(beta, mCfgType);
    mGamma.convertAndCopy(gamma, mCfgType);

    mHasBias = (bias.values != nullptr);
    if (mHasBias)
    {
        mBias.convertAndCopy(bias, mCfgType);
    }

    copyToDevice(mGamma, getWeightsSize(mGamma, mCfgType), mGammaDev);
    copyToDevice(mBeta, getWeightsSize(mBeta, mCfgType), mBetaDev);
    if (mHasBias)
    {
        copyToDevice(mBias, getWeightsSize(mBias, mCfgType), mBiasDev);
    }
}

SkipLayerNormPluginDynamic::SkipLayerNormPluginDynamic(const std::string name, const void* data, size_t length)
    : mLayerName(name)
    , mGammaDev(nullptr)
    , mBetaDev(nullptr)
    , mBiasDev(nullptr)
{
    BERT_DEBUG_MSG("SkipLayerNormPluginDynamic deserialize");

    // Deserialize in the same order as serialization
    deserialize_value(&data, &length, &mType);
    deserialize_value(&data, &length, &mCfgType);
    deserialize_value(&data, &length, &mLd);
    deserialize_value(&data, &length, &mHasBias);

    PLUGIN_VALIDATE(mCfgType == nvinfer1::DataType::kFLOAT || mCfgType == nvinfer1::DataType::kHALF);
    mParamWordsize = getElementSize(mCfgType);

    const char* d = static_cast<const char*>(data);
    mBeta.convertAndCopy(d, mLd, mCfgType);
    mGamma.convertAndCopy(d, mLd, mCfgType);
    if (mHasBias)
    {
        mBias.convertAndCopy(d, mLd, mCfgType);
    }

    copyToDevice(mGamma, getWeightsSize(mGamma, mCfgType), mGammaDev);
    copyToDevice(mBeta, getWeightsSize(mBeta, mCfgType), mBetaDev);
    if (mHasBias)
    {
        copyToDevice(mBias, getWeightsSize(mBias, mCfgType), mBiasDev);
    }
}

// IPluginV2DynamicExt Methods
IPluginV2DynamicExt* SkipLayerNormPluginDynamic::clone() const noexcept
{
    try
    {
        BERT_DEBUG_MSG("SkipLayerNormPluginDynamic clone");

        auto* p = new SkipLayerNormPluginDynamic(mLayerName, mType, mLd, mBeta, mGamma, mBias);
        p->initialize();
        p->setPluginNamespace(mNamespace.c_str());
        return p;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

DimsExprs SkipLayerNormPluginDynamic::getOutputDimensions(
    int outputIndex, const DimsExprs* inputs, int nbInputs, IExprBuilder& exprBuilder) noexcept
{
    PLUGIN_ASSERT(nbInputs == 2);
    PLUGIN_ASSERT(outputIndex == 0);
    PLUGIN_ASSERT(inputs[0].nbDims == inputs[1].nbDims);
    return inputs[0];
}

bool SkipLayerNormPluginDynamic::supportsFormatCombination(
    int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept
{
    PLUGIN_ASSERT(nbInputs == 2);
    PLUGIN_ASSERT(nbOutputs == 1);

    const PluginTensorDesc& in = inOut[pos];
    if (pos == 0)
    {
        // Since H = W = 1, we can report CHWx for any x
        if (mType == DataType::kINT8)
        {
            // won't work for hiddensize too small!
            TensorFormat myFmt = TensorFormat::kCHW32;
            if (mLd < 32)
            {
                myFmt = TensorFormat::kCHW4;
                BERT_DEBUG_VALUE("SkipLayerNormDQQ: TensorFormat CHW4 for LD=", mLd);
            }
            else
            {
                BERT_DEBUG_VALUE("SkipLayerNormDQQ: TensorFormat CHW32 for LD=", mLd);
            }
            // TODO do we need to check if the vectorization divides mLd?
            return ((in.type == mType) && (in.format == myFmt));
        }
        return (in.type == mType) && (in.format == TensorFormat::kLINEAR);
    }
    const PluginTensorDesc& prev = inOut[pos - 1];

    return in.type == prev.type && in.format == prev.format;
}

void SkipLayerNormPluginDynamic::configurePlugin(
    const DynamicPluginTensorDesc* inputs, int nbInputs, const DynamicPluginTensorDesc* outputs, int nbOutputs) noexcept
{
    BERT_DEBUG_MSG("SkipLayerNormPluginDynamic configurePlugin");

    // Validate input arguments
    PLUGIN_ASSERT(nbOutputs == 1);
    PLUGIN_ASSERT(nbInputs == 2);
    if (mType == DataType::kFLOAT || mType == DataType::kHALF)
    {
        PLUGIN_ASSERT(mType == inputs[0].desc.type);
        PLUGIN_ASSERT(mType == inputs[1].desc.type);
    }
    else
    {
        PLUGIN_ASSERT(mType == inputs[0].desc.type || DataType::kFLOAT == inputs[0].desc.type);
        PLUGIN_ASSERT(mType == inputs[1].desc.type || DataType::kFLOAT == inputs[1].desc.type);
    }
    const auto& inDims0 = inputs[0].desc.dims;
    const auto& inDims1 = inputs[1].desc.dims;
    TRT_UNUSED inDims1;
    PLUGIN_ASSERT(inDims0.nbDims == inDims1.nbDims);

    PLUGIN_ASSERT(std::equal(inDims0.d, inDims0.d + inDims0.nbDims, inDims1.d));

    PLUGIN_ASSERT(inDims0.nbDims == 5);
    mLd = inDims0.d[HDIM]; // hiddensize
    PLUGIN_ASSERT(inDims0.d[3] == 1);
    PLUGIN_ASSERT(inDims0.d[4] == 1);

    mCfgType = inputs[0].desc.type == DataType::kINT8 ? DataType::kHALF : inputs[0].desc.type;

    const auto paramType = getParamWordType(mCfgType);
    mParamWordsize = getElementSize(paramType);
}

size_t SkipLayerNormPluginDynamic::getWorkspaceSize(
    const PluginTensorDesc* inputs, int nbInputs, const PluginTensorDesc* outputs, int nbOutputs) const noexcept
{
    return 0;
}

int SkipLayerNormPluginDynamic::enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc,
    const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    const int inputVolume = volume(inputDesc[0].dims);
    int status = -1;
    DataType iType = inputDesc->type;

    // Our plugin outputs only one tensor
    // Launch CUDA kernel wrapper and save its return value
    if (iType == DataType::kFLOAT)
    {
        const auto* const input = static_cast<const float*>(inputs[0]);
        const auto* const skip = static_cast<const float*>(inputs[1]);
        auto* output = static_cast<float*>(outputs[0]);
        const auto* const bias = static_cast<const float*>(mBiasDev.get());
        const auto* const beta = static_cast<const float*>(mBetaDev.get());
        const auto* const gamma = static_cast<const float*>(mGammaDev.get());
        if (mHasBias)
        {
            status = computeSkipLayerNorm<float, true>(
                stream, static_cast<int>(mLd), inputVolume, input, skip, beta, gamma, output, bias);
        }
        else
        {
            status
                = computeSkipLayerNorm<float, false>(stream, static_cast<int>(mLd), inputVolume, input, skip, beta, gamma, output, bias);
        }
    }
    else if (iType == DataType::kHALF)
    {
        const auto* const input = static_cast<const half*>(inputs[0]);
        const auto* const skip = static_cast<const half*>(inputs[1]);
        auto* output = static_cast<half*>(outputs[0]);
        const auto* const bias = static_cast<const half*>(mBiasDev.get());
        const auto* const beta = static_cast<const half*>(mBetaDev.get());
        const auto* const gamma = static_cast<const half*>(mGammaDev.get());
        if (mHasBias)
        {
            status = computeSkipLayerNorm<half, true>(
                stream, static_cast<int>(mLd), inputVolume, input, skip, beta, gamma, output, bias);
        }
        else
        {
            status
                = computeSkipLayerNorm<half, false>(stream, static_cast<int>(mLd), inputVolume, input, skip, beta, gamma, output, bias);
        }
    }
    else if (iType == DataType::kINT8)
    {
        const float dqScaleIn = inputDesc[0].scale;
        const float dqScaleSkip = inputDesc[1].scale;
        const float qScale = 1.F / outputDesc[0].scale;
        const auto* const input = static_cast<const int8_t*>(inputs[0]);
        const auto* const skip = static_cast<const int8_t*>(inputs[1]);
        auto* output = static_cast<int8_t*>(outputs[0]);
        const auto* const bias = static_cast<const half*>(mBiasDev.get());
        const auto* const beta = static_cast<const half*>(mBetaDev.get());
        const auto* const gamma = static_cast<const half*>(mGammaDev.get());
        if (mHasBias)
        {
            status = computeSkipLayerNormDQQ<true>(stream, static_cast<int>(mLd), inputVolume, input, skip, beta, gamma,
                output, bias, dqScaleIn, dqScaleSkip, qScale);
        }
        else
        {
            status = computeSkipLayerNormDQQ<false>(
                stream, static_cast<int>(mLd), inputVolume, input, skip, beta, gamma, output, bias, dqScaleIn, dqScaleSkip, qScale);
        }
    }
    else
    {
        gLogError << "Unsupported type error, expected [kINT8,kHALF,kFLOAT], but received " << static_cast<int>(iType) << "." << std::endl;
        PLUGIN_ASSERT(false);
    }
    return status;
}

// IPluginV2Ext Methods
DataType SkipLayerNormPluginDynamic::getOutputDataType(int index, const DataType* inputTypes, int nbInputs) const noexcept
{
    PLUGIN_ASSERT(index == 0);
    PLUGIN_ASSERT(nbInputs == 2);
    return inputTypes[0];
}

// IPluginV2 Methods
const char* SkipLayerNormPluginDynamic::getPluginType() const noexcept
{
    return SKIP_LAYER_NORM_NAME;
}

const char* SkipLayerNormPluginDynamic::getPluginVersion() const noexcept
{
    return SKIP_LAYER_NORM_VERSION;
}

int SkipLayerNormPluginDynamic::getNbOutputs() const noexcept
{
    return 1;
}
int SkipLayerNormPluginDynamic::initialize() noexcept
{
    BERT_DEBUG_MSG("SkipLayerNormPluginDynamic initialize");
    return 0;
}

void SkipLayerNormPluginDynamic::terminate() noexcept
{
    BERT_DEBUG_MSG("SkipLayerNormPluginDynamic terminate");
}

size_t SkipLayerNormPluginDynamic::getSerializationSize() const noexcept
{
    const size_t biasSize = mHasBias ? (mLd * mParamWordsize) : 0;
    return 2 * mParamWordsize * mLd + 2 * sizeof(DataType) + sizeof(mLd) + biasSize + sizeof(mHasBias);
}

void SkipLayerNormPluginDynamic::serialize(void* buffer) const noexcept
{
    serialize_value(&buffer, mType);
    serialize_value(&buffer, mCfgType);
    serialize_value(&buffer, mLd);
    serialize_value(&buffer, mHasBias);

    char* d = static_cast<char*>(buffer);
    serFromDev(d, static_cast<char*>(mBetaDev.get()), mLd * mParamWordsize);
    serFromDev(d, static_cast<char*>(mGammaDev.get()), mLd * mParamWordsize);
    if (mHasBias)
    {
        serFromDev(d, static_cast<char*>(mBiasDev.get()), mLd * mParamWordsize);
    }
}

void SkipLayerNormPluginDynamic::destroy() noexcept
{
    BERT_DEBUG_MSG("SkipLayerNormPluginDynamic destroy");
    // This gets called when the network containing plugin is destroyed
    mGammaDev.reset(nullptr);
    mBetaDev.reset(nullptr);
    mBiasDev.reset(nullptr);
    delete this;
}

void SkipLayerNormPluginDynamic::setPluginNamespace(const char* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

const char* SkipLayerNormPluginDynamic::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

/////////////////////////////////////////////////////////

SkipLayerNormPluginDynamicCreator::SkipLayerNormPluginDynamicCreator()
{
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("ld"));
    mPluginAttributes.emplace_back(PluginField("type_id"));
    mPluginAttributes.emplace_back(PluginField("beta"));
    mPluginAttributes.emplace_back(PluginField("gamma"));
    mPluginAttributes.emplace_back(PluginField("bias"));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* SkipLayerNormPluginDynamicCreator::getPluginName() const noexcept
{
    return SKIP_LAYER_NORM_NAME;
}

const char* SkipLayerNormPluginDynamicCreator::getPluginVersion() const noexcept
{
    return SKIP_LAYER_NORM_VERSION;
}

const PluginFieldCollection* SkipLayerNormPluginDynamicCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* SkipLayerNormPluginDynamicCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
{
    try
    {
        BERT_DEBUG_MSG("SkipLayerNormPluginDynamicCreator createPlugin");

        int32_t ld = 0;
        Weights beta{DataType::kFLOAT, nullptr, 0};
        Weights gamma{DataType::kFLOAT, nullptr, 0};
        Weights bias{DataType::kFLOAT, nullptr, 0};
        int32_t typeId = -1;

        plugin::validateRequiredAttributesExist({"type_id", "beta", "ld", "gamma"}, fc);

        for (int32_t i = 0; i < fc->nbFields; i++)
        {
            std::string field_name(fc->fields[i].name);
            if (field_name.compare("ld") == 0)
            {
                ld = *static_cast<int32_t const*>(fc->fields[i].data);
                BERT_DEBUG_VALUE("Building ld: ", ld);
            }

            if (field_name.compare("type_id") == 0)
            {
                typeId = *static_cast<int32_t const*>(fc->fields[i].data);
                BERT_DEBUG_VALUE("Building typeId: ", typeId);
            }

            if (field_name.compare("beta") == 0)
            {
                BERT_DEBUG_MSG("Building beta...");
                beta.values = fc->fields[i].data;
                beta.count = fc->fields[i].length;
                beta.type = fieldTypeToDataType(fc->fields[i].type);
            }

            if (field_name.compare("gamma") == 0)
            {
                BERT_DEBUG_MSG("Building gamma...");
                gamma.values = fc->fields[i].data;
                gamma.count = fc->fields[i].length;
                gamma.type = fieldTypeToDataType(fc->fields[i].type);
            }

            if (field_name.compare("bias") == 0)
            {
                BERT_DEBUG_MSG("Building bias...");
                bias.values = fc->fields[i].data;
                bias.count = fc->fields[i].length;
                bias.type = fieldTypeToDataType(fc->fields[i].type);
            }
        }
        BERT_DEBUG_VALUE("Type ", typeId);

        if (typeId < 0 || typeId > 3)
        {
            gLogError << "SkipLayerNorm: Invalid type ID: " << typeId << std::endl;
        }

        if (beta.count <= 0 || beta.values == nullptr)
        {
            gLogError << "SkipLayerNorm: invalid beta" << std::endl;
        }

        if (gamma.count <= 0 || gamma.values == nullptr)
        {
            gLogError << "SkipLayerNorm: invalid gamma" << std::endl;
        }

        return new SkipLayerNormPluginDynamic(name, static_cast<DataType>(typeId), ld, beta, gamma, bias);
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* SkipLayerNormPluginDynamicCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength) noexcept
{
    // This object will be deleted when the network is destroyed, which will
    // call SkipLayerNormPluginDynamic::destroy()
    try
    {
        return new SkipLayerNormPluginDynamic(name, serialData, serialLength);
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

void SkipLayerNormPluginDynamicCreator::setPluginNamespace(const char* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

const char* SkipLayerNormPluginDynamicCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

SkipLayerNormVarSeqlenPlugin::SkipLayerNormVarSeqlenPlugin(
    const std::string name, const DataType type, const Weights& beta, const Weights& gamma, const Weights& bias)
    : mLayerName(name)
    , mGammaDev(nullptr)
    , mBetaDev(nullptr)
    , mLd(beta.count)
    , mType(type)
    , mBiasDev(nullptr)
{
    PLUGIN_VALIDATE(mLd > 0);
    PLUGIN_VALIDATE(beta.count == gamma.count);
    PLUGIN_VALIDATE(mType == nvinfer1::DataType::kFLOAT || mType == nvinfer1::DataType::kHALF
        || mType == nvinfer1::DataType::kINT8);
    // mCfgType is the dataType for beta, gamma bias weights, always fp16 or fp32
    // mType is the plugin IO datatype, can be int8
    mCfgType = mType == DataType::kINT8 ? DataType::kHALF :  mType;
    mParamWordsize = getElementSize(mCfgType);

    mBeta.convertAndCopy(beta, mCfgType);
    mGamma.convertAndCopy(gamma, mCfgType);

    mHasBias = (bias.values != nullptr);
    if (mHasBias)
    {
        mBias.convertAndCopy(bias, mCfgType);
    }

    copyToDevice(mGamma, getWeightsSize(mGamma, mCfgType), mGammaDev);
    copyToDevice(mBeta, getWeightsSize(mBeta, mCfgType), mBetaDev);
    if (mHasBias)
    {
        copyToDevice(mBias, getWeightsSize(mBias, mCfgType), mBiasDev);
    }
}

SkipLayerNormVarSeqlenPlugin::SkipLayerNormVarSeqlenPlugin(const std::string name, const void* data, size_t length)
    : mLayerName(name)
    , mGammaDev(nullptr)
    , mBetaDev(nullptr)
    , mBiasDev(nullptr)
{
    BERT_DEBUG_MSG("SkipLayerNormVarSeqlenPlugin deserialize");

    // Deserialize in the same order as serialization
    deserialize_value(&data, &length, &mType);
    deserialize_value(&data, &length, &mCfgType);
    deserialize_value(&data, &length, &mLd);
    deserialize_value(&data, &length, &mHasBias);

    PLUGIN_VALIDATE(mCfgType == nvinfer1::DataType::kFLOAT || mCfgType == nvinfer1::DataType::kHALF);
    mParamWordsize = getElementSize(mCfgType);

    const char* d = static_cast<const char*>(data);
    mBeta.convertAndCopy(d, mLd, mCfgType);
    mGamma.convertAndCopy(d, mLd, mCfgType);
    if (mHasBias)
    {
        mBias.convertAndCopy(d, mLd, mCfgType);
    }

    copyToDevice(mGamma, getWeightsSize(mGamma, mCfgType), mGammaDev);
    copyToDevice(mBeta, getWeightsSize(mBeta, mCfgType), mBetaDev);
    if (mHasBias)
    {
        copyToDevice(mBias, getWeightsSize(mBias, mCfgType), mBiasDev);
    }
}

// IPluginV2DynamicExt Methods
IPluginV2DynamicExt* SkipLayerNormVarSeqlenPlugin::clone() const noexcept
{
    try
    {
        BERT_DEBUG_MSG("SkipLayerNormVarSeqlenPlugin clone");

        auto* p = new SkipLayerNormVarSeqlenPlugin(mLayerName, mType, mBeta, mGamma, mBias);
        p->initialize();
        p->setPluginNamespace(mNamespace.c_str());
        return p;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

DimsExprs SkipLayerNormVarSeqlenPlugin::getOutputDimensions(
    int outputIndex, const DimsExprs* inputs, int nbInputs, IExprBuilder& exprBuilder) noexcept
{
    PLUGIN_ASSERT(nbInputs == 2);
    PLUGIN_ASSERT(outputIndex == 0);
    PLUGIN_ASSERT(inputs[0].nbDims == inputs[1].nbDims);
    return inputs[0];
}

bool SkipLayerNormVarSeqlenPlugin::supportsFormatCombination(
    int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept
{
    PLUGIN_ASSERT(nbInputs == 2);
    PLUGIN_ASSERT(nbOutputs == 1);

    const PluginTensorDesc& in = inOut[pos];

    if(mType != in.type) return false;
    if (pos == 0)
    {
        // Since H = W = 1, we can report CHWx for any x
        if (mType == DataType::kINT8)
        {
            // won't work for hiddensize too small!
            TensorFormat myFmt = TensorFormat::kCHW32;
            if (mLd < 32)
            {
                myFmt = TensorFormat::kCHW4;
                BERT_DEBUG_VALUE("SkipLayerNormDQQ: TensorFormat CHW4 for LD=", mLd);
            }
            else
            {
                BERT_DEBUG_VALUE("SkipLayerNormDQQ: TensorFormat CHW32 for LD=", mLd);
            }
            // TODO do we need to check if the vectorization divides mLd?
            return in.format == myFmt;
        }
        return in.format == TensorFormat::kLINEAR;
    }
    const PluginTensorDesc& prev = inOut[pos - 1];

    return in.format == prev.format;
}

void SkipLayerNormVarSeqlenPlugin::configurePlugin(
    const DynamicPluginTensorDesc* inputs, int nbInputs, const DynamicPluginTensorDesc* outputs, int nbOutputs) noexcept
{
    // Validate input arguments
    PLUGIN_ASSERT(nbOutputs == 1);
    PLUGIN_ASSERT(nbInputs == 2);
    if (mType == DataType::kFLOAT || mType == DataType::kHALF)
    {
        PLUGIN_ASSERT(mType == inputs[0].desc.type);
        PLUGIN_ASSERT(mType == inputs[1].desc.type);
    }
    else
    {
        PLUGIN_ASSERT(mType == inputs[0].desc.type || DataType::kFLOAT == inputs[0].desc.type);
        PLUGIN_ASSERT(mType == inputs[1].desc.type || DataType::kFLOAT == inputs[1].desc.type);
    }
    const auto& inDims0 = inputs[0].desc.dims;
    const auto& inDims1 = inputs[1].desc.dims;
    TRT_UNUSED inDims1;
    PLUGIN_ASSERT(inDims0.nbDims == inDims1.nbDims);

    PLUGIN_ASSERT(std::equal(inDims0.d, inDims0.d + inDims0.nbDims, inDims1.d));

    mCfgType = inputs[0].desc.type == DataType::kINT8 ? DataType::kHALF : inputs[0].desc.type;

    const auto paramType = getParamWordType(mCfgType);
    mParamWordsize = getElementSize(paramType);
}

size_t SkipLayerNormVarSeqlenPlugin::getWorkspaceSize(
    const PluginTensorDesc* inputs, int nbInputs, const PluginTensorDesc* outputs, int nbOutputs) const noexcept
{
    return 0;
}

int SkipLayerNormVarSeqlenPlugin::enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc,
    const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    const int inputVolume = volume(inputDesc[0].dims);
    PLUGIN_ASSERT(inputVolume % mLd == 0 && "inconsistent dimensions");
    int status = -1;
    DataType iType = inputDesc->type;

    // Our plugin outputs only one tensor
    // Launch CUDA kernel wrapper and save its return value
    if (iType == DataType::kFLOAT)
    {
        const auto* const input = static_cast<const float*>(inputs[0]);
        const auto* const skip = static_cast<const float*>(inputs[1]);
        auto* output = static_cast<float*>(outputs[0]);
        const auto* const bias = static_cast<const float*>(mBiasDev.get());
        const auto* const beta = static_cast<const float*>(mBetaDev.get());
        const auto* const gamma = static_cast<const float*>(mGammaDev.get());
        if (mHasBias)
        {
            status = computeSkipLayerNorm<float, true>(
                stream, static_cast<int>(mLd), inputVolume, input, skip, beta, gamma, output, bias);
        }
        else
        {
            status
                = computeSkipLayerNorm<float, false>(stream, static_cast<int>(mLd), inputVolume, input, skip, beta, gamma, output, bias);
        }
    }
    else if (iType == DataType::kHALF)
    {
        const auto* const input = static_cast<const half*>(inputs[0]);
        const auto* const skip = static_cast<const half*>(inputs[1]);
        auto* output = static_cast<half*>(outputs[0]);
        const auto* const bias = static_cast<const half*>(mBiasDev.get());
        const auto* const beta = static_cast<const half*>(mBetaDev.get());
        const auto* const gamma = static_cast<const half*>(mGammaDev.get());
        if (mHasBias)
        {
            status = computeSkipLayerNorm<half, true>(
                stream, static_cast<int>(mLd), inputVolume, input, skip, beta, gamma, output, bias);
        }
        else
        {
            status
                = computeSkipLayerNorm<half, false>(stream, static_cast<int>(mLd), inputVolume, input, skip, beta, gamma, output, bias);
        }
    }
    else if (iType == DataType::kINT8)
    {
        const float dqScaleIn = inputDesc[0].scale;
        const float dqScaleSkip = inputDesc[1].scale;
        const float qScale = 1.F / outputDesc[0].scale;
        const auto* const input = static_cast<const int8_t*>(inputs[0]);
        const auto* const skip = static_cast<const int8_t*>(inputs[1]);
        auto* output = static_cast<int8_t*>(outputs[0]);
        const auto* const bias = static_cast<const half*>(mBiasDev.get());
        const auto* const beta = static_cast<const half*>(mBetaDev.get());
        const auto* const gamma = static_cast<const half*>(mGammaDev.get());
        if (mHasBias)
        {
            status = computeSkipLayerNormDQQ<true>(stream, static_cast<int>(mLd), inputVolume, input, skip, beta, gamma,
                output, bias, dqScaleIn, dqScaleSkip, qScale);
        }
        else
        {
            status = computeSkipLayerNormDQQ<false>(
                stream, static_cast<int>(mLd), inputVolume, input, skip, beta, gamma, output, bias, dqScaleIn, dqScaleSkip, qScale);
        }
    }
    else
    {
        gLogError << "Unsupported type error, expected [kINT8,kHALF,kFLOAT], but received " << static_cast<int>(iType) << "." << std::endl;
        PLUGIN_ASSERT(false);
    }
    return status;
}

// IPluginV2Ext Methods
DataType SkipLayerNormVarSeqlenPlugin::getOutputDataType(int index, const DataType* inputTypes, int nbInputs) const noexcept
{
    PLUGIN_ASSERT(index == 0);
    PLUGIN_ASSERT(nbInputs == 2);
    return inputTypes[0];
}

// IPluginV2 Methods
const char* SkipLayerNormVarSeqlenPlugin::getPluginType() const noexcept
{
    return SKIP_LAYER_NORM_NAME;
}

const char* SkipLayerNormVarSeqlenPlugin::getPluginVersion() const noexcept
{
    return SKIP_LAYER_NORM_VAR_SEQLEN_VERSION;
}

int SkipLayerNormVarSeqlenPlugin::getNbOutputs() const noexcept
{
    return 1;
}
int SkipLayerNormVarSeqlenPlugin::initialize() noexcept
{
    BERT_DEBUG_MSG("SkipLayerNormVarSeqlenPlugin initialize");
    return 0;
}

void SkipLayerNormVarSeqlenPlugin::terminate() noexcept
{
    BERT_DEBUG_MSG("SkipLayerNormVarSeqlenPlugin terminate");
}

size_t SkipLayerNormVarSeqlenPlugin::getSerializationSize() const noexcept
{
    const size_t biasSize = mHasBias ? (mLd * mParamWordsize) : 0;
    return 2 * mParamWordsize * mLd + 2 * sizeof(DataType) + sizeof(mLd) + biasSize + sizeof(mHasBias);
}

void SkipLayerNormVarSeqlenPlugin::serialize(void* buffer) const noexcept
{
    serialize_value(&buffer, mType);
    serialize_value(&buffer, mCfgType);
    serialize_value(&buffer, mLd);
    serialize_value(&buffer, mHasBias);

    char* d = static_cast<char*>(buffer);
    serFromDev(d, static_cast<char*>(mBetaDev.get()), mLd * mParamWordsize);
    serFromDev(d, static_cast<char*>(mGammaDev.get()), mLd * mParamWordsize);
    if (mHasBias)
    {
        serFromDev(d, static_cast<char*>(mBiasDev.get()), mLd * mParamWordsize);
    }
}

void SkipLayerNormVarSeqlenPlugin::destroy() noexcept
{
    BERT_DEBUG_MSG("SkipLayerNormVarSeqlenPlugin destroy");
    // This gets called when the network containing plugin is destroyed
    mGammaDev.reset(nullptr);
    mBetaDev.reset(nullptr);
    mBiasDev.reset(nullptr);
    delete this;
}

void SkipLayerNormVarSeqlenPlugin::setPluginNamespace(const char* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

const char* SkipLayerNormVarSeqlenPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

/////////////////////////////////////////////////////////

SkipLayerNormVarSeqlenPluginCreator::SkipLayerNormVarSeqlenPluginCreator()
{
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("type_id"));
    mPluginAttributes.emplace_back(PluginField("beta"));
    mPluginAttributes.emplace_back(PluginField("gamma"));
    mPluginAttributes.emplace_back(PluginField("bias"));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* SkipLayerNormVarSeqlenPluginCreator::getPluginName() const noexcept
{
    return SKIP_LAYER_NORM_NAME;
}

const char* SkipLayerNormVarSeqlenPluginCreator::getPluginVersion() const noexcept
{
    return SKIP_LAYER_NORM_VAR_SEQLEN_VERSION;
}

const PluginFieldCollection* SkipLayerNormVarSeqlenPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* SkipLayerNormVarSeqlenPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
{
    try
    {
        BERT_DEBUG_MSG("SkipLayerNormVarSeqlenPluginCreator createPlugin");

        Weights beta{DataType::kFLOAT, nullptr, 0};
        Weights gamma{DataType::kFLOAT, nullptr, 0};
        Weights bias{DataType::kFLOAT, nullptr, 0};
        int32_t typeId = -1;

        plugin::validateRequiredAttributesExist({"type_id", "beta", "gamma"}, fc);

        for (int32_t i = 0; i < fc->nbFields; i++)
        {
            std::string field_name(fc->fields[i].name);

            if (field_name.compare("type_id") == 0)
            {
                typeId = *static_cast<int32_t const*>(fc->fields[i].data);
                BERT_DEBUG_VALUE("Building typeId: ", typeId);
            }

            if (field_name.compare("beta") == 0)
            {
                BERT_DEBUG_MSG("Building beta...");
                beta.values = fc->fields[i].data;
                beta.count = fc->fields[i].length;
                beta.type = fieldTypeToDataType(fc->fields[i].type);
            }

            if (field_name.compare("gamma") == 0)
            {
                BERT_DEBUG_MSG("Building gamma...");
                gamma.values = fc->fields[i].data;
                gamma.count = fc->fields[i].length;
                gamma.type = fieldTypeToDataType(fc->fields[i].type);
            }

            if (field_name.compare("bias") == 0)
            {
                BERT_DEBUG_MSG("Building bias...");
                bias.values = fc->fields[i].data;
                bias.count = fc->fields[i].length;
                bias.type = fieldTypeToDataType(fc->fields[i].type);
            }
        }
        BERT_DEBUG_VALUE("Type ", typeId);

        if (typeId < 0 || typeId > 3)
        {
            gLogError << "SkipLayerNorm: Invalid type ID: " << typeId << std::endl;
        }

        if (beta.count <= 0 || beta.values == nullptr)
        {
            gLogError << "SkipLayerNorm: invalid beta" << std::endl;
        }

        if (gamma.count <= 0 || gamma.values == nullptr)
        {
            gLogError << "SkipLayerNorm: invalid gamma" << std::endl;
        }

        return new SkipLayerNormVarSeqlenPlugin(name, static_cast<DataType>(typeId), beta, gamma, bias);
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* SkipLayerNormVarSeqlenPluginCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength) noexcept
{
    // This object will be deleted when the network is destroyed, which will
    // call SkipLayerNormVarSeqlenPlugin::destroy()
    try
    {
        return new SkipLayerNormVarSeqlenPlugin(name, serialData, serialLength);
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

void SkipLayerNormVarSeqlenPluginCreator::setPluginNamespace(const char* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

const char* SkipLayerNormVarSeqlenPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

#endif // CUDA_VERSION >= 10010
