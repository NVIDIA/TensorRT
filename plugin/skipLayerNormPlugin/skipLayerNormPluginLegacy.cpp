/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION &
 * AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
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
#include "skipLayerNormPluginLegacy.h"

#include <cstring>
#include <vector>

using namespace nvinfer1;
using namespace nvinfer1::plugin;
using namespace nvinfer1::plugin::bert;

// Clip plugin specific constants
namespace
{
constexpr char const* kSKIP_LAYER_NORM_VERSION{"1"};
constexpr char const* kSKIP_LAYER_NORM_NAME{"CustomSkipLayerNormPluginDynamic"};
constexpr char const* kSKIP_LAYER_NORM_VAR_SEQLEN_VERSION{"2"};
} // namespace

REGISTER_TENSORRT_PLUGIN(SkipLayerNormPluginDynamicCreator);
REGISTER_TENSORRT_PLUGIN(SkipLayerNormVarSeqlenPluginCreator);

SkipLayerNormPluginDynamic::SkipLayerNormPluginDynamic(const std::string name, const DataType type, int32_t const ld,
    Weights const& beta, Weights const& gamma, Weights const& bias)
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
    mCfgType = mType == DataType::kINT8 ? DataType::kHALF : mType;
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

SkipLayerNormPluginDynamic::SkipLayerNormPluginDynamic(const std::string name, void const* data, size_t length)
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

    char const* d = static_cast<char const*>(data);
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
    int32_t outputIndex, DimsExprs const* inputs, int32_t nbInputs, IExprBuilder& exprBuilder) noexcept
{
    try
    {
        PLUGIN_VALIDATE(inputs != nullptr);
        PLUGIN_VALIDATE(nbInputs == 2);
        PLUGIN_VALIDATE(outputIndex == 0);
        PLUGIN_VALIDATE(inputs[0].nbDims == inputs[1].nbDims);
        return inputs[0];
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return DimsExprs{};
}

bool SkipLayerNormPluginDynamic::supportsFormatCombination(
    int32_t pos, PluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    try
    {
        PLUGIN_VALIDATE(inOut != nullptr);
        PLUGIN_VALIDATE(nbInputs == 2);
        PLUGIN_VALIDATE(nbOutputs == 1);
        PLUGIN_VALIDATE(pos >= 0 && pos < (nbInputs + nbOutputs));

        PluginTensorDesc const& in = inOut[pos];
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
        PluginTensorDesc const& prev = inOut[pos - 1];

        return in.type == prev.type && in.format == prev.format;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return false;
}

void SkipLayerNormPluginDynamic::configurePlugin(DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
    DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) noexcept
{
    try
    {
        BERT_DEBUG_MSG("SkipLayerNormPluginDynamic configurePlugin");

        // Validate input arguments
        PLUGIN_VALIDATE(inputs != nullptr);
        PLUGIN_VALIDATE(outputs != nullptr);
        PLUGIN_VALIDATE(nbOutputs == 1);
        PLUGIN_VALIDATE(nbInputs == 2);
        if (mType == DataType::kFLOAT || mType == DataType::kHALF)
        {
            PLUGIN_VALIDATE(mType == inputs[0].desc.type);
            PLUGIN_VALIDATE(mType == inputs[1].desc.type);
        }
        else
        {
            PLUGIN_VALIDATE(mType == inputs[0].desc.type || DataType::kFLOAT == inputs[0].desc.type);
            PLUGIN_VALIDATE(mType == inputs[1].desc.type || DataType::kFLOAT == inputs[1].desc.type);
        }
        auto const& inDims0 = inputs[0].desc.dims;
        auto const& inDims1 = inputs[1].desc.dims;
        PLUGIN_VALIDATE(inDims0.nbDims == inDims1.nbDims);

        PLUGIN_VALIDATE(std::equal(inDims0.d, inDims0.d + inDims0.nbDims, inDims1.d));

        PLUGIN_VALIDATE(inDims0.nbDims == 5);
        mLd = inDims0.d[HDIM]; // hiddensize
        PLUGIN_VALIDATE(mLd != 0U);
        PLUGIN_VALIDATE(inDims0.d[3] == 1);
        PLUGIN_VALIDATE(inDims0.d[4] == 1);

        mCfgType = inputs[0].desc.type == DataType::kINT8 ? DataType::kHALF : inputs[0].desc.type;

        auto const paramType = mCfgType == DataType::kINT8 ? DataType::kHALF : mCfgType;
        mParamWordsize = getElementSize(paramType);
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
}

size_t SkipLayerNormPluginDynamic::getWorkspaceSize(
    PluginTensorDesc const* inputs, int32_t nbInputs, PluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept
{
    return 0;
}

int32_t SkipLayerNormPluginDynamic::enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc,
    void const* const* inputs, void* const* outputs, void* /* workspace */, cudaStream_t stream) noexcept
{
    int32_t status = -1;
    try
    {
        PLUGIN_VALIDATE(inputDesc != nullptr && outputDesc != nullptr && inputs != nullptr && outputs != nullptr);

        int32_t const inputVolume = volume(inputDesc[0].dims);
        DataType iType = inputDesc->type;

        // Our plugin outputs only one tensor
        // Launch CUDA kernel wrapper and save its return value
        if (iType == DataType::kFLOAT)
        {
            auto const* const input = static_cast<float const*>(inputs[0]);
            auto const* const skip = static_cast<float const*>(inputs[1]);
            auto* output = static_cast<float*>(outputs[0]);
            auto const* const bias = static_cast<float const*>(mBiasDev.get());
            auto const* const beta = static_cast<float const*>(mBetaDev.get());
            auto const* const gamma = static_cast<float const*>(mGammaDev.get());
            if (mHasBias)
            {
                status = computeSkipLayerNorm<float, true>(
                    stream, static_cast<int32_t>(mLd), inputVolume, input, skip, beta, gamma, output, bias);
            }
            else
            {
                status = computeSkipLayerNorm<float, false>(
                    stream, static_cast<int32_t>(mLd), inputVolume, input, skip, beta, gamma, output, bias);
            }
        }
        else if (iType == DataType::kHALF)
        {
            auto const* const input = static_cast<half const*>(inputs[0]);
            auto const* const skip = static_cast<half const*>(inputs[1]);
            auto* output = static_cast<half*>(outputs[0]);
            auto const* const bias = static_cast<half const*>(mBiasDev.get());
            auto const* const beta = static_cast<half const*>(mBetaDev.get());
            auto const* const gamma = static_cast<half const*>(mGammaDev.get());
            if (mHasBias)
            {
                status = computeSkipLayerNorm<half, true>(
                    stream, static_cast<int32_t>(mLd), inputVolume, input, skip, beta, gamma, output, bias);
            }
            else
            {
                status = computeSkipLayerNorm<half, false>(
                    stream, static_cast<int32_t>(mLd), inputVolume, input, skip, beta, gamma, output, bias);
            }
        }
        else if (iType == DataType::kINT8)
        {
            float const dqScaleIn = inputDesc[0].scale;
            float const dqScaleSkip = inputDesc[1].scale;
            PLUGIN_VALIDATE(outputDesc[0].scale != 0.0F);
            float const qScale = 1.F / outputDesc[0].scale;
            auto const* const input = static_cast<int8_t const*>(inputs[0]);
            auto const* const skip = static_cast<int8_t const*>(inputs[1]);
            auto* output = static_cast<int8_t*>(outputs[0]);
            auto const* const bias = static_cast<half const*>(mBiasDev.get());
            auto const* const beta = static_cast<half const*>(mBetaDev.get());
            auto const* const gamma = static_cast<half const*>(mGammaDev.get());
            if (mHasBias)
            {
                status = computeSkipLayerNormDQQ<true>(stream, static_cast<int32_t>(mLd), inputVolume, input, skip,
                    beta, gamma, output, bias, dqScaleIn, dqScaleSkip, qScale);
            }
            else
            {
                status = computeSkipLayerNormDQQ<false>(stream, static_cast<int32_t>(mLd), inputVolume, input, skip,
                    beta, gamma, output, bias, dqScaleIn, dqScaleSkip, qScale);
            }
        }
        else
        {
            PLUGIN_ERROR(("Unsupported type error, expected [kINT8,kHALF,kFLOAT], but received "
                + std::to_string(static_cast<int32_t>(iType)))
                             .c_str());
        }
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return status;
}

// IPluginV2Ext Methods
DataType SkipLayerNormPluginDynamic::getOutputDataType(
    int32_t index, DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    try
    {
        PLUGIN_VALIDATE(inputTypes != nullptr);
        PLUGIN_VALIDATE(index == 0);
        PLUGIN_VALIDATE(nbInputs == 2);
        return inputTypes[0];
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return DataType{};
}

// IPluginV2 Methods
char const* SkipLayerNormPluginDynamic::getPluginType() const noexcept
{
    return kSKIP_LAYER_NORM_NAME;
}

char const* SkipLayerNormPluginDynamic::getPluginVersion() const noexcept
{
    return kSKIP_LAYER_NORM_VERSION;
}

int32_t SkipLayerNormPluginDynamic::getNbOutputs() const noexcept
{
    return 1;
}
int32_t SkipLayerNormPluginDynamic::initialize() noexcept
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
    try
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
    catch (std::exception const& e)
    {
        caughtError(e);
    }
}

void SkipLayerNormPluginDynamic::destroy() noexcept
{
    try
    {
        BERT_DEBUG_MSG("SkipLayerNormPluginDynamic destroy");
        // This gets called when the network containing plugin is destroyed
        mGammaDev.reset(nullptr);
        mBetaDev.reset(nullptr);
        mBiasDev.reset(nullptr);
        delete this;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
}

void SkipLayerNormPluginDynamic::setPluginNamespace(char const* libNamespace) noexcept
{
    try
    {
        mNamespace = libNamespace;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
}

char const* SkipLayerNormPluginDynamic::getPluginNamespace() const noexcept
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

char const* SkipLayerNormPluginDynamicCreator::getPluginName() const noexcept
{
    return kSKIP_LAYER_NORM_NAME;
}

char const* SkipLayerNormPluginDynamicCreator::getPluginVersion() const noexcept
{
    return kSKIP_LAYER_NORM_VERSION;
}

PluginFieldCollection const* SkipLayerNormPluginDynamicCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* SkipLayerNormPluginDynamicCreator::createPlugin(char const* name, PluginFieldCollection const* fc) noexcept
{
    try
    {
        BERT_DEBUG_MSG("SkipLayerNormPluginDynamicCreator createPlugin");

        int32_t ld = 0;
        Weights beta{DataType::kFLOAT, nullptr, 0};
        Weights gamma{DataType::kFLOAT, nullptr, 0};
        Weights bias{DataType::kFLOAT, nullptr, 0};
        int32_t typeId = -1;

        PLUGIN_VALIDATE(fc != nullptr);

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

        PLUGIN_VALIDATE(
            typeId >= 0 && typeId <= 3, ("SkipLayerNorm: Invalid type ID: " + std::to_string(typeId)).c_str());

        PLUGIN_VALIDATE(beta.values != nullptr, "SkipLayerNorm: invalid beta");
        PLUGIN_VALIDATE(beta.count > 0, "SkipLayerNorm: invalid beta");

        PLUGIN_VALIDATE(gamma.values != nullptr, "SkipLayerNorm: invalid gamma");
        PLUGIN_VALIDATE(gamma.count > 0, "SkipLayerNorm: invalid gamma");

        return new SkipLayerNormPluginDynamic(name, static_cast<DataType>(typeId), ld, beta, gamma, bias);
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* SkipLayerNormPluginDynamicCreator::deserializePlugin(
    char const* name, void const* serialData, size_t serialLength) noexcept
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

void SkipLayerNormPluginDynamicCreator::setPluginNamespace(char const* libNamespace) noexcept
{
    try
    {
        mNamespace = libNamespace;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
}

char const* SkipLayerNormPluginDynamicCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

SkipLayerNormVarSeqlenPlugin::SkipLayerNormVarSeqlenPlugin(
    const std::string name, const DataType type, Weights const& beta, Weights const& gamma, Weights const& bias)
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
    mCfgType = mType == DataType::kINT8 ? DataType::kHALF : mType;
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

SkipLayerNormVarSeqlenPlugin::SkipLayerNormVarSeqlenPlugin(const std::string name, void const* data, size_t length)
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

    char const* d = static_cast<char const*>(data);
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
    int32_t outputIndex, DimsExprs const* inputs, int32_t nbInputs, IExprBuilder& exprBuilder) noexcept
{
    try
    {
        PLUGIN_VALIDATE(inputs != nullptr);
        PLUGIN_VALIDATE(nbInputs == 2);
        PLUGIN_VALIDATE(outputIndex == 0);
        PLUGIN_VALIDATE(inputs[0].nbDims == inputs[1].nbDims);
        return inputs[0];
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return DimsExprs{};
}

bool SkipLayerNormVarSeqlenPlugin::supportsFormatCombination(
    int32_t pos, PluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    try
    {
        PLUGIN_VALIDATE(inOut != nullptr);
        PLUGIN_VALIDATE(nbInputs == 2);
        PLUGIN_VALIDATE(nbOutputs == 1);
        PLUGIN_VALIDATE(pos >= 0 && pos < (nbInputs + nbOutputs));

        PluginTensorDesc const& in = inOut[pos];

        if (mType != in.type)
            return false;
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
        PluginTensorDesc const& prev = inOut[pos - 1];

        return in.format == prev.format;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return false;
}

void SkipLayerNormVarSeqlenPlugin::configurePlugin(DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
    DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) noexcept
{
    try
    {
        // Validate input arguments
        PLUGIN_VALIDATE(inputs != nullptr);
        PLUGIN_VALIDATE(outputs != nullptr);
        PLUGIN_VALIDATE(nbOutputs == 1);
        PLUGIN_VALIDATE(nbInputs == 2);

        if (mType == DataType::kFLOAT || mType == DataType::kHALF)
        {
            PLUGIN_VALIDATE(mType == inputs[0].desc.type);
            PLUGIN_VALIDATE(mType == inputs[1].desc.type);
        }
        else
        {
            PLUGIN_VALIDATE(mType == inputs[0].desc.type || DataType::kFLOAT == inputs[0].desc.type);
            PLUGIN_VALIDATE(mType == inputs[1].desc.type || DataType::kFLOAT == inputs[1].desc.type);
        }
        auto const& inDims0 = inputs[0].desc.dims;
        auto const& inDims1 = inputs[1].desc.dims;
        PLUGIN_VALIDATE(inDims0.nbDims == inDims1.nbDims);

        PLUGIN_VALIDATE(std::equal(inDims0.d, inDims0.d + inDims0.nbDims, inDims1.d));

        mCfgType = inputs[0].desc.type == DataType::kINT8 ? DataType::kHALF : inputs[0].desc.type;

        auto const paramType = mCfgType == DataType::kINT8 ? DataType::kHALF : mCfgType;
        mParamWordsize = getElementSize(paramType);
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
}

size_t SkipLayerNormVarSeqlenPlugin::getWorkspaceSize(
    PluginTensorDesc const* inputs, int32_t nbInputs, PluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept
{
    return 0;
}

int32_t SkipLayerNormVarSeqlenPlugin::enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc,
    void const* const* inputs, void* const* outputs, void* /* workspace */, cudaStream_t stream) noexcept
{
    int32_t status = -1;
    try
    {
        PLUGIN_VALIDATE(inputDesc != nullptr && outputDesc != nullptr && inputs != nullptr && outputs != nullptr);

        int32_t const inputVolume = volume(inputDesc[0].dims);
        PLUGIN_VALIDATE(inputVolume % mLd == 0 && "inconsistent dimensions");
        DataType iType = inputDesc->type;

        // Our plugin outputs only one tensor
        // Launch CUDA kernel wrapper and save its return value
        if (iType == DataType::kFLOAT)
        {
            auto const* const input = static_cast<float const*>(inputs[0]);
            auto const* const skip = static_cast<float const*>(inputs[1]);
            auto* output = static_cast<float*>(outputs[0]);
            auto const* const bias = static_cast<float const*>(mBiasDev.get());
            auto const* const beta = static_cast<float const*>(mBetaDev.get());
            auto const* const gamma = static_cast<float const*>(mGammaDev.get());
            if (mHasBias)
            {
                status = computeSkipLayerNorm<float, true>(
                    stream, static_cast<int32_t>(mLd), inputVolume, input, skip, beta, gamma, output, bias);
            }
            else
            {
                status = computeSkipLayerNorm<float, false>(
                    stream, static_cast<int32_t>(mLd), inputVolume, input, skip, beta, gamma, output, bias);
            }
        }
        else if (iType == DataType::kHALF)
        {
            auto const* const input = static_cast<half const*>(inputs[0]);
            auto const* const skip = static_cast<half const*>(inputs[1]);
            auto* output = static_cast<half*>(outputs[0]);
            auto const* const bias = static_cast<half const*>(mBiasDev.get());
            auto const* const beta = static_cast<half const*>(mBetaDev.get());
            auto const* const gamma = static_cast<half const*>(mGammaDev.get());
            if (mHasBias)
            {
                status = computeSkipLayerNorm<half, true>(
                    stream, static_cast<int32_t>(mLd), inputVolume, input, skip, beta, gamma, output, bias);
            }
            else
            {
                status = computeSkipLayerNorm<half, false>(
                    stream, static_cast<int32_t>(mLd), inputVolume, input, skip, beta, gamma, output, bias);
            }
        }
        else if (iType == DataType::kINT8)
        {
            float const dqScaleIn = inputDesc[0].scale;
            float const dqScaleSkip = inputDesc[1].scale;
            PLUGIN_VALIDATE(outputDesc[0].scale != 0.0F);
            float const qScale = 1.F / outputDesc[0].scale;
            auto const* const input = static_cast<int8_t const*>(inputs[0]);
            auto const* const skip = static_cast<int8_t const*>(inputs[1]);
            auto* output = static_cast<int8_t*>(outputs[0]);
            auto const* const bias = static_cast<half const*>(mBiasDev.get());
            auto const* const beta = static_cast<half const*>(mBetaDev.get());
            auto const* const gamma = static_cast<half const*>(mGammaDev.get());
            if (mHasBias)
            {
                status = computeSkipLayerNormDQQ<true>(stream, static_cast<int32_t>(mLd), inputVolume, input, skip,
                    beta, gamma, output, bias, dqScaleIn, dqScaleSkip, qScale);
            }
            else
            {
                status = computeSkipLayerNormDQQ<false>(stream, static_cast<int32_t>(mLd), inputVolume, input, skip,
                    beta, gamma, output, bias, dqScaleIn, dqScaleSkip, qScale);
            }
        }
        else
        {
            PLUGIN_VALIDATE(("Unsupported type error, expected [kINT8,kHALF,kFLOAT], but received "
                + std::to_string(static_cast<int32_t>(iType)))
                                .c_str());
        }
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return status;
}

// IPluginV2Ext Methods
DataType SkipLayerNormVarSeqlenPlugin::getOutputDataType(
    int32_t index, DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    PLUGIN_VALIDATE(inputTypes != nullptr);
    PLUGIN_VALIDATE(index == 0);
    PLUGIN_VALIDATE(nbInputs == 2);
    return inputTypes[0];
}

// IPluginV2 Methods
char const* SkipLayerNormVarSeqlenPlugin::getPluginType() const noexcept
{
    return kSKIP_LAYER_NORM_NAME;
}

char const* SkipLayerNormVarSeqlenPlugin::getPluginVersion() const noexcept
{
    return kSKIP_LAYER_NORM_VAR_SEQLEN_VERSION;
}

int32_t SkipLayerNormVarSeqlenPlugin::getNbOutputs() const noexcept
{
    return 1;
}
int32_t SkipLayerNormVarSeqlenPlugin::initialize() noexcept
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
    try
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
    catch (std::exception const& e)
    {
        caughtError(e);
    }
}

void SkipLayerNormVarSeqlenPlugin::destroy() noexcept
{
    try
    {
        BERT_DEBUG_MSG("SkipLayerNormVarSeqlenPlugin destroy");
        // This gets called when the network containing plugin is destroyed
        mGammaDev.reset(nullptr);
        mBetaDev.reset(nullptr);
        mBiasDev.reset(nullptr);
        delete this;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
}

void SkipLayerNormVarSeqlenPlugin::setPluginNamespace(char const* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

char const* SkipLayerNormVarSeqlenPlugin::getPluginNamespace() const noexcept
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

char const* SkipLayerNormVarSeqlenPluginCreator::getPluginName() const noexcept
{
    return kSKIP_LAYER_NORM_NAME;
}

char const* SkipLayerNormVarSeqlenPluginCreator::getPluginVersion() const noexcept
{
    return kSKIP_LAYER_NORM_VAR_SEQLEN_VERSION;
}

PluginFieldCollection const* SkipLayerNormVarSeqlenPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* SkipLayerNormVarSeqlenPluginCreator::createPlugin(char const* name, PluginFieldCollection const* fc) noexcept
{
    try
    {
        BERT_DEBUG_MSG("SkipLayerNormVarSeqlenPluginCreator createPlugin");

        Weights beta{DataType::kFLOAT, nullptr, 0};
        Weights gamma{DataType::kFLOAT, nullptr, 0};
        Weights bias{DataType::kFLOAT, nullptr, 0};
        int32_t typeId = -1;

        PLUGIN_VALIDATE(fc != nullptr);

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

        PLUGIN_VALIDATE(
            typeId >= 0 && typeId <= 3, ("SkipLayerNorm: Invalid type ID: " + std::to_string(typeId)).c_str());

        PLUGIN_VALIDATE(beta.values != nullptr, "SkipLayerNorm: invalid beta");
        PLUGIN_VALIDATE(beta.count > 0, "SkipLayerNorm: invalid beta");

        PLUGIN_VALIDATE(gamma.values != nullptr, "SkipLayerNorm: invalid gamma");
        PLUGIN_VALIDATE(gamma.count > 0, "SkipLayerNorm: invalid gamma");

        return new SkipLayerNormVarSeqlenPlugin(name, static_cast<DataType>(typeId), beta, gamma, bias);
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* SkipLayerNormVarSeqlenPluginCreator::deserializePlugin(
    char const* name, void const* serialData, size_t serialLength) noexcept
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

void SkipLayerNormVarSeqlenPluginCreator::setPluginNamespace(char const* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

char const* SkipLayerNormVarSeqlenPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

#endif // CUDA_VERSION >= 10010
