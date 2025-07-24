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
#include "skipLayerNormPlugin.h"

#include <cstring>
#include <vector>

using namespace nvinfer1;
using namespace nvinfer1::plugin;
using namespace nvinfer1::plugin::bert;

// Clip plugin specific constants
namespace
{
constexpr char const* kSKIP_LAYER_NORM_VERSION{"5"};
constexpr char const* kSKIP_LAYER_NORM_NAME{"CustomSkipLayerNormPluginDynamic"};
constexpr char const* kSKIP_LAYER_NORM_VAR_SEQLEN_VERSION{"6"};
} // namespace

REGISTER_TENSORRT_PLUGIN(SkipLayerNormPluginV3Creator);
REGISTER_TENSORRT_PLUGIN(SkipLayerNormVarSeqlenPluginV3Creator);

SkipLayerNormPluginV3::SkipLayerNormPluginV3(const std::string name, const DataType type, int32_t const ld,
    Weights const& beta, Weights const& gamma, Weights const& bias)
    : mLayerName(name)
    , mType(type)
    , mLd(ld)
    , mGammaDev(nullptr)
    , mBetaDev(nullptr)
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
    BERT_DEBUG_MSG("SkipLayerNormPluginV3 initialize");
}

SkipLayerNormPluginV3::~SkipLayerNormPluginV3()
{
    BERT_DEBUG_MSG("SkipLayerNormPluginV3 terminate");
    try
    {
        BERT_DEBUG_MSG("SkipLayerNormPluginV3 destroy");
        mGammaDev.reset(nullptr);
        mBetaDev.reset(nullptr);
        mBiasDev.reset(nullptr);
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
}

IPluginV3* SkipLayerNormPluginV3::clone() noexcept
{
    try
    {
        BERT_DEBUG_MSG("SkipLayerNormPluginV3 clone");

        auto* p = new SkipLayerNormPluginV3(mLayerName, mType, mLd, mBeta, mGamma, mBias);
        p->setPluginNamespace(mNamespace.c_str());
        return p;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

int32_t SkipLayerNormPluginV3::getOutputShapes(DimsExprs const* inputs, int32_t nbInputs, DimsExprs const* shapeInputs,
    int32_t nbShapeInputs, DimsExprs* outputs, int32_t nbOutputs, IExprBuilder& exprBuilder) noexcept
{
    try
    {
        PLUGIN_VALIDATE(inputs != nullptr);
        PLUGIN_VALIDATE(nbInputs == 2);
        PLUGIN_VALIDATE(inputs[0].nbDims == inputs[1].nbDims);
        outputs[0] = inputs[0];
        return pluginStatus_t::STATUS_SUCCESS;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return pluginStatus_t::STATUS_FAILURE;
}

bool SkipLayerNormPluginV3::supportsFormatCombination(
    int32_t pos, DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    try
    {
        PLUGIN_VALIDATE(inOut != nullptr);
        PLUGIN_VALIDATE(nbInputs == 2);
        PLUGIN_VALIDATE(nbOutputs == 1);
        PLUGIN_VALIDATE(pos >= 0 && pos < (nbInputs + nbOutputs));

        PluginTensorDesc const& in = inOut[pos].desc;
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
        PluginTensorDesc const& prev = inOut[pos - 1].desc;

        return in.type == prev.type && in.format == prev.format;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return false;
}

int32_t SkipLayerNormPluginV3::configurePlugin(DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
    DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) noexcept
{
    return pluginStatus_t::STATUS_SUCCESS;
}

size_t SkipLayerNormPluginV3::getWorkspaceSize(DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
    DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept
{
    return 0;
}

int32_t SkipLayerNormPluginV3::enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
    nvinfer1::PluginTensorDesc const* outputDesc, void const* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream) noexcept
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

int32_t SkipLayerNormPluginV3::getOutputDataTypes(
    DataType* outputTypes, int32_t nbOutputs, DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    try
    {
        PLUGIN_VALIDATE(outputTypes != nullptr);
        PLUGIN_VALIDATE(nbOutputs == 1);
        PLUGIN_VALIDATE(inputTypes != nullptr);
        PLUGIN_VALIDATE(nbInputs == 2);
        outputTypes[0] = inputTypes[0];
        return pluginStatus_t::STATUS_SUCCESS;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return pluginStatus_t::STATUS_FAILURE;
}

char const* SkipLayerNormPluginV3::getPluginVersion() const noexcept
{
    return kSKIP_LAYER_NORM_VERSION;
}

int32_t SkipLayerNormPluginV3::getNbOutputs() const noexcept
{
    return 1;
}

PluginFieldCollection const* SkipLayerNormPluginV3::getFieldsToSerialize() noexcept
{
    mDataToSerialize.clear();
    mDataToSerialize.emplace_back("type_id", &mType, PluginFieldType::kINT32, 1);
    mDataToSerialize.emplace_back("ld", &mLd, PluginFieldType::kINT32, 1);
    if (mCfgType == DataType::kHALF)
    {
        mDataToSerialize.emplace_back(
            "beta", static_cast<half const*>(mBeta.values), PluginFieldType::kFLOAT16, mBeta.count);
        PLUGIN_ASSERT(mBeta.type == mCfgType);
        mDataToSerialize.emplace_back(
            "gamma", static_cast<half const*>(mGamma.values), PluginFieldType::kFLOAT16, mGamma.count);
        PLUGIN_ASSERT(mGamma.type == mCfgType);
        if (mHasBias)
        {
            mDataToSerialize.emplace_back(
                "bias", static_cast<half const*>(mBias.values), PluginFieldType::kFLOAT16, mBias.count);
            PLUGIN_ASSERT(mBias.type == mCfgType);
        }
    }
    else
    {
        PLUGIN_ASSERT(mCfgType == DataType::kFLOAT);
        mDataToSerialize.emplace_back(
            "beta", static_cast<float const*>(mBeta.values), PluginFieldType::kFLOAT32, mBeta.count);
        PLUGIN_ASSERT(mBeta.type == mCfgType);
        mDataToSerialize.emplace_back(
            "gamma", static_cast<float const*>(mGamma.values), PluginFieldType::kFLOAT32, mGamma.count);
        PLUGIN_ASSERT(mGamma.type == mCfgType);
        if (mHasBias)
        {
            mDataToSerialize.emplace_back(
                "bias", static_cast<float const*>(mBias.values), PluginFieldType::kFLOAT32, mBias.count);
            PLUGIN_ASSERT(mBias.type == mCfgType);
        }
    }

    mFCToSerialize.nbFields = mDataToSerialize.size();
    mFCToSerialize.fields = mDataToSerialize.data();

    return &mFCToSerialize;
}
void SkipLayerNormPluginV3::setPluginNamespace(char const* libNamespace) noexcept
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

char const* SkipLayerNormPluginV3::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

char const* SkipLayerNormPluginV3::getPluginName() const noexcept
{
    return kSKIP_LAYER_NORM_NAME;
}

int32_t SkipLayerNormPluginV3::onShapeChange(
    PluginTensorDesc const* inputs, int32_t nbInputs, PluginTensorDesc const* outputs, int32_t nbOutputs) noexcept
{
    try
    {
        BERT_DEBUG_MSG("SkipLayerNormPluginV3 onShapeChange");

        // Validate input arguments
        PLUGIN_VALIDATE(inputs != nullptr);
        PLUGIN_VALIDATE(outputs != nullptr);
        PLUGIN_VALIDATE(nbOutputs == 1);
        PLUGIN_VALIDATE(nbInputs == 2);
        if (mType == DataType::kFLOAT || mType == DataType::kHALF)
        {
            PLUGIN_VALIDATE(mType == inputs[0].type);
            PLUGIN_VALIDATE(mType == inputs[1].type);
        }
        else
        {
            PLUGIN_VALIDATE(mType == inputs[0].type || DataType::kFLOAT == inputs[0].type);
            PLUGIN_VALIDATE(mType == inputs[1].type || DataType::kFLOAT == inputs[1].type);
        }
        auto const& inDims0 = inputs[0].dims;
        auto const& inDims1 = inputs[1].dims;
        PLUGIN_VALIDATE(inDims0.nbDims == inDims1.nbDims);

        PLUGIN_VALIDATE(std::equal(inDims0.d, inDims0.d + inDims0.nbDims, inDims1.d));

        PLUGIN_VALIDATE(inDims0.nbDims == 5);
        mLd = inDims0.d[HDIM]; // hiddensize
        PLUGIN_VALIDATE(mLd != 0);
        PLUGIN_VALIDATE(inDims0.d[3] == 1);
        PLUGIN_VALIDATE(inDims0.d[4] == 1);

        mCfgType = inputs[0].type == DataType::kINT8 ? DataType::kHALF : inputs[0].type;

        mParamWordsize = getElementSize(mCfgType);
        return pluginStatus_t::STATUS_SUCCESS;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return pluginStatus_t::STATUS_FAILURE;
}

IPluginV3* SkipLayerNormPluginV3::attachToContext(IPluginResourceContext* context) noexcept
{
    return clone();
}

IPluginCapability* SkipLayerNormPluginV3::getCapabilityInterface(PluginCapabilityType type) noexcept
{
    try
    {
        if (type == PluginCapabilityType::kBUILD)
        {
            return static_cast<IPluginV3OneBuild*>(this);
        }
        if (type == PluginCapabilityType::kRUNTIME)
        {
            return static_cast<IPluginV3OneRuntime*>(this);
        }
        PLUGIN_ASSERT(type == PluginCapabilityType::kCORE);
        return static_cast<IPluginV3OneCore*>(this);
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

////////////////////////// SkipLayerNormPluginV3 (version:5) Creator ///////////////////////////////

SkipLayerNormPluginV3Creator::SkipLayerNormPluginV3Creator()
{
    static std::mutex sMutex;
    std::lock_guard<std::mutex> guard(sMutex);
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("type_id"));
    mPluginAttributes.emplace_back(PluginField("ld"));
    mPluginAttributes.emplace_back(PluginField("beta"));
    mPluginAttributes.emplace_back(PluginField("gamma"));
    mPluginAttributes.emplace_back(PluginField("bias"));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

char const* SkipLayerNormPluginV3Creator::getPluginName() const noexcept
{
    return kSKIP_LAYER_NORM_NAME;
}

char const* SkipLayerNormPluginV3Creator::getPluginVersion() const noexcept
{
    return kSKIP_LAYER_NORM_VERSION;
}

PluginFieldCollection const* SkipLayerNormPluginV3Creator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV3* SkipLayerNormPluginV3Creator::createPlugin(
    char const* name, PluginFieldCollection const* fc, TensorRTPhase phase) noexcept
{
    try
    {
        BERT_DEBUG_MSG("SkipLayerNormPluginV3Creator createPlugin");

        int32_t ld = 0;
        Weights beta{DataType::kFLOAT, nullptr, 0};
        Weights gamma{DataType::kFLOAT, nullptr, 0};
        Weights bias{DataType::kFLOAT, nullptr, 0};
        int32_t typeId = -1;

        PLUGIN_VALIDATE(fc != nullptr);
        PLUGIN_VALIDATE(fc->fields != nullptr);

        plugin::validateRequiredAttributesExist({"type_id", "beta", "ld", "gamma"}, fc);

        for (int32_t i = 0; i < fc->nbFields; i++)
        {
            std::string fieldName(fc->fields[i].name);
            if (fieldName == "type_id")
            {
                typeId = *static_cast<int32_t const*>(fc->fields[i].data);
                BERT_DEBUG_VALUE("Building typeId: ", typeId);
            }
            else if (fieldName == "ld")
            {
                ld = *static_cast<int32_t const*>(fc->fields[i].data);
                BERT_DEBUG_VALUE("Building ld: ", ld);
            }
            // process the weight tensors beta, gamma, bias
            else if (fieldName == "beta" || fieldName == "gamma" || fieldName == "bias")
            {
                Weights* weightPtr = (fieldName == "beta") ? &beta : (fieldName == "gamma") ? &gamma : &bias;

                BERT_DEBUG_MSG(("Building " + fieldName + "...").c_str());
                weightPtr->type = fieldTypeToDataType(fc->fields[i].type);
                weightPtr->values = fc->fields[i].data;
                weightPtr->count = fc->fields[i].length;
            }
        }
        BERT_DEBUG_VALUE("Type ", typeId);

        PLUGIN_VALIDATE(
            typeId >= 0 && typeId <= 3, ("SkipLayerNorm: Invalid type ID: " + std::to_string(typeId)).c_str());

        PLUGIN_VALIDATE(beta.values != nullptr, "SkipLayerNorm: invalid beta");
        PLUGIN_VALIDATE(beta.count > 0, "SkipLayerNorm: invalid beta");
        PLUGIN_VALIDATE(gamma.values != nullptr, "SkipLayerNorm: invalid gamma");
        PLUGIN_VALIDATE(gamma.count > 0, "SkipLayerNorm: invalid gamma");
        if (bias.values != nullptr)
        {
            PLUGIN_VALIDATE(bias.count > 0, "SkipLayerNorm: invalid bias");
        }

        return new SkipLayerNormPluginV3(name, static_cast<DataType>(typeId), ld, beta, gamma, bias);
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

void SkipLayerNormPluginV3Creator::setPluginNamespace(char const* libNamespace) noexcept
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

char const* SkipLayerNormPluginV3Creator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

////////////////////////// SkipLayerNormVarSeqlenPluginV3 (skipLayerNorm version: 6) ///////////////////////////////

SkipLayerNormVarSeqlenPluginV3::SkipLayerNormVarSeqlenPluginV3(
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

SkipLayerNormVarSeqlenPluginV3::~SkipLayerNormVarSeqlenPluginV3()
{
    try
    {
        BERT_DEBUG_MSG("SkipLayerNormVarSeqlenPluginV3 destroy");
        mGammaDev.reset(nullptr);
        mBetaDev.reset(nullptr);
        mBiasDev.reset(nullptr);
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
}

IPluginV3* SkipLayerNormVarSeqlenPluginV3::clone() noexcept
{
    try
    {
        BERT_DEBUG_MSG("SkipLayerNormVarSeqlenPluginV3 clone");
        auto* p = new SkipLayerNormVarSeqlenPluginV3(mLayerName, mType, mBeta, mGamma, mBias);
        p->setPluginNamespace(mNamespace.c_str());
        return p;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

int32_t SkipLayerNormVarSeqlenPluginV3::getOutputShapes(DimsExprs const* inputs, int32_t nbInputs,
    DimsExprs const* shapeInputs, int32_t nbShapeInputs, DimsExprs* outputs, int32_t nbOutputs,
    IExprBuilder& exprBuilder) noexcept
{
    try
    {
        PLUGIN_VALIDATE(inputs != nullptr);
        PLUGIN_VALIDATE(nbInputs == 2);
        PLUGIN_VALIDATE(nbOutputs == 1);
        PLUGIN_VALIDATE(inputs[0].nbDims == inputs[1].nbDims);
        outputs[0] = inputs[0];
        return pluginStatus_t::STATUS_SUCCESS;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return pluginStatus_t::STATUS_FAILURE;
}

bool SkipLayerNormVarSeqlenPluginV3::supportsFormatCombination(
    int32_t pos, DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    try
    {
        PLUGIN_VALIDATE(inOut != nullptr);
        PLUGIN_VALIDATE(nbInputs == 2);
        PLUGIN_VALIDATE(nbOutputs == 1);
        PLUGIN_VALIDATE(pos >= 0 && pos < (nbInputs + nbOutputs));

        PluginTensorDesc const& in = inOut[pos].desc;

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
        PluginTensorDesc const& prev = inOut[pos - 1].desc;

        return in.format == prev.format;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return false;
}

int32_t SkipLayerNormVarSeqlenPluginV3::configurePlugin(DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
    DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) noexcept
{
    return pluginStatus_t::STATUS_SUCCESS;
}

size_t SkipLayerNormVarSeqlenPluginV3::getWorkspaceSize(DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
    DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept
{
    return 0;
}

int32_t SkipLayerNormVarSeqlenPluginV3::enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
    nvinfer1::PluginTensorDesc const* outputDesc, void const* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream) noexcept
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

int32_t SkipLayerNormVarSeqlenPluginV3::getOutputDataTypes(
    DataType* outputTypes, int32_t nbOutputs, DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    try
    {
        PLUGIN_VALIDATE(outputTypes != nullptr);
        PLUGIN_VALIDATE(nbOutputs == 1);
        PLUGIN_VALIDATE(inputTypes != nullptr);
        PLUGIN_VALIDATE(nbInputs == 2);
        outputTypes[0] = inputTypes[0];
        return pluginStatus_t::STATUS_SUCCESS;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return pluginStatus_t::STATUS_FAILURE;
}

char const* SkipLayerNormVarSeqlenPluginV3::getPluginVersion() const noexcept
{
    return kSKIP_LAYER_NORM_VAR_SEQLEN_VERSION;
}

int32_t SkipLayerNormVarSeqlenPluginV3::getNbOutputs() const noexcept
{
    return 1;
}
PluginFieldCollection const* SkipLayerNormVarSeqlenPluginV3::getFieldsToSerialize() noexcept
{
    mDataToSerialize.clear();
    mDataToSerialize.emplace_back("type_id", &mType, PluginFieldType::kINT32, 1);
    mDataToSerialize.emplace_back("ld", &mLd, PluginFieldType::kINT32, 1);
    if (mCfgType == DataType::kHALF)
    {
        mDataToSerialize.emplace_back(
            "beta", static_cast<half const*>(mBeta.values), PluginFieldType::kFLOAT16, mBeta.count);
        PLUGIN_ASSERT(mBeta.type == mCfgType);
        mDataToSerialize.emplace_back(
            "gamma", static_cast<half const*>(mGamma.values), PluginFieldType::kFLOAT16, mGamma.count);
        PLUGIN_ASSERT(mGamma.type == mCfgType);
        if (mHasBias)
        {
            mDataToSerialize.emplace_back(
                "bias", static_cast<half const*>(mBias.values), PluginFieldType::kFLOAT16, mBias.count);
            PLUGIN_ASSERT(mBias.type == mCfgType);
        }
    }
    else
    {
        PLUGIN_ASSERT(mCfgType == DataType::kFLOAT);
        mDataToSerialize.emplace_back(
            "beta", static_cast<float const*>(mBeta.values), PluginFieldType::kFLOAT32, mBeta.count);
        PLUGIN_ASSERT(mBeta.type == mCfgType);
        mDataToSerialize.emplace_back(
            "gamma", static_cast<float const*>(mGamma.values), PluginFieldType::kFLOAT32, mGamma.count);
        PLUGIN_ASSERT(mGamma.type == mCfgType);
        if (mHasBias)
        {
            mDataToSerialize.emplace_back(
                "bias", static_cast<float const*>(mBias.values), PluginFieldType::kFLOAT32, mBias.count);
            PLUGIN_ASSERT(mBias.type == mCfgType);
        }
    }

    mFCToSerialize.nbFields = mDataToSerialize.size();
    mFCToSerialize.fields = mDataToSerialize.data();

    return &mFCToSerialize;
}

void SkipLayerNormVarSeqlenPluginV3::setPluginNamespace(char const* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

char const* SkipLayerNormVarSeqlenPluginV3::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

char const* SkipLayerNormVarSeqlenPluginV3::getPluginName() const noexcept
{
    return kSKIP_LAYER_NORM_NAME;
}

int32_t SkipLayerNormVarSeqlenPluginV3::onShapeChange(
    PluginTensorDesc const* inputs, int32_t nbInputs, PluginTensorDesc const* outputs, int32_t nbOutputs) noexcept
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
            PLUGIN_VALIDATE(mType == inputs[0].type);
            PLUGIN_VALIDATE(mType == inputs[1].type);
        }
        else
        {
            PLUGIN_VALIDATE(mType == inputs[0].type || DataType::kFLOAT == inputs[0].type);
            PLUGIN_VALIDATE(mType == inputs[1].type || DataType::kFLOAT == inputs[1].type);
        }
        auto const& inDims0 = inputs[0].dims;
        auto const& inDims1 = inputs[1].dims;
        PLUGIN_VALIDATE(inDims0.nbDims == inDims1.nbDims);

        PLUGIN_VALIDATE(std::equal(inDims0.d, inDims0.d + inDims0.nbDims, inDims1.d));

        mCfgType = inputs[0].type == DataType::kINT8 ? DataType::kHALF : inputs[0].type;

        mParamWordsize = getElementSize(mCfgType);

        return pluginStatus_t::STATUS_SUCCESS;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return pluginStatus_t::STATUS_FAILURE;
}

IPluginV3* SkipLayerNormVarSeqlenPluginV3::attachToContext(IPluginResourceContext* context) noexcept
{
    return clone();
}

IPluginCapability* SkipLayerNormVarSeqlenPluginV3::getCapabilityInterface(PluginCapabilityType type) noexcept
{
    try
    {
        if (type == PluginCapabilityType::kBUILD)
        {
            return static_cast<IPluginV3OneBuild*>(this);
        }
        if (type == PluginCapabilityType::kRUNTIME)
        {
            return static_cast<IPluginV3OneRuntime*>(this);
        }
        PLUGIN_ASSERT(type == PluginCapabilityType::kCORE);
        return static_cast<IPluginV3OneCore*>(this);
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

////////////////////////// SkipLayerNormVarSeqlenPluginV3Creator ///////////////////////////////

SkipLayerNormVarSeqlenPluginV3Creator::SkipLayerNormVarSeqlenPluginV3Creator()
{
    static std::mutex sMutex;
    std::lock_guard<std::mutex> guard(sMutex);
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("type_id"));
    mPluginAttributes.emplace_back(PluginField("beta"));
    mPluginAttributes.emplace_back(PluginField("gamma"));
    mPluginAttributes.emplace_back(PluginField("bias"));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

char const* SkipLayerNormVarSeqlenPluginV3Creator::getPluginName() const noexcept
{
    return kSKIP_LAYER_NORM_NAME;
}

char const* SkipLayerNormVarSeqlenPluginV3Creator::getPluginVersion() const noexcept
{
    return kSKIP_LAYER_NORM_VAR_SEQLEN_VERSION;
}

PluginFieldCollection const* SkipLayerNormVarSeqlenPluginV3Creator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV3* SkipLayerNormVarSeqlenPluginV3Creator::createPlugin(
    char const* name, PluginFieldCollection const* fc, TensorRTPhase phase) noexcept
{
    try
    {
        BERT_DEBUG_MSG("SkipLayerNormVarSeqlenPluginV3Creator createPlugin");

        Weights beta{DataType::kFLOAT, nullptr, 0};
        Weights gamma{DataType::kFLOAT, nullptr, 0};
        Weights bias{DataType::kFLOAT, nullptr, 0};
        int32_t typeId = -1;

        PLUGIN_VALIDATE(fc != nullptr);

        plugin::validateRequiredAttributesExist({"type_id", "beta", "gamma"}, fc);

        for (int32_t i = 0; i < fc->nbFields; i++)
        {
            std::string fieldName(fc->fields[i].name);
            if (fieldName == "type_id")
            {
                typeId = *static_cast<int32_t const*>(fc->fields[i].data);
                BERT_DEBUG_VALUE("Building typeId: ", typeId);
            }
            // process the weight tensors beta, gamma, bias
            else if (fieldName == "beta" || fieldName == "gamma" || fieldName == "bias")
            {
                Weights* weightPtr = (fieldName == "beta") ? &beta : (fieldName == "gamma") ? &gamma : &bias;

                BERT_DEBUG_MSG(("Building " + fieldName + "...").c_str());
                weightPtr->type = fieldTypeToDataType(fc->fields[i].type);
                weightPtr->values = fc->fields[i].data;
                weightPtr->count = fc->fields[i].length;
            }
        }
        BERT_DEBUG_VALUE("Type ", typeId);

        PLUGIN_VALIDATE(
            typeId >= 0 && typeId <= 3, ("SkipLayerNorm: Invalid type ID: " + std::to_string(typeId)).c_str());

        PLUGIN_VALIDATE(beta.values != nullptr, "SkipLayerNorm: invalid beta");
        PLUGIN_VALIDATE(beta.count > 0, "SkipLayerNorm: invalid beta");

        PLUGIN_VALIDATE(gamma.values != nullptr, "SkipLayerNorm: invalid gamma");
        PLUGIN_VALIDATE(gamma.count > 0, "SkipLayerNorm: invalid gamma");

        return new SkipLayerNormVarSeqlenPluginV3(name, static_cast<DataType>(typeId), beta, gamma, bias);
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

void SkipLayerNormVarSeqlenPluginV3Creator::setPluginNamespace(char const* libNamespace) noexcept
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

char const* SkipLayerNormVarSeqlenPluginV3Creator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

#endif // CUDA_VERSION >= 10010
