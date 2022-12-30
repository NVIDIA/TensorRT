/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "qkvToContextWithPosBiasPlugin.h"
#include "NvInfer.h"
#include "common/bertCommon.h"
#include "common/serialize.hpp"
#include "swinQKVToContextPlugin/fmha_with_position_bias/fmha_with_position_bias_v2.h"
#include <cstdint>
#include <cstring>
#include <cuda.h>
#include <iostream>
#include <tuple>
#include <vector>

namespace nvinfer1
{
namespace plugin
{

namespace
{
char const* kQKV_TO_CONTEXT_WITH_POS_BIAS_PLUGIN_VERSION{"1"};
char const* kQKV_TO_CONTEXT_WITH_POS_BIAS_PLUGIN_NAME{"CustomQKVToContextWithPosBiasPluginDynamic"};
} // namespace

constexpr int32_t kIIDX = 0; // index of the input tensor
constexpr int32_t kMIDX = 1; // index of the mask

// Static class fields initialization
PluginFieldCollection QKVToContextWithPosBiasPluginCreator::mFC{};
std::vector<PluginField> QKVToContextWithPosBiasPluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(QKVToContextWithPosBiasPluginCreator);

QKVToContextWithPosBiasPlugin::QKVToContextWithPosBiasPlugin(std::string name, nvinfer1::DataType type,
    int32_t hiddenSize, int32_t numHeads, int32_t hasMask, float qkvScale, float dqProbs)
    : mLayerName(name)
    , mType(type)
    , mS(0)
    , mB(0)
    , mW(0)
    , mSM(bert::getSMVersion())
    , mHiddenSize(hiddenSize)
    , mNumHeads(numHeads)
    , mHasMask(hasMask)
    , mQKVScale(qkvScale)
    , mDqProbs(dqProbs)
    , mHdim(HDIM)
{
    PLUGIN_VALIDATE(numHeads != 0);
    mHeadSize = hiddenSize / numHeads;
}

QKVToContextWithPosBiasPlugin::QKVToContextWithPosBiasPlugin(std::string const& name, void const* data, size_t length)
    : mLayerName(name)
{
    PLUGIN_VALIDATE(data != nullptr);
    BERT_DEBUG_MSG("QKVToContextWithPosBiasPlugin: Deserialization start");
    deserialize_value(&data, &length, &mType);
    deserialize_value(&data, &length, &mNumHeads);
    deserialize_value(&data, &length, &mHeadSize);
    deserialize_value(&data, &length, &mHiddenSize);
    deserialize_value(&data, &length, &mSM);
    deserialize_value(&data, &length, &mS);
    deserialize_value(&data, &length, &mB);
    deserialize_value(&data, &length, &mW);
    deserialize_value(&data, &length, &mHasMask);
    deserialize_value(&data, &length, &mQKVScale);

    deserialize_value(&data, &length, &mDqProbs);

    deserialize_value(&data, &length, &mHdim);

    createMHARunner();
    PLUGIN_VALIDATE(dispatcher != nullptr);
    dispatcher->deserialize(data, length);

    BERT_DEBUG_MSG("QKVToContextWithPosBiasPlugin: Deserialization done");
}

void QKVToContextWithPosBiasPlugin::createMHARunner()
{
    if (dispatcher.get())
    {
        return;
    }

    if ((mSM == kSM_86 || mSM == kSM_80 || mSM == kSM_75) && (mHeadSize == 32))
    {
        int32_t headSize = mHeadSize;

        if (mType == nvinfer1::DataType::kHALF)
        {
            dispatcher.reset(new FusedMHARunnerFP16v2(mNumHeads, headSize, mSM, mHasMask, mQKVScale, 1.0F));
        }
        else if (mType == nvinfer1::DataType::kINT8)
        {
            dispatcher.reset(new FusedMHARunnerInt8v2(mNumHeads, headSize, mSM, mHasMask, mQKVScale, mDqProbs));
        }
        else
        {
            PLUGIN_ERROR("Conditions `mType == kINT8 || mType == kHALF` is not met");
        }
    }
    else
    {
        PLUGIN_ERROR(
            "Conditions `(mSM == kSM_86 || mSM == kSM_80 || mSM == kSM_75) && (mHeadSize == 32)` for support not "
            "fulfilled");
    }
}

// IPluginV2DynamicExt Methods
nvinfer1::IPluginV2DynamicExt* QKVToContextWithPosBiasPlugin::clone() const noexcept
{
    try
    {
        BERT_DEBUG_MSG("QKVToContextWithPosBiasPlugin: Clone");

        QKVToContextWithPosBiasPlugin* ret = nullptr;
        if (dispatcher.get())
        {
            std::vector<char> buff;
            buff.resize(getSerializationSize());
            serialize(buff.data());

            ret = new QKVToContextWithPosBiasPlugin(mLayerName, buff.data(), buff.size());
        }
        else
        {
            ret = new QKVToContextWithPosBiasPlugin(
                mLayerName, mType, mHiddenSize, mNumHeads, mHasMask, mQKVScale, mDqProbs);
        }

        ret->setPluginNamespace(mNamespace.c_str());
        BERT_DEBUG_MSG("QKVToContextWithPosBiasPlugin: Clone done");
        return ret;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

DimsExprs QKVToContextWithPosBiasPlugin::getOutputDimensions(
    int32_t outputIndex, DimsExprs const* inputs, int32_t nbInputs, IExprBuilder& exprBuilder) noexcept
{
    try
    {
        // Input is BxSx3*N*H, output should be BxSxN*H
        PLUGIN_VALIDATE(outputIndex == 0);
        PLUGIN_VALIDATE(inputs != nullptr);
        PLUGIN_VALIDATE(nbInputs == 3);

        // Copy over everything
        PLUGIN_VALIDATE(kIIDX < nbInputs);
        DimsExprs output(inputs[kIIDX]);

        // Divide last dim by three
        auto const* three = exprBuilder.constant(3);
        PLUGIN_VALIDATE(three != nullptr);
        PLUGIN_VALIDATE(mHdim < output.nbDims);
        PLUGIN_VALIDATE(mHdim < inputs[kIIDX].nbDims);
        PLUGIN_VALIDATE(inputs[kIIDX].d[mHdim] != nullptr);

        output.d[mHdim] = exprBuilder.operation(DimensionOperation::kFLOOR_DIV, *inputs[kIIDX].d[mHdim], *three);
        return output;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }

    return DimsExprs{};
}

bool QKVToContextWithPosBiasPlugin::supportsFormatCombination(
    int32_t pos, PluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    try
    {
        // we only support int8 IO in fused mha runner, and we only support fused mha runner on Turing and Ampere
        if (mType == nvinfer1::DataType::kINT8 && mSM != kSM_86 && mSM != kSM_80 && mSM != kSM_75)
        {
            BERT_DEBUG_VALUE("INT8 IO is only supported on Turing and Ampere for plugin ",
                kQKV_TO_CONTEXT_WITH_POS_BIAS_PLUGIN_NAME);
            return false;
        }

        PLUGIN_VALIDATE(pos >= 0);
        PLUGIN_VALIDATE(pos < 4);
        PLUGIN_VALIDATE(nbInputs == 3); // input, attention mask, position bias
        PLUGIN_VALIDATE(nbOutputs == 1);
        auto const* in = inOut;
        auto const* out = inOut + nbInputs;
        PLUGIN_VALIDATE((mType == nvinfer1::DataType::kHALF || mType == nvinfer1::DataType::kINT8),
            "Conditions for support not fulfilled");

        PLUGIN_VALIDATE(in != nullptr);
        PLUGIN_VALIDATE(out != nullptr);
        auto const inDims = in->dims;
        auto const outDims = out->dims;
        PLUGIN_VALIDATE(mHdim < inDims.nbDims);
        PLUGIN_VALIDATE(mHdim < outDims.nbDims);

        auto supportedFormat = TensorFormat::kLINEAR;
        if (mType == nvinfer1::DataType::kINT8)
        {
            supportedFormat = (inDims.d[mHdim] % 32U == 0) ? TensorFormat::kCHW32 : TensorFormat::kCHW4;
        }

        //[B, S, 3*hidden_dim, 1, 1] or [B, S, 3*hidden_dim]
        bool supportedHdim = (pos == 0) ? (inDims.d[mHdim] % 3U == 0) : (inDims.d[mHdim] / 3 == outDims.d[mHdim]);

        auto const& desc = inOut[pos];
        if (pos == 0 || pos == nbInputs)
        {                                           // check input and output
            PLUGIN_VALIDATE(desc.dims.nbDims == 5 || desc.dims.nbDims == 3);
            return (desc.type == mType) &&          // check type
                (desc.format == supportedFormat) && // check format
                (supportedHdim);                    // hidden dims multiple of 3 for qkv
        }

        if (pos == 1) // attention mask: input shape [nW, max_len*max_len];
        {
            PLUGIN_VALIDATE(desc.dims.nbDims == 2);
            return (desc.type == nvinfer1::DataType::kHALF) && // type
                (desc.format == TensorFormat::kLINEAR);        // format
        }

        if (pos == 2) // position bias: input shape [nH, max_len*max_len]
        {
            PLUGIN_VALIDATE(desc.dims.nbDims == 2);
            return (desc.type == nvinfer1::DataType::kHALF) && // type
                (desc.format == TensorFormat::kLINEAR) &&      // format
                (desc.dims.d[0] == mNumHeads);                 // head num check
        }

        return false;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return false;
}

void QKVToContextWithPosBiasPlugin::configurePlugin(
    DynamicPluginTensorDesc const* in, int32_t nbInputs, DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
    try
    {
        PLUGIN_VALIDATE(nbInputs == 3);
        PLUGIN_VALIDATE(nbOutputs == 1);
        PLUGIN_VALIDATE(in != nullptr);
        PLUGIN_VALIDATE(out != nullptr);
        PLUGIN_VALIDATE(kIIDX < nbInputs);
        PLUGIN_VALIDATE(kMIDX < nbInputs);
        PluginTensorDesc const& inDesc = in[kIIDX].desc;
        TRT_UNUSED inDesc;
        PluginTensorDesc const& outDesc = out->desc;
        TRT_UNUSED outDesc;
        PLUGIN_VALIDATE(mType == inDesc.type);
        PLUGIN_VALIDATE(mType == outDesc.type);
        PluginTensorDesc const& maskDesc = in[kMIDX].desc;
        PLUGIN_VALIDATE(0 < maskDesc.dims.nbDims);
        TRT_UNUSED outDesc;
        auto windowNum = maskDesc.dims.d[0];
        BERT_DEBUG_MSG("setting up MHA runner for variable sequence length");
        createMHARunner();
        // need to initialize S and B with somewhat useful values, they will be reset at enqueue for the actual
        // batchsize
        PLUGIN_VALIDATE(dispatcher != nullptr);
        dispatcher->setup(256, 1, windowNum);
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
}

size_t QKVToContextWithPosBiasPlugin::getWorkspaceSize(
    PluginTensorDesc const* inputs, int32_t nbInputs, PluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept
{
    try
    {
        PLUGIN_VALIDATE(dispatcher != nullptr);
        return dispatcher->getWorkspaceSize();
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return 0;
}

// IPluginV2Ext Methods
nvinfer1::DataType QKVToContextWithPosBiasPlugin::getOutputDataType(
    int32_t index, nvinfer1::DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    try
    {
        PLUGIN_VALIDATE(index == 0);
        PLUGIN_VALIDATE(inputTypes != nullptr);
        PLUGIN_VALIDATE(nbInputs == 3);
        return inputTypes[0];
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nvinfer1::DataType{};
}

// IPluginV2 Methods
char const* QKVToContextWithPosBiasPlugin::getPluginType() const noexcept
{
    return kQKV_TO_CONTEXT_WITH_POS_BIAS_PLUGIN_NAME;
}

char const* QKVToContextWithPosBiasPlugin::getPluginVersion() const noexcept
{
    return kQKV_TO_CONTEXT_WITH_POS_BIAS_PLUGIN_VERSION;
}

int32_t QKVToContextWithPosBiasPlugin::getNbOutputs() const noexcept
{
    return 1;
}

int32_t QKVToContextWithPosBiasPlugin::initialize() noexcept
{
    return 0;
}

void QKVToContextWithPosBiasPlugin::terminate() noexcept {}

size_t QKVToContextWithPosBiasPlugin::getSerializationSize() const noexcept
{
    try
    {
        PLUGIN_VALIDATE(dispatcher != nullptr);
        return sizeof(mNumHeads) + sizeof(mHeadSize) + sizeof(DataType) + sizeof(mHiddenSize) + sizeof(mSM) + sizeof(mS)
            + sizeof(mB) + sizeof(mW) + sizeof(mHasMask) + sizeof(mQKVScale) + sizeof(mDqProbs)
            + dispatcher->getSerializationSize() + sizeof(mHdim);
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return 0;
}

void QKVToContextWithPosBiasPlugin::serialize(void* buffer) const noexcept
{
    try
    {
        PLUGIN_VALIDATE(buffer != nullptr);
        PLUGIN_VALIDATE(dispatcher != nullptr);
        serialize_value(&buffer, mType);
        serialize_value(&buffer, mNumHeads);
        serialize_value(&buffer, mHeadSize);
        serialize_value(&buffer, mHiddenSize);
        serialize_value(&buffer, mSM);
        serialize_value(&buffer, mS);
        serialize_value(&buffer, mB);
        serialize_value(&buffer, mW);
        serialize_value(&buffer, mHasMask);
        serialize_value(&buffer, mQKVScale);

        serialize_value(&buffer, mDqProbs);
        serialize_value(&buffer, mHdim);
        dispatcher->serialize(buffer);
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
}

void QKVToContextWithPosBiasPlugin::destroy() noexcept
{
    delete this;
}

void QKVToContextWithPosBiasPlugin::setPluginNamespace(char const* libNamespace) noexcept
{
    try
    {
        PLUGIN_VALIDATE(libNamespace != nullptr);
        mNamespace = libNamespace;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
}

char const* QKVToContextWithPosBiasPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

int32_t QKVToContextWithPosBiasPlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
    nvinfer1::PluginTensorDesc const* outputDesc, void const* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream) noexcept
{

    try
    {
        PLUGIN_VALIDATE(inputDesc != nullptr);
        PLUGIN_VALIDATE(outputDesc != nullptr);
        PLUGIN_VALIDATE(inputs != nullptr);
        PLUGIN_VALIDATE(outputs != nullptr);
        PLUGIN_VALIDATE(dispatcher != nullptr);
        int32_t const windowNum = inputDesc[1].dims.d[0]; // attention mask is [window_num, max_seqlen*max_seqlen]
        int32_t const B
            = inputDesc[0].dims.d[0]; // qkv is [batch_size*window_num, window_len, head_num*3*size_per_head]
        int32_t const maxSeqLen = inputDesc[0].dims.d[1];

        int32_t S = dispatcher->getSFromMaxSeqLen(maxSeqLen);
        PLUGIN_VALIDATE((dispatcher->isValid(S)),
            "No implementation for variable sequence length multi-head attention plugin with sequence > 256.");
        // transforms mask & position bias
        dispatcher->setup(S, B, windowNum);
        dispatcher->run(inputDesc, outputDesc, inputs, outputs, workspace, stream);

        return cudaGetLastError();
    }
    catch (std::exception const& e)
    {
        caughtError(e);
        return -1;
    }
}

QKVToContextWithPosBiasPluginCreator::QKVToContextWithPosBiasPluginCreator()
{
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("type_id", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("hidden_size", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("num_heads", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("has_mask", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("qkv_scale", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("dq_probs", nullptr, PluginFieldType::kFLOAT32, 1));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

char const* QKVToContextWithPosBiasPluginCreator::getPluginName() const noexcept
{
    return kQKV_TO_CONTEXT_WITH_POS_BIAS_PLUGIN_NAME;
}

char const* QKVToContextWithPosBiasPluginCreator::getPluginVersion() const noexcept
{
    return kQKV_TO_CONTEXT_WITH_POS_BIAS_PLUGIN_VERSION;
}

PluginFieldCollection const* QKVToContextWithPosBiasPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* QKVToContextWithPosBiasPluginCreator::createPlugin(
    char const* name, PluginFieldCollection const* fc) noexcept
{
    try
    {
        PLUGIN_VALIDATE(name != nullptr);
        PLUGIN_VALIDATE(fc != nullptr);
        BERT_DEBUG_MSG("Creating QKV2ContextPlugin...");

        int32_t hiddenSize = 0;
        int32_t numHeads = 0;
        int32_t typeId = -1;
        int32_t hasMask = 0;

        float dqProbs = -1.F;
        float qkvScale = -1.F;

        PLUGIN_VALIDATE(fc->fields != nullptr);
        validateRequiredAttributesExist({"type_id", "hidden_size", "num_heads", "has_mask"}, fc);

        for (int32_t i = 0; i < fc->nbFields; i++)
        {
            std::string fieldName(fc->fields[i].name);

            if (fieldName.compare("type_id") == 0)
            {
                typeId = *static_cast<int32_t const*>(fc->fields[i].data);
                BERT_DEBUG_VALUE("Building typeId: ", typeId);
            }
            if (fieldName.compare("hidden_size") == 0)
            {
                hiddenSize = *static_cast<int32_t const*>(fc->fields[i].data);
                BERT_DEBUG_VALUE("Building hiddenSize: ", hiddenSize);
            }
            if (fieldName.compare("num_heads") == 0)
            {
                numHeads = *static_cast<int32_t const*>(fc->fields[i].data);
                BERT_DEBUG_VALUE("Building numHeads: ", numHeads);
            }

            if (fieldName.compare("has_mask") == 0)
            {
                hasMask = *static_cast<int32_t const*>(fc->fields[i].data);
                BERT_DEBUG_VALUE("Building hasMask: ", hasMask);
            }

            if (fieldName.compare("qkv_scale") == 0)
            {
                qkvScale = *static_cast<float const*>(fc->fields[i].data);
                BERT_DEBUG_VALUE("Building qkv_scale: ", qkvScale);
            }

            if (fieldName.compare("dq_probs") == 0)
            {
                dqProbs = *static_cast<float const*>(fc->fields[i].data);
                BERT_DEBUG_VALUE("Building dqProbs: ", dqProbs);
            }
        }
        if (qkvScale == -1.0F)
        {
            qkvScale = 1.F / sqrt(float(hiddenSize) / numHeads);
        }

        if (qkvScale <= 0)
        {
            gLogError << "QKV: Invalid qkvScale " << qkvScale << std::endl;
            return nullptr;
        }

        if (hasMask != 0 && hasMask != 1)
        {
            gLogError << "QKV: Invalid hasMask " << hasMask << std::endl;
            return nullptr;
        }

        if (typeId < 0 || typeId > 3)
        {
            gLogError << "QKV: Invalid TypeId " << typeId << std::endl;
            return nullptr;
        }

        if (hiddenSize <= 0)
        {
            gLogError << "QKV: Invalid hiddenSize " << hiddenSize << std::endl;
            return nullptr;
        }

        if (numHeads <= 0)
        {
            gLogError << "QKV: Invalid numHeads " << numHeads << std::endl;
            return nullptr;
        }

        BERT_DEBUG_MSG("Building the Plugin...");
        nvinfer1::DataType type = static_cast<nvinfer1::DataType>(typeId);
        if (type == nvinfer1::DataType::kINT8 && dqProbs < 0)
        {
            gLogInfo << "Using default scale factor\n";
            dqProbs = 1.F / 127.F;
        }

        if (type == nvinfer1::DataType::kINT8 && dqProbs <= 0)
        {
            gLogError << "QKV: Invalid dqProbs " << dqProbs << std::endl;
            return nullptr;
        }

        return new QKVToContextWithPosBiasPlugin(name, type, hiddenSize, numHeads, hasMask, qkvScale, dqProbs);
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* QKVToContextWithPosBiasPluginCreator::deserializePlugin(
    char const* name, void const* serialData, size_t serialLength) noexcept
{
    try
    {
        PLUGIN_VALIDATE(name != nullptr);
        PLUGIN_VALIDATE(serialData != nullptr);
        // This object will be deleted when the network is destroyed, which will
        // call QKVToContextWithPosBiasPlugin::destroy()
        return new QKVToContextWithPosBiasPlugin(name, serialData, serialLength);
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

void QKVToContextWithPosBiasPluginCreator::setPluginNamespace(char const* libNamespace) noexcept
{
    try
    {
        PLUGIN_VALIDATE(libNamespace != nullptr);
        mNamespace = libNamespace;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
}

char const* QKVToContextWithPosBiasPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

} // namespace plugin
} // namespace nvinfer1
