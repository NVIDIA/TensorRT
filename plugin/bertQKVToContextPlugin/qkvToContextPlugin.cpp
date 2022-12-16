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

// Need 10.1 for cublasGemmStridedBatchedEx
#include <cuda.h>
#if CUDA_VERSION >= 10010

#include "NvInfer.h"
#include "bertQKVToContextPlugin/fused_multihead_attention/include/fused_multihead_attention.h"
#include "bertQKVToContextPlugin/fused_multihead_attention_v2/include/fused_multihead_attention_v2.h"
#include "common/bertCommon.h"
#include "common/serialize.hpp"
#include "qkvToContextPlugin.h"

#include <cstdint>
#include <cstring>
#include <iostream>
#include <tuple>
#include <vector>

using namespace nvinfer1;
using namespace nvinfer1::plugin;
using namespace nvinfer1::plugin::bert;

namespace
{
const char* QKV_TO_CONTEXT_PLUGIN_VERSION{"1"};
const char* QKV_TO_CONTEXT_VAR_SEQLEN_PLUGIN_VERSION{"2"};
const char* QKV_TO_CONTEXT_PLUGIN_NAME{"CustomQKVToContextPluginDynamic"};
} // namespace

// Static class fields initialization
PluginFieldCollection QKVToContextPluginDynamicCreator::mFC{};
std::vector<PluginField> QKVToContextPluginDynamicCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(QKVToContextPluginDynamicCreator);

constexpr uint32_t IIDX = 0; // index of the input tensor
constexpr uint32_t MIDX = 1; // index of the mask

// Static class fields initialization
PluginFieldCollection QKVToContextVarSeqlenPluginCreator::mFC{};
std::vector<PluginField> QKVToContextVarSeqlenPluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(QKVToContextVarSeqlenPluginCreator);

QKVToContextPluginDynamic::QKVToContextPluginDynamic(const std::string name, const DataType type,
    const int32_t hiddenSize, const int32_t numHeads, const float dqProbs, bool hasImask)
    : mLayerName(name)
    , mS(0)
    , mB(0)
    , mHeadSize(hiddenSize / numHeads)
    , mHiddenSize(hiddenSize)
    , mNumHeads(numHeads)
    , mHasImask(hasImask)
    , mType(type)
    , mDqProbs(dqProbs)

{
    mSM = getSMVersion();
}

QKVToContextPluginDynamic::QKVToContextPluginDynamic(const std::string name, const void* data, size_t length)
    : mLayerName(name)
{
    BERT_DEBUG_MSG("QKV Deser Start");
    deserialize_value(&data, &length, &mType);
    deserialize_value(&data, &length, &mNumHeads);
    deserialize_value(&data, &length, &mHeadSize);
    deserialize_value(&data, &length, &mHasImask);
    deserialize_value(&data, &length, &mHiddenSize);
    deserialize_value(&data, &length, &mSM);
    deserialize_value(&data, &length, &mS);
    deserialize_value(&data, &length, &mB);

    deserialize_value(&data, &length, &mDqProbs);

    createMHARunner();

    int32_t hasUnfusedRunner = 0;
    deserialize_value(&data, &length, &hasUnfusedRunner);
    if (hasUnfusedRunner)
    {
        PLUGIN_ASSERT(unfusedDispatcher.get());
        unfusedDispatcher->deserialize(data, length);
    }

    BERT_DEBUG_MSG("QKV Deser done");
}

void QKVToContextPluginDynamic::createMHARunner()
{
    if (!fusedDispatcher.get())
    {
        if (mType == DataType::kHALF)
        {
            fusedDispatcher.reset(new FusedMHARunnerFP16(mNumHeads, mHeadSize, mSM));
        }
        else if (mType == DataType::kINT8)
        {
            fusedDispatcher.reset(new FusedMHARunnerInt8(mNumHeads, mHeadSize, mSM, mDqProbs));
        }
    }

    if (!unfusedDispatcher.get())
    {
        unfusedDispatcher.reset(new UnfusedMHARunner(mType, mNumHeads, mHeadSize, mSM));
    }
}

// IPluginV2DynamicExt Methods
nvinfer1::IPluginV2DynamicExt* QKVToContextPluginDynamic::clone() const noexcept
{
    BERT_DEBUG_MSG("QKV Clone");

    QKVToContextPluginDynamic* ret = nullptr;
    // the workspacesize is 0 if we have not call setup the dispatcher yet.
    if (unfusedDispatcher.get() && unfusedDispatcher->getWorkspaceSize())
    {
        std::vector<char> buff;
        buff.resize(getSerializationSize());
        serialize(buff.data());

        ret = new QKVToContextPluginDynamic(mLayerName, buff.data(), buff.size());
    }
    else
    {
        ret = new QKVToContextPluginDynamic(mLayerName, mType, mHiddenSize, mNumHeads, mDqProbs, mHasImask);
    }

    ret->setPluginNamespace(mNamespace.c_str());
    BERT_DEBUG_MSG("QKV Clone done");
    return ret;
}

DimsExprs QKVToContextPluginDynamic::getOutputDimensions(
    int32_t outputIndex, const DimsExprs* inputs, int32_t /*nbInputs*/, IExprBuilder& exprBuilder) noexcept
{
    // Input is BxSx3*N*H, output should be BxSxN*H
    PLUGIN_ASSERT(outputIndex == 0);
    // Copy over everything
    DimsExprs output(inputs[IIDX]);
    // Divide last dim by three
    const auto* three = exprBuilder.constant(3);
    output.d[HDIM] = exprBuilder.operation(DimensionOperation::kFLOOR_DIV, *inputs[IIDX].d[HDIM], *three);
    return output;
}
bool QKVToContextPluginDynamic::supportsFormatCombination(
    int32_t pos, const PluginTensorDesc* inOut, int32_t nbInputs, int32_t /*nbOutputs*/) noexcept
{
    PLUGIN_ASSERT(pos >= 0);
    PLUGIN_ASSERT(pos < 2 + mHasImask);
    PLUGIN_ASSERT(nbInputs == 1 + mHasImask);
    const auto* in = inOut;
    const auto* out = inOut + nbInputs;
    int32_t packedSize = getMHAMaskPackedSize(mSM, mType, in->dims.d[SDIM]);

    // we only support int8 IO in fused mha runner, and we only support fused mha runner on Xavier, Turing and Ampere
    if (mType == DataType::kINT8)
    {
        if (mSM != kSM_75 && mSM != kSM_80 && mSM != kSM_86 && mSM != kSM_87 && mSM != kSM_89 && mSM != kSM_90)
        {
            gLogError << "INT8 IO is only supported on Turing, Ampere and Hopper for plugin " << QKV_TO_CONTEXT_PLUGIN_NAME
                      << std::endl;
            return false;
        }
        if (in->dims.d[SDIM] == -1)
        {
            gLogError << "INT8 IO not support dynamic shape in sequence dimension for plugin "
                      << QKV_TO_CONTEXT_PLUGIN_NAME << std::endl;
            return false;
        }
        if (packedSize == unfusedMaskSize)
        {
            gLogError << "INT8 IO only support sequence length 128,384 for plugin " << QKV_TO_CONTEXT_PLUGIN_NAME
                      << std::endl;
            return false;
        }
    }
    if (mType == DataType::kHALF)
    {
        if (mSM < kSM_53)
        {
            gLogError
                << "Half-precision floating-point is only supported on compute capability 5.3 and later for plugin "
                << QKV_TO_CONTEXT_PLUGIN_NAME << std::endl;
            return false;
        }
    }

    if (pos == 0)
    {
        bool isFormatSupported = in->format == TensorFormat::kLINEAR;
        if (mType == DataType::kINT8)
        {
            if (in->dims.d[HDIM] % 32U == 0)
            {
                isFormatSupported = in->format == TensorFormat::kCHW32;
            }
            else
            {
                isFormatSupported = in->format == TensorFormat::kCHW4;
            }
        }

        // must not check descriptions > pos
        return (in->type == mType) &&         // precision
            isFormatSupported &&              // format
            (in->dims.nbDims == 5) &&         // num dims
            ((in->dims.d[HDIM] % 3U) == 0) && // see getOutputDimensions
            ((in->dims.d[3]) == 1) &&         // for fc
            ((in->dims.d[4]) == 1)            // for fc
            ;
    }

    // pos==1
    if ((mHasImask && pos == 1)) // pos 1 is the mask
    {
        auto const* inMask = &inOut[1];
        if (inMask->dims.d[1] != -1 && inMask->dims.d[1] != packedSize)
        {
            gLogError << "CustomEmbLayerNormPluginDynamic returned mask with pack size " << inMask->dims.d[1]
                      << ", but " << QKV_TO_CONTEXT_PLUGIN_NAME << " expects mask pack size " << packedSize
                      << std::endl;
            return false;
        }

        // detect full mask and check that it was produced
        return (inMask->type == DataType::kFLOAT) &&     // precision
            (inMask->format == TensorFormat::kLINEAR) && // format
            (inMask->dims.nbDims == 2) &&                // Bx2*maskSize
            (inMask->dims.d[0] == in->dims.d[BDIM]);
    }

    if (!mHasImask || pos == 2) // output pos
    {
        bool isFormatSupported = out->format == TensorFormat::kLINEAR;
        if (mType == DataType::kINT8)
        {
            if (out->dims.d[HDIM] % 32U == 0)
            {
                isFormatSupported = out->format == TensorFormat::kCHW32;
            }
            else
            {
                isFormatSupported = out->format == TensorFormat::kCHW4;
            }
        }

        return (in->type == out->type) &&                      // precision
            isFormatSupported &&                               // format
            (out->dims.nbDims == 5) &&                         // num dims
            ((in->dims.d[HDIM] / 3) == (out->dims.d[HDIM])) && // div 3
            ((out->dims.d[3]) == 1) &&                         // for fc
            ((out->dims.d[4]) == 1) &&                         // for fc
            ((out->dims.d[BDIM]) == in->dims.d[BDIM]) &&       // check B
            ((out->dims.d[SDIM]) == in->dims.d[SDIM])          // check S
            ;
    }

    return false;
}
void QKVToContextPluginDynamic::configurePlugin(
    const DynamicPluginTensorDesc* in, int32_t nbInputs, const DynamicPluginTensorDesc* out, int32_t nbOutputs) noexcept
{
    PLUGIN_ASSERT(nbInputs == 1 + mHasImask);
    PLUGIN_ASSERT(nbOutputs == 1);
    const PluginTensorDesc& inDesc = in[IIDX].desc;
    TRT_UNUSED inDesc;
    const PluginTensorDesc& outDesc = out->desc;
    TRT_UNUSED outDesc;
    PLUGIN_ASSERT(mType == inDesc.type);
    PLUGIN_ASSERT(mType == outDesc.type);
    PLUGIN_ASSERT(inDesc.dims.d[BDIM] == outDesc.dims.d[BDIM]);
    PLUGIN_ASSERT(inDesc.dims.d[SDIM] == outDesc.dims.d[SDIM]);
    PLUGIN_ASSERT(inDesc.dims.d[HDIM] == 3 * outDesc.dims.d[HDIM]);
    if (mHasImask)
    {
        const PluginTensorDesc& maskDesc = in[MIDX].desc;
        TRT_UNUSED maskDesc;
        PLUGIN_ASSERT(maskDesc.dims.d[0] == inDesc.dims.d[BDIM]);
    }

    createMHARunner();

    const int32_t S = inDesc.dims.d[SDIM];
    const int32_t B = inDesc.dims.d[BDIM] <= 0 ? in->max.d[BDIM] : inDesc.dims.d[BDIM];
    if (S <= 0)
    {
        // in dynamic shape build stage, we setup with max sequence that cannot fused
        const int32_t Smin = in->min.d[SDIM];
        const int32_t Smax = in->max.d[SDIM];

        if (fusedDispatcher.get())
        {
            for (int32_t i = Smax; i >= Smin; --i)
            {
                if (!fusedDispatcher->isValid(i))
                {
                    unfusedDispatcher->setup(i, B);
                    mS = i;
                    mB = B;
                    break;
                }
            }
        }
        else
        {
            unfusedDispatcher->setup(Smax, B);
            mS = Smax;
            mB = B;
        }
    }
    else
    {
        // in inference stage or in static shape build stage
        if (fusedDispatcher.get() && fusedDispatcher->isValid(S))
        {
            fusedDispatcher->setup(S, B);
        }
        else
        {
            unfusedDispatcher->setup(S, B);
        }
        mS = S;
        mB = B;
    }
}

size_t QKVToContextPluginDynamic::getWorkspaceSize(const PluginTensorDesc* /*inputs*/, int32_t /*nbInputs*/,
    const PluginTensorDesc* /*outputs*/, int32_t /*nbOutputs*/) const noexcept
{
    // only unfused kernel need workspace, and we need larger workspace for larger sequence length
    // we have already setup unfusedDispatcher with max sequence in configurePlugin
    // if unfusedDispatcher is not initialized in configurePlugin
    PLUGIN_ASSERT(unfusedDispatcher.get());
    return unfusedDispatcher->getWorkspaceSize();
}

// IPluginV2Ext Methods
DataType QKVToContextPluginDynamic::getOutputDataType(
    int32_t index, const nvinfer1::DataType* inputTypes, int32_t /*nbInputs*/) const noexcept
{
    PLUGIN_ASSERT(index == 0);
    PLUGIN_ASSERT(inputTypes[0] == DataType::kFLOAT || inputTypes[0] == DataType::kHALF || inputTypes[0] == DataType::kINT8);
    return inputTypes[0];
}

// IPluginV2 Methods
const char* QKVToContextPluginDynamic::getPluginType() const noexcept
{
    return QKV_TO_CONTEXT_PLUGIN_NAME;
}

const char* QKVToContextPluginDynamic::getPluginVersion() const noexcept
{
    return QKV_TO_CONTEXT_PLUGIN_VERSION;
}

int32_t QKVToContextPluginDynamic::getNbOutputs() const noexcept
{
    return 1;
}

int32_t QKVToContextPluginDynamic::initialize() noexcept
{
    return 0;
}

void QKVToContextPluginDynamic::terminate() noexcept {}

size_t QKVToContextPluginDynamic::getSerializationSize() const noexcept
{
    PLUGIN_ASSERT(unfusedDispatcher.get());
    return sizeof(mNumHeads) + sizeof(mHeadSize) + sizeof(DataType) + sizeof(mHasImask) + sizeof(mHiddenSize)
        + sizeof(mSM) + sizeof(mS) + sizeof(mB) + sizeof(mDqProbs) + sizeof(int)
        + unfusedDispatcher->getSerializationSize();
}

void QKVToContextPluginDynamic::serialize(void* buffer) const noexcept
{
    serialize_value(&buffer, mType);
    serialize_value(&buffer, mNumHeads);
    serialize_value(&buffer, mHeadSize);
    serialize_value(&buffer, mHasImask);
    serialize_value(&buffer, mHiddenSize);
    serialize_value(&buffer, mSM);
    serialize_value(&buffer, mS);
    serialize_value(&buffer, mB);

    serialize_value(&buffer, mDqProbs);
    if (unfusedDispatcher.get() && unfusedDispatcher->getWorkspaceSize())
    {
        int32_t hasUnfusedRunner = 1;
        serialize_value(&buffer, hasUnfusedRunner);
        unfusedDispatcher->serialize(buffer);
    }
    else
    {
        int32_t hasUnfusedRunner = 0;
        serialize_value(&buffer, hasUnfusedRunner);
    }
}

void QKVToContextPluginDynamic::destroy() noexcept
{
    delete this;
}

void QKVToContextPluginDynamic::setPluginNamespace(const char* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

const char* QKVToContextPluginDynamic::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

int32_t QKVToContextPluginDynamic::enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc,
    const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    PLUGIN_ASSERT(mS == inputDesc->dims.d[SDIM]);
    PLUGIN_ASSERT(mB == inputDesc->dims.d[BDIM]);

    try
    {
        void const* const maskPtr = mHasImask ? inputs[1] : nullptr;
        if (fusedDispatcher.get() && fusedDispatcher->isValid(inputDesc->dims.d[SDIM]))
        {
            fusedDispatcher->run(inputDesc[0], outputDesc[0], inputs[0], maskPtr, outputs[0], workspace, stream);
        }
        else
        {
            PLUGIN_VALIDATE(unfusedDispatcher.get(), "The Unfused MHARunner is uninitialized, no MHARunner available!");
            unfusedDispatcher->run(inputDesc[0], outputDesc[0], inputs[0], maskPtr, outputs[0], workspace, stream);
        }
    }
    catch (std::exception const& e)
    {
        caughtError(e);
        return -1;
    }
    return 0;
}

QKVToContextPluginDynamicCreator::QKVToContextPluginDynamicCreator()
{
    mPluginAttributes.emplace_back(PluginField("type_id", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("hidden_size", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("num_heads", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("has_mask", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("dq_probs", nullptr, PluginFieldType::kFLOAT32, 1));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* QKVToContextPluginDynamicCreator::getPluginName() const noexcept
{
    return QKV_TO_CONTEXT_PLUGIN_NAME;
}

const char* QKVToContextPluginDynamicCreator::getPluginVersion() const noexcept
{
    return QKV_TO_CONTEXT_PLUGIN_VERSION;
}

const PluginFieldCollection* QKVToContextPluginDynamicCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* QKVToContextPluginDynamicCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
{
    try
    {
        BERT_DEBUG_MSG("Creating QKV2ContextPlugin...");
        PLUGIN_VALIDATE(fc != nullptr);
        int32_t hiddenSize = 0;
        // Since numHeads must always exist or validateRequiredAttributes will fail,
        // we can set numHeads to -1 so that static analysis tools don't warn about
        // a division by zero in QKVToContextPluginDynamic constructor.
        int32_t numHeads{-1};
        bool hasMask = false;
        int32_t typeId = -1;

        float dqProbs = -1;

        PLUGIN_VALIDATE(fc->fields != nullptr);
        plugin::validateRequiredAttributesExist({"type_id", "hidden_size", "num_heads", "has_mask"}, fc);

        for (int32_t i = 0; i < fc->nbFields; i++)
        {
            PLUGIN_VALIDATE(fc->fields[i].name != nullptr);
            PLUGIN_VALIDATE(fc->fields[i].data != nullptr);
            std::string field_name(fc->fields[i].name);

            if (field_name.compare("type_id") == 0)
            {
                typeId = *static_cast<int32_t const*>(fc->fields[i].data);
                PLUGIN_VALIDATE(typeId >= 0 && typeId <= 2, ("QKV: Invalid TypeId " + std::to_string(typeId)).c_str());
                BERT_DEBUG_VALUE("Building typeId: ", typeId);
            }
            if (field_name.compare("hidden_size") == 0)
            {
                hiddenSize = *static_cast<int32_t const*>(fc->fields[i].data);
                PLUGIN_VALIDATE(hiddenSize > 0, ("QKV: Invalid hiddenSize " + std::to_string(hiddenSize)).c_str());
                BERT_DEBUG_VALUE("Building hiddenSize: ", hiddenSize);
            }
            if (field_name.compare("num_heads") == 0)
            {
                numHeads = *static_cast<int32_t const*>(fc->fields[i].data);
                PLUGIN_VALIDATE(numHeads > 0, ("QKV: Invalid numHeads " + std::to_string(numHeads)).c_str());
                BERT_DEBUG_VALUE("Building numHeads: ", numHeads);
            }
            if (field_name.compare("has_mask") == 0)
            {
                auto hasMaskValue = *static_cast<int32_t const*>(fc->fields[i].data);
                PLUGIN_VALIDATE(hasMaskValue == 0 || hasMaskValue == 1,
                    ("QKV: Invalid hasMask " + std::to_string(hasMaskValue)).c_str());
                hasMask = static_cast<bool>(hasMaskValue);
                BERT_DEBUG_VALUE("Building hasMask: ", hasMask);
            }

            if (field_name.compare("dq_probs") == 0)
            {
                dqProbs = *static_cast<float const*>(fc->fields[i].data);
                PLUGIN_VALIDATE(dqProbs > 0.0F, ("QKV: Invalid dqProbs " + std::to_string(dqProbs)).c_str());
                BERT_DEBUG_VALUE("Building dqProbs: ", dqProbs);
            }
        }

        BERT_DEBUG_MSG("Building the Plugin...");
        auto type = static_cast<DataType>(typeId);
        if (type == DataType::kINT8 && dqProbs < 0)
        {
            BERT_DEBUG_MSG("Using default scale factor");
            dqProbs = 1.F / 127.F;
        }

        auto* p = new QKVToContextPluginDynamic(name, type, hiddenSize, numHeads, dqProbs, hasMask);
        return p;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* QKVToContextPluginDynamicCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength)  noexcept
{
    // This object will be deleted when the network is destroyed, which will
    // call QKVToContextPluginDynamic::destroy()
    return new QKVToContextPluginDynamic(name, serialData, serialLength);
}

void QKVToContextPluginDynamicCreator::setPluginNamespace(const char* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

const char* QKVToContextPluginDynamicCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

QKVToContextVarSeqlenPlugin::QKVToContextVarSeqlenPlugin(std::string const name, DataType const type,
    int32_t const hiddenSize, int32_t const numHeads, float const dqProbs, bool hasImask, bool varSeqlen, bool useInt8ScaleMax)
    : mLayerName(name)
    , mS(0)
    , mB(0)
    , mHeadSize(hiddenSize / numHeads)
    , mHiddenSize(hiddenSize)
    , mNumHeads(numHeads)
    , mHasImask(hasImask)
    , mType(type)
    , mDqProbs(dqProbs)
    , mHdim(HDIM)
    , mUseVarSeqlen(varSeqlen)
    , mUseInt8ScaleMax(useInt8ScaleMax)
{
    mSM = getSMVersion();

    if (varSeqlen)
    {
        // variable sequence length is only supported with the fused MHA kernels
        // we should not override mS!
        PLUGIN_ASSERT((mSM == kSM_90 || mSM == kSM_87 || mSM == kSM_86 || mSM == kSM_89 || mSM == kSM_80 || mSM == kSM_75 || mSM == kSM_72)
            && (type == DataType::kINT8 || type == DataType::kHALF)
            && "requesting maxSeqlen not compatible with GPU arch");
        // the layout changes: SxB will be a combined \sum_i s_i and hdim will be the 2nd dimension instead of the third
        mHdim = 1;
    }
}

QKVToContextVarSeqlenPlugin::QKVToContextVarSeqlenPlugin(const std::string name, const void* data, size_t length)
    : mLayerName(name)
{
    BERT_DEBUG_MSG("QKV Deser Start");
    deserialize_value(&data, &length, &mType);
    deserialize_value(&data, &length, &mNumHeads);
    deserialize_value(&data, &length, &mHeadSize);
    deserialize_value(&data, &length, &mHasImask);
    deserialize_value(&data, &length, &mHiddenSize);
    deserialize_value(&data, &length, &mSM);
    deserialize_value(&data, &length, &mS);
    deserialize_value(&data, &length, &mB);

    deserialize_value(&data, &length, &mDqProbs);

    deserialize_value(&data, &length, &mUseVarSeqlen);
    deserialize_value(&data, &length, &mHdim);
    deserialize_value(&data, &length, &mUseInt8ScaleMax);

    createMHARunner();
    dispatcher->deserialize(data, length);

    BERT_DEBUG_MSG("QKV Deser done");
}

void QKVToContextVarSeqlenPlugin::createMHARunner()
{
    if (dispatcher.get())
    {
        return;
    }

    if (mSM == kSM_90 || mSM == kSM_87 || mSM == kSM_86 || mSM == kSM_89 || mSM == kSM_80 || mSM == kSM_75 || mSM == kSM_72)
    {
        int32_t headSize = mHeadSize;
        if (mHeadSize != 32 && mHeadSize != 64)
        {
            patcher.reset(new QkvPaddingRunner(mHeadSize, mType));
            headSize = patcher->getPaddingHeadSize();
        }

        if (mType == DataType::kHALF)
        {
            dispatcher.reset(new FusedMHARunnerFP16v2(mNumHeads, headSize, mSM));
        }
        else if (mType == DataType::kINT8)
        {
            dispatcher.reset(new FusedMHARunnerInt8v2(mNumHeads, headSize, mSM, mDqProbs, mUseInt8ScaleMax));
        }
    }
    else
    {
        PLUGIN_ASSERT(!mUseVarSeqlen);
        dispatcher.reset(new UnfusedMHARunner(mType, mNumHeads, mHeadSize, mSM));
    }
}

// IPluginV2DynamicExt Methods
nvinfer1::IPluginV2DynamicExt* QKVToContextVarSeqlenPlugin::clone() const noexcept
{
    BERT_DEBUG_MSG("QKV Clone");

    QKVToContextVarSeqlenPlugin* ret = nullptr;
    if (dispatcher.get())
    {
        std::vector<char> buff;
        buff.resize(getSerializationSize());
        serialize(buff.data());

        ret = new QKVToContextVarSeqlenPlugin(mLayerName, buff.data(), buff.size());
    }
    else
    {
        ret = new QKVToContextVarSeqlenPlugin(
            mLayerName, mType, mHiddenSize, mNumHeads, mDqProbs, mHasImask, mUseVarSeqlen, mUseInt8ScaleMax);
    }

    ret->setPluginNamespace(mNamespace.c_str());
    BERT_DEBUG_MSG("QKV Clone done");
    return ret;
}

DimsExprs QKVToContextVarSeqlenPlugin::getOutputDimensions(
    int32_t outputIndex, const DimsExprs* inputs, int32_t /*nbInputs*/, IExprBuilder& exprBuilder) noexcept
{
    // Input is BxSx3*N*H, output should be BxSxN*H
    PLUGIN_ASSERT(outputIndex == 0);
    // Copy over everything
    DimsExprs output(inputs[IIDX]);
    // Divide last dim by three
    const auto* three = exprBuilder.constant(3);
    output.d[mHdim] = exprBuilder.operation(DimensionOperation::kFLOOR_DIV, *inputs[IIDX].d[mHdim], *three);
    return output;
}

bool QKVToContextVarSeqlenPlugin::supportsFormatCombination(
    int32_t pos, const PluginTensorDesc* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    // we only support int8 IO in fused mha runner, and we only support fused mha runner on Turing and Ampere
    if (mType == DataType::kINT8 && mSM != kSM_90 && mSM != kSM_89 && mSM != kSM_87 && mSM != kSM_86 && mSM != kSM_80 && mSM != kSM_75 && mSM != kSM_72)
    {
        BERT_DEBUG_VALUE(
            "INT8 IO is only supported on Xavier, Turing and Ampere for plugin ", QKV_TO_CONTEXT_PLUGIN_NAME);
        return false;
    }

    PLUGIN_ASSERT(pos >= 0);
    PLUGIN_ASSERT(pos < 2 + mHasImask + 2 * mUseVarSeqlen);
    PLUGIN_ASSERT(nbInputs == 1 + mHasImask + 2 * mUseVarSeqlen);
    PLUGIN_ASSERT(nbOutputs == 1);
    const auto* in = inOut;
    const auto* out = inOut + nbInputs;
    if (mUseVarSeqlen)
    {
        PLUGIN_ASSERT((mType == DataType::kHALF || mType == DataType::kINT8)
            && "Conditions for variable seqlen support not fulfilled");
        // qkv, mask, cu_seqlens, dummy
        PLUGIN_ASSERT(nbInputs == 4 && "for varseqlen, expected 4 inputs");
    }

    const auto inDims = in->dims;
    // const auto inType = in->type;
    // const auto inFmt = in->format;
    // const auto outType = out->type;
    const auto outDims = out->dims;
    // const auto outFmt = out->format;

    auto supportedFormat = TensorFormat::kLINEAR;
    if (mType == DataType::kINT8)
    {
        supportedFormat = (inDims.d[mHdim] % 32U == 0) ? TensorFormat::kCHW32 : TensorFormat::kCHW4;
    }

    int32_t supportedNbDims = 5;
    if (mUseVarSeqlen)
    {
        supportedNbDims = 4;
    }

    bool supportedHdim = (pos == 0) ? (inDims.d[mHdim] % 3U == 0) : (inDims.d[mHdim] / 3 == outDims.d[mHdim]);

    if (pos == 0 || pos == nbInputs)
    { // check input and output
        const auto& desc = inOut[pos];
        return (desc.type == mType) &&               // check type
            (desc.format == supportedFormat) &&      // check format
            (desc.dims.nbDims == supportedNbDims) && // check dims:
            (supportedHdim) &&                       // - hidden dims multiple of 3 for qkv
            (desc.dims.d[mHdim + 1] == 1) &&         // - dummy 1 or h
            (desc.dims.d[mHdim + 2] == 1)            // - dummy 1 or w
            ;
    }

    PLUGIN_ASSERT(mHasImask);
    if (pos == 1)
    { // must be input mask
        const auto* mask = &inOut[pos];
        if (mUseVarSeqlen)
        {
            // dummy input
            return true;
        }

        return mask->format == TensorFormat::kLINEAR && (mask->type == DataType::kINT32) && // precision
            (mask->dims.nbDims == 1)                                                        // num dims
            ;
    }
    PLUGIN_ASSERT(mUseVarSeqlen);
    if (pos == 2)
    { // must be cuSeqlens
        // cuSeqlens is a int array of size B+1
        const auto* seqlens = &inOut[pos];
        return (seqlens->type == DataType::kINT32) && (seqlens->format == TensorFormat::kLINEAR);
    }
    if (pos == 3)
    {
        // this is the dummy input
        return inOut[pos].dims.nbDims == 1;
    }
    return false;
}

void QKVToContextVarSeqlenPlugin::configurePlugin(
    const DynamicPluginTensorDesc* in, int32_t nbInputs, const DynamicPluginTensorDesc* out, int32_t nbOutputs) noexcept
{
    PLUGIN_ASSERT(nbInputs == 1 + mHasImask + 2 * mUseVarSeqlen);
    PLUGIN_ASSERT(nbOutputs == 1);
    const PluginTensorDesc& inDesc = in[IIDX].desc;
    TRT_UNUSED inDesc;
    const PluginTensorDesc& outDesc = out->desc;
    TRT_UNUSED outDesc;
    PLUGIN_ASSERT(mType == inDesc.type);
    PLUGIN_ASSERT(mType == outDesc.type);
    if (!mUseVarSeqlen)
    {
        PLUGIN_ASSERT(inDesc.dims.d[BDIM] == outDesc.dims.d[BDIM]);
        PLUGIN_ASSERT(inDesc.dims.d[SDIM] == outDesc.dims.d[SDIM]);
        PLUGIN_ASSERT(inDesc.dims.d[mHdim] == 3 * outDesc.dims.d[mHdim]);
        if (mHasImask)
        {
            const PluginTensorDesc& maskDesc = in[MIDX].desc;
            TRT_UNUSED maskDesc;
            PLUGIN_ASSERT(maskDesc.dims.d[0] == inDesc.dims.d[BDIM]);
        }

        const int32_t S = inDesc.dims.d[SDIM] <= 0 ? in->max.d[SDIM] : inDesc.dims.d[SDIM];
        const int32_t B = inDesc.dims.d[BDIM] <= 0 ? in->max.d[BDIM] : inDesc.dims.d[BDIM];

        if (S != mS || B != mB)
        {
            BERT_DEBUG_MSG("setting up MHA runner for single sequence length");
            createMHARunner();
            this->dispatcher->setup(S, B);
            mS = S;
            mB = B;
        }
    }
    else
    {
        BERT_DEBUG_MSG("setting up MHA runner for variable sequence length");
        createMHARunner();
        // need to initialize S and B with somewhat useful values, they will be reset at enqueue for the actual
        // batchsize
        this->dispatcher->setup(256, 1);
    }
}

size_t QKVToContextVarSeqlenPlugin::getWorkspaceSize(
    const PluginTensorDesc* inputs, int32_t nbInputs, const PluginTensorDesc* outputs, int32_t nbOutputs) const noexcept
{
    size_t paddingWorkpaceSize = 0;
    if (patcher)
    {
        paddingWorkpaceSize = patcher->getWorkspaceSize(inputs[0].dims.d[0], mNumHeads);
    }
    return this->dispatcher->getWorkspaceSize() + paddingWorkpaceSize;
}

// IPluginV2Ext Methods
DataType QKVToContextVarSeqlenPlugin::getOutputDataType(
    int32_t index, const nvinfer1::DataType* inputTypes, int32_t /*nbInputs*/) const noexcept
{
    PLUGIN_ASSERT(index == 0);
    PLUGIN_ASSERT(inputTypes[0] == DataType::kFLOAT || inputTypes[0] == DataType::kHALF || inputTypes[0] == DataType::kINT8);
    return inputTypes[0];
}

// IPluginV2 Methods
const char* QKVToContextVarSeqlenPlugin::getPluginType() const noexcept
{
    return QKV_TO_CONTEXT_PLUGIN_NAME;
}

const char* QKVToContextVarSeqlenPlugin::getPluginVersion() const noexcept
{
    return QKV_TO_CONTEXT_VAR_SEQLEN_PLUGIN_VERSION;
}

int32_t QKVToContextVarSeqlenPlugin::getNbOutputs() const noexcept
{
    return 1;
}

int32_t QKVToContextVarSeqlenPlugin::initialize() noexcept
{
    return 0;
}

void QKVToContextVarSeqlenPlugin::terminate() noexcept {}

size_t QKVToContextVarSeqlenPlugin::getSerializationSize() const noexcept
{
    return sizeof(mNumHeads) + sizeof(mHeadSize) + sizeof(DataType) + sizeof(mHasImask) + sizeof(mHiddenSize)
        + sizeof(mSM) + sizeof(mS) + sizeof(mB) + sizeof(mDqProbs) + dispatcher->getSerializationSize()
        + sizeof(mUseVarSeqlen) + sizeof(mHdim) + sizeof(mUseInt8ScaleMax);
}

void QKVToContextVarSeqlenPlugin::serialize(void* buffer) const noexcept
{
    serialize_value(&buffer, mType);
    serialize_value(&buffer, mNumHeads);
    serialize_value(&buffer, mHeadSize);
    serialize_value(&buffer, mHasImask);
    serialize_value(&buffer, mHiddenSize);
    serialize_value(&buffer, mSM);
    serialize_value(&buffer, mS);
    serialize_value(&buffer, mB);

    serialize_value(&buffer, mDqProbs);
    serialize_value(&buffer, mUseVarSeqlen);
    serialize_value(&buffer, mHdim);
    serialize_value(&buffer, mUseInt8ScaleMax);
    dispatcher->serialize(buffer);
}

void QKVToContextVarSeqlenPlugin::destroy() noexcept
{
    delete this;
}

void QKVToContextVarSeqlenPlugin::setPluginNamespace(const char* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

const char* QKVToContextVarSeqlenPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

int32_t QKVToContextVarSeqlenPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream) noexcept
{

    if (mUseVarSeqlen)
    {
        const int32_t B = inputDesc[2].dims.d[0] - 1;
        const int32_t maxS = inputDesc[3].dims.d[0];
        PLUGIN_ASSERT((maxS <= 512)
            && "No implementation for variable sequence length multi-head attention plugin with sequence > 512.");

        int32_t S = 512;
        if (DataType::kHALF == mType && maxS <= 64)
        {
            S = 64;
        }
        else if (DataType::kHALF == mType && maxS <= 96)
        {
            S = 96;
        }
        else if (maxS <= 128)
        {
            S = 128;
        }
        else if (maxS <= 192)
        {
            S = 192;
            if(mType == DataType::kHALF)
            {
                S = 256;
            }

        }
        else if (maxS <= 256)
        {
            S = 256;
        }
        else if (maxS <= 384)
        {
            S = 384;
        }

        this->dispatcher->setup(S, B);

        if (patcher)
        {
            auto sumSeqLen = inputDesc[0].dims.d[0];
            auto paddingWorkspace = patcher->get16BytesAlignedPointer(workspace, dispatcher->getWorkspaceSize());
            auto ret = patcher->pad(inputs[0], paddingWorkspace, sumSeqLen, mNumHeads, mHeadSize, stream);
            if (ret != cudaSuccess)
            {
                return ret;
            }

            MhaRunParameter paddingArgs
                = patcher->patchMhaArgs(inputDesc, outputDesc, inputs, outputs, paddingWorkspace, sumSeqLen, mNumHeads);
            try
            {
                this->dispatcher->run(paddingArgs.inputDesc, paddingArgs.outputDesc, paddingArgs.inputs,
                    paddingArgs.outputs, workspace, stream);
            }
            catch (std::exception const& e)
            {
                caughtError(e);
                return -1;
            }

            ret = patcher->unpad(paddingArgs.outputs[0], outputs[0], sumSeqLen, mNumHeads, mHeadSize, stream);
            if (ret != cudaSuccess)
            {
                return ret;
            }
        }
        else
        {
            try
            {
                this->dispatcher->run(inputDesc, outputDesc, inputs, outputs, workspace, stream);
            }
            catch (std::exception const& e)
            {
                caughtError(e);
                return -1;
            }
        }

        return cudaGetLastError();
    }

    PLUGIN_ASSERT(mS == inputDesc->dims.d[SDIM]);
    PLUGIN_ASSERT(mB == inputDesc->dims.d[BDIM]);

    void const* maskPtr = mHasImask ? inputs[1] : nullptr;
    this->dispatcher->run(inputDesc[0], outputDesc[0], inputs[0], maskPtr, outputs[0], workspace, stream);
    return cudaGetLastError();
}

QKVToContextVarSeqlenPluginCreator::QKVToContextVarSeqlenPluginCreator()
{
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("type_id", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("hidden_size", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("num_heads", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("has_mask", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("dq_probs", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("var_seqlen", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("use_int8_scale_max", nullptr, PluginFieldType::kINT32, 1));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* QKVToContextVarSeqlenPluginCreator::getPluginName() const noexcept
{
    return QKV_TO_CONTEXT_PLUGIN_NAME;
}

const char* QKVToContextVarSeqlenPluginCreator::getPluginVersion() const noexcept
{
    return QKV_TO_CONTEXT_VAR_SEQLEN_PLUGIN_VERSION;
}

const PluginFieldCollection* QKVToContextVarSeqlenPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* QKVToContextVarSeqlenPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
{
    BERT_DEBUG_MSG("Creating QKV2ContextPlugin...");

    int32_t hiddenSize = 0;
    // Since numHeads must always exist or validateRequiredAttributes will fail,
    // we can set numHeads to -1 so that static analysis tools don't warn about
    // a division by zero in QKVToContextVarSeqelnPlugin constructor.
    int32_t numHeads{-1};
    bool hasMask = false;
    int32_t typeId = -1;

    int32_t varSeqlen = 0;

    float dqProbs = -1;
    int32_t useInt8ScaleMax{-1};

    plugin::validateRequiredAttributesExist({"type_id", "hidden_size", "num_heads", "has_mask"}, fc);
    for (int32_t i = 0; i < fc->nbFields; i++)
    {
        std::string field_name(fc->fields[i].name);

        if (field_name.compare("type_id") == 0)
        {
            typeId = *static_cast<int32_t const*>(fc->fields[i].data);
            PLUGIN_VALIDATE(typeId >= 0 && typeId <= 2, ("QKV: Invalid TypeId " + std::to_string(typeId)).c_str());
            BERT_DEBUG_VALUE("Building typeId: ", typeId);
        }
        if (field_name.compare("hidden_size") == 0)
        {
            hiddenSize = *static_cast<int32_t const*>(fc->fields[i].data);
            PLUGIN_VALIDATE(hiddenSize > 0, ("QKV: Invalid hiddenSize " + std::to_string(hiddenSize)).c_str());
            BERT_DEBUG_VALUE("Building hiddenSize: ", hiddenSize);
        }
        if (field_name.compare("num_heads") == 0)
        {
            numHeads = *static_cast<int32_t const*>(fc->fields[i].data);
            PLUGIN_VALIDATE(numHeads > 0, ("QKV: Invalid numHeads " + std::to_string(numHeads)).c_str());
            BERT_DEBUG_VALUE("Building numHeads: ", numHeads);
        }
        if (field_name.compare("has_mask") == 0)
        {
            hasMask = *static_cast<bool const*>(fc->fields[i].data);
            PLUGIN_VALIDATE(hasMask == 0 || hasMask == 1, ("QKV: Invalid hasMask " + std::to_string(hasMask)).c_str());
            BERT_DEBUG_VALUE("Building hasMask: ", hasMask);
        }

        if (field_name.compare("dq_probs") == 0)
        {
            dqProbs = *static_cast<float const*>(fc->fields[i].data);
            PLUGIN_VALIDATE(dqProbs > 0.0F, ("QKV: Invalid dqProbs " + std::to_string(dqProbs)).c_str());
            BERT_DEBUG_VALUE("Building dqProbs: ", dqProbs);
        }
        if (field_name.compare("var_seqlen") == 0)
        {
            varSeqlen = *static_cast<int32_t const*>(fc->fields[i].data);
            BERT_DEBUG_VALUE("Building var_seqlen: ", varSeqlen);
        }
        if (field_name.compare("use_int8_scale_max") == 0)
        {
            useInt8ScaleMax = *static_cast<int32_t const*>(fc->fields[i].data);
            PLUGIN_VALIDATE(useInt8ScaleMax == 0 || useInt8ScaleMax == 1,
                ("QKV: Invalid useInt8ScaleMax " + std::to_string(useInt8ScaleMax)).c_str());
            BERT_DEBUG_VALUE("Building useInt8ScaleMax: ", useInt8ScaleMax);
        }
    }

    if (useInt8ScaleMax < 0)
    {
        gLogInfo << "Using default for use_int8_scale_max: true" << std::endl;
        useInt8ScaleMax = 1;
    }

    BERT_DEBUG_MSG("Building the Plugin...");
    DataType type = static_cast<DataType>(typeId);
    if (type == DataType::kINT8 && dqProbs < 0)
    {
        gLogInfo << "Using default scale factor\n";
        dqProbs = 1.F / 127.F;
    }

    auto const useInt8ScaleMaxFlag = static_cast<bool>(useInt8ScaleMax);

    QKVToContextVarSeqlenPlugin* p
        = new QKVToContextVarSeqlenPlugin(name, type, hiddenSize, numHeads, dqProbs, hasMask, varSeqlen, useInt8ScaleMaxFlag);
    return p;
}

IPluginV2* QKVToContextVarSeqlenPluginCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength)  noexcept
{
    // This object will be deleted when the network is destroyed, which will
    // call QKVToContextVarSeqlenPlugin::destroy()
    return new QKVToContextVarSeqlenPlugin(name, serialData, serialLength);
}

void QKVToContextVarSeqlenPluginCreator::setPluginNamespace(const char* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

const char* QKVToContextVarSeqlenPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

#endif // CUDA_VERSION >= 10010
