/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "bertQKVToContextPlugin/fused_multihead_attention/fused_multihead_attention.h"
#include "bertQKVToContextPlugin/fused_multihead_attention_v2/fused_multihead_attention_v2.h"
#include "common/bertCommon.h"
#include "common/serialize.hpp"
#include "mhaRunner.h"
#include "qkvToContextPluginLegacy.h"

#include <cstdint>
#include <cstring>
#include <iostream>
#include <tuple>
#include <vector>

using namespace nvinfer1;
using namespace nvinfer1::plugin;
using namespace nvinfer1::plugin::bert;
using namespace nvinfer1::pluginInternal;

namespace
{
char const* const kQKV_TO_CONTEXT_PLUGIN_LEGACY_VERSION{"1"};
char const* const kQKV_TO_CONTEXT_VAR_SEQLEN_LEGACY_PLUGIN_VERSION{"2"};
char const* const kQKV_TO_CONTEXT_PLUGIN_NAME{"CustomQKVToContextPluginDynamic"};
} // namespace

REGISTER_TENSORRT_PLUGIN(QKVToContextPluginDynamicLegacyCreator);

constexpr uint32_t kIIDX = 0; // index of the input tensor
constexpr uint32_t kMIDX = 1; // index of the mask

REGISTER_TENSORRT_PLUGIN(QKVToContextVarSeqlenPluginLegacyCreator);

QKVToContextPluginDynamicLegacy::QKVToContextPluginDynamicLegacy(std::string const name, DataType const type,
    int32_t const hiddenSize, int32_t const numHeads, float const dqProbs, bool hasImask)
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
    mSM = getSmVersion();
}

QKVToContextPluginDynamicLegacy::QKVToContextPluginDynamicLegacy(
    std::string const name, void const* data, size_t length)
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

void QKVToContextPluginDynamicLegacy::createMHARunner()
{
    if (!fusedDispatcher.get())
    {
        if (mType == DataType::kHALF)
        {
            fusedDispatcher.reset(new FusedMHARunnerFP16(mNumHeads, mSM));
        }
        else if (mType == DataType::kINT8)
        {
            fusedDispatcher.reset(new FusedMHARunnerInt8(mNumHeads, mSM, mDqProbs));
        }
    }

    if (!unfusedDispatcher.get())
    {
        unfusedDispatcher.reset(new UnfusedMHARunner(mType, mNumHeads, mSM));
    }
}

// IPluginV2DynamicExt Methods
nvinfer1::IPluginV2DynamicExt* QKVToContextPluginDynamicLegacy::clone() const noexcept
{
    BERT_DEBUG_MSG("QKV Clone");

    QKVToContextPluginDynamicLegacy* ret = nullptr;
    // the workspacesize is 0 if we have not call setup the dispatcher yet.
    if (unfusedDispatcher.get() && unfusedDispatcher->getWorkspaceSize())
    {
        std::vector<char> buff;
        buff.resize(getSerializationSize());
        serialize(buff.data());

        ret = new QKVToContextPluginDynamicLegacy(mLayerName, buff.data(), buff.size());
    }
    else
    {
        ret = new QKVToContextPluginDynamicLegacy(mLayerName, mType, mHiddenSize, mNumHeads, mDqProbs, mHasImask);
    }

    ret->setPluginNamespace(mNamespace.c_str());
    BERT_DEBUG_MSG("QKV Clone done");
    return ret;
}

DimsExprs QKVToContextPluginDynamicLegacy::getOutputDimensions(
    int32_t outputIndex, DimsExprs const* inputs, int32_t /*nbInputs*/, IExprBuilder& exprBuilder) noexcept
{
    // Input is BxSx3*N*H, output should be BxSxN*H
    PLUGIN_ASSERT(outputIndex == 0);
    // Copy over everything
    DimsExprs output(inputs[kIIDX]);
    // Divide last dim by three
    auto const* three = exprBuilder.constant(3);
    output.d[HDIM] = exprBuilder.operation(DimensionOperation::kFLOOR_DIV, *inputs[kIIDX].d[HDIM], *three);
    return output;
}
bool QKVToContextPluginDynamicLegacy::supportsFormatCombination(
    int32_t pos, PluginTensorDesc const* inOut, int32_t nbInputs, int32_t /*nbOutputs*/) noexcept
{
    PLUGIN_ASSERT(pos >= 0);
    PLUGIN_ASSERT(pos < 2 + mHasImask);
    PLUGIN_ASSERT(nbInputs == 1 + mHasImask);
    auto const* in = inOut;
    auto const* out = inOut + nbInputs;
    int32_t packedSize = getMHAMaskPackedSize(mSM, mType, in->dims.d[SDIM]);

    // we only support int8 IO in fused mha runner, and we only support fused mha runner on Xavier, Turing and Ampere
    if (mType == DataType::kINT8)
    {
        if (!elem(mSM, {kSM_75, kSM_80, kSM_86, kSM_87, kSM_89, kSM_90, kSM_100, kSM_120}))
        {
            gLogError << "INT8 IO is only supported on Turing, Ampere, Hopper and Blackwell for plugin "
                      << kQKV_TO_CONTEXT_PLUGIN_NAME << std::endl;
            return false;
        }
        if (in->dims.d[SDIM] == -1)
        {
            gLogError << "INT8 IO not support dynamic shape in sequence dimension for plugin "
                      << kQKV_TO_CONTEXT_PLUGIN_NAME << std::endl;
            return false;
        }
        if (packedSize == unfusedMaskSize)
        {
            gLogError << "INT8 IO only support sequence length 128,384 for plugin " << kQKV_TO_CONTEXT_PLUGIN_NAME
                      << std::endl;
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
                      << ", but " << kQKV_TO_CONTEXT_PLUGIN_NAME << " expects mask pack size " << packedSize
                      << std::endl;
            return false;
        }

        // detect full mask and check that it was produced
        return (inMask->type == DataType::kINT32) &&     // precision
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
void QKVToContextPluginDynamicLegacy::configurePlugin(
    DynamicPluginTensorDesc const* in, int32_t nbInputs, DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
    PLUGIN_ASSERT(nbInputs == 1 + mHasImask);
    PLUGIN_ASSERT(nbOutputs == 1);
    PluginTensorDesc const& inDesc = in[kIIDX].desc;
    TRT_UNUSED inDesc;
    PluginTensorDesc const& outDesc = out->desc;
    TRT_UNUSED outDesc;
    PLUGIN_ASSERT(mType == inDesc.type);
    PLUGIN_ASSERT(mType == outDesc.type);
    PLUGIN_ASSERT(inDesc.dims.d[BDIM] == outDesc.dims.d[BDIM]);
    PLUGIN_ASSERT(inDesc.dims.d[SDIM] == outDesc.dims.d[SDIM]);
    PLUGIN_ASSERT(inDesc.dims.d[HDIM] == 3 * outDesc.dims.d[HDIM]);
    if (mHasImask)
    {
        PluginTensorDesc const& maskDesc = in[kMIDX].desc;
        TRT_UNUSED maskDesc;
        PLUGIN_ASSERT(maskDesc.dims.d[0] == inDesc.dims.d[BDIM]);
    }

    createMHARunner();

    int32_t const S = inDesc.dims.d[SDIM];
    int32_t const B = inDesc.dims.d[BDIM] <= 0 ? in->max.d[BDIM] : inDesc.dims.d[BDIM];
    if (S <= 0)
    {
        // in dynamic shape build stage, we setup with max sequence that cannot fused
        int32_t const Smin = in->min.d[SDIM];
        int32_t const Smax = in->max.d[SDIM];

        if (fusedDispatcher.get())
        {
            for (int32_t i = Smax; i >= Smin; --i)
            {
                if (!fusedDispatcher->isValid(mHeadSize, i))
                {
                    unfusedDispatcher->setup(i, B, mHeadSize);
                    mS = i;
                    mB = B;
                    break;
                }
            }
        }
        else
        {
            unfusedDispatcher->setup(Smax, B, mHeadSize);
            mS = Smax;
            mB = B;
        }
    }
    else
    {
        // in inference stage or in static shape build stage
        if (fusedDispatcher.get() && fusedDispatcher->isValid(mHeadSize, S))
        {
            fusedDispatcher->setup(S, B, mHeadSize);
        }
        else
        {
            unfusedDispatcher->setup(S, B, mHeadSize);
        }
        mS = S;
        mB = B;
    }
}

size_t QKVToContextPluginDynamicLegacy::getWorkspaceSize(PluginTensorDesc const* /*inputs*/, int32_t /*nbInputs*/,
    PluginTensorDesc const* /*outputs*/, int32_t /*nbOutputs*/) const noexcept
{
    // only unfused kernel need workspace, and we need larger workspace for larger sequence length
    // we have already setup unfusedDispatcher with max sequence in configurePlugin
    // if unfusedDispatcher is not initialized in configurePlugin
    PLUGIN_ASSERT(unfusedDispatcher.get());
    return unfusedDispatcher->getWorkspaceSize();
}

// IPluginV2Ext Methods
DataType QKVToContextPluginDynamicLegacy::getOutputDataType(
    int32_t index, nvinfer1::DataType const* inputTypes, int32_t /*nbInputs*/) const noexcept
{
    PLUGIN_ASSERT(index == 0);
    PLUGIN_ASSERT(
        inputTypes[0] == DataType::kFLOAT || inputTypes[0] == DataType::kHALF || inputTypes[0] == DataType::kINT8);
    return inputTypes[0];
}

void QKVToContextPluginDynamicLegacy::attachToContext(
    cudnnContext* cudnn, cublasContext* cublas, nvinfer1::IGpuAllocator* allocator) noexcept
{
    try
    {
        mCublasWrapper = createPluginCublasWrapper(allocator);
        mCublas = mCublasWrapper->getCublasHandle();
        PLUGIN_VALIDATE(mCublas != nullptr);
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
}

// IPluginV2 Methods
char const* QKVToContextPluginDynamicLegacy::getPluginType() const noexcept
{
    return kQKV_TO_CONTEXT_PLUGIN_NAME;
}

char const* QKVToContextPluginDynamicLegacy::getPluginVersion() const noexcept
{
    return kQKV_TO_CONTEXT_PLUGIN_LEGACY_VERSION;
}

int32_t QKVToContextPluginDynamicLegacy::getNbOutputs() const noexcept
{
    return 1;
}

int32_t QKVToContextPluginDynamicLegacy::initialize() noexcept
{
    return 0;
}

void QKVToContextPluginDynamicLegacy::terminate() noexcept {}

size_t QKVToContextPluginDynamicLegacy::getSerializationSize() const noexcept
{
    PLUGIN_ASSERT(unfusedDispatcher.get());
    return sizeof(mNumHeads) + sizeof(mHeadSize) + sizeof(DataType) + sizeof(mHasImask) + sizeof(mHiddenSize)
        + sizeof(mSM) + sizeof(mS) + sizeof(mB) + sizeof(mDqProbs) + sizeof(int32_t)
        + unfusedDispatcher->getSerializationSize();
}

void QKVToContextPluginDynamicLegacy::serialize(void* buffer) const noexcept
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

void QKVToContextPluginDynamicLegacy::destroy() noexcept
{
    delete this;
}

void QKVToContextPluginDynamicLegacy::setPluginNamespace(char const* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

char const* QKVToContextPluginDynamicLegacy::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

int32_t QKVToContextPluginDynamicLegacy::enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc,
    void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    PLUGIN_VALIDATE(inputDesc != nullptr && outputDesc != nullptr && inputs != nullptr && outputs != nullptr);
    PLUGIN_ASSERT(mS == inputDesc->dims.d[SDIM]);
    PLUGIN_ASSERT(mB == inputDesc->dims.d[BDIM]);

    try
    {
        void const* const maskPtr = mHasImask ? inputs[1] : nullptr;
        if (mHasImask && fusedDispatcher.get() && fusedDispatcher->isValid(mHeadSize, inputDesc->dims.d[SDIM]))
        {
            fusedDispatcher->run(
                inputDesc[0], outputDesc[0], inputs[0], maskPtr, outputs[0], workspace, stream, mCublas);
        }
        else
        {
            PLUGIN_VALIDATE(unfusedDispatcher.get(), "The Unfused MHARunner is uninitialized, no MHARunner available!");
            PLUGIN_VALIDATE(mType != DataType::kINT8, "The Unfused MHARunner does not support INT8!");
            unfusedDispatcher->run(
                inputDesc[0], outputDesc[0], inputs[0], maskPtr, outputs[0], workspace, stream, mCublas);
        }
    }
    catch (std::exception const& e)
    {
        caughtError(e);
        return -1;
    }
    return 0;
}

QKVToContextPluginDynamicLegacyCreator::QKVToContextPluginDynamicLegacyCreator()
{
    mPluginAttributes.emplace_back(PluginField("type_id", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("hidden_size", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("num_heads", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("has_mask", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("dq_probs", nullptr, PluginFieldType::kFLOAT32, 1));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

char const* QKVToContextPluginDynamicLegacyCreator::getPluginName() const noexcept
{
    return kQKV_TO_CONTEXT_PLUGIN_NAME;
}

char const* QKVToContextPluginDynamicLegacyCreator::getPluginVersion() const noexcept
{
    return kQKV_TO_CONTEXT_PLUGIN_LEGACY_VERSION;
}

PluginFieldCollection const* QKVToContextPluginDynamicLegacyCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* QKVToContextPluginDynamicLegacyCreator::createPlugin(
    char const* name, PluginFieldCollection const* fc) noexcept
{
    try
    {
        BERT_DEBUG_MSG("Creating QKV2ContextPlugin...");
        PLUGIN_VALIDATE(fc != nullptr);
        int32_t hiddenSize = 0;
        // Since numHeads must always exist or validateRequiredAttributes will fail,
        // we can set numHeads to -1 so that static analysis tools don't warn about
        // a division by zero in QKVToContextPluginDynamicLegacy constructor.
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

        auto* p = new QKVToContextPluginDynamicLegacy(name, type, hiddenSize, numHeads, dqProbs, hasMask);
        return p;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* QKVToContextPluginDynamicLegacyCreator::deserializePlugin(
    char const* name, void const* serialData, size_t serialLength) noexcept
{
    // This object will be deleted when the network is destroyed, which will
    // call QKVToContextPluginDynamicLegacy::destroy()
    return new QKVToContextPluginDynamicLegacy(name, serialData, serialLength);
}

void QKVToContextPluginDynamicLegacyCreator::setPluginNamespace(char const* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

char const* QKVToContextPluginDynamicLegacyCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

QKVToContextVarSeqlenPluginLegacy::QKVToContextVarSeqlenPluginLegacy(std::string const name, DataType const type,
    int32_t const hiddenSize, int32_t const numHeads, float const dqProbs, bool hasImask, bool varSeqlen,
    bool useInt8ScaleMax)
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
    mSM = getSmVersion();

    if (varSeqlen)
    {
        // variable sequence length is only supported with the fused MHA kernels
        // we should not override mS!
        bool isSMSupported = elem(mSM, {kSM_75, kSM_80, kSM_86, kSM_87, kSM_89, kSM_90, kSM_100, kSM_120});
        PLUGIN_ASSERT(isSMSupported && (type == DataType::kINT8 || type == DataType::kHALF)
            && "requesting maxSeqlen not compatible with GPU arch");
        // the layout changes: SxB will be a combined \sum_i s_i and hdim will be the 2nd dimension instead of the third
        mHdim = 1;
    }
}

QKVToContextVarSeqlenPluginLegacy::QKVToContextVarSeqlenPluginLegacy(
    std::string const name, void const* data, size_t length)
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
    mDispatcher->deserialize(data, length);

    BERT_DEBUG_MSG("QKV Deser done");
}

void QKVToContextVarSeqlenPluginLegacy::createMHARunner()
{
    if (mDispatcher.get())
    {
        return;
    }

    if (mUseVarSeqlen)
    {
        PLUGIN_ASSERT(mHeadSize <= 64);
        {
            if (mHeadSize != 64)
            {
                mPatcher.reset(new QkvPaddingRunner(mType));
            }
            if (mType == DataType::kHALF)
            {
                mDispatcher.reset(new FusedMHARunnerFP16v2(mNumHeads, mSM));
            }
            else if (mType == DataType::kINT8)
            {
                mDispatcher.reset(new FusedMHARunnerInt8v2(mNumHeads, mSM, mDqProbs, mUseInt8ScaleMax));
            }
        }
    }
    else
    {
        PLUGIN_ASSERT(mType != DataType::kINT8);
        mDispatcher.reset(new UnfusedMHARunner(mType, mNumHeads, mSM));
    }
}

// IPluginV2DynamicExt Methods
nvinfer1::IPluginV2DynamicExt* QKVToContextVarSeqlenPluginLegacy::clone() const noexcept
{
    BERT_DEBUG_MSG("QKV Clone");

    QKVToContextVarSeqlenPluginLegacy* ret = nullptr;
    if (mDispatcher.get())
    {
        std::vector<char> buff;
        buff.resize(getSerializationSize());
        serialize(buff.data());

        ret = new QKVToContextVarSeqlenPluginLegacy(mLayerName, buff.data(), buff.size());
    }
    else
    {
        ret = new QKVToContextVarSeqlenPluginLegacy(
            mLayerName, mType, mHiddenSize, mNumHeads, mDqProbs, mHasImask, mUseVarSeqlen, mUseInt8ScaleMax);
    }

    ret->setPluginNamespace(mNamespace.c_str());
    BERT_DEBUG_MSG("QKV Clone done");
    return ret;
}

DimsExprs QKVToContextVarSeqlenPluginLegacy::getOutputDimensions(
    int32_t outputIndex, DimsExprs const* inputs, int32_t /*nbInputs*/, IExprBuilder& exprBuilder) noexcept
{
    // Input is BxSx3*N*H, output should be BxSxN*H
    PLUGIN_ASSERT(outputIndex == 0);
    // Copy over everything
    DimsExprs output(inputs[kIIDX]);
    // Divide last dim by three
    auto const* three = exprBuilder.constant(3);
    output.d[mHdim] = exprBuilder.operation(DimensionOperation::kFLOOR_DIV, *inputs[kIIDX].d[mHdim], *three);
    return output;
}

bool QKVToContextVarSeqlenPluginLegacy::supportsFormatCombination(
    int32_t pos, PluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    // we only support variable sequence and int8 IO in fused mha runner, and we only support fused mha runner on
    // Turing, Ampere, Hopper and Blackwell
    bool const hasV2Kernels = elem(mSM, {kSM_75, kSM_80, kSM_86, kSM_87, kSM_89, kSM_90, kSM_100, kSM_120});
    PLUGIN_ASSERT((mType != DataType::kINT8 || hasV2Kernels)
        && "INT8 IO is only supported on Xavier, Turing, Ampere, Hopper and Blackwell!");
    PLUGIN_ASSERT((!mUseVarSeqlen || hasV2Kernels)
        && "Variable sequence is only supported on Xavier, Turing, Ampere, Hopper and Blackwell!");

    PLUGIN_ASSERT(pos >= 0);
    PLUGIN_ASSERT(pos < 2 + mHasImask + 2 * mUseVarSeqlen);
    PLUGIN_ASSERT(nbInputs == 1 + mHasImask + 2 * mUseVarSeqlen);
    PLUGIN_ASSERT(nbOutputs == 1);
    auto const* in = inOut;
    auto const* out = inOut + nbInputs;
    if (mUseVarSeqlen)
    {
        PLUGIN_ASSERT((mType == DataType::kHALF || mType == DataType::kINT8)
            && "Conditions for variable seqlen support not fulfilled");
        // qkv, mask, cu_seqlens, dummy
        PLUGIN_ASSERT(nbInputs == 4 && "for varseqlen, expected 4 inputs");
    }

    auto const inDims = in->dims;
    auto const outDims = out->dims;

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
        auto const& desc = inOut[pos];
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
        auto const* mask = &inOut[pos];
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
        // cuSeqlens is a int32_t array of size B+1
        auto const* seqlens = &inOut[pos];
        return (seqlens->type == DataType::kINT32) && (seqlens->format == TensorFormat::kLINEAR);
    }
    if (pos == 3)
    {
        // this is the dummy input
        return inOut[pos].dims.nbDims == 1;
    }
    return false;
}

void QKVToContextVarSeqlenPluginLegacy::configurePlugin(
    DynamicPluginTensorDesc const* in, int32_t nbInputs, DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
    PLUGIN_ASSERT(nbInputs == 1 + mHasImask + 2 * mUseVarSeqlen);
    PLUGIN_ASSERT(nbOutputs == 1);
    PluginTensorDesc const& inDesc = in[kIIDX].desc;
    TRT_UNUSED inDesc;
    PluginTensorDesc const& outDesc = out->desc;
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
            PluginTensorDesc const& maskDesc = in[kMIDX].desc;
            TRT_UNUSED maskDesc;
            PLUGIN_ASSERT(maskDesc.dims.d[0] == inDesc.dims.d[BDIM]);
        }

        int32_t const S = inDesc.dims.d[SDIM] <= 0 ? in->max.d[SDIM] : inDesc.dims.d[SDIM];
        int32_t const B = inDesc.dims.d[BDIM] <= 0 ? in->max.d[BDIM] : inDesc.dims.d[BDIM];

        if (S != mS || B != mB)
        {
            BERT_DEBUG_MSG("setting up MHA runner for single sequence length");
            createMHARunner();
            this->mDispatcher->setup(S, B, mHeadSize);
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
        this->mDispatcher->setup(256, 1, mHeadSize);
    }
}

size_t QKVToContextVarSeqlenPluginLegacy::getWorkspaceSize(PluginTensorDesc const* inputs, int32_t /* nbInputs */,
    PluginTensorDesc const* /* outputs */, int32_t /* nbOutputs */) const noexcept
{
    size_t paddingWorkpaceSize = mPatcher ? mPatcher->getWorkspaceSize(inputs[0].dims.d[0], mNumHeads) : 0;
    return mDispatcher->getWorkspaceSize() + paddingWorkpaceSize;
}

// IPluginV2Ext Methods
DataType QKVToContextVarSeqlenPluginLegacy::getOutputDataType(
    int32_t index, nvinfer1::DataType const* inputTypes, int32_t /*nbInputs*/) const noexcept
{
    PLUGIN_ASSERT(index == 0);
    PLUGIN_ASSERT(
        inputTypes[0] == DataType::kFLOAT || inputTypes[0] == DataType::kHALF || inputTypes[0] == DataType::kINT8);
    return inputTypes[0];
}

void QKVToContextVarSeqlenPluginLegacy::attachToContext(
    cudnnContext* cudnn, cublasContext* cublas, nvinfer1::IGpuAllocator* allocator) noexcept
{
    try
    {
        mCublasWrapper = createPluginCublasWrapper(allocator);
        mCublas = mCublasWrapper->getCublasHandle();
        PLUGIN_VALIDATE(mCublas != nullptr);
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
}

// IPluginV2 Methods
char const* QKVToContextVarSeqlenPluginLegacy::getPluginType() const noexcept
{
    return kQKV_TO_CONTEXT_PLUGIN_NAME;
}

char const* QKVToContextVarSeqlenPluginLegacy::getPluginVersion() const noexcept
{
    return kQKV_TO_CONTEXT_VAR_SEQLEN_LEGACY_PLUGIN_VERSION;
}

int32_t QKVToContextVarSeqlenPluginLegacy::getNbOutputs() const noexcept
{
    return 1;
}

int32_t QKVToContextVarSeqlenPluginLegacy::initialize() noexcept
{
    return 0;
}

void QKVToContextVarSeqlenPluginLegacy::terminate() noexcept {}

size_t QKVToContextVarSeqlenPluginLegacy::getSerializationSize() const noexcept
{
    return sizeof(mNumHeads) + sizeof(mHeadSize) + sizeof(DataType) + sizeof(mHasImask) + sizeof(mHiddenSize)
        + sizeof(mSM) + sizeof(mS) + sizeof(mB) + sizeof(mDqProbs) + mDispatcher->getSerializationSize()
        + sizeof(mUseVarSeqlen) + sizeof(mHdim) + sizeof(mUseInt8ScaleMax);
}

void QKVToContextVarSeqlenPluginLegacy::serialize(void* buffer) const noexcept
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
    mDispatcher->serialize(buffer);
}

void QKVToContextVarSeqlenPluginLegacy::destroy() noexcept
{
    delete this;
}

void QKVToContextVarSeqlenPluginLegacy::setPluginNamespace(char const* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

char const* QKVToContextVarSeqlenPluginLegacy::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

int32_t QKVToContextVarSeqlenPluginLegacy::enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
    nvinfer1::PluginTensorDesc const* outputDesc, void const* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream) noexcept
{
    PLUGIN_VALIDATE(inputDesc != nullptr && outputDesc != nullptr && inputs != nullptr && outputs != nullptr);

    if (mUseVarSeqlen)
    {
        int32_t const B = inputDesc[2].dims.d[0] - 1;
        int32_t const maxS = inputDesc[3].dims.d[0];
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
            if (mType == DataType::kHALF)
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

        auto runV2Kernel = [this, &S, &B, &workspace, &inputDesc, &outputDesc, &stream, &inputs, &outputs](
                               MHARunner* dispatcher, QkvPaddingRunner* patcher, int32_t padSize) {
            PLUGIN_ASSERT(dispatcher);
            // Validate that we can padding to the dispatch required head size also there is kernel exist for this
            // sequence length.
            if (mHeadSize > padSize || !dispatcher->isValid(padSize, S))
            {
                return false;
            }
            dispatcher->setup(S, B, padSize);

            // Need pad and unpad to run the V2 kernel.
            if (mHeadSize < padSize)
            {
                PLUGIN_ASSERT(patcher);
                PLUGIN_ASSERT(padSize <= patcher->getMaxPaddingHeadSize());
                auto sumSeqLen = inputDesc[0].dims.d[0];
                auto paddingWorkspace = patcher->get16BytesAlignedPointer(workspace, dispatcher->getWorkspaceSize());
                auto ret = mPatcher->pad(inputs[0], paddingWorkspace, sumSeqLen, mNumHeads, mHeadSize, padSize, stream);
                if (ret != cudaSuccess)
                {
                    return false;
                }

                MhaRunParameter paddingArgs = patcher->patchMhaArgs(
                    inputDesc, outputDesc, inputs, outputs, paddingWorkspace, sumSeqLen, mNumHeads, padSize);
                try
                {
                    dispatcher->run(paddingArgs.inputDesc, paddingArgs.outputDesc, paddingArgs.inputs,
                        paddingArgs.outputs, workspace, stream, mCublas);
                }
                catch (std::exception const& e)
                {
                    caughtError(e);
                    return false;
                }

                ret = patcher->unpad(
                    paddingArgs.outputs[0], outputs[0], sumSeqLen, mNumHeads, mHeadSize, padSize, stream);
                return ret == cudaSuccess;
            }
            else
            {
                // No pad/unpad is needed.
                try
                {
                    dispatcher->run(inputDesc, outputDesc, inputs, outputs, workspace, stream, mCublas);
                }
                catch (std::exception const& e)
                {
                    caughtError(e);
                    return false;
                }
                return true;
            }
        };
        // Try pad head size to 32 first, if it failed, then try to pad head size to 64.
        if (!runV2Kernel(mDispatcher.get(), mPatcher.get(), 32) && !runV2Kernel(mDispatcher.get(), mPatcher.get(), 64))
        {
            return false;
        }

        return cudaGetLastError();
    }

    PLUGIN_ASSERT(mS == inputDesc->dims.d[SDIM]);
    PLUGIN_ASSERT(mB == inputDesc->dims.d[BDIM]);

    void const* maskPtr = mHasImask ? inputs[1] : nullptr;
    mDispatcher->run(inputDesc[0], outputDesc[0], inputs[0], maskPtr, outputs[0], workspace, stream, mCublas);
    return cudaGetLastError();
}

QKVToContextVarSeqlenPluginLegacyCreator::QKVToContextVarSeqlenPluginLegacyCreator()
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

char const* QKVToContextVarSeqlenPluginLegacyCreator::getPluginName() const noexcept
{
    return kQKV_TO_CONTEXT_PLUGIN_NAME;
}

char const* QKVToContextVarSeqlenPluginLegacyCreator::getPluginVersion() const noexcept
{
    return kQKV_TO_CONTEXT_VAR_SEQLEN_LEGACY_PLUGIN_VERSION;
}

PluginFieldCollection const* QKVToContextVarSeqlenPluginLegacyCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* QKVToContextVarSeqlenPluginLegacyCreator::createPlugin(
    char const* name, PluginFieldCollection const* fc) noexcept
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

    QKVToContextVarSeqlenPluginLegacy* p = new QKVToContextVarSeqlenPluginLegacy(
        name, type, hiddenSize, numHeads, dqProbs, hasMask, varSeqlen, useInt8ScaleMaxFlag);
    return p;
}

IPluginV2* QKVToContextVarSeqlenPluginLegacyCreator::deserializePlugin(
    char const* name, void const* serialData, size_t serialLength) noexcept
{
    // This object will be deleted when the network is destroyed, which will
    // call QKVToContextVarSeqlenPluginLegacy::destroy()
    return new QKVToContextVarSeqlenPluginLegacy(name, serialData, serialLength);
}

void QKVToContextVarSeqlenPluginLegacyCreator::setPluginNamespace(char const* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

char const* QKVToContextVarSeqlenPluginLegacyCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

#endif // CUDA_VERSION >= 10010
