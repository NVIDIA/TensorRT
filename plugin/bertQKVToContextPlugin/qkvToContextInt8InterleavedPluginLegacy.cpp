/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "qkvToContextInt8InterleavedPluginLegacy.h"
#include "NvInfer.h"
#include "common/bertCommon.h"
#include "common/plugin.h"
#include "common/serialize.hpp"

#include <cstring>
#include <cuda.h>
#include <iostream>
#include <tuple>
#include <vector>

#include "bertQKVToContextPlugin/fused_multihead_attention_v2/fused_multihead_attention_v2.h"

using namespace nvinfer1;
using namespace nvinfer1::plugin;
using namespace nvinfer1::plugin::bert;

namespace
{
char const* const kQKV_TO_CONTEXT_INTERLEAVED_PLUGIN_LEGACY_VERSION{"3"};
char const* const kQKV_TO_CONTEXT_INTERLEAVED_PLUGIN_LEGACY_NAME{"CustomQKVToContextPluginDynamic"};
} // namespace

REGISTER_TENSORRT_PLUGIN(QKVToContextInterleavedPluginLegacyCreator);

constexpr uint32_t kIIDX = 0; // index of the input tensor

QKVToContextInterleavedPluginLegacy::QKVToContextInterleavedPluginLegacy(std::string const& name, int32_t hiddenSize,
    int32_t numHeads, float dqProbs, bool useInt8ScaleMax, bool useExplicitInt8, float qkvScale, float ctxScale)
    : mLayerName(name)
    , mS(0)
    , mB(0)
    , mHeadSize(hiddenSize / numHeads)
    , mHiddenSize(hiddenSize)
    , mNumHeads(numHeads)
    , mDqProbs(dqProbs)
    , mUseInt8ScaleMax(useInt8ScaleMax)
    , mUseExplicitInt8(useExplicitInt8)
    , mQkvScale(qkvScale)
    , mCtxScale(ctxScale)
{
    mSM = getSmVersion();
    // variable sequence length is only supported with the fused MHA kernels
    // we should not override mS!
    bool isSMSupported = elem(mSM,
        {kSM_AMPERE_100, kSM_AMPERE_10X, kSM_AMPERE_10B, kSM_TURING, kSM_XAVIER, kSM_ADA_10X, kSM_HOPPER_100,
            kSM_BLACKWELL_100, kSM_BLACKWELL_120});
    PLUGIN_VALIDATE(isSMSupported && "requesting maxSeqlen not compatible with GPU arch");
    // the layout changes: SxB will be a combined \sum_i s_i and hdim will be the 2nd dimension instead of the third
    mXmmaKernel = getXMMAKernelsV2(DATA_TYPE_INT8, mSM);
}

QKVToContextInterleavedPluginLegacy::QKVToContextInterleavedPluginLegacy(
    std::string const& name, void const* data, size_t length)
    : mLayerName(name)
{
    deserialize_value(&data, &length, &mNumHeads);
    deserialize_value(&data, &length, &mHeadSize);
    deserialize_value(&data, &length, &mHiddenSize);
    deserialize_value(&data, &length, &mSM);
    deserialize_value(&data, &length, &mS);
    deserialize_value(&data, &length, &mB);
    deserialize_value(&data, &length, &mDqProbs);
    deserialize_value(&data, &length, &mUseInt8ScaleMax);
    deserialize_value(&data, &length, &mUseExplicitInt8);
    deserialize_value(&data, &length, &mQkvScale);
    deserialize_value(&data, &length, &mCtxScale);
}

// IPluginV2DynamicExt Methods
nvinfer1::IPluginV2DynamicExt* QKVToContextInterleavedPluginLegacy::clone() const noexcept
{
    try
    {
        QKVToContextInterleavedPluginLegacy* ret = new QKVToContextInterleavedPluginLegacy(
            mLayerName, mHiddenSize, mNumHeads, mDqProbs, mUseInt8ScaleMax, mUseExplicitInt8, mQkvScale, mCtxScale);

        ret->setPluginNamespace(mNamespace.c_str());
        return ret;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

DimsExprs QKVToContextInterleavedPluginLegacy::getOutputDimensions(
    int32_t outputIndex, DimsExprs const* inputs, int32_t nbInputs, IExprBuilder& exprBuilder) noexcept
{
    // Input SHAPE is 1x(3*N*H)xTotalx1 (NCHW)
    // Output SHAPE is 1x(N*H)xTotalx1
    // In SupportsFormatCombination, we force the layout to be CHW, i.e.
    // Input: 3xNx(H/32)xsumSx32, Output: 1xNx(H/32)xsumSx32
    PLUGIN_ASSERT(outputIndex == 0);
    // Copy over everything
    DimsExprs output(inputs[kIIDX]);
    // output.d[0] = exprBuilder.constant(1);
    // Divide last dim by three
    auto const* three = exprBuilder.constant(3);
    output.d[1] = exprBuilder.operation(DimensionOperation::kFLOOR_DIV, *inputs[kIIDX].d[1], *three);
    return output;
}
bool QKVToContextInterleavedPluginLegacy::supportsFormatCombination(
    int32_t pos, PluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    PLUGIN_ASSERT(nbInputs == 3);
    PLUGIN_ASSERT(nbOutputs == 1);
    // 3 inputs:
    // 0: qkv
    // 1: cu_seqlens
    // 2: dummy
    // 1 output
    if (pos == 0 || pos == nbInputs)
    {
        return (inOut[pos].type == DataType::kINT8) && (inOut[pos].format == TensorFormat::kCHW32);
    }

    if (pos == 1)
    {
        // cuSeqlens is a int32_t array of size B+1
        auto const* seqlens = &inOut[pos];
        return (seqlens->type == DataType::kINT32) && (seqlens->format == TensorFormat::kLINEAR);
    }
    if (pos == 2)
    {
        // this is the dummy input
        return inOut[pos].dims.nbDims == 1;
    }
    return false;
}

void QKVToContextInterleavedPluginLegacy::configurePlugin(
    DynamicPluginTensorDesc const* in, int32_t nbInputs, DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
}

size_t QKVToContextInterleavedPluginLegacy::getWorkspaceSize(
    PluginTensorDesc const* inputs, int32_t nbInputs, PluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept
{
    return 0;
}

// IPluginV2Ext Methods
DataType QKVToContextInterleavedPluginLegacy::getOutputDataType(
    int32_t index, nvinfer1::DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    PLUGIN_ASSERT(index == 0);
    return DataType::kINT8;
}

// IPluginV2 Methods
char const* QKVToContextInterleavedPluginLegacy::getPluginType() const noexcept
{
    return kQKV_TO_CONTEXT_INTERLEAVED_PLUGIN_LEGACY_NAME;
}

char const* QKVToContextInterleavedPluginLegacy::getPluginVersion() const noexcept
{
    return kQKV_TO_CONTEXT_INTERLEAVED_PLUGIN_LEGACY_VERSION;
}

int32_t QKVToContextInterleavedPluginLegacy::getNbOutputs() const noexcept
{
    return 1;
}

int32_t QKVToContextInterleavedPluginLegacy::initialize() noexcept
{
    return 0;
}

void QKVToContextInterleavedPluginLegacy::terminate() noexcept {}

size_t QKVToContextInterleavedPluginLegacy::getSerializationSize() const noexcept
{
    return sizeof(mNumHeads) + sizeof(mHeadSize) + sizeof(mHiddenSize) + sizeof(mSM) + sizeof(mS) + sizeof(mB)
        + sizeof(mDqProbs) + sizeof(mUseInt8ScaleMax) + sizeof(mUseExplicitInt8) + sizeof(mQkvScale)
        + sizeof(mCtxScale);
}

void QKVToContextInterleavedPluginLegacy::serialize(void* buffer) const noexcept
{
    serialize_value(&buffer, mNumHeads);
    serialize_value(&buffer, mHeadSize);
    serialize_value(&buffer, mHiddenSize);
    serialize_value(&buffer, mSM);
    serialize_value(&buffer, mS);
    serialize_value(&buffer, mB);
    serialize_value(&buffer, mDqProbs);
    serialize_value(&buffer, mUseInt8ScaleMax);
    serialize_value(&buffer, mUseExplicitInt8);
    serialize_value(&buffer, mQkvScale);
    serialize_value(&buffer, mCtxScale);
}

void QKVToContextInterleavedPluginLegacy::destroy() noexcept
{
    delete this;
}

void QKVToContextInterleavedPluginLegacy::setPluginNamespace(char const* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

char const* QKVToContextInterleavedPluginLegacy::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

int32_t QKVToContextInterleavedPluginLegacy::enqueue(PluginTensorDesc const* inputDesc,
    PluginTensorDesc const* outputDesc, void const* const* inputs, void* const* outputs, void* /* workspace */,
    cudaStream_t stream) noexcept
{
    PLUGIN_VALIDATE(inputDesc != nullptr && outputDesc != nullptr && inputs != nullptr && outputs != nullptr);

    int32_t const total = inputDesc[0].dims.d[2];
    int32_t const B = inputDesc[1].dims.d[0] - 1;
    int32_t const maxS = inputDesc[2].dims.d[0];
    int32_t S = 384;
    if (maxS <= 128)
    {
        S = 128;
    }
    else if (maxS <= 192)
    {
        S = 192;
    }
    else if (maxS <= 256)
    {
        S = 256;
    }
    Fused_multihead_attention_params_v2 params{};
    params.b = B;
    params.s = S;
    params.h = mNumHeads;
    params.d = mHeadSize;

    params.interleaved = true;

    params.o_ptr = outputs[0];
    params.qkv_ptr = const_cast<void*>(inputs[0]);
    params.cu_seqlens = static_cast<int32_t*>(const_cast<void*>(inputs[1]));

    float scaleQkv = mUseExplicitInt8 ? mQkvScale : inputDesc[0].scale;
    float scaleCtx = mUseExplicitInt8 ? mCtxScale : outputDesc[0].scale;

    float scaleBmm1 = scaleQkv * scaleQkv * 0.125; // 1 / sqrt(64)
    float scaleBmm2 = mDqProbs * scaleQkv / scaleCtx;
    float scaleSoftmax = 1.F / mDqProbs;

    params.scale_bmm1 = reinterpret_cast<uint32_t const&>(scaleBmm1);
    params.scale_bmm2 = reinterpret_cast<uint32_t const&>(scaleBmm2);
    params.scale_softmax = reinterpret_cast<uint32_t const&>(scaleSoftmax);

    params.qkv_stride_in_bytes = total;
    params.o_stride_in_bytes = total;

    params.use_int8_scale_max = mUseInt8ScaleMax;
    params.enable_i2f_trick
        = -double(1 << 22) * double(scaleBmm2) <= -128.F && double(1 << 22) * double(scaleBmm2) >= 127.F;

    try
    {
        mXmmaKernel->run(params, stream);
        return cudaPeekAtLastError();
    }
    catch (std::exception const& e)
    {
        caughtError(e);
        return -1;
    }
}

QKVToContextInterleavedPluginLegacyCreator::QKVToContextInterleavedPluginLegacyCreator()
{
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("hidden_size", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("num_heads", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("dq_probs", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("use_int8_scale_max", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("use_explicit_int8", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("input_qkv_scale", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("output_ctx_scale", nullptr, PluginFieldType::kFLOAT32, 1));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

char const* QKVToContextInterleavedPluginLegacyCreator::getPluginName() const noexcept
{
    return kQKV_TO_CONTEXT_INTERLEAVED_PLUGIN_LEGACY_NAME;
}

char const* QKVToContextInterleavedPluginLegacyCreator::getPluginVersion() const noexcept
{
    return kQKV_TO_CONTEXT_INTERLEAVED_PLUGIN_LEGACY_VERSION;
}

PluginFieldCollection const* QKVToContextInterleavedPluginLegacyCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* QKVToContextInterleavedPluginLegacyCreator::createPlugin(
    char const* name, PluginFieldCollection const* fc) noexcept
{
    try
    {
        int32_t hiddenSize = 0;
        // Since numHeads must always exist or validateRequiredAttributes will fail,
        // we can set numHeads to -1 so that static analysis tools don't warn about
        // a division by zero in QKVToContextInterleavedPluginLegacy constructor.
        int32_t numHeads{-1};

        float dqProbs = -1;
        int32_t useInt8ScaleMax{-1};

        int32_t useExplicitInt8{};
        float qkvScale{1.F};
        float ctxScale{1.F};

        plugin::validateRequiredAttributesExist({"hidden_size", "num_heads"}, fc);

        for (int32_t i = 0; i < fc->nbFields; i++)
        {
            std::string field_name(fc->fields[i].name);

            if (field_name.compare("hidden_size") == 0)
            {
                hiddenSize = *static_cast<int32_t const*>(fc->fields[i].data);
                PLUGIN_VALIDATE(hiddenSize > 0, ("QKV: Invalid hiddenSize " + std::to_string(hiddenSize)).c_str());
                BERT_DEBUG_VALUE("Building hiddenSize: ", hiddenSize);
            }
            else if (field_name.compare("num_heads") == 0)
            {
                numHeads = *static_cast<int32_t const*>(fc->fields[i].data);
                PLUGIN_VALIDATE(numHeads > 0, ("QKV: Invalid numHeads " + std::to_string(numHeads)).c_str());
                BERT_DEBUG_VALUE("Building numHeads: ", numHeads);
            }
            else if (field_name.compare("dq_probs") == 0)
            {
                dqProbs = *static_cast<float const*>(fc->fields[i].data);
                PLUGIN_VALIDATE(dqProbs > 0.0F, ("QKV: Invalid dqProbs " + std::to_string(dqProbs)).c_str());
                BERT_DEBUG_VALUE("Building dqProbs: ", dqProbs);
            }
            else if (field_name.compare("use_int8_scale_max") == 0)
            {
                useInt8ScaleMax = *static_cast<int32_t const*>(fc->fields[i].data);
                PLUGIN_VALIDATE(useInt8ScaleMax == 0 || useInt8ScaleMax == 1,
                    ("QKV: Invalid useInt8ScaleMax " + std::to_string(useInt8ScaleMax)).c_str());
                BERT_DEBUG_VALUE("Building useInt8ScaleMax: ", useInt8ScaleMax);
            }
            else if (field_name.compare("use_explicit_int8") == 0)
            {
                useExplicitInt8 = *static_cast<int32_t const*>(fc->fields[i].data);
                BERT_DEBUG_VALUE("Building use_explicit_int8: ", useExplicitInt8);
            }
            else if (field_name.compare("input_qkv_scale") == 0)
            {
                qkvScale = *static_cast<float const*>(fc->fields[i].data);
                PLUGIN_VALIDATE(qkvScale > 0, ("QKV: Invalid input_qkv_scale" + std::to_string(qkvScale)).c_str());
                BERT_DEBUG_VALUE("Building input_qkv_scale: ", qkvScale);
            }
            else if (field_name.compare("output_ctx_scale") == 0)
            {
                ctxScale = *static_cast<float const*>(fc->fields[i].data);
                PLUGIN_VALIDATE(ctxScale > 0, ("QKV: Invalid output_ctx_scale " + std::to_string(ctxScale)).c_str());
                BERT_DEBUG_VALUE("Building output_ctx_scale: ", ctxScale);
            }
        }

        if (dqProbs < 0)
        {
            gLogInfo << "Using default scale factor\n";
            dqProbs = 1.F / 127.F;
        }

        if (useInt8ScaleMax < 0)
        {
            gLogInfo << "Using default for use_int8_scale_max: true" << std::endl;
            useInt8ScaleMax = 1;
        }

        auto const useInt8ScaleMaxFlag = static_cast<bool>(useInt8ScaleMax);

        QKVToContextInterleavedPluginLegacy* p = new QKVToContextInterleavedPluginLegacy(
            name, hiddenSize, numHeads, dqProbs, useInt8ScaleMaxFlag, useExplicitInt8 != 0, qkvScale, ctxScale);
        return p;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* QKVToContextInterleavedPluginLegacyCreator::deserializePlugin(
    char const* name, void const* serialData, size_t serialLength) noexcept
{
    try
    {
        // This object will be deleted when the network is destroyed, which will
        // call QKVToContextInterleavedPluginLegacy::destroy() noexcept
        return new QKVToContextInterleavedPluginLegacy(name, serialData, serialLength);
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

void QKVToContextInterleavedPluginLegacyCreator::setPluginNamespace(char const* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

char const* QKVToContextInterleavedPluginLegacyCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}
