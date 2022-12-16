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

#include "qkvToContextInt8InterleavedPlugin.h"
#include "NvInfer.h"
#include "common/bertCommon.h"
#include "common/plugin.h"
#include "common/serialize.hpp"
#include <cstring>
#include <cuda.h>
#include <iostream>
#include <tuple>
#include <vector>

#include "bertQKVToContextPlugin/fused_multihead_attention_v2/include/fused_multihead_attention_v2.h"

using namespace nvinfer1;
using namespace nvinfer1::plugin;
using namespace nvinfer1::plugin::bert;

namespace
{
const char* QKV_TO_CONTEXT_INTERLEAVED_PLUGIN_VERSION{"3"};
const char* QKV_TO_CONTEXT_INTERLEAVED_PLUGIN_NAME{"CustomQKVToContextPluginDynamic"};
} // namespace

// Static class fields initialization
PluginFieldCollection QKVToContextInterleavedPluginCreator::mFC{};
std::vector<PluginField> QKVToContextInterleavedPluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(QKVToContextInterleavedPluginCreator);

constexpr uint32_t IIDX = 0; // index of the input tensor

QKVToContextInterleavedPlugin::QKVToContextInterleavedPlugin(std::string const& name, int32_t const hiddenSize,
    int32_t const numHeads, float const dqProbs, bool const useInt8ScaleMax)
    : mLayerName(name)
    , mS(0)
    , mB(0)
    , mHeadSize(hiddenSize / numHeads)
    , mHiddenSize(hiddenSize)
    , mNumHeads(numHeads)
    , mDqProbs(dqProbs)
    , mUseInt8ScaleMax(useInt8ScaleMax)
{
    mSM = getSMVersion();
    // variable sequence length is only supported with the fused MHA kernels
    // we should not override mS!
    PLUGIN_VALIDATE((mSM == kSM_AMPERE_100 || mSM == kSM_AMPERE_10X || mSM == kSM_AMPERE_10B || mSM == kSM_TURING
               || mSM == kSM_XAVIER || mSM == kSM_ADA_10X || mSM == kSM_HOPPER_100)
        && "requesting maxSeqlen not compatible with GPU arch");
    // the layout changes: SxB will be a combined \sum_i s_i and hdim will be the 2nd dimension instead of the third
    mXmmaKernel = getXMMAKernelsV2(DATA_TYPE_INT8, mSM);
}

QKVToContextInterleavedPlugin::QKVToContextInterleavedPlugin(std::string const& name, void const* data, size_t length)
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
}

int QKVToContextInterleavedPlugin::getSMVersion() const noexcept
{
    int device{-1};
    PLUGIN_CHECK(cudaGetDevice(&device));
    cudaDeviceProp props;
    PLUGIN_CHECK(cudaGetDeviceProperties(&props, device));
    return getTrtSMVersionDec(props.major, props.minor);
}

// IPluginV2DynamicExt Methods
nvinfer1::IPluginV2DynamicExt* QKVToContextInterleavedPlugin::clone() const noexcept
{
    try
    {
        QKVToContextInterleavedPlugin* ret
            = new QKVToContextInterleavedPlugin(mLayerName, mHiddenSize, mNumHeads, mDqProbs, mUseInt8ScaleMax);

        ret->setPluginNamespace(mNamespace.c_str());
        return ret;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

DimsExprs QKVToContextInterleavedPlugin::getOutputDimensions(
    int outputIndex, const DimsExprs* inputs, int nbInputs, IExprBuilder& exprBuilder) noexcept
{
    // Input SHAPE is 1x(3*N*H)xTotalx1 (NCHW)
    // Output SHAPE is 1x(N*H)xTotalx1
    // In SupportsFormatCombination, we force the layout to be CHW, i.e.
    // Input: 3xNx(H/32)xsumSx32, Output: 1xNx(H/32)xsumSx32
    PLUGIN_ASSERT(outputIndex == 0);
    // Copy over everything
    DimsExprs output(inputs[IIDX]);
    // output.d[0] = exprBuilder.constant(1);
    // Divide last dim by three
    const auto* three = exprBuilder.constant(3);
    output.d[1] = exprBuilder.operation(DimensionOperation::kFLOOR_DIV, *inputs[IIDX].d[1], *three);
    return output;
}
bool QKVToContextInterleavedPlugin::supportsFormatCombination(
    int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept
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
        // cuSeqlens is a int array of size B+1
        const auto* seqlens = &inOut[pos];
        return (seqlens->type == DataType::kINT32) && (seqlens->format == TensorFormat::kLINEAR);
    }
    if (pos == 2)
    {
        // this is the dummy input
        return inOut[pos].dims.nbDims == 1;
    }
    return false;
}

void QKVToContextInterleavedPlugin::configurePlugin(
    const DynamicPluginTensorDesc* in, int nbInputs, const DynamicPluginTensorDesc* out, int nbOutputs) noexcept
{
}

size_t QKVToContextInterleavedPlugin::getWorkspaceSize(
    const PluginTensorDesc* inputs, int nbInputs, const PluginTensorDesc* outputs, int nbOutputs) const noexcept
{
    return 0;
}

// IPluginV2Ext Methods
DataType QKVToContextInterleavedPlugin::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept
{
    PLUGIN_ASSERT(index == 0);
    return DataType::kINT8;
}

// IPluginV2 Methods
const char* QKVToContextInterleavedPlugin::getPluginType() const noexcept
{
    return QKV_TO_CONTEXT_INTERLEAVED_PLUGIN_NAME;
}

const char* QKVToContextInterleavedPlugin::getPluginVersion() const noexcept
{
    return QKV_TO_CONTEXT_INTERLEAVED_PLUGIN_VERSION;
}

int QKVToContextInterleavedPlugin::getNbOutputs() const noexcept
{
    return 1;
}

int QKVToContextInterleavedPlugin::initialize() noexcept
{
    return 0;
}

void QKVToContextInterleavedPlugin::terminate() noexcept {}

size_t QKVToContextInterleavedPlugin::getSerializationSize() const noexcept
{
    return sizeof(mNumHeads) + sizeof(mHeadSize) + sizeof(mHiddenSize) + sizeof(mSM) + sizeof(mS) + sizeof(mB)
        + sizeof(mDqProbs) + sizeof(mUseInt8ScaleMax);
}

void QKVToContextInterleavedPlugin::serialize(void* buffer) const noexcept
{
    serialize_value(&buffer, mNumHeads);
    serialize_value(&buffer, mHeadSize);
    serialize_value(&buffer, mHiddenSize);
    serialize_value(&buffer, mSM);
    serialize_value(&buffer, mS);
    serialize_value(&buffer, mB);
    serialize_value(&buffer, mDqProbs);
    serialize_value(&buffer, mUseInt8ScaleMax);
}

void QKVToContextInterleavedPlugin::destroy() noexcept
{
    delete this;
}

void QKVToContextInterleavedPlugin::setPluginNamespace(const char* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

const char* QKVToContextInterleavedPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

int QKVToContextInterleavedPlugin::enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc,
    const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{

    const int total = inputDesc[0].dims.d[2];
    const int B = inputDesc[1].dims.d[0] - 1;
    const int maxS = inputDesc[2].dims.d[0];
    int S = 384;
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
    params.cu_seqlens = static_cast<int*>(const_cast<void*>(inputs[1]));

    float scaleQkv = inputDesc[0].scale;
    float scaleCtx = outputDesc[0].scale;

    float scaleBmm1 = scaleQkv * scaleQkv * 0.125; // 1 / sqrt(64)
    float scaleBmm2 = mDqProbs * scaleQkv / scaleCtx;
    float scaleSoftmax = 1.F / mDqProbs;

    params.scale_bmm1 = reinterpret_cast<const uint32_t&>(scaleBmm1);
    params.scale_bmm2 = reinterpret_cast<const uint32_t&>(scaleBmm2);
    params.scale_softmax = reinterpret_cast<const uint32_t&>(scaleSoftmax);

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

QKVToContextInterleavedPluginCreator::QKVToContextInterleavedPluginCreator()
{
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("hidden_size", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("num_heads", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("dq_probs", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("use_int8_scale_max", nullptr, PluginFieldType::kINT32, 1));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* QKVToContextInterleavedPluginCreator::getPluginName() const noexcept
{
    return QKV_TO_CONTEXT_INTERLEAVED_PLUGIN_NAME;
}

const char* QKVToContextInterleavedPluginCreator::getPluginVersion() const noexcept
{
    return QKV_TO_CONTEXT_INTERLEAVED_PLUGIN_VERSION;
}

const PluginFieldCollection* QKVToContextInterleavedPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* QKVToContextInterleavedPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
{
    try
    {
        int32_t hiddenSize = 0;
        // Since numHeads must always exist or validateRequiredAttributes will fail,
        // we can set numHeads to -1 so that static analysis tools don't warn about
        // a division by zero in QKVToContextInterleavedPlugin constructor.
        int32_t numHeads{-1};

        float dqProbs = -1;
        int32_t useInt8ScaleMax{-1};

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
            if (field_name.compare("num_heads") == 0)
            {
                numHeads = *static_cast<int32_t const*>(fc->fields[i].data);
                PLUGIN_VALIDATE(numHeads > 0, ("QKV: Invalid numHeads " + std::to_string(numHeads)).c_str());
                BERT_DEBUG_VALUE("Building numHeads: ", numHeads);
            }
            if (field_name.compare("dq_probs") == 0)
            {
                dqProbs = *static_cast<float const*>(fc->fields[i].data);
                PLUGIN_VALIDATE(dqProbs > 0.0F, ("QKV: Invalid dqProbs " + std::to_string(dqProbs)).c_str());
                BERT_DEBUG_VALUE("Building dqProbs: ", dqProbs);
            }
            if (field_name.compare("use_int8_scale_max") == 0)
            {
                useInt8ScaleMax = *static_cast<int32_t const*>(fc->fields[i].data);
                PLUGIN_VALIDATE(useInt8ScaleMax == 0 || useInt8ScaleMax == 1,
                    ("QKV: Invalid useInt8ScaleMax " + std::to_string(useInt8ScaleMax)).c_str());
                BERT_DEBUG_VALUE("Building useInt8ScaleMax: ", useInt8ScaleMax);
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

        QKVToContextInterleavedPlugin* p = new QKVToContextInterleavedPlugin(name, hiddenSize, numHeads, dqProbs, useInt8ScaleMaxFlag);
        return p;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* QKVToContextInterleavedPluginCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength)  noexcept
{
    try
    {
        // This object will be deleted when the network is destroyed, which will
        // call QKVToContextInterleavedPlugin::destroy() noexcept
        return new QKVToContextInterleavedPlugin(name, serialData, serialLength);
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

void QKVToContextInterleavedPluginCreator::setPluginNamespace(const char* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

const char* QKVToContextInterleavedPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}
