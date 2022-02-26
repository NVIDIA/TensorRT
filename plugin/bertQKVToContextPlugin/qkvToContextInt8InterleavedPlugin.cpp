/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

#include "NvInfer.h"
#include "bertCommon.h"
#include "qkvToContextInt8InterleavedPlugin.h"
#include "serialize.hpp"

#include <cassert>
#include <cstring>
#include <cuda.h>
#include <iostream>
#include <tuple>
#include <vector>

#include <fused_multihead_attention_v2.h>

using namespace nvinfer1;
// using namespace fused_multihead_attention;

namespace bert
{

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

QKVToContextInterleavedPlugin::QKVToContextInterleavedPlugin(
    const std::string name, const int hiddenSize, const int numHeads, const float dqProbs)
    : mLayerName(name)
    , mS(0)
    , mB(0)
    , mHeadSize(hiddenSize / numHeads)
    , mHiddenSize(hiddenSize)
    , mNumHeads(numHeads)
    , mDqProbs(dqProbs)

{
    mSM = getSMVersion();
    // variable sequence length is only supported with the fused MHA kernels
    // we should not override mS!
    assert((mSM == kSM_AMPERE_100 || mSM == kSM_AMPERE_10X || mSM == kSM_TURING || mSM == kSM_XAVIER)
        && "requesting maxSeqlen not compatible with GPU arch");
    // the layout changes: SxB will be a combined \sum_i s_i and hdim will be the 2nd dimension instead of the third
    mXmmaKernel = getXMMAKernelsV2(DATA_TYPE_INT8, mSM);
}

QKVToContextInterleavedPlugin::QKVToContextInterleavedPlugin(const std::string name, const void* data, size_t length)
    : mLayerName(name)
{
    deserialize_value(&data, &length, &mNumHeads);
    deserialize_value(&data, &length, &mHeadSize);
    deserialize_value(&data, &length, &mHiddenSize);
    deserialize_value(&data, &length, &mSM);
    deserialize_value(&data, &length, &mS);
    deserialize_value(&data, &length, &mB);
    deserialize_value(&data, &length, &mDqProbs);
}

int QKVToContextInterleavedPlugin::getSMVersion() const noexcept
{
    int device{-1};
    CHECK(cudaGetDevice(&device));
    cudaDeviceProp props;
    CHECK(cudaGetDeviceProperties(&props, device));
    return props.major * 10 + props.minor;
}

// IPluginV2DynamicExt Methods
nvinfer1::IPluginV2DynamicExt* QKVToContextInterleavedPlugin::clone() const noexcept
{
    QKVToContextInterleavedPlugin* ret
        = new QKVToContextInterleavedPlugin(mLayerName, mHiddenSize, mNumHeads, mDqProbs);

    ret->setPluginNamespace(mNamespace.c_str());
    return ret;
}

DimsExprs QKVToContextInterleavedPlugin::getOutputDimensions(
    int outputIndex, const DimsExprs* inputs, int nbInputs, IExprBuilder& exprBuilder) noexcept
{
    // Input SHAPE is 1x(3*N*H)xTotalx1 (NCHW)
    // Output SHAPE is 1x(N*H)xTotalx1
    // In SupportsFormatCombination, we force the layout to be CHW, i.e.
    // Input: 3xNx(H/32)xsumSx32, Output: 1xNx(H/32)xsumSx32
    assert(outputIndex == 0);
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
    assert(nbInputs == 3);
    assert(nbOutputs == 1);
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
    assert(index == 0);
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
        + sizeof(mDqProbs);
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

    params.use_int8_scale_max = true;
    params.enable_i2f_trick
        = -double(1 << 22) * double(scaleBmm2) <= -128.F && double(1 << 22) * double(scaleBmm2) >= 127.F;

    mXmmaKernel->run(params, stream);
    return cudaPeekAtLastError();
}

QKVToContextInterleavedPluginCreator::QKVToContextInterleavedPluginCreator()
{
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("hidden_size", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("num_heads", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("dq_probs", nullptr, PluginFieldType::kFLOAT32, 1));

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
    int hiddenSize = 0;
    int numHeads = 0;

    float dqProbs = -1;

    for (int i = 0; i < fc->nbFields; i++)
    {
        std::string field_name(fc->fields[i].name);

        if (field_name.compare("hidden_size") == 0)
        {
            hiddenSize = *static_cast<const int*>(fc->fields[i].data);
            BERT_DEBUG_VALUE("Building hiddenSize: ", hiddenSize);
        }
        if (field_name.compare("num_heads") == 0)
        {
            numHeads = *static_cast<const int*>(fc->fields[i].data);
            BERT_DEBUG_VALUE("Building numHeads: ", numHeads);
        }
        if (field_name.compare("dq_probs") == 0)
        {
            dqProbs = *static_cast<const float*>(fc->fields[i].data);
            BERT_DEBUG_VALUE("Building dqProbs: ", dqProbs);
        }
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

    if (dqProbs < 0)
    {
        gLogInfo << "Using default scale factor\n";
        dqProbs = 1.F / 127.F;
    }

    QKVToContextInterleavedPlugin* p = new QKVToContextInterleavedPlugin(name, hiddenSize, numHeads, dqProbs);
    return p;
}

IPluginV2* QKVToContextInterleavedPluginCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength)  noexcept
{
    // This object will be deleted when the network is destroyed, which will
    // call QKVToContextInterleavedPlugin::destroy() noexcept
    return new QKVToContextInterleavedPlugin(name, serialData, serialLength);
}

void QKVToContextInterleavedPluginCreator::setPluginNamespace(const char* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

const char* QKVToContextInterleavedPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}
} // namespace bert
