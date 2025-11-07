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

#include "qkvToContextInt8InterleavedPlugin.h"
#include "NvInfer.h"
#include "common/bertCommon.h"
#include "common/plugin.h"
#include "common/serialize.hpp"

#include <cstring>
#include <cuda.h>
#include <iostream>
#include <optional>
#include <tuple>
#include <vector>

#include "bertQKVToContextPlugin/fused_multihead_attention_v2/fused_multihead_attention_v2.h"

using namespace nvinfer1;
using namespace nvinfer1::plugin;
using namespace nvinfer1::plugin::bert;

namespace
{
char const* const kQKV_TO_CONTEXT_INTERLEAVED_PLUGIN_VERSION{"6"};
char const* const kQKV_TO_CONTEXT_INTERLEAVED_PLUGIN_NAME{"CustomQKVToContextPluginDynamic"};
} // namespace

REGISTER_TENSORRT_PLUGIN(QKVToContextInterleavedPluginCreator);

constexpr uint32_t kIIDX = 0; // index of the input tensor

QKVToContextInterleavedPlugin::QKVToContextInterleavedPlugin(std::string const& name, int32_t hiddenSize,
    int32_t numHeads, float dqProbs, bool useInt8ScaleMax, bool useExplicitInt8, float qkvScale, float ctxScale)
    : mLayerName(name)
    , mHeadSize(hiddenSize / numHeads)
    , mHiddenSize(hiddenSize)
    , mNumHeads(numHeads)
    , mDqProbs(dqProbs)
    , mQkvScale(qkvScale)
    , mCtxScale(ctxScale)
{
    mSM = getSmVersion();
    mUseInt8ScaleMax = static_cast<int32_t>(useInt8ScaleMax);
    mUseExplicitInt8 = static_cast<int32_t>(useExplicitInt8);
    // variable sequence length is only supported with the fused MHA kernels
    // we should not override mS!
    bool isSMSupported = elem(mSM,
        {kSM_AMPERE_100, kSM_AMPERE_10X, kSM_AMPERE_10B, kSM_TURING, kSM_XAVIER, kSM_ADA_10X, kSM_HOPPER_100,
            kSM_BLACKWELL_100, kSM_BLACKWELL_120});
    PLUGIN_VALIDATE(isSMSupported && "requesting maxSeqlen not compatible with GPU arch");
    // the layout changes: SxB will be a combined \sum_i s_i and hdim will be the 2nd dimension instead of the third
    mXmmaKernel = getXMMAKernelsV2(DATA_TYPE_INT8, mSM);
}

QKVToContextInterleavedPlugin::~QKVToContextInterleavedPlugin() {}

IPluginV3* QKVToContextInterleavedPlugin::attachToContext(IPluginResourceContext* context) noexcept
{
    return clone();
}

IPluginCapability* QKVToContextInterleavedPlugin::getCapabilityInterface(PluginCapabilityType type) noexcept
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

IPluginV3* QKVToContextInterleavedPlugin::clone() noexcept
{
    try
    {
        QKVToContextInterleavedPlugin* ret = new QKVToContextInterleavedPlugin(
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

int32_t QKVToContextInterleavedPlugin::getOutputShapes(DimsExprs const* inputs, int32_t nbInputs,
    DimsExprs const* shapeInputs, int32_t nbShapeInputs, DimsExprs* outputs, int32_t nbOutputs,
    IExprBuilder& exprBuilder) noexcept
{
    try
    {
        // Input SHAPE is 1x(3*N*H)xTotalx1 (NCHW)
        // Output SHAPE is 1x(N*H)xTotalx1
        // In SupportsFormatCombination, we force the layout to be CHW, i.e.
        // Input: 3xNx(H/32)xsumSx32, Output: 1xNx(H/32)xsumSx32
        PLUGIN_ASSERT(inputs != nullptr);
        PLUGIN_ASSERT(nbInputs == 3);
        PLUGIN_ASSERT(nbShapeInputs == 0);
        PLUGIN_ASSERT(outputs != nullptr);
        PLUGIN_ASSERT(nbOutputs == 1);
        outputs[kIIDX] = inputs[kIIDX];
        // Divide last dim by three
        auto const* three = exprBuilder.constant(3);
        outputs[kIIDX].d[1] = exprBuilder.operation(DimensionOperation::kFLOOR_DIV, *inputs[kIIDX].d[1], *three);
        return pluginStatus_t::STATUS_SUCCESS;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return pluginStatus_t::STATUS_FAILURE;
}

bool QKVToContextInterleavedPlugin::supportsFormatCombination(
    int32_t pos, DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t /*nbOutputs*/) noexcept
{
    PLUGIN_ASSERT(pos >= 0);
    PLUGIN_ASSERT(nbInputs == 3);
    PLUGIN_ASSERT(pos <= nbInputs);
    PLUGIN_ASSERT(inOut != nullptr);
    // 3 inputs:
    // 0: qkv
    // 1: cu_seqlens
    // 2: dummy
    // 1 output
    if (pos == 0 || pos == nbInputs)
    {
        return (inOut[pos].desc.type == DataType::kINT8) && (inOut[pos].desc.format == TensorFormat::kCHW32);
    }

    if (pos == 1)
    {
        // cuSeqlens is a int32_t array of size B+1
        auto const* seqlens = &inOut[pos].desc;
        return (seqlens->type == DataType::kINT32) && (seqlens->format == TensorFormat::kLINEAR);
    }
    if (pos == 2)
    {
        // this is the dummy input
        return inOut[pos].desc.dims.nbDims == 1;
    }
    return false;
}

int32_t QKVToContextInterleavedPlugin::onShapeChange(
    PluginTensorDesc const* in, int32_t nbInputs, PluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
    return pluginStatus_t::STATUS_SUCCESS;
}

int32_t QKVToContextInterleavedPlugin::configurePlugin(
    DynamicPluginTensorDesc const* in, int32_t nbInputs, DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
    return pluginStatus_t::STATUS_SUCCESS;
}

size_t QKVToContextInterleavedPlugin::getWorkspaceSize(DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
    DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept
{
    return 0;
}

int32_t QKVToContextInterleavedPlugin::getOutputDataTypes(
    DataType* outputTypes, int32_t nbOutputs, DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    try
    {
        PLUGIN_ASSERT(nbOutputs == 1);
        outputTypes[0] = DataType::kINT8;
        return pluginStatus_t::STATUS_SUCCESS;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return pluginStatus_t::STATUS_FAILURE;
}

char const* QKVToContextInterleavedPlugin::getPluginVersion() const noexcept
{
    return kQKV_TO_CONTEXT_INTERLEAVED_PLUGIN_VERSION;
}

int32_t QKVToContextInterleavedPlugin::getNbOutputs() const noexcept
{
    return 1;
}

void QKVToContextInterleavedPlugin::setPluginNamespace(char const* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

char const* QKVToContextInterleavedPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

char const* QKVToContextInterleavedPlugin::getPluginName() const noexcept
{
    return kQKV_TO_CONTEXT_INTERLEAVED_PLUGIN_NAME;
}

int32_t QKVToContextInterleavedPlugin::enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc,
    void const* const* inputs, void* const* outputs, void* /* workspace */, cudaStream_t stream) noexcept
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

PluginFieldCollection const* QKVToContextInterleavedPlugin::getFieldsToSerialize() noexcept
{
    mDataToSerialize.clear();

    mDataToSerialize.emplace_back("hidden_size", &mHiddenSize, PluginFieldType::kINT32, 1);
    mDataToSerialize.emplace_back("num_heads", &mNumHeads, PluginFieldType::kINT32, 1);
    mDataToSerialize.emplace_back("use_int8_scale_max", &mUseInt8ScaleMax, PluginFieldType::kINT32, 1);
    mDataToSerialize.emplace_back("use_explicit_int8", &mUseExplicitInt8, PluginFieldType::kINT32, 1);
    mDataToSerialize.emplace_back("input_qkv_scale", &mQkvScale, PluginFieldType::kFLOAT32, 1);
    mDataToSerialize.emplace_back("output_ctx_scale", &mCtxScale, PluginFieldType::kFLOAT32, 1);

    if (mDqProbs >= 0)
    {
        mDataToSerialize.emplace_back("dq_probs", &mDqProbs, PluginFieldType::kFLOAT32, 1);
    }

    mFCToSerialize.nbFields = mDataToSerialize.size();
    mFCToSerialize.fields = mDataToSerialize.data();

    return &mFCToSerialize;
}

///////////////////////// Creator methods ////////////////////////

QKVToContextInterleavedPluginCreator::QKVToContextInterleavedPluginCreator()
{
    static std::mutex sMutex;
    std::lock_guard<std::mutex> lock(sMutex);
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

char const* QKVToContextInterleavedPluginCreator::getPluginName() const noexcept
{
    return kQKV_TO_CONTEXT_INTERLEAVED_PLUGIN_NAME;
}

char const* QKVToContextInterleavedPluginCreator::getPluginVersion() const noexcept
{
    return kQKV_TO_CONTEXT_INTERLEAVED_PLUGIN_VERSION;
}

PluginFieldCollection const* QKVToContextInterleavedPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV3* QKVToContextInterleavedPluginCreator::createPlugin(
    char const* name, PluginFieldCollection const* fc, TensorRTPhase phase) noexcept
{
    try
    {
        // Since numHeads must always exist or validateRequiredAttributes will fail,
        // we can set numHeads to -1 so that static analysis tools don't warn about
        // a division by zero in QKVToContextInterleavedPlugin constructor.
        int32_t numHeads{-1};
        int32_t hiddenSize{0};
        std::optional<int32_t> useInt8ScaleMax;
        std::optional<int32_t> useExplicitInt8;
        std::optional<float> qkvScale;
        std::optional<float> ctxScale;
        std::optional<float> dqProbs;

        if (phase == TensorRTPhase::kBUILD)
        {

            plugin::validateRequiredAttributesExist({"hidden_size", "num_heads"}, fc);
        }
        else
        {
            PLUGIN_ASSERT(phase == TensorRTPhase::kRUNTIME);
            plugin::validateRequiredAttributesExist({"hidden_size", "num_heads", "use_int8_scale_max",
                                                        "use_explicit_int8", "input_qkv_scale", "output_ctx_scale"},
                fc);
        }

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
                PLUGIN_VALIDATE(
                    dqProbs.value() > 0.0F, ("QKV: Invalid dqProbs " + std::to_string(dqProbs.value())).c_str());
                BERT_DEBUG_VALUE("Building dqProbs: ", dqProbs.value());
            }
            else if (field_name.compare("use_int8_scale_max") == 0)
            {
                useInt8ScaleMax = *static_cast<int32_t const*>(fc->fields[i].data);
                PLUGIN_VALIDATE(useInt8ScaleMax.value() == 0 || useInt8ScaleMax.value() == 1,
                    ("QKV: Invalid useInt8ScaleMax " + std::to_string(useInt8ScaleMax.value())).c_str());
                BERT_DEBUG_VALUE("Building useInt8ScaleMax: ", useInt8ScaleMax.value());
            }
            else if (field_name.compare("use_explicit_int8") == 0)
            {
                useExplicitInt8 = *static_cast<int32_t const*>(fc->fields[i].data);
                PLUGIN_VALIDATE(useExplicitInt8.value() == 0 || useExplicitInt8.value() == 1,
                    ("QKV: Invalid useExplicitInt8 " + std::to_string(useExplicitInt8.value())).c_str());
                BERT_DEBUG_VALUE("Building use_explicit_int8: ", useExplicitInt8.value());
            }
            else if (field_name.compare("input_qkv_scale") == 0)
            {
                qkvScale = *static_cast<float const*>(fc->fields[i].data);
                PLUGIN_VALIDATE(
                    qkvScale.value() > 0, ("QKV: Invalid input_qkv_scale" + std::to_string(qkvScale.value())).c_str());
                BERT_DEBUG_VALUE("Building input_qkv_scale: ", qkvScale.value());
            }
            else if (field_name.compare("output_ctx_scale") == 0)
            {
                ctxScale = *static_cast<float const*>(fc->fields[i].data);
                PLUGIN_VALIDATE(ctxScale.value() > 0,
                    ("QKV: Invalid output_ctx_scale " + std::to_string(ctxScale.value())).c_str());
                BERT_DEBUG_VALUE("Building output_ctx_scale: ", ctxScale.value());
            }
        }

        if (!dqProbs.has_value())
        {
            gLogInfo << "Using default scale factor: 1.F/127.F" << std::endl;
            dqProbs = 1.F / 127.F;
        }
        if (!useInt8ScaleMax.has_value())
        {
            gLogInfo << "Using default for use_int8_scale_max: 1" << std::endl;
            useInt8ScaleMax = 1;
        }
        if (!useExplicitInt8.has_value())
        {
            gLogInfo << "Using default for use_explicit_int8: 0" << std::endl;
            useExplicitInt8 = 0;
        }
        if (!qkvScale.has_value())
        {
            gLogInfo << "Using default for qkvScale: 1.F" << std::endl;
            qkvScale = 1.F;
        }
        if (!ctxScale.has_value())
        {
            gLogInfo << "Using default for ctxScale: 1.F" << std::endl;
            ctxScale = 1.F;
        }

        return new QKVToContextInterleavedPlugin(name, hiddenSize, numHeads, dqProbs.value(),
            useInt8ScaleMax.value() != 0, useExplicitInt8.value() != 0, qkvScale.value(), ctxScale.value());
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

void QKVToContextInterleavedPluginCreator::setPluginNamespace(char const* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

char const* QKVToContextInterleavedPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}
