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
#include "qkvToContextPlugin.h"

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
char const* const kQKV_TO_CONTEXT_PLUGIN_VERSION{"4"};
char const* const kQKV_TO_CONTEXT_VAR_SEQLEN_PLUGIN_VERSION{"5"};
char const* const kQKV_TO_CONTEXT_PLUGIN_NAME{"CustomQKVToContextPluginDynamic"};
} // namespace

REGISTER_TENSORRT_PLUGIN(QKVToContextPluginDynamicCreator);

constexpr uint32_t kIIDX = 0; // index of the input tensor
constexpr uint32_t kMIDX = 1; // index of the mask

REGISTER_TENSORRT_PLUGIN(QKVToContextVarSeqlenPluginCreator);
QKVToContextPluginDynamic::~QKVToContextPluginDynamic() {}

QKVToContextPluginDynamic::QKVToContextPluginDynamic(const std::string name, const DataType type,
    const int32_t hiddenSize, const int32_t numHeads, float const dqProbs, bool hasImask)
    : mLayerName(name)
    , mS(0)
    , mB(0)
    , mHeadSize(hiddenSize / numHeads)
    , mHiddenSize(hiddenSize)
    , mNumHeads(numHeads)
    , mType(type)
    , mDqProbs(dqProbs)

{
    mHasImask = static_cast<int32_t>(hasImask);
    mSM = getSmVersion();
}

QKVToContextPluginDynamic::QKVToContextPluginDynamic(const std::string name, const DataType type, const int32_t S,
    const int32_t B, const int32_t SM, const int32_t hiddenSize, const int32_t numHeads, float const dqProbs,
    bool hasImask, bool hasUnfusedDispatcher, void const* runnerStateBuffer)
    : mLayerName(name)
    , mS(S)
    , mB(B)
    , mSM(SM)
    , mHeadSize(hiddenSize / numHeads)
    , mHiddenSize(hiddenSize)
    , mNumHeads(numHeads)
    , mType(type)
    , mDqProbs(dqProbs)

{
    BERT_DEBUG_MSG("MHA Runner Deser");
    mHasImask = static_cast<int32_t>(hasImask);
    mHasUnfusedDispatcher = static_cast<int32_t>(hasUnfusedDispatcher);

    createMHARunner();

    if (hasUnfusedDispatcher)
    {
        PLUGIN_ASSERT(unfusedDispatcher.get());
        PLUGIN_ASSERT(runnerStateBuffer != nullptr);
        auto length = unfusedDispatcher->getSerializationSize();
        unfusedDispatcher->deserialize(runnerStateBuffer, length);
    }

    BERT_DEBUG_MSG("MHA Runner Deser Done");
}


IPluginCapability* QKVToContextPluginDynamic::getCapabilityInterface(PluginCapabilityType type) noexcept
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

void QKVToContextPluginDynamic::createMHARunner()
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

IPluginV3* QKVToContextPluginDynamic::clone() noexcept
{
    BERT_DEBUG_MSG("QKV Clone");

    QKVToContextPluginDynamic* ret = nullptr;
    mHasUnfusedDispatcher = 0;
    char* bufferData = nullptr;
    // the workspacesize is 0 if we have not call setup the dispatcher yet.
    if (unfusedDispatcher.get() && unfusedDispatcher->getWorkspaceSize())
    {
        mHasUnfusedDispatcher = 1;
        mRunnerStateBuffer.resize(unfusedDispatcher->getSerializationSize());
        unfusedDispatcher->serialize(mRunnerStateBuffer.data());
        bufferData = mRunnerStateBuffer.data();
    }

    ret = new QKVToContextPluginDynamic(mLayerName, mType, mS, mB, mSM, mHiddenSize, mNumHeads, mDqProbs,
        static_cast<bool>(mHasImask), mHasUnfusedDispatcher, static_cast<void const*>(bufferData));
    ret->setPluginNamespace(mNamespace.c_str());
    BERT_DEBUG_MSG("QKV Clone done");
    return ret;
}

int32_t QKVToContextPluginDynamic::getOutputShapes(DimsExprs const* inputs, int32_t nbInputs,
    DimsExprs const* shapeInputs, int32_t nbShapeInputs, DimsExprs* outputs, int32_t nbOutputs,
    IExprBuilder& exprBuilder) noexcept
{
    try
    {
        PLUGIN_ASSERT(inputs != nullptr);
        PLUGIN_ASSERT(nbInputs == 1 + mHasImask);
        PLUGIN_ASSERT(nbShapeInputs == 0);
        PLUGIN_ASSERT(outputs != nullptr);
        PLUGIN_ASSERT(nbOutputs == 1);
        // Input is BxSx3*N*H, output should be BxSxN*H
        // Copy over everything
        outputs[kIIDX] = inputs[kIIDX];
        // Divide last dim by three
        auto const* three = exprBuilder.constant(3);
        outputs[kIIDX].d[HDIM] = exprBuilder.operation(DimensionOperation::kFLOOR_DIV, *inputs[kIIDX].d[HDIM], *three);
        return pluginStatus_t::STATUS_SUCCESS;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return pluginStatus_t::STATUS_FAILURE;
}

bool QKVToContextPluginDynamic::supportsFormatCombination(
    int32_t pos, DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t /*nbOutputs*/) noexcept
{
    PLUGIN_ASSERT(pos >= 0);
    PLUGIN_ASSERT(pos < 2 + mHasImask);
    PLUGIN_ASSERT(nbInputs == 1 + mHasImask);
    auto const* in = inOut;
    auto const* out = inOut + nbInputs;
    int32_t packedSize = getMHAMaskPackedSize(mSM, mType, in->desc.dims.d[SDIM]);

    // we only support int8 IO in fused mha runner, and we only support fused mha runner on Xavier, Turing and Ampere
    if (mType == DataType::kINT8)
    {
        if (!elem(mSM, {kSM_75, kSM_80, kSM_86, kSM_87, kSM_89, kSM_90, kSM_100, kSM_120}))
        {
            gLogError << "INT8 IO is only supported on Turing, Ampere, Hopper and Blackwell for plugin "
                      << kQKV_TO_CONTEXT_PLUGIN_NAME << std::endl;
            return false;
        }
        if (in->desc.dims.d[SDIM] == -1)
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
        bool isFormatSupported = in->desc.format == TensorFormat::kLINEAR;
        if (mType == DataType::kINT8)
        {
            if (in->desc.dims.d[HDIM] % 32U == 0)
            {
                isFormatSupported = in->desc.format == TensorFormat::kCHW32;
            }
            else
            {
                isFormatSupported = in->desc.format == TensorFormat::kCHW4;
            }
        }

        // must not check descriptions > pos
        return (in->desc.type == mType) &&         // precision
            isFormatSupported &&                   // format
            (in->desc.dims.nbDims == 5) &&         // num dims
            ((in->desc.dims.d[HDIM] % 3U) == 0) && // see getOutputDimensions
            ((in->desc.dims.d[3]) == 1) &&         // for fc
            ((in->desc.dims.d[4]) == 1)            // for fc
            ;
    }

    // pos==1
    if ((mHasImask && pos == 1)) // pos 1 is the mask
    {
        auto const* inMask = &inOut[1].desc;
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
            (inMask->dims.d[0] == in->desc.dims.d[BDIM]);
    }

    if (!mHasImask || pos == 2) // output pos
    {
        bool isFormatSupported = out->desc.format == TensorFormat::kLINEAR;
        if (mType == DataType::kINT8)
        {
            if (out->desc.dims.d[HDIM] % 32U == 0)
            {
                isFormatSupported = out->desc.format == TensorFormat::kCHW32;
            }
            else
            {
                isFormatSupported = out->desc.format == TensorFormat::kCHW4;
            }
        }

        return (in->desc.type == out->desc.type) &&                      // precision
            isFormatSupported &&                                         // format
            (out->desc.dims.nbDims == 5) &&                              // num dims
            ((in->desc.dims.d[HDIM] / 3) == (out->desc.dims.d[HDIM])) && // div 3
            ((out->desc.dims.d[3]) == 1) &&                              // for fc
            ((out->desc.dims.d[4]) == 1) &&                              // for fc
            ((out->desc.dims.d[BDIM]) == in->desc.dims.d[BDIM]) &&       // check B
            ((out->desc.dims.d[SDIM]) == in->desc.dims.d[SDIM])          // check S
            ;
    }

    return false;
}

int32_t QKVToContextPluginDynamic::onShapeChange(
    PluginTensorDesc const* in, int32_t nbInputs, PluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
    try
    {
        PLUGIN_ASSERT(in != nullptr);
        PLUGIN_ASSERT(nbInputs == 1 + mHasImask);
        PLUGIN_ASSERT(nbOutputs == 1);
        PluginTensorDesc const& inDesc = in[kIIDX];
        TRT_UNUSED inDesc;
        PLUGIN_ASSERT(out != nullptr);
        PluginTensorDesc const& outDesc = out[0];
        TRT_UNUSED outDesc;
        PLUGIN_ASSERT(mType == inDesc.type);
        PLUGIN_ASSERT(mType == outDesc.type);
        PLUGIN_ASSERT(inDesc.dims.d[BDIM] == outDesc.dims.d[BDIM]);
        PLUGIN_ASSERT(inDesc.dims.d[SDIM] == outDesc.dims.d[SDIM]);
        PLUGIN_ASSERT(inDesc.dims.d[HDIM] == 3 * outDesc.dims.d[HDIM]);
        if (mHasImask)
        {
            PluginTensorDesc const& maskDesc = in[kMIDX];
            TRT_UNUSED maskDesc;
            PLUGIN_ASSERT(maskDesc.dims.d[0] == inDesc.dims.d[BDIM]);
        }

        createMHARunner();

        // mS and mB that are set by configurePlugin() may be stale
        mS = inDesc.dims.d[SDIM];
        mB = inDesc.dims.d[BDIM];
        PLUGIN_ASSERT(mS);
        PLUGIN_ASSERT(mB);
        if (fusedDispatcher.get() && fusedDispatcher->isValid(mHeadSize, mS))
        {
            fusedDispatcher->setup(mS, mB, mHeadSize);
        }
        else
        {
            unfusedDispatcher->setup(mS, mB, mHeadSize);
        }

        return pluginStatus_t::STATUS_SUCCESS;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return pluginStatus_t::STATUS_FAILURE;
}

int32_t QKVToContextPluginDynamic::configurePlugin(
    DynamicPluginTensorDesc const* in, int32_t nbInputs, DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
    try
    {
        PLUGIN_ASSERT(in != nullptr);
        PLUGIN_ASSERT(nbInputs == 1 + mHasImask);
        PLUGIN_ASSERT(nbOutputs == 1);
        PluginTensorDesc const& inDesc = in[kIIDX].desc;
        TRT_UNUSED inDesc;
        PLUGIN_ASSERT(out != nullptr);
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
        return pluginStatus_t::STATUS_SUCCESS;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return pluginStatus_t::STATUS_FAILURE;
}

size_t QKVToContextPluginDynamic::getWorkspaceSize(DynamicPluginTensorDesc const* /*inputs*/, int32_t /*nbInputs*/,
    DynamicPluginTensorDesc const* /*outputs*/, int32_t /*nbOutputs*/) const noexcept
{
    // only unfused kernel need workspace, and we need larger workspace for larger sequence length
    // we have already setup unfusedDispatcher with max sequence in configurePlugin
    // if unfusedDispatcher is not initialized in configurePlugin
    PLUGIN_ASSERT(unfusedDispatcher.get());
    return unfusedDispatcher->getWorkspaceSize();
}

// IPluginV2Ext Methods
int32_t QKVToContextPluginDynamic::getOutputDataTypes(
    DataType* outputTypes, int32_t nbOutputs, DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    try
    {
        PLUGIN_ASSERT(
            inputTypes[0] == DataType::kFLOAT || inputTypes[0] == DataType::kHALF || inputTypes[0] == DataType::kINT8);
        outputTypes[0] = inputTypes[0];
        return pluginStatus_t::STATUS_SUCCESS;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return pluginStatus_t::STATUS_FAILURE;
}

void QKVToContextPluginDynamic::setCublasResources(std::shared_ptr<CublasWrapper> cublasWrapper)
{
    mCublasWrapper = cublasWrapper;
    // The shared cublasWrapper resource owns the handle.
    // but `this` instance has a non-owning pointer to the handle.
    // Note that the cublasWrapper inits the handle and checks for nullptr
    // so we don't have to do that here.
    mCublasHandle = mCublasWrapper->getCublasHandle();
}

IPluginV3* QKVToContextPluginDynamic::attachToContext(IPluginResourceContext* context) noexcept
{
    try
    {
        auto p = static_cast<QKVToContextPluginDynamic*>(clone());
        // the clone has shared ownership of underling cublasWrapper instance
        // that is mapped to current context
        p->setCublasResources(createPluginCublasWrapper(context));
        return p;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return nullptr;
}

char const* QKVToContextPluginDynamic::getPluginVersion() const noexcept
{
    return kQKV_TO_CONTEXT_PLUGIN_VERSION;
}

int32_t QKVToContextPluginDynamic::getNbOutputs() const noexcept
{
    return 1;
}

char const* QKVToContextPluginDynamic::getPluginName() const noexcept
{
    return kQKV_TO_CONTEXT_PLUGIN_NAME;
}


void QKVToContextPluginDynamic::setPluginNamespace(char const* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

char const* QKVToContextPluginDynamic::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

int32_t QKVToContextPluginDynamic::enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc,
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
                inputDesc[0], outputDesc[0], inputs[0], maskPtr, outputs[0], workspace, stream, mCublasHandle);
        }
        else
        {
            PLUGIN_VALIDATE(unfusedDispatcher.get(), "The Unfused MHARunner is uninitialized, no MHARunner available!");
            PLUGIN_VALIDATE(mType != DataType::kINT8, "The Unfused MHARunner does not support INT8!");
            unfusedDispatcher->run(
                inputDesc[0], outputDesc[0], inputs[0], maskPtr, outputs[0], workspace, stream, mCublasHandle);
        }
    }
    catch (std::exception const& e)
    {
        caughtError(e);
        return -1;
    }
    return 0;
}

PluginFieldCollection const* QKVToContextPluginDynamic::getFieldsToSerialize() noexcept
{
    mDataToSerialize.clear();

    mDataToSerialize.emplace_back("type_id", &mType, PluginFieldType::kINT32, 1);
    mDataToSerialize.emplace_back("hidden_size", &mHiddenSize, PluginFieldType::kINT32, 1);
    mDataToSerialize.emplace_back("num_heads", &mNumHeads, PluginFieldType::kINT32, 1);
    mDataToSerialize.emplace_back("has_mask", &mHasImask, PluginFieldType::kINT32, 1);
    mDataToSerialize.emplace_back("S", &mS, PluginFieldType::kINT32, 1);
    mDataToSerialize.emplace_back("B", &mB, PluginFieldType::kINT32, 1);
    mDataToSerialize.emplace_back("SM", &mSM, PluginFieldType::kINT32, 1);

    if (unfusedDispatcher.get() && unfusedDispatcher->getWorkspaceSize())
    {
        mHasUnfusedDispatcher = 1;
        mRunnerStateBuffer.resize(unfusedDispatcher->getSerializationSize());
        unfusedDispatcher->serialize(mRunnerStateBuffer.data());
        mDataToSerialize.emplace_back("runnerStateBuffer", (void const*) mRunnerStateBuffer.data(),
            PluginFieldType::kUNKNOWN, mRunnerStateBuffer.size());
    }
    else
    {
        mHasUnfusedDispatcher = 0;
    }

    mDataToSerialize.emplace_back("hasUnfusedDispatcher", &mHasUnfusedDispatcher, PluginFieldType::kINT32, 1);

    if (mDqProbs >= 0)
    {
        mDataToSerialize.emplace_back("dq_probs", &mDqProbs, PluginFieldType::kFLOAT32, 1);
    }

    mFCToSerialize.nbFields = mDataToSerialize.size();
    mFCToSerialize.fields = mDataToSerialize.data();

    return &mFCToSerialize;
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

char const* QKVToContextPluginDynamicCreator::getPluginName() const noexcept
{
    return kQKV_TO_CONTEXT_PLUGIN_NAME;
}

char const* QKVToContextPluginDynamicCreator::getPluginVersion() const noexcept
{
    return kQKV_TO_CONTEXT_PLUGIN_VERSION;
}

PluginFieldCollection const* QKVToContextPluginDynamicCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV3* QKVToContextPluginDynamicCreator::createPlugin(
    char const* name, PluginFieldCollection const* fc, TensorRTPhase phase) noexcept
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
        int32_t s = -1;
        int32_t b = -1;
        int32_t sm = -1;
        bool hasUnfusedDispatcher = false;
        void const* runnerStateBuffer = nullptr;
        float dqProbs = -1;

        PLUGIN_VALIDATE(fc->fields != nullptr);
        if (phase == TensorRTPhase::kBUILD)
        {
            plugin::validateRequiredAttributesExist({"type_id", "hidden_size", "num_heads", "has_mask"}, fc);
        }
        else
        {
            PLUGIN_ASSERT(phase == TensorRTPhase::kRUNTIME);
            plugin::validateRequiredAttributesExist(
                {"type_id", "S", "B", "hidden_size", "num_heads", "has_mask", "SM", "hasUnfusedDispatcher"}, fc);
        }

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
            else if (field_name.compare("hidden_size") == 0)
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
            else if (field_name.compare("has_mask") == 0)
            {
                auto hasMaskValue = *static_cast<int32_t const*>(fc->fields[i].data);
                PLUGIN_VALIDATE(hasMaskValue == 0 || hasMaskValue == 1,
                    ("QKV: Invalid hasMask " + std::to_string(hasMaskValue)).c_str());
                hasMask = static_cast<bool>(hasMaskValue);
                BERT_DEBUG_VALUE("Building hasMask: ", hasMask);
            }
            else if (field_name.compare("dq_probs") == 0)
            {
                dqProbs = *static_cast<float const*>(fc->fields[i].data);
                PLUGIN_VALIDATE(dqProbs > 0.0F, ("QKV: Invalid dqProbs " + std::to_string(dqProbs)).c_str());
                BERT_DEBUG_VALUE("Building dqProbs: ", dqProbs);
            }
            else if (field_name.compare("S") == 0)
            {
                PLUGIN_ASSERT(phase == TensorRTPhase::kRUNTIME);
                s = *static_cast<int32_t const*>(fc->fields[i].data);
                BERT_DEBUG_VALUE("Building S: ", s);
            }
            else if (field_name.compare("B") == 0)
            {
                PLUGIN_ASSERT(phase == TensorRTPhase::kRUNTIME);
                b = *static_cast<int32_t const*>(fc->fields[i].data);
                BERT_DEBUG_VALUE("Building B: ", b);
            }
            else if (field_name.compare("SM") == 0)
            {
                PLUGIN_ASSERT(phase == TensorRTPhase::kRUNTIME);
                sm = *static_cast<int32_t const*>(fc->fields[i].data);
                BERT_DEBUG_VALUE("Building SM: ", sm);
            }
            else if (field_name.compare("hasUnfusedDispatcher") == 0)
            {
                PLUGIN_ASSERT(phase == TensorRTPhase::kRUNTIME);
                auto hasUnfusedDispatcherValue = *static_cast<int32_t const*>(fc->fields[i].data);
                PLUGIN_VALIDATE(hasUnfusedDispatcherValue == 0 || hasUnfusedDispatcherValue == 1,
                    ("QKV: Invalid hasUnfusedDispatcher " + std::to_string(hasUnfusedDispatcherValue)).c_str());
                hasUnfusedDispatcher = static_cast<bool>(hasUnfusedDispatcherValue);
                BERT_DEBUG_VALUE("Building hasUnfusedDispatcher: ", hasUnfusedDispatcher);
            }
            else if (field_name.compare("runnerStateBuffer") == 0)
            {
                PLUGIN_ASSERT(phase == TensorRTPhase::kRUNTIME);
                runnerStateBuffer = static_cast<void const*>(fc->fields[i].data);
            }
        }

        BERT_DEBUG_MSG("Building the Plugin...");
        auto type = static_cast<DataType>(typeId);
        if (type == DataType::kINT8 && dqProbs < 0)
        {
            BERT_DEBUG_MSG("Using default scale factor");
            dqProbs = 1.F / 127.F;
        }

        if (phase == TensorRTPhase::kBUILD)
        {
            return new QKVToContextPluginDynamic(name, type, hiddenSize, numHeads, dqProbs, hasMask);
        }

        PLUGIN_VALIDATE(s != -1, "invalid S during runtime plugin creation");
        PLUGIN_VALIDATE(b != -1, "invalid B during runtime plugin creation");
        PLUGIN_VALIDATE(sm != -1, "invalid SM during runtime plugin creation");
        if (hasUnfusedDispatcher == 1)
        {
            PLUGIN_VALIDATE(runnerStateBuffer != nullptr, "invalid runnerStateBuffer during runtime plugin creation");
        }
        else
        {
            PLUGIN_VALIDATE(runnerStateBuffer == nullptr, "invalid runnerStateBuffer during runtime plugin creation");
        }

        return new QKVToContextPluginDynamic(
            name, type, s, b, sm, hiddenSize, numHeads, dqProbs, hasMask, hasUnfusedDispatcher, runnerStateBuffer);
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}


void QKVToContextPluginDynamicCreator::setPluginNamespace(char const* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

char const* QKVToContextPluginDynamicCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}


///// QKVToContextVarSeqlenPlugin (CustomQKVToContextPluginDynamic v5) ////

QKVToContextVarSeqlenPlugin::~QKVToContextVarSeqlenPlugin() {}

QKVToContextVarSeqlenPlugin::QKVToContextVarSeqlenPlugin(std::string const name, DataType const type,
    int32_t const hiddenSize, int32_t const numHeads, float const dqProbs, bool hasImask, bool varSeqlen,
    bool useInt8ScaleMax)
    : mLayerName(name)
    , mS(0)
    , mB(0)
    , mHeadSize(hiddenSize / numHeads)
    , mHiddenSize(hiddenSize)
    , mNumHeads(numHeads)
    , mType(type)
    , mDqProbs(dqProbs)
    , mHdim(HDIM)
{
    mSM = getSmVersion();
    mUseVarSeqlen = static_cast<int32_t>(varSeqlen);
    mUseInt8ScaleMax = static_cast<int32_t>(useInt8ScaleMax);
    mHasImask = static_cast<int32_t>(hasImask);

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

QKVToContextVarSeqlenPlugin::QKVToContextVarSeqlenPlugin(std::string const name, int32_t const S, int32_t const B,
    DataType const type, int32_t const hiddenSize, int32_t const numHeads, float const dqProbs, bool hasImask,
    bool varSeqlen, bool useInt8ScaleMax, void const* runnerStateBuffer)
    : mLayerName(name)
    , mS(S)
    , mB(B)
    , mHeadSize(hiddenSize / numHeads)
    , mHiddenSize(hiddenSize)
    , mNumHeads(numHeads)
    , mType(type)
    , mDqProbs(dqProbs)
    , mHdim(HDIM)
{
    mSM = getSmVersion();
    mUseVarSeqlen = static_cast<int32_t>(varSeqlen);
    mUseInt8ScaleMax = static_cast<int32_t>(useInt8ScaleMax);
    mHasImask = static_cast<int32_t>(hasImask);

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

    createMHARunner();

    PLUGIN_ASSERT(runnerStateBuffer != nullptr);
    auto length = mDispatcher->getSerializationSize();
    mDispatcher->deserialize(runnerStateBuffer, length);
}

IPluginCapability* QKVToContextVarSeqlenPlugin::getCapabilityInterface(PluginCapabilityType type) noexcept
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

void QKVToContextVarSeqlenPlugin::createMHARunner()
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


IPluginV3* QKVToContextVarSeqlenPlugin::clone() noexcept
{
    BERT_DEBUG_MSG("QKV Clone");

    QKVToContextVarSeqlenPlugin* ret = nullptr;

    char* bufferData = nullptr;
    // the workspacesize is 0 if we have not call setup the dispatcher yet.
    if (mDispatcher.get())
    {
        mRunnerStateBuffer.resize(mDispatcher->getSerializationSize());
        mDispatcher->serialize(mRunnerStateBuffer.data());
        bufferData = mRunnerStateBuffer.data();

        ret = new QKVToContextVarSeqlenPlugin(mLayerName, mS, mB, mType, mHiddenSize, mNumHeads, mDqProbs, mHasImask,
            mUseVarSeqlen, mUseInt8ScaleMax, static_cast<void const*>(bufferData));
    }
    else
    {
        // dispatcher not setup yet, use type 1 constructor
        ret = new QKVToContextVarSeqlenPlugin(
            mLayerName, mType, mHiddenSize, mNumHeads, mDqProbs, mHasImask, mUseVarSeqlen, mUseInt8ScaleMax);
    }

    ret->setPluginNamespace(mNamespace.c_str());
    BERT_DEBUG_MSG("QKV Clone done");
    return ret;
}

int32_t QKVToContextVarSeqlenPlugin::getOutputShapes(DimsExprs const* inputs, int32_t nbInputs,
    DimsExprs const* shapeInputs, int32_t nbShapeInputs, DimsExprs* outputs, int32_t nbOutputs,
    IExprBuilder& exprBuilder) noexcept
{
    try
    {
        PLUGIN_ASSERT(inputs != nullptr);
        PLUGIN_ASSERT(nbInputs == 1 + mHasImask + 2 * mUseVarSeqlen);
        PLUGIN_ASSERT(nbShapeInputs == 0);
        PLUGIN_ASSERT(outputs != nullptr);
        PLUGIN_ASSERT(nbOutputs == 1);
        // Input is BxSx3*N*H, output should be BxSxN*H
        // Copy over everything
        outputs[kIIDX] = inputs[kIIDX];
        // Divide last dim by three
        auto const* three = exprBuilder.constant(3);
        // mHdim is 2 for fixed seqlen and is 1 for varseqlen
        outputs[kIIDX].d[mHdim]
            = exprBuilder.operation(DimensionOperation::kFLOOR_DIV, *inputs[kIIDX].d[mHdim], *three);
        return pluginStatus_t::STATUS_SUCCESS;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return pluginStatus_t::STATUS_FAILURE;
}

bool QKVToContextVarSeqlenPlugin::supportsFormatCombination(
    int32_t pos, DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
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

    auto const inDims = in->desc.dims;
    auto const outDims = out->desc.dims;

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
        auto const& desc = inOut[pos].desc;
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
        auto const* mask = &inOut[pos].desc;
        if (mUseVarSeqlen)
        {
            // dummy input
            return true;
        }

        return mask->format == TensorFormat::kLINEAR && (mask->type == DataType::kINT32) && // precision
            (mask->dims.nbDims == 1);                                                       // num dims
    }
    PLUGIN_ASSERT(mUseVarSeqlen);
    if (pos == 2)
    { // must be cuSeqlens
        // cuSeqlens is a int32_t array of size B+1
        auto const* seqlens = &inOut[pos].desc;
        return (seqlens->type == DataType::kINT32) && (seqlens->format == TensorFormat::kLINEAR);
    }
    if (pos == 3)
    {
        // this is the dummy input
        return inOut[pos].desc.dims.nbDims == 1;
    }
    return false;
}

int32_t QKVToContextVarSeqlenPlugin::onShapeChange(
    PluginTensorDesc const* in, int32_t nbInputs, PluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
    try
    {
        PLUGIN_ASSERT(in != nullptr);
        PLUGIN_ASSERT(nbInputs == 1 + mHasImask + 2 * mUseVarSeqlen);
        PLUGIN_ASSERT(nbOutputs == 1);
        PluginTensorDesc const& inDesc = in[kIIDX];
        TRT_UNUSED inDesc;
        PluginTensorDesc const& outDesc = out[0];
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
                PluginTensorDesc const& maskDesc = in[kMIDX];
                TRT_UNUSED maskDesc;
                PLUGIN_ASSERT(maskDesc.dims.d[0] == inDesc.dims.d[BDIM]);
            }

            // during build, configurePlugin() should have set mS, mB appropriately
            // during inference, the engine should have mS, mB information
            PLUGIN_ASSERT(mS);
            PLUGIN_ASSERT(mB);

            BERT_DEBUG_MSG("setting up MHA runner for single sequence length");
            createMHARunner();
            this->mDispatcher->setup(mS, mB, mHeadSize);
        }
        else
        {
            BERT_DEBUG_MSG("setting up MHA runner for variable sequence length");
            createMHARunner();
            // need to initialize S and B with somewhat useful values, they will be reset at enqueue for the actual
            // batchsize
            this->mDispatcher->setup(256, 1, mHeadSize);
        }

        return pluginStatus_t::STATUS_SUCCESS;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return pluginStatus_t::STATUS_FAILURE;
}

int32_t QKVToContextVarSeqlenPlugin::configurePlugin(
    DynamicPluginTensorDesc const* in, int32_t nbInputs, DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
    try
    {
        PLUGIN_ASSERT(in != nullptr);
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

            const int32_t S = inDesc.dims.d[SDIM] <= 0 ? in->max.d[SDIM] : inDesc.dims.d[SDIM];
            const int32_t B = inDesc.dims.d[BDIM] <= 0 ? in->max.d[BDIM] : inDesc.dims.d[BDIM];

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

        return pluginStatus_t::STATUS_SUCCESS;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return pluginStatus_t::STATUS_FAILURE;
}

size_t QKVToContextVarSeqlenPlugin::getWorkspaceSize(DynamicPluginTensorDesc const* inputs, int32_t /* nbInputs */,
    DynamicPluginTensorDesc const* /* outputs */, int32_t /* nbOutputs */) const noexcept
{
    size_t paddingWorkpaceSize = mPatcher ? mPatcher->getWorkspaceSize(inputs[0].desc.dims.d[0], mNumHeads) : 0;
    return mDispatcher->getWorkspaceSize() + paddingWorkpaceSize;
}

int32_t QKVToContextVarSeqlenPlugin::getOutputDataTypes(
    DataType* outputTypes, int32_t nbOutputs, DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    try
    {
        PLUGIN_ASSERT(
            inputTypes[0] == DataType::kFLOAT || inputTypes[0] == DataType::kHALF || inputTypes[0] == DataType::kINT8);
        outputTypes[0] = inputTypes[0];
        return pluginStatus_t::STATUS_SUCCESS;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return pluginStatus_t::STATUS_FAILURE;
}

void QKVToContextVarSeqlenPlugin::setCublasResources(std::shared_ptr<CublasWrapper> cublasWrapper)
{
    mCublasWrapper = cublasWrapper;
    // The shared cublasWrapper resource owns the handle.
    // but `this` instance has a non-owning pointer to the handle.
    // Note that the cublasWrapper inits the handle and checks for nullptr
    // so we don't have to do that here.
    mCublasHandle = mCublasWrapper->getCublasHandle();
}

IPluginV3* QKVToContextVarSeqlenPlugin::attachToContext(IPluginResourceContext* context) noexcept
{
    try
    {
        auto p = static_cast<QKVToContextVarSeqlenPlugin*>(clone());
        // the clone has shared ownership of underling cublasWrapper instance
        // that is mapped to current context
        p->setCublasResources(createPluginCublasWrapper(context));
        return p;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return nullptr;
}

char const* QKVToContextVarSeqlenPlugin::getPluginVersion() const noexcept
{
    return kQKV_TO_CONTEXT_VAR_SEQLEN_PLUGIN_VERSION;
}

int32_t QKVToContextVarSeqlenPlugin::getNbOutputs() const noexcept
{
    return 1;
}

char const* QKVToContextVarSeqlenPlugin::getPluginName() const noexcept
{
    return kQKV_TO_CONTEXT_PLUGIN_NAME;
}


void QKVToContextVarSeqlenPlugin::setPluginNamespace(char const* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

char const* QKVToContextVarSeqlenPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

int32_t QKVToContextVarSeqlenPlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
    nvinfer1::PluginTensorDesc const* outputDesc, void const* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream) noexcept
{
    PLUGIN_VALIDATE(inputDesc != nullptr && outputDesc != nullptr && inputs != nullptr && outputs != nullptr);

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
                        paddingArgs.outputs, workspace, stream, mCublasHandle);
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
                    dispatcher->run(inputDesc, outputDesc, inputs, outputs, workspace, stream, mCublasHandle);
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
    mDispatcher->run(inputDesc[0], outputDesc[0], inputs[0], maskPtr, outputs[0], workspace, stream, mCublasHandle);
    return cudaGetLastError();
}

PluginFieldCollection const* QKVToContextVarSeqlenPlugin::getFieldsToSerialize() noexcept
{
    mDataToSerialize.clear();

    mDataToSerialize.emplace_back("type_id", &mType, PluginFieldType::kINT32, 1);
    mDataToSerialize.emplace_back("hidden_size", &mHiddenSize, PluginFieldType::kINT32, 1);
    mDataToSerialize.emplace_back("num_heads", &mNumHeads, PluginFieldType::kINT32, 1);
    mDataToSerialize.emplace_back("has_mask", &mHasImask, PluginFieldType::kINT32, 1);
    mDataToSerialize.emplace_back("var_seqlen", &mUseVarSeqlen, PluginFieldType::kINT32, 1);
    mDataToSerialize.emplace_back("use_int8_scale_max", &mUseInt8ScaleMax, PluginFieldType::kINT32, 1);
    mDataToSerialize.emplace_back("S", &mS, PluginFieldType::kINT32, 1);
    mDataToSerialize.emplace_back("B", &mB, PluginFieldType::kINT32, 1);

    mRunnerStateBuffer.resize(mDispatcher->getSerializationSize());
    mDispatcher->serialize(mRunnerStateBuffer.data());
    mDataToSerialize.emplace_back("runnerStateBuffer", (void const*) mRunnerStateBuffer.data(),
        PluginFieldType::kUNKNOWN, mRunnerStateBuffer.size());

    if (mDqProbs >= 0)
    {
        mDataToSerialize.emplace_back("dq_probs", &mDqProbs, PluginFieldType::kFLOAT32, 1);
    }

    mFCToSerialize.nbFields = mDataToSerialize.size();
    mFCToSerialize.fields = mDataToSerialize.data();

    return &mFCToSerialize;
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

char const* QKVToContextVarSeqlenPluginCreator::getPluginName() const noexcept
{
    return kQKV_TO_CONTEXT_PLUGIN_NAME;
}

char const* QKVToContextVarSeqlenPluginCreator::getPluginVersion() const noexcept
{
    return kQKV_TO_CONTEXT_VAR_SEQLEN_PLUGIN_VERSION;
}

PluginFieldCollection const* QKVToContextVarSeqlenPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV3* QKVToContextVarSeqlenPluginCreator::createPlugin(
    char const* name, PluginFieldCollection const* fc, TensorRTPhase phase) noexcept
{
    try
    {
        BERT_DEBUG_MSG("Creating QKV2ContextPlugin...");
        PLUGIN_VALIDATE(fc != nullptr);
        int32_t hiddenSize = 0;
        // Since numHeads must always exist or validateRequiredAttributes will fail,
        // we can set numHeads to -1 so that static analysis tools don't warn about
        // a division by zero in QKVToContextVarSeqelnPlugin constructor.
        int32_t numHeads = -1;
        bool hasMask = false;
        int32_t typeId = -1;
        int32_t s = -1;
        int32_t b = -1;
        void const* runnerStateBuffer = nullptr;
        int32_t varSeqlen = 0;
        float dqProbs = -1;
        int32_t useInt8ScaleMax = -1;

        PLUGIN_VALIDATE(fc->fields != nullptr);

        if (phase == TensorRTPhase::kBUILD)
        {
            plugin::validateRequiredAttributesExist({"type_id", "hidden_size", "num_heads", "has_mask"}, fc);
        }
        else
        {
            PLUGIN_ASSERT(phase == TensorRTPhase::kRUNTIME);
            // since fc is from a deserialized engine,
            // we expect all attributes (except dq_probs) to be present during runtime
            plugin::validateRequiredAttributesExist({"type_id", "S", "B", "hidden_size", "num_heads", "has_mask",
                                                        "var_seqlen", "use_int8_scale_max", "runnerStateBuffer"},
                fc);
        }
        for (int32_t i = 0; i < fc->nbFields; i++)
        {
            std::string field_name(fc->fields[i].name);

            if (field_name.compare("type_id") == 0)
            {
                typeId = *static_cast<int32_t const*>(fc->fields[i].data);
                PLUGIN_VALIDATE(typeId >= 0 && typeId <= 2, ("QKV: Invalid TypeId " + std::to_string(typeId)).c_str());
                BERT_DEBUG_VALUE("Building typeId: ", typeId);
            }
            else if (field_name.compare("hidden_size") == 0)
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
            else if (field_name.compare("has_mask") == 0)
            {
                hasMask = *static_cast<bool const*>(fc->fields[i].data);
                PLUGIN_VALIDATE(
                    hasMask == 0 || hasMask == 1, ("QKV: Invalid hasMask " + std::to_string(hasMask)).c_str());
                BERT_DEBUG_VALUE("Building hasMask: ", hasMask);
            }

            else if (field_name.compare("dq_probs") == 0)
            {
                dqProbs = *static_cast<float const*>(fc->fields[i].data);
                PLUGIN_VALIDATE(dqProbs > 0.0F, ("QKV: Invalid dqProbs " + std::to_string(dqProbs)).c_str());
                BERT_DEBUG_VALUE("Building dqProbs: ", dqProbs);
            }
            else if (field_name.compare("var_seqlen") == 0)
            {
                varSeqlen = *static_cast<int32_t const*>(fc->fields[i].data);
                BERT_DEBUG_VALUE("Building var_seqlen: ", varSeqlen);
            }
            else if (field_name.compare("use_int8_scale_max") == 0)
            {
                useInt8ScaleMax = *static_cast<int32_t const*>(fc->fields[i].data);
                PLUGIN_VALIDATE(useInt8ScaleMax == 0 || useInt8ScaleMax == 1,
                    ("QKV: Invalid useInt8ScaleMax " + std::to_string(useInt8ScaleMax)).c_str());
                BERT_DEBUG_VALUE("Building useInt8ScaleMax: ", useInt8ScaleMax);
            }
            else if (field_name.compare("S") == 0)
            {
                PLUGIN_ASSERT(phase == TensorRTPhase::kRUNTIME);
                s = *static_cast<int32_t const*>(fc->fields[i].data);
                BERT_DEBUG_VALUE("Building S: ", s);
            }
            else if (field_name.compare("B") == 0)
            {
                PLUGIN_ASSERT(phase == TensorRTPhase::kRUNTIME);
                b = *static_cast<int32_t const*>(fc->fields[i].data);
                BERT_DEBUG_VALUE("Building B: ", b);
            }
            else if (field_name.compare("runnerStateBuffer") == 0)
            {
                PLUGIN_ASSERT(phase == TensorRTPhase::kRUNTIME);
                runnerStateBuffer = static_cast<void const*>(fc->fields[i].data);
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

        if (phase == TensorRTPhase::kBUILD)
        {
            return new QKVToContextVarSeqlenPlugin(
                name, type, hiddenSize, numHeads, dqProbs, hasMask, varSeqlen, useInt8ScaleMaxFlag);
        }

        PLUGIN_VALIDATE(s != -1, "invalid S during runtime plugin creation");
        PLUGIN_VALIDATE(b != -1, "invalid B during runtime plugin creation");
        PLUGIN_VALIDATE(runnerStateBuffer != nullptr, "invalid runnerStateBuffer during runtime plugin creation");

        return new QKVToContextVarSeqlenPlugin(name, s, b, type, hiddenSize, numHeads, dqProbs, hasMask, varSeqlen,
            useInt8ScaleMaxFlag, runnerStateBuffer);
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

void QKVToContextVarSeqlenPluginCreator::setPluginNamespace(char const* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

char const* QKVToContextVarSeqlenPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

#endif // CUDA_VERSION >= 10010
