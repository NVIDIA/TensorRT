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

#if defined(ENABLE_SM75) || defined(ENABLE_SM80) || defined(ENABLE_SM86) || defined(ENABLE_SM89)
#include "fmhaPlugin.h"
#include "fmha.h"
#include "fmha_flash_attention/include/commonDatatype.h"
#include "fmha_flash_attention/include/fmha_flash_attention.h"

namespace
{
static char const* PLUGIN_NAME{"fMHA_V2"};
static char const* PLUGIN_VERSION{"1"};
} // namespace

namespace nvinfer1
{
namespace plugin
{
FMHAPlugin::FMHAPlugin(std::string const& name)
    : mLayerName(name)
{
    init();
}

FMHAPlugin::FMHAPlugin(std::string const& name, void const* data, size_t length)
    : mLayerName(name)
{
    memcpy(&mSerializationData, data, sizeof(mSerializationData));
}

void FMHAPlugin::init(bool loadCubins)
{
    try
    {
        mSM = bert::getSMVersion();

        // initialize seqlens buffer
        allocateSeqlens(mSerializationData.mMaxBatchSize);
        initializeSeqlens(mSerializationData.mOptBatchSize, mSerializationData.mOptSeqLen, mCuSeqLen.get());

        if (loadCubins)
        {
            createMHARunner();
        }
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
}

void FMHAPlugin::createMHARunner()
{
    switch (mSerializationData.mDataType)
    {
    case DataType::kFLOAT: mKernels = getFMHAFlashCubinKernels(plugin::DATA_TYPE_FP32, mSM); break;
    case DataType::kHALF: mKernels = getFMHAFlashCubinKernels(plugin::DATA_TYPE_FP16, mSM); break;
    default: break;
    }
}

size_t FMHAPlugin::getSerializationSize() const noexcept
{
    return sizeof(mSerializationData);
}

void FMHAPlugin::serialize(void* buffer) const noexcept
{
    memcpy(buffer, &mSerializationData, sizeof(mSerializationData));
}

IPluginV2DynamicExt* FMHAPlugin::clone() const noexcept
{
    try
    {
        std::vector<char> buff;
        buff.resize(getSerializationSize());
        serialize(buff.data());

        auto p = new FMHAPlugin(mLayerName, buff.data(), buff.size());
        p->mCuSeqLen = mCuSeqLen;
        p->setPluginNamespace(mNamespace.c_str());
        p->init(true);
        return p;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

int32_t FMHAPlugin::getNbOutputs() const noexcept
{
    return 1;
}

DimsExprs FMHAPlugin::getOutputDimensions(
    int32_t outputIndex, DimsExprs const* inputs, int32_t nbInputs, IExprBuilder& exprBuilder) noexcept
{
    // input0 qkv_packed in [b, s, h, 3, d]
    // input1 cu_seqlens in [b + 1]
    // output O in [b, s, h, d]
    DimsExprs out;
    out.nbDims = 4;
    out.d[0] = inputs[0].d[0];
    out.d[1] = inputs[0].d[1];
    out.d[2] = inputs[0].d[2];
    out.d[3] = inputs[0].d[4];
    return out;
}

bool FMHAPlugin::supportsFormatCombination(
    int32_t pos, PluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    bool res = false;
    try
    {
        // load kernel and check if we have any implementations.
        auto hasImplement = [this](DataType dt) {
            switch (dt)
            {
            case DataType::kFLOAT:
                return getFMHAFlashCubinKernels(plugin::DATA_TYPE_FP32, mSM)->isValid(/*dummy seq*/ 128);
            case DataType::kHALF:
                return getFMHAFlashCubinKernels(plugin::DATA_TYPE_FP16, mSM)->isValid(/*dummy seq*/ 128);
            default: break;
            }
            return false;
        };

        if (inOut[pos].format != TensorFormat::kLINEAR)
        {
            return false;
        }

        switch (pos)
        {
        case 0: res = hasImplement(inOut[pos].type) && inOut[pos].dims.nbDims == 5; break;
        case 1: res = inOut[pos].type == inOut[0].type && inOut[pos].dims.nbDims == 4; break;
        default: // should NOT be here
            break;
        }
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return res;
}

DataType FMHAPlugin::getOutputDataType(int32_t outputIndex, DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    return inputTypes[0];
}

void FMHAPlugin::allocateSeqlens(int32_t maxBatchSize)
{
    // allocate seqlens buffer
    if (!mCuSeqLen && maxBatchSize)
    {
        void* cudaMem{nullptr};
        PLUGIN_CHECK(cudaMalloc(&cudaMem, sizeof(int32_t) * (maxBatchSize + 1)));
        bert::make_cuda_shared(mCuSeqLen, cudaMem);
    }

    mSerializationData.mMaxBatchSize = maxBatchSize;
}

void FMHAPlugin::initializeSeqlens(int32_t b, int32_t s, void* cu_seqlens_d, cudaStream_t stream)
{
    if (!b || !s)
    {
        return;
    }

    std::vector<int32_t> cuSeqLens(b + 1, 0);
    // Compute the prefix sum of the seqlen
    for (int32_t it = 0; it < b; it++)
    {
        cuSeqLens[it + 1] = cuSeqLens[it] + s;
    }

    PLUGIN_CUASSERT(cudaMemcpyAsync(
        cu_seqlens_d, cuSeqLens.data(), sizeof(int32_t) * cuSeqLens.size(), cudaMemcpyHostToDevice, stream));
    mSerializationData.mOptBatchSize = b;
    mSerializationData.mOptSeqLen = s;
}

void FMHAPlugin::configurePlugin(
    DynamicPluginTensorDesc const* in, int32_t nbInputs, DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
    try
    {
        int32_t const batchSize = in[0].max.d[0];
        int32_t const seqLen = in[0].max.d[1];

        allocateSeqlens(batchSize);
        if (batchSize != mSerializationData.mOptBatchSize || seqLen != mSerializationData.mOptSeqLen)
        {
            initializeSeqlens(batchSize, seqLen, mCuSeqLen.get());
        }

        mSerializationData.mDataType = in[0].desc.type;
        createMHARunner();
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
}

size_t FMHAPlugin::getWorkspaceSize(
    PluginTensorDesc const* inputs, int32_t nbInputs, PluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept
{
    return 0;
}

void FMHAPlugin::setPluginNamespace(char const* szNamespace) noexcept
{
    mNamespace = szNamespace;
}
char const* FMHAPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}
char const* FMHAPlugin::getPluginType() const noexcept
{
    return PLUGIN_NAME;
}
char const* FMHAPlugin::getPluginVersion() const noexcept
{
    return PLUGIN_VERSION;
}
int32_t FMHAPlugin::initialize() noexcept
{
    return 0;
}
void FMHAPlugin::terminate() noexcept {}

void FMHAPlugin::destroy() noexcept
{
    delete this;
}

int32_t FMHAPlugin::enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc,
    void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    // input[ 0]:  [float16],  (b, s, h, 3, d)
    // output[0]:  [float16],  (b,s,h,d)
    int32_t result{-1};
    try
    {
        PLUGIN_VALIDATE(mKernels);
        PLUGIN_VALIDATE(mSM);
        PLUGIN_VALIDATE(mCuSeqLen);

        // update cuseqlens when bs or seq changed.
        int32_t const batchSize = inputDesc[0].dims.d[0];
        int32_t const seqLen = inputDesc[0].dims.d[1];
        if (batchSize != mSerializationData.mOptBatchSize || seqLen != mSerializationData.mOptSeqLen)
        {
            initializeSeqlens(batchSize, seqLen, mCuSeqLen.get(), stream);
        }

        // launch kernel.
        int32_t const head_num = inputDesc[0].dims.d[2];
        int32_t const size_per_head = inputDesc[0].dims.d[4];
        size_t const total = mSerializationData.mOptBatchSize * mSerializationData.mOptSeqLen;
        result = runFMHFAKernel(inputs[0], mCuSeqLen.get(), outputs[0], total, mSM, mKernels,
            mSerializationData.mOptBatchSize, head_num, size_per_head, mSerializationData.mOptSeqLen, stream);
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return result;
}

PluginFieldCollection FMHAPluginCreator::mFc{};
std::vector<PluginField> FMHAPluginCreator::mPluginAttributes;

FMHAPluginCreator::FMHAPluginCreator()
{
    mFc.nbFields = mPluginAttributes.size();
    mFc.fields = mPluginAttributes.data();
}

IPluginV2* FMHAPluginCreator::createPlugin(char const* name, PluginFieldCollection const* fc) noexcept
{
    try
    {
        return new FMHAPlugin(name);
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* FMHAPluginCreator::deserializePlugin(char const* name, void const* serialData, size_t serialLength) noexcept
{
    try
    {
        auto p = new FMHAPlugin(name, serialData, serialLength);
        p->init(true);
        return p;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

void FMHAPluginCreator::setPluginNamespace(char const* szNamespace) noexcept
{
    mNamespace = szNamespace;
}

char const* FMHAPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

char const* FMHAPluginCreator::getPluginName() const noexcept
{
    return PLUGIN_NAME;
}

char const* FMHAPluginCreator::getPluginVersion() const noexcept
{
    return PLUGIN_VERSION;
}

PluginFieldCollection const* FMHAPluginCreator::getFieldNames() noexcept
{
    return &mFc;
}

} // namespace plugin
} // namespace nvinfer1
#endif // defined(ENABLE_SM75) || defined(ENABLE_SM80) || defined(ENABLE_SM86) || defined(ENABLE_SM89)
