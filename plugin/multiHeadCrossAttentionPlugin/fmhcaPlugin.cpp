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
#include "fmhcaPlugin.h"
#include "fmhca.h"
#include "fmha_cross_attention/include/commonDatatype.h"
#include "fmha_cross_attention/include/fmha_cross_attention.h"

namespace
{
static char const* PLUGIN_NAME{"fMHCA"};
static char const* PLUGIN_VERSION{"1"};
} // namespace

namespace nvinfer1
{
namespace plugin
{
FMHCAPlugin::FMHCAPlugin(std::string const& name)
    : mLayerName(name)
{
    init();
}

FMHCAPlugin::FMHCAPlugin(std::string const& name, void const* data, size_t length)
    : mLayerName(name)
{
    memcpy(&mSerializationData, data, sizeof(mSerializationData));
}

void FMHCAPlugin::init(bool loadCubins)
{
    try
    {
        mSM = bert::getSMVersion();

        // initialize seqlens buffer
        allocateSeqlens(mSerializationData.mMaxBatchSize);
        mSerializationData.mOptSeqLenQ
            = initializeSeqlens(mSerializationData.mOptBatchSize, mSerializationData.mOptSeqLenQ, mCuSeqLensQ.get());
        mSerializationData.mOptSeqLenKV
            = initializeSeqlens(mSerializationData.mOptBatchSize, mSerializationData.mOptSeqLenKV, mCuSeqLensKV.get());

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

void FMHCAPlugin::createMHARunner()
{
    switch (mSerializationData.mDataType)
    {
    case DataType::kFLOAT: mKernels = getFMHCACubinKernels(plugin::DATA_TYPE_FP32, mSM); break;
    case DataType::kHALF: mKernels = getFMHCACubinKernels(plugin::DATA_TYPE_FP16, mSM); break;
    default: break;
    }
}

size_t FMHCAPlugin::getSerializationSize() const noexcept
{
    return sizeof(mSerializationData);
}

void FMHCAPlugin::serialize(void* buffer) const noexcept
{
    memcpy(buffer, &mSerializationData, sizeof(mSerializationData));
}

IPluginV2DynamicExt* FMHCAPlugin::clone() const noexcept
{
    try
    {
        std::vector<char> buff;
        buff.resize(getSerializationSize());
        serialize(buff.data());

        auto p = new FMHCAPlugin(mLayerName, buff.data(), buff.size());
        p->mCuSeqLensQ = mCuSeqLensQ;
        p->mCuSeqLensKV = mCuSeqLensKV;
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

int32_t FMHCAPlugin::getNbOutputs() const noexcept
{
    return 1;
}

// input0 q_packed in [b, s_q, h, d]
// input1 kv_packed in [b, s_kv, h, 2, d]
// output 0 in [b, s_q, h, d]
DimsExprs FMHCAPlugin::getOutputDimensions(
    int32_t outputIndex, DimsExprs const* inputs, int32_t nbInputs, IExprBuilder& exprBuilder) noexcept
{
    DimsExprs out{};
    out.nbDims = 4;
    out.d[0] = inputs[0].d[0];
    out.d[1] = inputs[0].d[1];
    out.d[2] = inputs[0].d[2];
    out.d[3] = inputs[0].d[3];
    return out;
}

bool FMHCAPlugin::supportsFormatCombination(
    int32_t pos, PluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    bool res = false;
    try
    {
        // load kernel and check if we have any implementations.
        auto hasImplement = [this](DataType dt) {
            switch (dt)
            {
            case DataType::kFLOAT: return getFMHCACubinKernels(plugin::DATA_TYPE_FP32, mSM)->isValid(/*dummy seq*/ 128);
            case DataType::kHALF: return getFMHCACubinKernels(plugin::DATA_TYPE_FP16, mSM)->isValid(/*dummy seq*/ 128);
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
        case 0: res = hasImplement(inOut[pos].type) && inOut[pos].dims.nbDims == 4; break;
        case 1: res = inOut[pos].type == inOut[0].type && inOut[pos].dims.nbDims == 5; break;
        case 2: res = inOut[pos].type == inOut[0].type && inOut[pos].dims.nbDims == 4; break;
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

void FMHCAPlugin::allocateSeqlens(int32_t maxBatchSize)
{
    // allocate seqlens buffer
    auto allocBuffer = [&maxBatchSize](bert::cuda_shared_ptr<void>& dptr) {
        if (!dptr && maxBatchSize)
        {
            void* cudaMem{nullptr};
            PLUGIN_CHECK(cudaMalloc(&cudaMem, sizeof(int32_t) * (maxBatchSize + 1)));
            bert::make_cuda_shared(dptr, cudaMem);
        }
    };
    allocBuffer(mCuSeqLensQ);
    allocBuffer(mCuSeqLensKV);
    mSerializationData.mMaxBatchSize = maxBatchSize;
}

int32_t FMHCAPlugin::initializeSeqlens(int32_t b, int32_t s, void* cuSeqlensDev, cudaStream_t stream)
{
    if (!b || !s)
    {
        return s;
    }

    std::vector<int32_t> cuSeqlens(b + 1, 0);
    // Compute the prefix sum of the sequence lenghts.
    for (int32_t it = 0; it < b; it++)
    {
        cuSeqlens[it + 1] = cuSeqlens[it] + s;
    }

    PLUGIN_CUASSERT(cudaMemcpyAsync(
        cuSeqlensDev, cuSeqlens.data(), sizeof(int32_t) * cuSeqlens.size(), cudaMemcpyHostToDevice, stream));
    mSerializationData.mOptBatchSize = b;
    return s;
}

DataType FMHCAPlugin::getOutputDataType(
    int32_t outputIndex, DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    return inputTypes[0];
}

void FMHCAPlugin::configurePlugin(
    DynamicPluginTensorDesc const* in, int32_t nbInputs, DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
    try
    {
        int32_t const batchSize = in[0].max.d[0];
        int32_t const seqLenQ = in[0].max.d[1];
        int32_t const seqLenKV = in[1].max.d[1];

        allocateSeqlens(batchSize);
        if (batchSize != mSerializationData.mOptBatchSize || mSerializationData.mOptSeqLenQ != seqLenQ
            || mSerializationData.mOptSeqLenKV != seqLenKV)
        {
            mSerializationData.mOptSeqLenQ = initializeSeqlens(batchSize, seqLenQ, mCuSeqLensQ.get());
            mSerializationData.mOptSeqLenKV = initializeSeqlens(batchSize, seqLenKV, mCuSeqLensKV.get());
        }

        mSerializationData.mDataType = in[0].desc.type;
        createMHARunner();
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
}

size_t FMHCAPlugin::getWorkspaceSize(
    PluginTensorDesc const* inputs, int32_t nbInputs, PluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept
{
    return 0;
}

void FMHCAPlugin::setPluginNamespace(char const* szNamespace) noexcept
{
    mNamespace = szNamespace;
}
char const* FMHCAPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}
char const* FMHCAPlugin::getPluginType() const noexcept
{
    return PLUGIN_NAME;
}
char const* FMHCAPlugin::getPluginVersion() const noexcept
{
    return PLUGIN_VERSION;
}
int32_t FMHCAPlugin::initialize() noexcept
{
    return 0;
}
void FMHCAPlugin::terminate() noexcept {}

void FMHCAPlugin::destroy() noexcept
{
    delete this;
}

int32_t FMHCAPlugin::enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc,
    void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    int32_t result{-1};
    try
    {
        PLUGIN_VALIDATE(mKernels);
        PLUGIN_VALIDATE(mSM);
        PLUGIN_VALIDATE(mCuSeqLensQ);
        PLUGIN_VALIDATE(mCuSeqLensKV);

        constexpr int32_t seqLenKvPadded = 128;
        int32_t const batchSize = inputDesc[0].dims.d[0];
        int32_t const seqLenQ = inputDesc[0].dims.d[1];
        int32_t const seqLenKV = inputDesc[1].dims.d[1];
        int32_t const headNum = inputDesc[0].dims.d[2];
        int32_t const sizePerHead = inputDesc[0].dims.d[3];

        if (batchSize != mSerializationData.mOptBatchSize || mSerializationData.mOptSeqLenQ != seqLenQ
            || mSerializationData.mOptSeqLenKV != seqLenKV)
        {
            mSerializationData.mOptSeqLenQ = initializeSeqlens(batchSize, seqLenQ, mCuSeqLensQ.get(), stream);
            mSerializationData.mOptSeqLenKV = initializeSeqlens(batchSize, seqLenKV, mCuSeqLensKV.get(), stream);
        }

        result = runFMHCAKernel(inputs[0], inputs[1], mCuSeqLensQ.get(), mCuSeqLensKV.get(), outputs[0], mSM, mKernels,
            batchSize, headNum, sizePerHead, seqLenQ, seqLenKvPadded, stream);
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return result;
}

PluginFieldCollection FMHCAPluginCreator::mFC{};
std::vector<PluginField> FMHCAPluginCreator::mPluginAttributes;

FMHCAPluginCreator::FMHCAPluginCreator()
{
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

IPluginV2* FMHCAPluginCreator::createPlugin(char const* name, PluginFieldCollection const* fc) noexcept
{
    try
    {
        return new FMHCAPlugin(name);
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* FMHCAPluginCreator::deserializePlugin(char const* name, void const* serialData, size_t serialLength) noexcept
{
    try
    {
        auto p = new FMHCAPlugin(name, serialData, serialLength);
        p->init(true);
        return p;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

void FMHCAPluginCreator::setPluginNamespace(char const* szNamespace) noexcept
{
    mNamespace = szNamespace;
}

char const* FMHCAPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

char const* FMHCAPluginCreator::getPluginName() const noexcept
{
    return PLUGIN_NAME;
}

char const* FMHCAPluginCreator::getPluginVersion() const noexcept
{
    return PLUGIN_VERSION;
}

PluginFieldCollection const* FMHCAPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

} // namespace plugin
} // namespace nvinfer1
#endif // defined(ENABLE_SM75) || defined(ENABLE_SM80) || defined(ENABLE_SM86) || defined(ENABLE_SM89)
