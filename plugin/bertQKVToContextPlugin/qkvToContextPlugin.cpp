/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

// Need 10.1 for cublasGemmStridedBatchedEx
#include <cuda.h>
#if CUDA_VERSION >= 10010

#include "NvInfer.h"
#include "bertCommon.h"
#include "fused_multihead_attention.h"
#include "fused_multihead_attention_v2.h"
#include "qkvToContextPlugin.h"
#include "serialize.hpp"

#include <cassert>
#include <cstring>
#include <iostream>
#include <tuple>
#include <vector>

using namespace nvinfer1;

namespace bert
{

namespace
{
static const char* QKV_TO_CONTEXT_PLUGIN_VERSION{"1"};
static const char* QKV_TO_CONTEXT_VAR_SEQLEN_PLUGIN_VERSION{"2"};
static const char* QKV_TO_CONTEXT_PLUGIN_NAME{"CustomQKVToContextPluginDynamic"};
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

QKVToContextPluginDynamic::QKVToContextPluginDynamic(const std::string name, const DataType type, const int hiddenSize,
    const int numHeads, const float dqProbs, bool hasImask)
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
    gLogVerbose << "QKV Deser Start" << std::endl;
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

    int hasUnfusedRunner = 0;
    deserialize_value(&data, &length, &hasUnfusedRunner);
    if (hasUnfusedRunner)
    {
        ASSERT(unfusedDispatcher.get());
        unfusedDispatcher->deserialize(data, length);
    }

    gLogVerbose << "QKV Deser done" << std::endl;
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
        unfusedDispatcher.reset(new UnfusedMHARunner(mType, mNumHeads, mHeadSize));
    }
}

// IPluginV2DynamicExt Methods
nvinfer1::IPluginV2DynamicExt* QKVToContextPluginDynamic::clone() const
{
    gLogVerbose << "QKV Clone" << std::endl;

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
    gLogVerbose << "QKV Clone done" << std::endl;
    return ret;
}

DimsExprs QKVToContextPluginDynamic::getOutputDimensions(
    int outputIndex, const DimsExprs* inputs, int nbInputs, IExprBuilder& exprBuilder)
{
    // Input is BxSx3*N*H, output should be BxSxN*H
    assert(outputIndex == 0);
    // Copy over everything
    DimsExprs output(inputs[IIDX]);
    // Divide last dim by three
    auto three = exprBuilder.constant(3);
    output.d[HDIM] = exprBuilder.operation(DimensionOperation::kFLOOR_DIV, *inputs[IIDX].d[HDIM], *three);
    return output;
}
bool QKVToContextPluginDynamic::supportsFormatCombination(
    int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs)
{
    assert(pos >= 0);
    assert(pos < 2 + mHasImask);
    assert(nbInputs == 1 + mHasImask);
    const auto* in = inOut;
    const auto* out = inOut + nbInputs;
    int packedSize = getMHAMaskPackedSize(mSM, mType, in->dims.d[SDIM]);

    // we only support int8 IO in fused mha runner, and we only support fused mha runner on Xavier, Turing and Ampere
    if (mType == DataType::kINT8)
    {
        if (mSM != kSM_75 && mSM != kSM_80 && mSM != kSM_86)
        {
            gLogError << "INT8 IO is only supported on Turing and Ampere for plugin " << QKV_TO_CONTEXT_PLUGIN_NAME
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

    if (pos == 0)
    {
        bool isFormatSupported = in->format == TensorFormat::kLINEAR;
        if (mType == DataType::kINT8)
        {
            if (in->dims.d[HDIM] % 32 == 0)
            {
                isFormatSupported = in->format == TensorFormat::kCHW32;
            }
            else
            {
                isFormatSupported = in->format == TensorFormat::kCHW4;
            }
        }

        // must not check descriptions > pos
        return (in->type == mType) &&        // precision
            isFormatSupported &&             // format
            (in->dims.nbDims == 5) &&        // num dims
            ((in->dims.d[HDIM] % 3) == 0) && // see getOutputDimensions
            ((in->dims.d[3]) == 1) &&        // for fc
            ((in->dims.d[4]) == 1)           // for fc
            ;
    }
    else
    {                                // pos==1
        if ((mHasImask && pos == 1)) // pos 1 is the mask
        {
            const auto* inMask = &inOut[1];
            if (inMask->dims.d[1] != -1 && inMask->dims.d[1] != packedSize)
            {
                gLogError << "CustomEmbLayerNormPluginDynamic returned mask with pack size " << inMask->dims.d[1]
                          << ", but " << QKV_TO_CONTEXT_PLUGIN_NAME << " expects mask pack size " << packedSize
                          << std::endl;
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
                if (out->dims.d[HDIM] % 32 == 0)
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
    }
    return false;
}
void QKVToContextPluginDynamic::configurePlugin(
    const DynamicPluginTensorDesc* in, int nbInputs, const DynamicPluginTensorDesc* out, int nbOutputs)
{
    assert(nbInputs == 1 + mHasImask);
    assert(nbOutputs == 1);
    const PluginTensorDesc& inDesc = in[IIDX].desc;
    TRT_UNUSED inDesc;
    const PluginTensorDesc& outDesc = out->desc;
    TRT_UNUSED outDesc;
    assert(mType == inDesc.type);
    assert(mType == outDesc.type);
    assert(inDesc.dims.d[BDIM] == outDesc.dims.d[BDIM]);
    assert(inDesc.dims.d[SDIM] == outDesc.dims.d[SDIM]);
    assert(inDesc.dims.d[HDIM] == 3 * outDesc.dims.d[HDIM]);
    if (mHasImask)
    {
        const PluginTensorDesc& maskDesc = in[MIDX].desc;
        TRT_UNUSED maskDesc;
        assert(maskDesc.dims.d[0] == inDesc.dims.d[BDIM]);
    }

    createMHARunner();

    const int S = inDesc.dims.d[SDIM];
    const int B = inDesc.dims.d[BDIM] <= 0 ? in->max.d[BDIM] : inDesc.dims.d[BDIM];
    if (S <= 0)
    {
        // in dynamic shape build stage, we setup with max sequence that cannot fused
        const int Smin = in->min.d[SDIM];
        const int Smax = in->max.d[SDIM];

        if (fusedDispatcher.get())
        {
            for (int i = Smax; i >= Smin; --i)
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

size_t QKVToContextPluginDynamic::getWorkspaceSize(
    const PluginTensorDesc* inputs, int nbInputs, const PluginTensorDesc* outputs, int nbOutputs) const
{
    // only unfused kernel need workspace, and we need larger workspace for larger sequence length
    // we have already setup unfusedDispatcher with max sequence in configurePlugin
    // if unfusedDispatcher is not initialized in configurePlugin
    ASSERT(unfusedDispatcher.get());
    return unfusedDispatcher->getWorkspaceSize();
}

// IPluginV2Ext Methods
DataType QKVToContextPluginDynamic::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes, int nbInputs) const
{
    assert(index == 0);
    assert(inputTypes[0] == DataType::kFLOAT || inputTypes[0] == DataType::kHALF || inputTypes[0] == DataType::kINT8);
    return inputTypes[0];
}

// IPluginV2 Methods
const char* QKVToContextPluginDynamic::getPluginType() const
{
    return QKV_TO_CONTEXT_PLUGIN_NAME;
}

const char* QKVToContextPluginDynamic::getPluginVersion() const
{
    return QKV_TO_CONTEXT_PLUGIN_VERSION;
}

int QKVToContextPluginDynamic::getNbOutputs() const
{
    return 1;
}

int QKVToContextPluginDynamic::initialize()
{
    return 0;
}

void QKVToContextPluginDynamic::terminate() {}

size_t QKVToContextPluginDynamic::getSerializationSize() const
{
    ASSERT(unfusedDispatcher.get());
    return sizeof(mNumHeads) + sizeof(mHeadSize) + sizeof(DataType) + sizeof(mHasImask) + sizeof(mHiddenSize)
        + sizeof(mSM) + sizeof(mS) + sizeof(mB) + sizeof(mDqProbs) + sizeof(int)
        + unfusedDispatcher->getSerializationSize();
}

void QKVToContextPluginDynamic::serialize(void* buffer) const
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
        int hasUnfusedRunner = 1;
        serialize_value(&buffer, hasUnfusedRunner);
        unfusedDispatcher->serialize(buffer);
    }
    else
    {
        int hasUnfusedRunner = 0;
        serialize_value(&buffer, hasUnfusedRunner);
    }
}

void QKVToContextPluginDynamic::destroy()
{
    delete this;
}

void QKVToContextPluginDynamic::setPluginNamespace(const char* libNamespace)
{
    mNamespace = libNamespace;
}

const char* QKVToContextPluginDynamic::getPluginNamespace() const
{
    return mNamespace.c_str();
}

int QKVToContextPluginDynamic::enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc,
    const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream)
{
    assert(mS == inputDesc->dims.d[SDIM]);
    assert(mB == inputDesc->dims.d[BDIM]);

    const void* maskPtr = mHasImask ? inputs[1] : nullptr;
    if (fusedDispatcher.get() && fusedDispatcher->isValid(inputDesc->dims.d[SDIM]))
    {
        fusedDispatcher->run(inputDesc[0], outputDesc[0], inputs[0], maskPtr, outputs[0], workspace, stream);
    }
    else
    {
        ASSERT(unfusedDispatcher.get());
        unfusedDispatcher->run(inputDesc[0], outputDesc[0], inputs[0], maskPtr, outputs[0], workspace, stream);
    }
    return 0;
}

QKVToContextPluginDynamicCreator::QKVToContextPluginDynamicCreator()
{
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* QKVToContextPluginDynamicCreator::getPluginName() const
{
    return QKV_TO_CONTEXT_PLUGIN_NAME;
}

const char* QKVToContextPluginDynamicCreator::getPluginVersion() const
{
    return QKV_TO_CONTEXT_PLUGIN_VERSION;
}

const PluginFieldCollection* QKVToContextPluginDynamicCreator::getFieldNames()
{
    return &mFC;
}

IPluginV2* QKVToContextPluginDynamicCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
{
    gLogVerbose << "Creating QKV2ContextPlugin...\n";

    int hiddenSize = 0;
    int numHeads = 0;
    bool hasMask = false;
    int typeId = -1;

    float dqProbs = -1;

    for (int i = 0; i < fc->nbFields; i++)
    {
        std::string field_name(fc->fields[i].name);

        if (field_name.compare("type_id") == 0)
        {
            typeId = *static_cast<const int*>(fc->fields[i].data);
            gLogVerbose << "Building typeId: " << typeId << std::endl;
        }
        if (field_name.compare("hidden_size") == 0)
        {
            hiddenSize = *static_cast<const int*>(fc->fields[i].data);
            gLogVerbose << "Building hiddenSize: " << hiddenSize << std::endl;
        }
        if (field_name.compare("num_heads") == 0)
        {
            numHeads = *static_cast<const int*>(fc->fields[i].data);
            gLogVerbose << "Building numHeads: " << numHeads << std::endl;
        }
        if (field_name.compare("has_mask") == 0)
        {
            hasMask = *static_cast<const bool*>(fc->fields[i].data);
            gLogVerbose << "Building hasMask: " << hasMask << std::endl;
        }

        if (field_name.compare("dq_probs") == 0)
        {
            dqProbs = *static_cast<const float*>(fc->fields[i].data);
            gLogVerbose << "Building dqProbs: " << dqProbs << std::endl;
        }
    }
    if (typeId < 0 || typeId > 3)
    {
        gLogError << "QKV: Invalid TypeId " << typeId << std::endl;
    }

    if (hiddenSize <= 0)
    {
        gLogError << "QKV: Invalid hiddenSize " << hiddenSize << std::endl;
    }

    if (numHeads <= 0)
    {
        gLogError << "QKV: Invalid numHeads " << numHeads << std::endl;
    }

    gLogVerbose << "Building the Plugin...\n";
    DataType type = static_cast<DataType>(typeId);
    if (type == DataType::kINT8 && dqProbs < 0)
    {
        gLogInfo << "Using default scale factor\n";
        dqProbs = 1.f / 127.f;
    }

    QKVToContextPluginDynamic* p
        = new QKVToContextPluginDynamic(name, type, hiddenSize, numHeads, dqProbs, hasMask);
    return p;
}

IPluginV2* QKVToContextPluginDynamicCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength)
{
    // This object will be deleted when the network is destroyed, which will
    // call QKVToContextPluginDynamic::destroy()
    return new QKVToContextPluginDynamic(name, serialData, serialLength);
}

void QKVToContextPluginDynamicCreator::setPluginNamespace(const char* libNamespace)
{
    mNamespace = libNamespace;
}

const char* QKVToContextPluginDynamicCreator::getPluginNamespace() const
{
    return mNamespace.c_str();
}

QKVToContextVarSeqlenPlugin::QKVToContextVarSeqlenPlugin(const std::string name, const DataType type,
    const int hiddenSize, const int numHeads, const float dqProbs, bool hasImask, bool varSeqlen)
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

{
    mSM = getSMVersion();

    if (varSeqlen)
    {
        // variable sequence length is only supported with the fused MHA kernels
        // we should not override mS!
        assert((mSM == kSM_86 || mSM == kSM_80 || mSM == kSM_75 || mSM == kSM_72) && (type == DataType::kINT8 || type == DataType::kHALF)
            && "requesting maxSeqlen not compatible with GPU arch");
        // the layout changes: SxB will be a combined \sum_i s_i and hdim will be the 2nd dimension instead of the third
        mHdim = 1;
    }
}

QKVToContextVarSeqlenPlugin::QKVToContextVarSeqlenPlugin(const std::string name, const void* data, size_t length)
    : mLayerName(name)
{
    gLogVerbose << "QKV Deser Start" << std::endl;
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

    createMHARunner();
    dispatcher->deserialize(data, length);

    gLogVerbose << "QKV Deser done" << std::endl;
}

void QKVToContextVarSeqlenPlugin::createMHARunner()
{
    if (dispatcher.get())
    {
        return;
    }

    if (mSM == kSM_86 || mSM == kSM_80 || mSM == kSM_75 || mSM == kSM_72)
    {
        if (mType == DataType::kHALF)
        {
            dispatcher.reset(new FusedMHARunnerFP16v2(mNumHeads, mHeadSize, mSM));
        }
        else if (mType == DataType::kINT8)
        {
            dispatcher.reset(new FusedMHARunnerInt8v2(mNumHeads, mHeadSize, mSM, mDqProbs));
        }
    }
    else
    {
        assert(!mUseVarSeqlen);
        dispatcher.reset(new UnfusedMHARunner(mType, mNumHeads, mHeadSize));
    }
}

// IPluginV2DynamicExt Methods
nvinfer1::IPluginV2DynamicExt* QKVToContextVarSeqlenPlugin::clone() const
{
    gLogVerbose << "QKV Clone" << std::endl;

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
            mLayerName, mType, mHiddenSize, mNumHeads, mDqProbs, mHasImask, mUseVarSeqlen);
    }

    ret->setPluginNamespace(mNamespace.c_str());
    gLogVerbose << "QKV Clone done" << std::endl;
    return ret;
}

DimsExprs QKVToContextVarSeqlenPlugin::getOutputDimensions(
    int outputIndex, const DimsExprs* inputs, int nbInputs, IExprBuilder& exprBuilder)
{
    // Input is BxSx3*N*H, output should be BxSxN*H
    assert(outputIndex == 0);
    // Copy over everything
    DimsExprs output(inputs[IIDX]);
    // Divide last dim by three
    auto three = exprBuilder.constant(3);
    output.d[mHdim] = exprBuilder.operation(DimensionOperation::kFLOOR_DIV, *inputs[IIDX].d[mHdim], *three);
    return output;
}

bool QKVToContextVarSeqlenPlugin::supportsFormatCombination(
    int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs)
{
    // we only support int8 IO in fused mha runner, and we only support fused mha runner on Turing and Ampere
    if (mType == DataType::kINT8 && mSM != kSM_86 && mSM != kSM_80 && mSM != kSM_75 && mSM != kSM_72)
    {
        gLogVerbose << "INT8 IO is only supported on Xavier, Turing and Ampere for plugin " << QKV_TO_CONTEXT_PLUGIN_NAME
                    << std::endl;
        return false;
    }

    assert(pos >= 0);
    assert(pos < 2 + mHasImask + 2 * mUseVarSeqlen);
    assert(nbInputs == 1 + mHasImask + 2 * mUseVarSeqlen);
    assert(nbOutputs == 1);
    const auto* in = inOut;
    const auto* out = inOut + nbInputs;
    if (mUseVarSeqlen)
    {
        assert((mType == DataType::kHALF || mType == DataType::kINT8)
            && "Conditions for variable seqlen support not fulfilled");
        // qkv, mask, cu_seqlens, dummy
        assert(nbInputs == 4 && "for varseqlen, expected 4 inputs");
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
        supportedFormat = (inDims.d[mHdim] % 32 == 0) ? TensorFormat::kCHW32 : TensorFormat::kCHW4;
    }

    int supportedNbDims = 5;
    if (mUseVarSeqlen)
    {
        supportedNbDims = 4;
    }

    bool supportedHdim = (pos == 0) ? (inDims.d[mHdim] % 3 == 0) : (inDims.d[mHdim] / 3 == outDims.d[mHdim]);

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

    assert(mHasImask);
    if (pos == 1)
    { // must be input mask
        const auto* mask = &inOut[pos];
        const auto maskType = mask->type;
        const auto maskFmt = mask->format;
        const auto maskDims = mask->dims;
        if (maskFmt != TensorFormat::kLINEAR)
            return false;

        if (mUseVarSeqlen) // use full mask for fused MHA of shape B x 2*MHAmaskSize
        {
            return (maskType == DataType::kHALF) && // precision
                (maskDims.nbDims == 2)              // Bx2*maskSize
                ;
        }
        return (mask->type == DataType::kINT32) && // precision
            (mask->dims.nbDims == 1)               // num dims
            ;
    }
    assert(mUseVarSeqlen);
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
    const DynamicPluginTensorDesc* in, int nbInputs, const DynamicPluginTensorDesc* out, int nbOutputs)
{
    assert(nbInputs == 1 + mHasImask + 2 * mUseVarSeqlen);
    assert(nbOutputs == 1);
    const PluginTensorDesc& inDesc = in[IIDX].desc;
    TRT_UNUSED inDesc;
    const PluginTensorDesc& outDesc = out->desc;
    TRT_UNUSED outDesc;
    assert(mType == inDesc.type);
    assert(mType == outDesc.type);
    if (!mUseVarSeqlen)
    {
        assert(inDesc.dims.d[BDIM] == outDesc.dims.d[BDIM]);
        assert(inDesc.dims.d[SDIM] == outDesc.dims.d[SDIM]);
        assert(inDesc.dims.d[mHdim] == 3 * outDesc.dims.d[mHdim]);
        if (mHasImask)
        {
            const PluginTensorDesc& maskDesc = in[MIDX].desc;
            TRT_UNUSED maskDesc;
            assert(maskDesc.dims.d[0] == inDesc.dims.d[BDIM]);
        }

        const int S = inDesc.dims.d[SDIM] <= 0 ? in->max.d[SDIM] : inDesc.dims.d[SDIM];
        const int B = inDesc.dims.d[BDIM] <= 0 ? in->max.d[BDIM] : inDesc.dims.d[BDIM];

        if (S != mS || B != mB)
        {
            // gLogVerbose << "setting up MHA runner for single sequence length" << std::endl;
            createMHARunner();
            this->dispatcher->setup(S, B);
            mS = S;
            mB = B;
        }
    }
    else
    {
        // gLogVerbose << "setting up MHA runner for variable sequence length" << std::endl;
        createMHARunner();
        // need to initialize S and B with somewhat useful values, they will be reset at enqueue for the actual
        // batchsize
        this->dispatcher->setup(256, 1);
    }
}

size_t QKVToContextVarSeqlenPlugin::getWorkspaceSize(
    const PluginTensorDesc* inputs, int nbInputs, const PluginTensorDesc* outputs, int nbOutputs) const
{
    return this->dispatcher->getWorkspaceSize();
}

// IPluginV2Ext Methods
DataType QKVToContextVarSeqlenPlugin::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes, int nbInputs) const
{
    assert(index == 0);
    assert(inputTypes[0] == DataType::kFLOAT || inputTypes[0] == DataType::kHALF || inputTypes[0] == DataType::kINT8);
    return inputTypes[0];
}

// IPluginV2 Methods
const char* QKVToContextVarSeqlenPlugin::getPluginType() const
{
    return QKV_TO_CONTEXT_PLUGIN_NAME;
}

const char* QKVToContextVarSeqlenPlugin::getPluginVersion() const
{
    return QKV_TO_CONTEXT_VAR_SEQLEN_PLUGIN_VERSION;
}

int QKVToContextVarSeqlenPlugin::getNbOutputs() const
{
    return 1;
}

int QKVToContextVarSeqlenPlugin::initialize()
{
    return 0;
}

void QKVToContextVarSeqlenPlugin::terminate() {}

size_t QKVToContextVarSeqlenPlugin::getSerializationSize() const
{
    return sizeof(mNumHeads) + sizeof(mHeadSize) + sizeof(DataType) + sizeof(mHasImask) + sizeof(mHiddenSize)
        + sizeof(mSM) + sizeof(mS) + sizeof(mB) + sizeof(mDqProbs) + dispatcher->getSerializationSize()
        + sizeof(mUseVarSeqlen) + sizeof(mHdim);
}

void QKVToContextVarSeqlenPlugin::serialize(void* buffer) const
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
    dispatcher->serialize(buffer);
}

void QKVToContextVarSeqlenPlugin::destroy()
{
    delete this;
}

void QKVToContextVarSeqlenPlugin::setPluginNamespace(const char* libNamespace)
{
    mNamespace = libNamespace;
}

const char* QKVToContextVarSeqlenPlugin::getPluginNamespace() const
{
    return mNamespace.c_str();
}

int QKVToContextVarSeqlenPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream)
{

    if (mUseVarSeqlen)
    {
        const int B = inputDesc[1].dims.d[0];
        const int maxS = inputDesc[3].dims.d[0];
        int S = 384;
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

        this->dispatcher->setup(S, B);
        this->dispatcher->run(inputDesc, outputDesc, inputs, outputs, workspace, stream);
    }
    else
    {
        assert(mS == inputDesc->dims.d[SDIM]);
        assert(mB == inputDesc->dims.d[BDIM]);

        const void* maskPtr = mHasImask ? inputs[1] : nullptr;
        this->dispatcher->run(inputDesc[0], outputDesc[0], inputs[0], maskPtr, outputs[0], workspace, stream);
        return 0;
    }

    return 0;
}

QKVToContextVarSeqlenPluginCreator::QKVToContextVarSeqlenPluginCreator()
{
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* QKVToContextVarSeqlenPluginCreator::getPluginName() const
{
    return QKV_TO_CONTEXT_PLUGIN_NAME;
}

const char* QKVToContextVarSeqlenPluginCreator::getPluginVersion() const
{
    return QKV_TO_CONTEXT_VAR_SEQLEN_PLUGIN_VERSION;
}

const PluginFieldCollection* QKVToContextVarSeqlenPluginCreator::getFieldNames()
{
    return &mFC;
}

IPluginV2* QKVToContextVarSeqlenPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
{
    gLogVerbose << "Creating QKV2ContextPlugin...\n";

    int hiddenSize = 0;
    int numHeads = 0;
    bool hasMask = false;
    int typeId = -1;

    int varSeqlen = 0;

    float dqProbs = -1;

    for (int i = 0; i < fc->nbFields; i++)
    {
        std::string field_name(fc->fields[i].name);

        if (field_name.compare("type_id") == 0)
        {
            typeId = *static_cast<const int*>(fc->fields[i].data);
            gLogVerbose << "Building typeId: " << typeId << std::endl;
        }
        if (field_name.compare("hidden_size") == 0)
        {
            hiddenSize = *static_cast<const int*>(fc->fields[i].data);
            gLogVerbose << "Building hiddenSize: " << hiddenSize << std::endl;
        }
        if (field_name.compare("num_heads") == 0)
        {
            numHeads = *static_cast<const int*>(fc->fields[i].data);
            gLogVerbose << "Building numHeads: " << numHeads << std::endl;
        }
        if (field_name.compare("has_mask") == 0)
        {
            hasMask = *static_cast<const bool*>(fc->fields[i].data);
            gLogVerbose << "Building hasMask: " << hasMask << std::endl;
        }

        if (field_name.compare("dq_probs") == 0)
        {
            dqProbs = *static_cast<const float*>(fc->fields[i].data);
            gLogVerbose << "Building dqProbs: " << dqProbs << std::endl;
        }
        if (field_name.compare("var_seqlen") == 0)
        {
            varSeqlen = *static_cast<const int*>(fc->fields[i].data);
            gLogVerbose << "Building var_seqlen: " << varSeqlen << std::endl;
        }
    }
    if (typeId < 0 || typeId > 3)
    {
        gLogError << "QKV: Invalid TypeId " << typeId << std::endl;
    }

    if (hiddenSize <= 0)
    {
        gLogError << "QKV: Invalid hiddenSize " << hiddenSize << std::endl;
    }

    if (numHeads <= 0)
    {
        gLogError << "QKV: Invalid numHeads " << numHeads << std::endl;
    }

    gLogVerbose << "Building the Plugin...\n";
    DataType type = static_cast<DataType>(typeId);
    if (type == DataType::kINT8 && dqProbs < 0)
    {
        gLogInfo << "Using default scale factor\n";
        dqProbs = 1.f / 127.f;
    }

    QKVToContextVarSeqlenPlugin* p
        = new QKVToContextVarSeqlenPlugin(name, type, hiddenSize, numHeads, dqProbs, hasMask, varSeqlen);
    return p;
}

IPluginV2* QKVToContextVarSeqlenPluginCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength)
{
    // This object will be deleted when the network is destroyed, which will
    // call QKVToContextVarSeqlenPlugin::destroy()
    return new QKVToContextVarSeqlenPlugin(name, serialData, serialLength);
}

void QKVToContextVarSeqlenPluginCreator::setPluginNamespace(const char* libNamespace)
{
    mNamespace = libNamespace;
}

const char* QKVToContextVarSeqlenPluginCreator::getPluginNamespace() const
{
    return mNamespace.c_str();
}
} // namespace bert

#endif // CUDA_VERSION >= 10010
