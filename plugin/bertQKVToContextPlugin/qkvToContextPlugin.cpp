/*
 * Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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
static const char* QKV_TO_CONTEXT_PLUGIN_NAME{"CustomQKVToContextPluginDynamic"};
} // namespace

// Static class fields initialization
PluginFieldCollection QKVToContextPluginDynamicCreator::mFC{};
std::vector<PluginField> QKVToContextPluginDynamicCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(QKVToContextPluginDynamicCreator);

constexpr size_t kAlignment = 256;
constexpr uint32_t IIDX = 0; // index of the input tensor
constexpr uint32_t MIDX = 1; // index of the mask

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
    dispatcher->deserialize(data, length);

    gLogVerbose << "QKV Deser done" << std::endl;
}

int QKVToContextPluginDynamic::getSMVersion() const
{
    int device{-1};
    CHECK(cudaGetDevice(&device));
    cudaDeviceProp props;
    CHECK(cudaGetDeviceProperties(&props, device));
    return props.major * 10 + props.minor;
}

void QKVToContextPluginDynamic::createMHARunner()
{
    assert(getSMVersion() == mSM);

    if (mType == DataType::kHALF)
    {
        dispatcher.reset(new FusedMHARunnerFP16(mNumHeads, mHeadSize, mSM));
    }
    else if (mType == DataType::kINT8)
    {
        dispatcher.reset(new FusedMHARunnerInt8(mNumHeads, mHeadSize, mSM, mDqProbs));
    }
    if (!dispatcher || !dispatcher->isValid())
    {
        dispatcher.reset(new UnfusedMHARunner(mType, mNumHeads, mHeadSize));
    }
}

// IPluginV2DynamicExt Methods
nvinfer1::IPluginV2DynamicExt* QKVToContextPluginDynamic::clone() const
{
    gLogVerbose << "QKV Clone" << std::endl;

    QKVToContextPluginDynamic* ret = nullptr;
    if (dispatcher.get())
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

    // TODO cleanup
    // TODO add CHW32
    assert(pos >= 0);
    assert(pos < 2 + mHasImask);
    assert(nbInputs == 1 + mHasImask);
    const auto* in = inOut;
    const auto* out = inOut + nbInputs;

    // we only support int8 IO in fused mha runner, and we only support fused mha runner on Turing and Ampere
    if (mType == DataType::kINT8 && mSM != bert::kSM_AMPERE && mSM != bert::kSM_TURING)
    {
        gLogVerbose << "INT8 IO is only supported on Turing and Ampere for plugin " << QKV_TO_CONTEXT_PLUGIN_NAME
                    << std::endl;
        return false;
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
            // detect full mask and check that it was produced
            const bool useFullMask = (mType == DataType::kHALF || mType == DataType::kINT8)
                && (in->dims.d[SDIM] == 128 || in->dims.d[SDIM] == 384) && (mSM == kSM_TURING || mSM == kSM_AMPERE);

            if (useFullMask)
            {
                return (inMask->type == DataType::kHALF) &&      // precision
                    (inMask->format == TensorFormat::kLINEAR) && // format
                    (inMask->dims.nbDims == 2) &&                // Bx2*maskSize
                    ((inMask->dims.d[0]) == in->dims.d[BDIM]);
            }
            return (inMask->type == DataType::kINT32) &&     // precision
                (inMask->format == TensorFormat::kLINEAR) && // format
                (inMask->dims.nbDims == 1) &&                // num dims
                ((inMask->dims.d[0]) == in->dims.d[BDIM])    // check B
                ;
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

    const int S = inDesc.dims.d[SDIM] <= 0 ? in->max.d[SDIM] : inDesc.dims.d[SDIM];
    const int B = inDesc.dims.d[BDIM] <= 0 ? in->max.d[BDIM] : inDesc.dims.d[BDIM];

    if (S != mS || B != mB)
    {
        createMHARunner();
        this->dispatcher->setup(S, B);
        mS = S;
        mB = B;
    }
}

size_t QKVToContextPluginDynamic::getWorkspaceSize(
    const PluginTensorDesc* inputs, int nbInputs, const PluginTensorDesc* outputs, int nbOutputs) const
{
    return this->dispatcher->getWorkspaceSize();
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
    return sizeof(mNumHeads) + sizeof(mHeadSize) + sizeof(DataType) + sizeof(mHasImask) + sizeof(mHiddenSize)
        + sizeof(mSM) + sizeof(mS) + sizeof(mB) + sizeof(mDqProbs) + dispatcher->getSerializationSize();
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
    dispatcher->serialize(buffer);
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
    this->dispatcher->run(inputDesc[0], outputDesc[0], inputs[0], maskPtr, outputs[0], workspace, stream);
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

    QKVToContextPluginDynamic* p = new QKVToContextPluginDynamic(name, type, hiddenSize, numHeads, dqProbs, hasMask);
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
} // namespace bert

#endif // CUDA_VERSION >= 10010
