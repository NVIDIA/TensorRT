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

#include <cuda.h>
#if CUDA_VERSION >= 10010

#include "NvInfer.h"
#include "serialize.hpp"
#include "skipLayerNormPlugin.h"

#include <cassert>
#include <cstring>
#include <vector>

using namespace nvinfer1;

namespace bert
{

// Clip plugin specific constants
namespace
{
static const char* SKIP_LAYER_NORM_VERSION{"1"};
static const char* SKIP_LAYER_NORM_NAME{"CustomSkipLayerNormPluginDynamic"};
} // namespace

// Static class fields initialization
PluginFieldCollection SkipLayerNormPluginDynamicCreator::mFC{};
std::vector<PluginField> SkipLayerNormPluginDynamicCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(SkipLayerNormPluginDynamicCreator);

static inline DataType getParamWordType(DataType cfgType)
{
    if (cfgType == DataType::kINT8)
    {
        return DataType::kHALF;
    }

    return cfgType;
}

SkipLayerNormPluginDynamic::SkipLayerNormPluginDynamic(const std::string name, const DataType type, const int ld,
    const Weights& beta, const Weights& gamma, const Weights& bias)
    : mLayerName(name)
    , mGammaDev(nullptr)
    , mBetaDev(nullptr)
    , mLd(ld)
    , mType(type)
    , mBiasDev(nullptr)
{
    assert(mType == nvinfer1::DataType::kFLOAT || mType == nvinfer1::DataType::kHALF
        || mType == nvinfer1::DataType::kINT8);
    // mCfgType is the dataType for beta, gamma bias weights, always fp16 or fp32
    // mType is the plugin IO datatype, can be int8
    mCfgType = mType == DataType::kINT8 ? DataType::kHALF : mType;
    mParamWordsize = getElementSize(mCfgType);

    mBeta.convertAndCopy(beta, mCfgType);
    mGamma.convertAndCopy(gamma, mCfgType);

    mHasBias = (bias.values != nullptr);
    if (mHasBias)
    {
        mBias.convertAndCopy(bias, mCfgType);
    }
}

SkipLayerNormPluginDynamic::SkipLayerNormPluginDynamic(const std::string name, const void* data, size_t length)
    : mLayerName(name)
    , mGammaDev(nullptr)
    , mBetaDev(nullptr)
    , mBiasDev(nullptr)
{
    gLogVerbose << "SkipLayerNormPluginDynamic deserialize\n";

    // Deserialize in the same order as serialization
    deserialize_value(&data, &length, &mType);
    deserialize_value(&data, &length, &mCfgType);
    deserialize_value(&data, &length, &mLd);
    deserialize_value(&data, &length, &mHasBias);

    assert(mCfgType == nvinfer1::DataType::kFLOAT || mCfgType == nvinfer1::DataType::kHALF);
    mParamWordsize = getElementSize(mCfgType);

    const char* d = static_cast<const char*>(data);
    mBeta.convertAndCopy(d, mLd, mCfgType);
    mGamma.convertAndCopy(d, mLd, mCfgType);
    if (mHasBias)
    {
        mBias.convertAndCopy(d, mLd, mCfgType);
    }
}

// IPluginV2DynamicExt Methods
IPluginV2DynamicExt* SkipLayerNormPluginDynamic::clone() const
{
    gLogVerbose << "SkipLayerNormPluginDynamic clone\n";

    auto p = new SkipLayerNormPluginDynamic(mLayerName, mType, mLd, mBeta, mGamma, mBias);
    p->initialize();
    p->setPluginNamespace(mNamespace.c_str());
    return p;
}

DimsExprs SkipLayerNormPluginDynamic::getOutputDimensions(
    int outputIndex, const DimsExprs* inputs, int nbInputs, IExprBuilder& exprBuilder)
{
    assert(nbInputs == 2);
    assert(outputIndex == 0);
    assert(inputs[0].nbDims == inputs[1].nbDims);
    return inputs[0];
}

bool SkipLayerNormPluginDynamic::supportsFormatCombination(
    int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs)
{
    assert(nbInputs == 2);
    assert(nbOutputs == 1);

    const PluginTensorDesc& in = inOut[pos];
    if (pos == 0)
    {
        // Since H = W = 1, we can report CHWx for any x
        if (mType == DataType::kINT8)
        {
            // won't work for hiddensize too small!
            TensorFormat myFmt = TensorFormat::kCHW32;
            if (mLd < 32)
            {
                myFmt = TensorFormat::kCHW4;
                gLogVerbose << "SkipLayerNormDQQ: TensorFormat CHW4"
                            << " for LD=" << mLd << std::endl;
            }
            else
            {
                gLogVerbose << "SkipLayerNormDQQ: TensorFormat CHW32"
                            << " for LD=" << mLd << std::endl;
            }
            // TODO do we need to check if the vectorization divides mLd?
            return ((in.type == mType) && (in.format == myFmt));
        }
        return (in.type == mType) && (in.format == TensorFormat::kLINEAR);
    }
    const PluginTensorDesc& prev = inOut[pos - 1];

    return in.type == prev.type && in.format == prev.format;
}

void SkipLayerNormPluginDynamic::configurePlugin(
    const DynamicPluginTensorDesc* inputs, int nbInputs, const DynamicPluginTensorDesc* outputs, int nbOutputs)
{
    gLogVerbose << "SkipLayerNormPluginDynamic configurePlugin\n";

    // Validate input arguments
    assert(nbOutputs == 1);
    assert(nbInputs == 2);
    if (mType == DataType::kFLOAT || mType == DataType::kHALF)
    {
        assert(mType == inputs[0].desc.type);
        assert(mType == inputs[1].desc.type);
    }
    else
    {
        assert(mType == inputs[0].desc.type || DataType::kFLOAT == inputs[0].desc.type);
        assert(mType == inputs[1].desc.type || DataType::kFLOAT == inputs[1].desc.type);
    }
    const auto& inDims0 = inputs[0].desc.dims;
    const auto& inDims1 = inputs[1].desc.dims;
    TRT_UNUSED inDims1;
    assert(inDims0.nbDims == inDims1.nbDims);

    assert(std::equal(inDims0.d, inDims0.d + inDims0.nbDims, inDims1.d));

    assert(inDims0.nbDims == 5);
    mLd = inDims0.d[HDIM]; // hiddensize
    assert(inDims0.d[3] == 1);
    assert(inDims0.d[4] == 1);

    mCfgType = inputs[0].desc.type == DataType::kINT8 ? DataType::kHALF : inputs[0].desc.type;

    const auto paramType = getParamWordType(mCfgType);
    mParamWordsize = getElementSize(paramType);

    copyToDevice(mGamma, getWeightsSize(mGamma, paramType), mGammaDev);
    copyToDevice(mBeta, getWeightsSize(mBeta, paramType), mBetaDev);
    if (mHasBias)
    {
        copyToDevice(mBias, getWeightsSize(mBias, paramType), mBiasDev);
    }
}

size_t SkipLayerNormPluginDynamic::getWorkspaceSize(
    const PluginTensorDesc* inputs, int nbInputs, const PluginTensorDesc* outputs, int nbOutputs) const
{
    return 0;
}

int SkipLayerNormPluginDynamic::enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc,
    const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream)
{
    const int inputVolume = volume(inputDesc[0].dims);
    int status = -1;
    DataType iType = inputDesc->type;

    // Our plugin outputs only one tensor
    // Launch CUDA kernel wrapper and save its return value
    if (iType == DataType::kFLOAT)
    {
        const auto input = static_cast<const float*>(inputs[0]);
        const auto skip = static_cast<const float*>(inputs[1]);
        auto output = static_cast<float*>(outputs[0]);
        const auto bias = static_cast<const float*>(mBiasDev.get());
        const auto beta = static_cast<const float*>(mBetaDev.get());
        const auto gamma = static_cast<const float*>(mGammaDev.get());
        if (mHasBias)
        {
            status = computeSkipLayerNorm<float, true>(
                stream, static_cast<int>(mLd), inputVolume, input, skip, beta, gamma, output, bias);
        }
        else
        {
            status = computeSkipLayerNorm<float, false>(
                stream, static_cast<int>(mLd), inputVolume, input, skip, beta, gamma, output, bias);
        }
    }
    else if (iType == DataType::kHALF)
    {
        const auto input = static_cast<const half*>(inputs[0]);
        const auto skip = static_cast<const half*>(inputs[1]);
        auto output = static_cast<half*>(outputs[0]);
        const auto bias = static_cast<const half*>(mBiasDev.get());
        const auto beta = static_cast<const half*>(mBetaDev.get());
        const auto gamma = static_cast<const half*>(mGammaDev.get());
        if (mHasBias)
        {
            status = computeSkipLayerNorm<half, true>(
                stream, static_cast<int>(mLd), inputVolume, input, skip, beta, gamma, output, bias);
        }
        else
        {
            status = computeSkipLayerNorm<half, false>(
                stream, static_cast<int>(mLd), inputVolume, input, skip, beta, gamma, output, bias);
        }
    }
    else if (iType == DataType::kINT8)
    {
        const float dqScaleIn = inputDesc[0].scale;
        const float dqScaleSkip = inputDesc[1].scale;
        const float qScale = 1.f / outputDesc[0].scale;
        const auto input = static_cast<const int8_t*>(inputs[0]);
        const auto skip = static_cast<const int8_t*>(inputs[1]);
        auto output = static_cast<int8_t*>(outputs[0]);
        const auto bias = static_cast<const half*>(mBiasDev.get());
        const auto beta = static_cast<const half*>(mBetaDev.get());
        const auto gamma = static_cast<const half*>(mGammaDev.get());
        if (mHasBias)
        {
            status = computeSkipLayerNormDQQ<true>(stream, static_cast<int>(mLd), inputVolume, input, skip, beta, gamma,
                output, bias, dqScaleIn, dqScaleSkip, qScale);
        }
        else
        {
            status = computeSkipLayerNormDQQ<false>(stream, static_cast<int>(mLd), inputVolume, input, skip, beta,
                gamma, output, bias, dqScaleIn, dqScaleSkip, qScale);
        }
    }
    else
    {
        gLogError << "Unsupported type error, expected [kINT8,kHALF,kFLOAT], but received " << static_cast<int>(iType)
                  << "." << std::endl;
        assert(false);
    }
    return status;
}

// IPluginV2Ext Methods
DataType SkipLayerNormPluginDynamic::getOutputDataType(int index, const DataType* inputTypes, int nbInputs) const
{
    assert(index == 0);
    assert(nbInputs == 2);
    assert(inputTypes[0] == inputTypes[1]);
    return inputTypes[0];
}

// IPluginV2 Methods
const char* SkipLayerNormPluginDynamic::getPluginType() const
{
    return SKIP_LAYER_NORM_NAME;
}

const char* SkipLayerNormPluginDynamic::getPluginVersion() const
{
    return SKIP_LAYER_NORM_VERSION;
}

int SkipLayerNormPluginDynamic::getNbOutputs() const
{
    return 1;
}
int SkipLayerNormPluginDynamic::initialize()
{
    gLogVerbose << "SkipLayerNormPluginDynamic initialize\n";
    return 0;
}

void SkipLayerNormPluginDynamic::terminate()
{
    gLogVerbose << "SkipLayerNormPluginDynamic terminate\n";
}

size_t SkipLayerNormPluginDynamic::getSerializationSize() const
{
    const size_t biasSize = mHasBias ? (mLd * mParamWordsize) : 0;
    return 2 * mParamWordsize * mLd + 2 * sizeof(DataType) + sizeof(mLd) + biasSize + sizeof(mHasBias);
}

void SkipLayerNormPluginDynamic::serialize(void* buffer) const
{
    serialize_value(&buffer, mType);
    serialize_value(&buffer, mCfgType);
    serialize_value(&buffer, mLd);
    serialize_value(&buffer, mHasBias);

    char* d = static_cast<char*>(buffer);
    serFromDev(d, static_cast<char*>(mBetaDev.get()), mLd * mParamWordsize);
    serFromDev(d, static_cast<char*>(mGammaDev.get()), mLd * mParamWordsize);
    if (mHasBias)
    {
        serFromDev(d, static_cast<char*>(mBiasDev.get()), mLd * mParamWordsize);
    }
}

void SkipLayerNormPluginDynamic::destroy()
{
    gLogVerbose << "SkipLayerNormPluginDynamic destroy\n";
    // This gets called when the network containing plugin is destroyed
    mGammaDev.release();
    mBetaDev.release();
    mBiasDev.release();
    delete this;
}

void SkipLayerNormPluginDynamic::setPluginNamespace(const char* libNamespace)
{
    mNamespace = libNamespace;
}

const char* SkipLayerNormPluginDynamic::getPluginNamespace() const
{
    return mNamespace.c_str();
}

/////////////////////////////////////////////////////////

SkipLayerNormPluginDynamicCreator::SkipLayerNormPluginDynamicCreator()
{
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* SkipLayerNormPluginDynamicCreator::getPluginName() const
{
    return SKIP_LAYER_NORM_NAME;
}

const char* SkipLayerNormPluginDynamicCreator::getPluginVersion() const
{
    return SKIP_LAYER_NORM_VERSION;
}

const PluginFieldCollection* SkipLayerNormPluginDynamicCreator::getFieldNames()
{
    return &mFC;
}

IPluginV2* SkipLayerNormPluginDynamicCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
{
    gLogVerbose << "SkipLayerNormPluginDynamicCreator createPlugin\n";

    int ld = 0;
    Weights beta{DataType::kFLOAT, nullptr, 0};
    Weights gamma{DataType::kFLOAT, nullptr, 0};
    Weights bias{DataType::kFLOAT, nullptr, 0};
    int typeId = -1;

    for (int i = 0; i < fc->nbFields; i++)
    {
        std::string field_name(fc->fields[i].name);
        if (field_name.compare("ld") == 0)
        {
            ld = *static_cast<const int*>(fc->fields[i].data);
            gLogVerbose << "Building ld: " << ld << std::endl;
        }

        if (field_name.compare("type_id") == 0)
        {
            typeId = *static_cast<const int*>(fc->fields[i].data);
            gLogVerbose << "Building typeId: " << typeId << std::endl;
        }

        if (field_name.compare("beta") == 0)
        {
            gLogVerbose << "Building beta...\n";
            beta.values = fc->fields[i].data;
            beta.count = fc->fields[i].length;
            beta.type = fieldTypeToDataType(fc->fields[i].type);
        }

        if (field_name.compare("gamma") == 0)
        {
            gLogVerbose << "Building gamma...\n";
            gamma.values = fc->fields[i].data;
            gamma.count = fc->fields[i].length;
            gamma.type = fieldTypeToDataType(fc->fields[i].type);
        }

        if (field_name.compare("bias") == 0)
        {
            gLogVerbose << "Building bias...\n";
            bias.values = fc->fields[i].data;
            bias.count = fc->fields[i].length;
            bias.type = fieldTypeToDataType(fc->fields[i].type);
        }
    }
    gLogVerbose << "Type " << typeId << std::endl;

    if (typeId < 0 || typeId > 3)
    {
        gLogError << "SkipLayerNorm: Invalid type ID: " << typeId << std::endl;
    }

    if (beta.count <= 0 || beta.values == nullptr)
    {
        gLogError << "SkipLayerNorm: invalid beta" << std::endl;
    }

    if (gamma.count <= 0 || gamma.values == nullptr)
    {
        gLogError << "SkipLayerNorm: invalid gamma" << std::endl;
    }

    return new SkipLayerNormPluginDynamic(name, static_cast<DataType>(typeId), ld, beta, gamma, bias);
}

IPluginV2* SkipLayerNormPluginDynamicCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength)
{
    // This object will be deleted when the network is destroyed, which will
    // call SkipLayerNormPluginDynamic::destroy()
    return new SkipLayerNormPluginDynamic(name, serialData, serialLength);
}

void SkipLayerNormPluginDynamicCreator::setPluginNamespace(const char* libNamespace)
{
    mNamespace = libNamespace;
}

const char* SkipLayerNormPluginDynamicCreator::getPluginNamespace() const
{
    return mNamespace.c_str();
}
} // namespace bert

#endif // CUDA_VERSION >= 10010
