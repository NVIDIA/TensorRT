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

#include <cuda.h>
#include "NvInfer.h"
#include "serialize.hpp"
#include "skipLayerNormInt8InterleavedPlugin.h"

#include <cstring>
#include <vector>

using namespace nvinfer1;

namespace bert
{

void launch_small(cudaStream_t stream, const int ld, const int total, const int8_t* input, const int8_t* skip,
    const half* beta, const half* gamma, int8_t* output, const float dqScaleIn, const float dqScaleSkip,
    const float qScale);

void launch_large(cudaStream_t stream, const int ld, const int total, const int8_t* input, const int8_t* skip,
    const half* beta, const half* gamma, int8_t* output, const float dqScaleIn, const float dqScaleSkip,
    const float qScale);

// Clip plugin specific constants
namespace
{
static const char* SKIP_LAYER_NORM_INTERLEAVED_VERSION{"3"};
static const char* SKIP_LAYER_NORM_INTERLEAVED_NAME{"CustomSkipLayerNormPluginDynamic"};
} // namespace

// Static class fields initialization
PluginFieldCollection SkipLayerNormInterleavedPluginCreator::mFC{};
std::vector<PluginField> SkipLayerNormInterleavedPluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(SkipLayerNormInterleavedPluginCreator);

constexpr auto param_type = DataType::kHALF;

static inline DataType getParamWordType(DataType cfgType)
{
    if (cfgType == DataType::kINT8)
    {
        return DataType::kHALF;
    }

    return cfgType;
}

SkipLayerNormInterleavedPlugin::SkipLayerNormInterleavedPlugin(
    const std::string name, const Weights& beta, const Weights& gamma)
    : mLayerName(name)
    , mGammaDev(nullptr)
    , mBetaDev(nullptr)
    , mLd(beta.count)
    , mParamsOnDevice(false)
{
    ASSERT(mLd > 0);
    ASSERT(beta.count == gamma.count);
    // dataType for beta, gamma weights is always fp16

    mParamWordsize = getElementSize(param_type);

    mBeta.convertAndCopy(beta, param_type);
    mGamma.convertAndCopy(gamma, param_type);
}

SkipLayerNormInterleavedPlugin::SkipLayerNormInterleavedPlugin(const std::string name, const void* data, size_t length)
    : mLayerName(name)
    , mGammaDev(nullptr)
    , mBetaDev(nullptr)
    , mParamsOnDevice(false)
{
    gLogVerbose << "SkipLayerNormInterleavedPlugin deserialize\n";

    // Deserialize in the same order as serialization
    deserialize_value(&data, &length, &mLd);

    mParamWordsize = getElementSize(param_type);

    const char* d = static_cast<const char*>(data);
    mBeta.convertAndCopy(d, mLd, param_type);
    mGamma.convertAndCopy(d, mLd, param_type);
}

// IPluginV2DynamicExt Methods
IPluginV2DynamicExt* SkipLayerNormInterleavedPlugin::clone() const
{
    gLogVerbose << "SkipLayerNormInterleavedPlugin clone\n";

    auto p = new SkipLayerNormInterleavedPlugin(mLayerName, mBeta, mGamma);
    p->initialize();
    p->setPluginNamespace(mNamespace.c_str());
    return p;
}

DimsExprs SkipLayerNormInterleavedPlugin::getOutputDimensions(
    int outputIndex, const DimsExprs* inputs, int nbInputs, IExprBuilder& exprBuilder)
{
    ASSERT(nbInputs == 2);
    ASSERT(outputIndex == 0);
    ASSERT(inputs[0].nbDims == inputs[1].nbDims);
    return inputs[0];
}

bool SkipLayerNormInterleavedPlugin::supportsFormatCombination(
    int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs)
{
    ASSERT(nbInputs == 2);
    ASSERT(nbOutputs == 1);

    const PluginTensorDesc& desc = inOut[pos];
    return desc.type == DataType::kINT8 && desc.format == TensorFormat::kCHW32;
}

void SkipLayerNormInterleavedPlugin::configurePlugin(
    const DynamicPluginTensorDesc* inputs, int nbInputs, const DynamicPluginTensorDesc* outputs, int nbOutputs)
{
    // Validate input arguments
    ASSERT(nbOutputs == 1);
    ASSERT(nbInputs == 2);
    ASSERT(DataType::kINT8 == inputs[0].desc.type);
    ASSERT(DataType::kINT8 == inputs[1].desc.type);

    const auto& inDims0 = inputs[0].desc.dims;
    const auto& inDims1 = inputs[1].desc.dims;
    TRT_UNUSED inDims1;

    ASSERT(inDims0.nbDims == inDims1.nbDims);
    ASSERT(std::equal(inDims0.d, inDims0.d + inDims0.nbDims, inDims1.d));

    mParamWordsize = getElementSize(param_type);

    if (!mParamsOnDevice)
    {
        copyToDevice(mGamma, getWeightsSize(mGamma, param_type), mGammaDev);
        copyToDevice(mBeta, getWeightsSize(mBeta, param_type), mBetaDev);
        mParamsOnDevice = true;
    }
}

size_t SkipLayerNormInterleavedPlugin::getWorkspaceSize(
    const PluginTensorDesc* inputs, int nbInputs, const PluginTensorDesc* outputs, int nbOutputs) const
{
    return 0;
}

int SkipLayerNormInterleavedPlugin::enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc,
    const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream)
{
    //Input shape: 1x(hxd)xtotalx1
    const auto iDesc = inputDesc[0];
    const auto sDesc = inputDesc[1];
    const auto oDesc = outputDesc[0];
    ASSERT(iDesc.dims.nbDims == 4);
    ASSERT(iDesc.dims.nbDims == sDesc.dims.nbDims);
    ASSERT(std::equal(iDesc.dims.d, iDesc.dims.d + iDesc.dims.nbDims, sDesc.dims.d));
    ASSERT(std::equal(iDesc.dims.d, iDesc.dims.d + iDesc.dims.nbDims, oDesc.dims.d));
    ASSERT(iDesc.dims.d[0] == 1);
    ASSERT(iDesc.dims.d[3] == 1);
    ASSERT(iDesc.format == TensorFormat::kCHW32);
    ASSERT(iDesc.type == DataType::kINT8);
    ASSERT(iDesc.format == sDesc.format);
    ASSERT(iDesc.format == oDesc.format);
    ASSERT(iDesc.type == sDesc.type);
    ASSERT(iDesc.type == oDesc.type);
    const int ld = iDesc.dims.d[1];
    const int total = iDesc.dims.d[2];
    const float dqScaleIn = iDesc.scale;
    const float dqScaleSkip = sDesc.scale;
    const float qScale = 1.f / oDesc.scale;
    const int8_t* input = static_cast<const int8_t*>(inputs[0]);
    const int8_t* skip = static_cast<const int8_t*>(inputs[1]);
    int8_t* output = static_cast<int8_t*>(outputs[0]);
    const half* gamma = static_cast<const half*>(mGammaDev.get());
    const half* beta = static_cast<const half*>(mBetaDev.get());

    if(total < 4096){
        launch_small(stream, ld, total, input, skip, beta, gamma, output, dqScaleIn, dqScaleSkip, qScale);
    }else{
        launch_large(stream, ld, total, input, skip, beta, gamma, output, dqScaleIn, dqScaleSkip, qScale);
    }

    return 0;
}

// IPluginV2Ext Methods
DataType SkipLayerNormInterleavedPlugin::getOutputDataType(int index, const DataType* inputTypes, int nbInputs) const
{
    ASSERT(index == 0);
    ASSERT(nbInputs == 2);
    return inputTypes[0];
}

// IPluginV2 Methods
const char* SkipLayerNormInterleavedPlugin::getPluginType() const
{
    return SKIP_LAYER_NORM_INTERLEAVED_NAME;
}

const char* SkipLayerNormInterleavedPlugin::getPluginVersion() const
{
    return SKIP_LAYER_NORM_INTERLEAVED_VERSION;
}

int SkipLayerNormInterleavedPlugin::getNbOutputs() const
{
    return 1;
}
int SkipLayerNormInterleavedPlugin::initialize()
{
    gLogVerbose << "SkipLayerNormInterleavedPlugin initialize\n";
    return 0;
}

void SkipLayerNormInterleavedPlugin::terminate()
{
    gLogVerbose << "SkipLayerNormInterleavedPlugin terminate\n";
}

size_t SkipLayerNormInterleavedPlugin::getSerializationSize() const
{
    return 2 * mParamWordsize * mLd + sizeof(mLd);
}

void SkipLayerNormInterleavedPlugin::serialize(void* buffer) const
{
    serialize_value(&buffer, mLd);

    char* d = static_cast<char*>(buffer);
    serFromDev(d, static_cast<char*>(mBetaDev.get()), mLd * mParamWordsize);
    serFromDev(d, static_cast<char*>(mGammaDev.get()), mLd * mParamWordsize);
}

void SkipLayerNormInterleavedPlugin::destroy()
{
    gLogVerbose << "SkipLayerNormInterleavedPlugin destroy\n";
    // This gets called when the network containing plugin is destroyed
    mGammaDev.release();
    mBetaDev.release();
    delete this;
}

void SkipLayerNormInterleavedPlugin::setPluginNamespace(const char* libNamespace)
{
    mNamespace = libNamespace;
}

const char* SkipLayerNormInterleavedPlugin::getPluginNamespace() const
{
    return mNamespace.c_str();
}

/////////////////////////////////////////////////////////

SkipLayerNormInterleavedPluginCreator::SkipLayerNormInterleavedPluginCreator()
{
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* SkipLayerNormInterleavedPluginCreator::getPluginName() const
{
    return SKIP_LAYER_NORM_INTERLEAVED_NAME;
}

const char* SkipLayerNormInterleavedPluginCreator::getPluginVersion() const
{
    return SKIP_LAYER_NORM_INTERLEAVED_VERSION;
}

const PluginFieldCollection* SkipLayerNormInterleavedPluginCreator::getFieldNames()
{
    return &mFC;
}

IPluginV2* SkipLayerNormInterleavedPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
{
    gLogVerbose << "SkipLayerNormInterleavedPluginCreator createPlugin\n";

    Weights beta{DataType::kFLOAT, nullptr, 0};
    Weights gamma{DataType::kFLOAT, nullptr, 0};

    for (int i = 0; i < fc->nbFields; i++)
    {
        std::string field_name(fc->fields[i].name);

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
    }

    if (beta.count <= 0 || beta.values == nullptr)
    {
        gLogError << "SkipLayerNorm: invalid beta" << std::endl;
    }

    if (gamma.count <= 0 || gamma.values == nullptr)
    {
        gLogError << "SkipLayerNorm: invalid gamma" << std::endl;
    }

    return new SkipLayerNormInterleavedPlugin(name, beta, gamma);
}

IPluginV2* SkipLayerNormInterleavedPluginCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength)
{
    // This object will be deleted when the network is destroyed, which will
    // call SkipLayerNormInterleavedPlugin::destroy()
    return new SkipLayerNormInterleavedPlugin(name, serialData, serialLength);
}

void SkipLayerNormInterleavedPluginCreator::setPluginNamespace(const char* libNamespace)
{
    mNamespace = libNamespace;
}

const char* SkipLayerNormInterleavedPluginCreator::getPluginNamespace() const
{
    return mNamespace.c_str();
}
} // namespace bert

