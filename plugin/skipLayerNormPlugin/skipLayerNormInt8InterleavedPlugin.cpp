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

#include "skipLayerNormInt8InterleavedPlugin.h"
#include "NvInfer.h"
#include "common/serialize.hpp"
#include <cuda.h>

#include <cstring>
#include <vector>

using namespace nvinfer1;
using namespace nvinfer1::plugin;
using namespace nvinfer1::plugin::bert;

// Clip plugin specific constants
namespace
{
const char* SKIP_LAYER_NORM_INTERLEAVED_VERSION_HFACE{"3"};
const char* SKIP_LAYER_NORM_INTERLEAVED_VERSION_MTRON{"4"};
const char* SKIP_LAYER_NORM_INTERLEAVED_NAME{"CustomSkipLayerNormPluginDynamic"};
} // namespace

// Static class fields initialization
PluginFieldCollection SkipLayerNormInterleavedPluginBaseCreator::mFC{};
std::vector<PluginField> SkipLayerNormInterleavedPluginBaseCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(SkipLayerNormInterleavedPluginHFaceCreator);
REGISTER_TENSORRT_PLUGIN(SkipLayerNormInterleavedPluginMTronCreator);

constexpr auto param_type = DataType::kHALF;

static inline DataType getParamWordType(DataType cfgType)
{
    if (cfgType == DataType::kINT8)
    {
        return DataType::kHALF;
    }

    return cfgType;
}

SkipLayerNormInterleavedPluginBase::SkipLayerNormInterleavedPluginBase(
    std::string const& name, Weights const& beta, Weights const& gamma)
    : mLayerName(name)
    , mGammaDev(nullptr)
    , mBetaDev(nullptr)
    , mLd(beta.count)
    , mParamsOnDevice(false)
{
    PLUGIN_VALIDATE(mLd > 0);
    PLUGIN_VALIDATE(beta.count == gamma.count);
    // dataType for beta, gamma weights is always fp16

    mParamWordsize = getElementSize(param_type);

    mBeta.convertAndCopy(beta, param_type);
    mGamma.convertAndCopy(gamma, param_type);
}

SkipLayerNormInterleavedPluginHFace::SkipLayerNormInterleavedPluginHFace(
    std::string const& name, Weights const& beta, Weights const& gamma)
    : SkipLayerNormInterleavedPluginBase(name, beta, gamma)
{
}

SkipLayerNormInterleavedPluginMTron::SkipLayerNormInterleavedPluginMTron(
    std::string const& name, Weights const& beta, Weights const& gamma)
    : SkipLayerNormInterleavedPluginBase(name, beta, gamma)
{
}

SkipLayerNormInterleavedPluginBase::SkipLayerNormInterleavedPluginBase(
    std::string const& name, const void* data, size_t length)
    : mLayerName(name)
    , mGammaDev(nullptr)
    , mBetaDev(nullptr)
    , mParamsOnDevice(false)
{
    // Deserialize in the same order as serialization
    deserialize_value(&data, &length, &mLd);

    mParamWordsize = getElementSize(param_type);

    const char* d = static_cast<const char*>(data);
    mBeta.convertAndCopy(d, mLd, param_type);
    mGamma.convertAndCopy(d, mLd, param_type);
}

SkipLayerNormInterleavedPluginHFace::SkipLayerNormInterleavedPluginHFace(
    std::string const& name, void const* data, size_t length)
    : SkipLayerNormInterleavedPluginBase(name, data, length)
{
    BERT_DEBUG_MSG("SkipLayerNormInterleavedPluginHFace deserialize");
}

SkipLayerNormInterleavedPluginMTron::SkipLayerNormInterleavedPluginMTron(
    std::string const& name, void const* data, size_t length)
    : SkipLayerNormInterleavedPluginBase(name, data, length)
{
    BERT_DEBUG_MSG("SkipLayerNormInterleavedPluginMTron deserialize");
}

// IPluginV2DynamicExt Methods
IPluginV2DynamicExt* SkipLayerNormInterleavedPluginHFace::clone() const noexcept
{
    try
    {
        BERT_DEBUG_MSG("SkipLayerNormInterleavedPluginHFace clone");

        auto* p = new SkipLayerNormInterleavedPluginHFace(mLayerName, mBeta, mGamma);
        p->initialize();
        p->setPluginNamespace(mNamespace.c_str());
        return p;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2DynamicExt* SkipLayerNormInterleavedPluginMTron::clone() const noexcept
{
    try
    {
        BERT_DEBUG_MSG("SkipLayerNormInterleavedPluginMTron clone");

        auto* p = new SkipLayerNormInterleavedPluginMTron(mLayerName, mBeta, mGamma);
        p->initialize();
        p->setPluginNamespace(mNamespace.c_str());
        return p;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

DimsExprs SkipLayerNormInterleavedPluginBase::getOutputDimensions(
    int32_t outputIndex, const DimsExprs* inputs, int32_t nbInputs, IExprBuilder& exprBuilder) noexcept
{
    PLUGIN_ASSERT(nbInputs == 2);
    PLUGIN_ASSERT(outputIndex >= 0 && outputIndex < getNbOutputs());
    PLUGIN_ASSERT(inputs[0].nbDims == inputs[1].nbDims);
    return inputs[0];
}

bool SkipLayerNormInterleavedPluginBase::supportsFormatCombination(
    int32_t pos, const PluginTensorDesc* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    PLUGIN_ASSERT(nbInputs == 2);
    PLUGIN_ASSERT(nbOutputs == getNbOutputs());

    const PluginTensorDesc& desc = inOut[pos];
    return desc.type == DataType::kINT8 && desc.format == TensorFormat::kCHW32;
}

void SkipLayerNormInterleavedPluginBase::configurePlugin(const DynamicPluginTensorDesc* inputs, int32_t nbInputs,
    const DynamicPluginTensorDesc* outputs, int32_t nbOutputs) noexcept
{
    // Validate input arguments
    PLUGIN_ASSERT(nbOutputs == getNbOutputs());
    PLUGIN_ASSERT(nbInputs == 2);
    PLUGIN_ASSERT(DataType::kINT8 == inputs[0].desc.type);
    PLUGIN_ASSERT(DataType::kINT8 == inputs[1].desc.type);

    const auto& inDims0 = inputs[0].desc.dims;
    const auto& inDims1 = inputs[1].desc.dims;
    TRT_UNUSED inDims1;

    PLUGIN_ASSERT(inDims0.nbDims == inDims1.nbDims);
    PLUGIN_ASSERT(std::equal(inDims0.d, inDims0.d + inDims0.nbDims, inDims1.d));

    mParamWordsize = getElementSize(param_type);

    if (!mParamsOnDevice)
    {
        copyToDevice(mGamma, getWeightsSize(mGamma, param_type), mGammaDev);
        copyToDevice(mBeta, getWeightsSize(mBeta, param_type), mBetaDev);
        mParamsOnDevice = true;
    }
}

size_t SkipLayerNormInterleavedPluginBase::getWorkspaceSize(
    const PluginTensorDesc* inputs, int32_t nbInputs, const PluginTensorDesc* outputs, int32_t nbOutputs) const noexcept
{
    return 0;
}

void checkDescs(const PluginTensorDesc& iDesc, const PluginTensorDesc& sDesc, const PluginTensorDesc& oDesc)
{
    PLUGIN_ASSERT(iDesc.dims.nbDims == 4);
    PLUGIN_ASSERT(iDesc.dims.nbDims == sDesc.dims.nbDims);
    PLUGIN_ASSERT(std::equal(iDesc.dims.d, iDesc.dims.d + iDesc.dims.nbDims, sDesc.dims.d));
    PLUGIN_ASSERT(std::equal(iDesc.dims.d, iDesc.dims.d + iDesc.dims.nbDims, oDesc.dims.d));
    PLUGIN_ASSERT(iDesc.dims.d[0] == 1);
    PLUGIN_ASSERT(iDesc.dims.d[3] == 1);
    PLUGIN_ASSERT(iDesc.format == TensorFormat::kCHW32);
    PLUGIN_ASSERT(iDesc.type == DataType::kINT8);
    PLUGIN_ASSERT(iDesc.format == sDesc.format);
    PLUGIN_ASSERT(iDesc.format == oDesc.format);
    PLUGIN_ASSERT(iDesc.type == sDesc.type);
    PLUGIN_ASSERT(iDesc.type == oDesc.type);
}

int32_t SkipLayerNormInterleavedPluginHFace::enqueue(const PluginTensorDesc* inputDesc,
    const PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream) noexcept
{
    // Input shape: 1x(hxd)xtotalx1
    const auto iDesc = inputDesc[0];
    const auto sDesc = inputDesc[1];
    const auto oDesc = outputDesc[0];
    checkDescs(iDesc, sDesc, oDesc);

    const int32_t ld = iDesc.dims.d[1];
    const int32_t total = iDesc.dims.d[2];
    const float dqScaleIn = iDesc.scale;
    const float dqScaleSkip = sDesc.scale;
    const float qScale = 1.F / oDesc.scale;
    const int8_t* input = static_cast<const int8_t*>(inputs[0]);
    const int8_t* skip = static_cast<const int8_t*>(inputs[1]);
    int8_t* output = static_cast<int8_t*>(outputs[0]);
    const half* gamma = static_cast<const half*>(mGammaDev.get());
    const half* beta = static_cast<const half*>(mBetaDev.get());

    if (total < 4096)
    {
        return launch_small_hface(stream, ld, total, input, skip, beta, gamma, output, dqScaleIn, dqScaleSkip, qScale);
    }

    return launch_large_hface(stream, ld, total, input, skip, beta, gamma, output, dqScaleIn, dqScaleSkip, qScale);
}

int32_t SkipLayerNormInterleavedPluginMTron::enqueue(PluginTensorDesc const* inputDesc,
    PluginTensorDesc const* outputDesc, void const* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream) noexcept
{
    // Input shape: 1x(hxd)xtotalx1
    const auto iDesc = inputDesc[0];
    const auto sDesc = inputDesc[1];
    const auto oDesc = outputDesc[0];
    const auto pDesc = outputDesc[1];
    checkDescs(iDesc, sDesc, oDesc);
    PLUGIN_ASSERT(std::equal(iDesc.dims.d, iDesc.dims.d + iDesc.dims.nbDims, pDesc.dims.d));

    const int32_t ld = iDesc.dims.d[1];
    const int32_t total = iDesc.dims.d[2];
    const float dqScaleIn = iDesc.scale;
    const float dqScaleSkip = sDesc.scale;
    const float qScale = 1.F / oDesc.scale;
    const float qSkipScale = 1.F / pDesc.scale;
    const int8_t* input = static_cast<const int8_t*>(inputs[0]);
    const int8_t* skip = static_cast<const int8_t*>(inputs[1]);
    int8_t* output = static_cast<int8_t*>(outputs[0]);
    int8_t* preln = static_cast<int8_t*>(outputs[1]);
    const half* gamma = static_cast<const half*>(mGammaDev.get());
    const half* beta = static_cast<const half*>(mBetaDev.get());

    if (total < 4096)
    {
        return launch_small_mtron(
            stream, ld, total, input, skip, beta, gamma, output, preln, dqScaleIn, dqScaleSkip, qScale, qSkipScale);
    }

    return launch_large_mtron(
        stream, ld, total, input, skip, beta, gamma, output, preln, dqScaleIn, dqScaleSkip, qScale, qSkipScale);
}

// IPluginV2Ext Methods
DataType SkipLayerNormInterleavedPluginBase::getOutputDataType(
    int32_t index, const DataType* inputTypes, int32_t nbInputs) const noexcept
{
    PLUGIN_ASSERT(index >= 0 && index < getNbOutputs());
    PLUGIN_ASSERT(nbInputs == 2);
    return inputTypes[0];
}

// IPluginV2 Methods
const char* SkipLayerNormInterleavedPluginBase::getPluginType() const noexcept
{
    return SKIP_LAYER_NORM_INTERLEAVED_NAME;
}

const char* SkipLayerNormInterleavedPluginHFace::getPluginVersion() const noexcept
{
    return SKIP_LAYER_NORM_INTERLEAVED_VERSION_HFACE;
}

const char* SkipLayerNormInterleavedPluginMTron::getPluginVersion() const noexcept
{
    return SKIP_LAYER_NORM_INTERLEAVED_VERSION_MTRON;
}

int32_t SkipLayerNormInterleavedPluginHFace::getNbOutputs() const noexcept
{
    return 1;
}

int32_t SkipLayerNormInterleavedPluginMTron::getNbOutputs() const noexcept
{
    return 2;
}

int32_t SkipLayerNormInterleavedPluginHFace::initialize() noexcept
{
    BERT_DEBUG_MSG("SkipLayerNormInterleavedPluginHFace initialize");
    return 0;
}

int32_t SkipLayerNormInterleavedPluginMTron::initialize() noexcept
{
    BERT_DEBUG_MSG("SkipLayerNormInterleavedPluginMTron initialize");
    return 0;
}

void SkipLayerNormInterleavedPluginHFace::terminate() noexcept
{
    BERT_DEBUG_MSG("SkipLayerNormInterleavedPluginHFace terminate");
}

void SkipLayerNormInterleavedPluginMTron::terminate() noexcept
{
    BERT_DEBUG_MSG("SkipLayerNormInterleavedPluginMTron terminate");
}

size_t SkipLayerNormInterleavedPluginBase::getSerializationSize() const noexcept
{
    return 2 * mParamWordsize * mLd + sizeof(mLd);
}

void SkipLayerNormInterleavedPluginBase::serialize(void* buffer) const noexcept
{
    serialize_value(&buffer, mLd);

    char* d = static_cast<char*>(buffer);
    serFromDev(d, static_cast<char*>(mBetaDev.get()), mLd * mParamWordsize);
    serFromDev(d, static_cast<char*>(mGammaDev.get()), mLd * mParamWordsize);
}

void SkipLayerNormInterleavedPluginBase::destroy() noexcept
{
    // This gets called when the network containing plugin is destroyed
    mGammaDev.reset(nullptr);
    mBetaDev.reset(nullptr);
    delete this;
}

void SkipLayerNormInterleavedPluginHFace::destroy() noexcept
{
    BERT_DEBUG_MSG("SkipLayerNormInterleavedPluginHFace destroy");
    SkipLayerNormInterleavedPluginBase::destroy();
}

void SkipLayerNormInterleavedPluginMTron::destroy() noexcept
{
    BERT_DEBUG_MSG("SkipLayerNormInterleavedPluginMTron destroy");
    SkipLayerNormInterleavedPluginBase::destroy();
}

void SkipLayerNormInterleavedPluginBase::setPluginNamespace(const char* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

const char* SkipLayerNormInterleavedPluginBase::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

/////////////////////////////////////////////////////////

SkipLayerNormInterleavedPluginBaseCreator::SkipLayerNormInterleavedPluginBaseCreator()
{
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("beta"));
    mPluginAttributes.emplace_back(PluginField("gamma"));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

SkipLayerNormInterleavedPluginHFaceCreator::SkipLayerNormInterleavedPluginHFaceCreator()
    : SkipLayerNormInterleavedPluginBaseCreator()
{
}

SkipLayerNormInterleavedPluginMTronCreator::SkipLayerNormInterleavedPluginMTronCreator()
    : SkipLayerNormInterleavedPluginBaseCreator()
{
}

const char* SkipLayerNormInterleavedPluginBaseCreator::getPluginName() const noexcept
{
    return SKIP_LAYER_NORM_INTERLEAVED_NAME;
}

const char* SkipLayerNormInterleavedPluginHFaceCreator::getPluginVersion() const noexcept
{
    return SKIP_LAYER_NORM_INTERLEAVED_VERSION_HFACE;
}

const char* SkipLayerNormInterleavedPluginMTronCreator::getPluginVersion() const noexcept
{
    return SKIP_LAYER_NORM_INTERLEAVED_VERSION_MTRON;
}

const PluginFieldCollection* SkipLayerNormInterleavedPluginBaseCreator::getFieldNames() noexcept
{
    return &mFC;
}

void buildBetaAndGamma(const PluginFieldCollection* fc, Weights& beta, Weights& gamma)
{
    plugin::validateRequiredAttributesExist({"beta", "gamma"}, fc);

    for (int32_t i = 0; i < fc->nbFields; i++)
    {
        std::string field_name(fc->fields[i].name);

        if (field_name.compare("beta") == 0)
        {
            BERT_DEBUG_MSG("Building beta...");
            beta.values = fc->fields[i].data;
            beta.count = fc->fields[i].length;
            beta.type = fieldTypeToDataType(fc->fields[i].type);
        }

        if (field_name.compare("gamma") == 0)
        {
            BERT_DEBUG_MSG("Building gamma...");
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
}

IPluginV2* SkipLayerNormInterleavedPluginHFaceCreator::createPlugin(
    const char* name, const PluginFieldCollection* fc) noexcept
{
    try
    {
        BERT_DEBUG_MSG("SkipLayerNormInterleavedPluginHFaceCreator createPlugin");

        Weights beta{DataType::kFLOAT, nullptr, 0};
        Weights gamma{DataType::kFLOAT, nullptr, 0};
        buildBetaAndGamma(fc, beta, gamma);

        return new SkipLayerNormInterleavedPluginHFace(name, beta, gamma);
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* SkipLayerNormInterleavedPluginMTronCreator::createPlugin(
    const char* name, const PluginFieldCollection* fc) noexcept
{
    try
    {
        BERT_DEBUG_MSG("SkipLayerNormInterleavedPluginMTronCreator createPlugin");

        Weights beta{DataType::kFLOAT, nullptr, 0};
        Weights gamma{DataType::kFLOAT, nullptr, 0};
        buildBetaAndGamma(fc, beta, gamma);

        return new SkipLayerNormInterleavedPluginMTron(name, beta, gamma);
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* SkipLayerNormInterleavedPluginHFaceCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength) noexcept
{
    // This object will be deleted when the network is destroyed, which will
    // call SkipLayerNormInterleavedPlugin::destroy()
    try
    {
        BERT_DEBUG_MSG("SkipLayerNormInterleavedPluginHFaceCreator deserializePlugin");
        return new SkipLayerNormInterleavedPluginHFace(name, serialData, serialLength);
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* SkipLayerNormInterleavedPluginMTronCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength) noexcept
{
    // This object will be deleted when the network is destroyed, which will
    // call SkipLayerNormInterleavedPlugin::destroy()
    try
    {
        BERT_DEBUG_MSG("SkipLayerNormInterleavedPluginMTronCreator deserializePlugin");
        return new SkipLayerNormInterleavedPluginMTron(name, serialData, serialLength);
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

void SkipLayerNormInterleavedPluginBaseCreator::setPluginNamespace(const char* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

const char* SkipLayerNormInterleavedPluginBaseCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}
