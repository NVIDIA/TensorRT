/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION &
 * AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
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

#include "skipLayerNormInt8InterleavedPluginLegacy.h"
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
constexpr char const* kSKIP_LAYER_NORM_INTERLEAVED_VERSION_HFACE_LEGACY{"3"};
constexpr char const* kSKIP_LAYER_NORM_INTERLEAVED_VERSION_MTRON_LEGACY{"4"};
constexpr char const* kSKIP_LAYER_NORM_INTERLEAVED_NAME{"CustomSkipLayerNormPluginDynamic"};

void buildBetaAndGamma(PluginFieldCollection const* fc, Weights& beta, Weights& gamma)
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

    PLUGIN_VALIDATE(beta.values != nullptr, "SkipLayerNorm: invalid beta");
    PLUGIN_VALIDATE(beta.count > 0, "SkipLayerNorm: invalid beta");

    PLUGIN_VALIDATE(gamma.values != nullptr, "SkipLayerNorm: invalid gamma");
    PLUGIN_VALIDATE(gamma.count > 0, "SkipLayerNorm: invalid gamma");
}

void checkDescs(PluginTensorDesc const& iDesc, PluginTensorDesc const& sDesc, PluginTensorDesc const& oDesc)
{
    PLUGIN_VALIDATE(iDesc.dims.nbDims == 4);
    PLUGIN_VALIDATE(iDesc.dims.nbDims == sDesc.dims.nbDims);
    PLUGIN_VALIDATE(std::equal(iDesc.dims.d, iDesc.dims.d + iDesc.dims.nbDims, sDesc.dims.d));
    PLUGIN_VALIDATE(std::equal(iDesc.dims.d, iDesc.dims.d + iDesc.dims.nbDims, oDesc.dims.d));
    PLUGIN_VALIDATE(iDesc.dims.d[0] == 1);
    PLUGIN_VALIDATE(iDesc.dims.d[3] == 1);
    PLUGIN_VALIDATE(iDesc.format == TensorFormat::kCHW32);
    PLUGIN_VALIDATE(iDesc.type == DataType::kINT8);
    PLUGIN_VALIDATE(iDesc.format == sDesc.format);
    PLUGIN_VALIDATE(iDesc.format == oDesc.format);
    PLUGIN_VALIDATE(iDesc.type == sDesc.type);
    PLUGIN_VALIDATE(iDesc.type == oDesc.type);
}
} // namespace

REGISTER_TENSORRT_PLUGIN(SkipLayerNormInterleavedPluginHFaceLegacyCreator);
REGISTER_TENSORRT_PLUGIN(SkipLayerNormInterleavedPluginMTronLegacyCreator);

constexpr auto kPARAM_TYPE = DataType::kHALF;

SkipLayerNormInterleavedPluginBaseLegacy::SkipLayerNormInterleavedPluginBaseLegacy(
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

    mParamWordsize = getElementSize(kPARAM_TYPE);

    mBeta.convertAndCopy(beta, kPARAM_TYPE);
    mGamma.convertAndCopy(gamma, kPARAM_TYPE);
}

SkipLayerNormInterleavedPluginHFaceLegacy::SkipLayerNormInterleavedPluginHFaceLegacy(
    std::string const& name, Weights const& beta, Weights const& gamma)
    : SkipLayerNormInterleavedPluginBaseLegacy(name, beta, gamma)
{
}

SkipLayerNormInterleavedPluginMTronLegacy::SkipLayerNormInterleavedPluginMTronLegacy(
    std::string const& name, Weights const& beta, Weights const& gamma)
    : SkipLayerNormInterleavedPluginBaseLegacy(name, beta, gamma)
{
}

SkipLayerNormInterleavedPluginBaseLegacy::SkipLayerNormInterleavedPluginBaseLegacy(
    std::string const& name, void const* data, size_t length)
    : mLayerName(name)
    , mGammaDev(nullptr)
    , mBetaDev(nullptr)
    , mParamsOnDevice(false)
{
    // Deserialize in the same order as serialization
    deserialize_value(&data, &length, &mLd);

    mParamWordsize = getElementSize(kPARAM_TYPE);

    char const* d = static_cast<char const*>(data);
    mBeta.convertAndCopy(d, mLd, kPARAM_TYPE);
    mGamma.convertAndCopy(d, mLd, kPARAM_TYPE);
}

SkipLayerNormInterleavedPluginHFaceLegacy::SkipLayerNormInterleavedPluginHFaceLegacy(
    std::string const& name, void const* data, size_t length)
    : SkipLayerNormInterleavedPluginBaseLegacy(name, data, length)
{
    BERT_DEBUG_MSG("SkipLayerNormInterleavedPluginHFaceLegacy deserialize");
}

SkipLayerNormInterleavedPluginMTronLegacy::SkipLayerNormInterleavedPluginMTronLegacy(
    std::string const& name, void const* data, size_t length)
    : SkipLayerNormInterleavedPluginBaseLegacy(name, data, length)
{
    BERT_DEBUG_MSG("SkipLayerNormInterleavedPluginMTronLegacy deserialize");
}

// IPluginV2DynamicExt Methods
IPluginV2DynamicExt* SkipLayerNormInterleavedPluginHFaceLegacy::clone() const noexcept
{
    try
    {
        BERT_DEBUG_MSG("SkipLayerNormInterleavedPluginHFaceLegacy clone");

        auto* p = new SkipLayerNormInterleavedPluginHFaceLegacy(mLayerName, mBeta, mGamma);
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

IPluginV2DynamicExt* SkipLayerNormInterleavedPluginMTronLegacy::clone() const noexcept
{
    try
    {
        BERT_DEBUG_MSG("SkipLayerNormInterleavedPluginMTronLegacy clone");

        auto* p = new SkipLayerNormInterleavedPluginMTronLegacy(mLayerName, mBeta, mGamma);
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

DimsExprs SkipLayerNormInterleavedPluginBaseLegacy::getOutputDimensions(
    int32_t outputIndex, DimsExprs const* inputs, int32_t nbInputs, IExprBuilder& exprBuilder) noexcept
{
    try
    {
        PLUGIN_VALIDATE(inputs != nullptr);
        PLUGIN_VALIDATE(nbInputs == 2);
        PLUGIN_VALIDATE(outputIndex >= 0 && outputIndex < getNbOutputs());
        PLUGIN_VALIDATE(inputs[0].nbDims == inputs[1].nbDims);
        return inputs[0];
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return DimsExprs{};
}

bool SkipLayerNormInterleavedPluginBaseLegacy::supportsFormatCombination(
    int32_t pos, PluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    try
    {
        PLUGIN_VALIDATE(inOut != nullptr);
        PLUGIN_VALIDATE(nbInputs == 2);
        PLUGIN_VALIDATE(nbOutputs == getNbOutputs());
        PLUGIN_VALIDATE(pos >= 0 && pos < (nbInputs + nbOutputs));

        PluginTensorDesc const& desc = inOut[pos];
        return desc.type == DataType::kINT8 && desc.format == TensorFormat::kCHW32;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return false;
}

void SkipLayerNormInterleavedPluginBaseLegacy::configurePlugin(DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
    DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) noexcept
{
    try
    {
        // Validate input arguments
        PLUGIN_VALIDATE(inputs != nullptr);
        PLUGIN_VALIDATE(outputs != nullptr);
        PLUGIN_VALIDATE(nbOutputs == getNbOutputs());
        PLUGIN_VALIDATE(nbInputs == 2);
        PLUGIN_VALIDATE(DataType::kINT8 == inputs[0].desc.type);
        PLUGIN_VALIDATE(DataType::kINT8 == inputs[1].desc.type);

        auto const& inDims0 = inputs[0].desc.dims;
        auto const& inDims1 = inputs[1].desc.dims;
        TRT_UNUSED inDims1;

        PLUGIN_VALIDATE(inDims0.nbDims == inDims1.nbDims);
        PLUGIN_VALIDATE(std::equal(inDims0.d, inDims0.d + inDims0.nbDims, inDims1.d));

        mParamWordsize = getElementSize(kPARAM_TYPE);

        if (!mParamsOnDevice)
        {
            copyToDevice(mGamma, getWeightsSize(mGamma, kPARAM_TYPE), mGammaDev);
            copyToDevice(mBeta, getWeightsSize(mBeta, kPARAM_TYPE), mBetaDev);
            mParamsOnDevice = true;
        }
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
}

size_t SkipLayerNormInterleavedPluginBaseLegacy::getWorkspaceSize(
    PluginTensorDesc const* inputs, int32_t nbInputs, PluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept
{
    return 0;
}

int32_t SkipLayerNormInterleavedPluginHFaceLegacy::enqueue(PluginTensorDesc const* inputDesc,
    PluginTensorDesc const* outputDesc, void const* const* inputs, void* const* outputs, void* /* workspace */,
    cudaStream_t stream) noexcept
{
    try
    {
        PLUGIN_VALIDATE(inputDesc != nullptr && outputDesc != nullptr && inputs != nullptr && outputs != nullptr);

        // Input shape: 1x(hxd)xtotalx1
        auto const iDesc = inputDesc[0];
        auto const sDesc = inputDesc[1];
        auto const oDesc = outputDesc[0];
        checkDescs(iDesc, sDesc, oDesc);

        int32_t const ld = iDesc.dims.d[1];
        int32_t const total = iDesc.dims.d[2];
        float const dqScaleIn = iDesc.scale;
        float const dqScaleSkip = sDesc.scale;
        float const qScale = 1.F / oDesc.scale;
        int8_t const* input = static_cast<int8_t const*>(inputs[0]);
        int8_t const* skip = static_cast<int8_t const*>(inputs[1]);
        int8_t* output = static_cast<int8_t*>(outputs[0]);
        half const* gamma = static_cast<half const*>(mGammaDev.get());
        half const* beta = static_cast<half const*>(mBetaDev.get());

        if (total < 4096)
        {
            return launch_small_hface(
                stream, ld, total, input, skip, beta, gamma, output, dqScaleIn, dqScaleSkip, qScale);
        }

        return launch_large_hface(stream, ld, total, input, skip, beta, gamma, output, dqScaleIn, dqScaleSkip, qScale);
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return -1;
}

int32_t SkipLayerNormInterleavedPluginMTronLegacy::enqueue(PluginTensorDesc const* inputDesc,
    PluginTensorDesc const* outputDesc, void const* const* inputs, void* const* outputs, void* /* workspace */,
    cudaStream_t stream) noexcept
{
    try
    {
        PLUGIN_VALIDATE(inputDesc != nullptr && outputDesc != nullptr && inputs != nullptr && outputs != nullptr);

        // Input shape: 1x(hxd)xtotalx1
        auto const iDesc = inputDesc[0];
        auto const sDesc = inputDesc[1];
        auto const oDesc = outputDesc[0];
        auto const pDesc = outputDesc[1];
        checkDescs(iDesc, sDesc, oDesc);
        PLUGIN_VALIDATE(std::equal(iDesc.dims.d, iDesc.dims.d + iDesc.dims.nbDims, pDesc.dims.d));

        int32_t const ld = iDesc.dims.d[1];
        int32_t const total = iDesc.dims.d[2];
        float const dqScaleIn = iDesc.scale;
        float const dqScaleSkip = sDesc.scale;
        float const qScale = 1.F / oDesc.scale;
        float const qSkipScale = 1.F / pDesc.scale;
        int8_t const* input = static_cast<int8_t const*>(inputs[0]);
        int8_t const* skip = static_cast<int8_t const*>(inputs[1]);
        int8_t* output = static_cast<int8_t*>(outputs[0]);
        int8_t* preln = static_cast<int8_t*>(outputs[1]);
        half const* gamma = static_cast<half const*>(mGammaDev.get());
        half const* beta = static_cast<half const*>(mBetaDev.get());

        if (total < 4096)
        {
            return launch_small_mtron(
                stream, ld, total, input, skip, beta, gamma, output, preln, dqScaleIn, dqScaleSkip, qScale, qSkipScale);
        }

        return launch_large_mtron(
            stream, ld, total, input, skip, beta, gamma, output, preln, dqScaleIn, dqScaleSkip, qScale, qSkipScale);
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return -1;
}

// IPluginV2Ext Methods
DataType SkipLayerNormInterleavedPluginBaseLegacy::getOutputDataType(
    int32_t index, DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    try
    {
        PLUGIN_VALIDATE(inputTypes != nullptr);
        PLUGIN_VALIDATE(index >= 0 && index < getNbOutputs());
        PLUGIN_VALIDATE(nbInputs == 2);
        return inputTypes[0];
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return DataType{};
}

// IPluginV2 Methods
char const* SkipLayerNormInterleavedPluginBaseLegacy::getPluginType() const noexcept
{
    return kSKIP_LAYER_NORM_INTERLEAVED_NAME;
}

char const* SkipLayerNormInterleavedPluginHFaceLegacy::getPluginVersion() const noexcept
{
    return kSKIP_LAYER_NORM_INTERLEAVED_VERSION_HFACE_LEGACY;
}

char const* SkipLayerNormInterleavedPluginMTronLegacy::getPluginVersion() const noexcept
{
    return kSKIP_LAYER_NORM_INTERLEAVED_VERSION_MTRON_LEGACY;
}

int32_t SkipLayerNormInterleavedPluginHFaceLegacy::getNbOutputs() const noexcept
{
    return 1;
}

int32_t SkipLayerNormInterleavedPluginMTronLegacy::getNbOutputs() const noexcept
{
    return 2;
}

int32_t SkipLayerNormInterleavedPluginHFaceLegacy::initialize() noexcept
{
    BERT_DEBUG_MSG("SkipLayerNormInterleavedPluginHFaceLegacy initialize");
    return 0;
}

int32_t SkipLayerNormInterleavedPluginMTronLegacy::initialize() noexcept
{
    BERT_DEBUG_MSG("SkipLayerNormInterleavedPluginMTronLegacy initialize");
    return 0;
}

void SkipLayerNormInterleavedPluginHFaceLegacy::terminate() noexcept
{
    BERT_DEBUG_MSG("SkipLayerNormInterleavedPluginHFaceLegacy terminate");
}

void SkipLayerNormInterleavedPluginMTronLegacy::terminate() noexcept
{
    BERT_DEBUG_MSG("SkipLayerNormInterleavedPluginMTronLegacy terminate");
}

size_t SkipLayerNormInterleavedPluginBaseLegacy::getSerializationSize() const noexcept
{
    return 2 * mParamWordsize * mLd + sizeof(mLd);
}

void SkipLayerNormInterleavedPluginBaseLegacy::serialize(void* buffer) const noexcept
{
    try
    {
        serialize_value(&buffer, mLd);

        char* d = static_cast<char*>(buffer);
        serFromDev(d, static_cast<char*>(mBetaDev.get()), mLd * mParamWordsize);
        serFromDev(d, static_cast<char*>(mGammaDev.get()), mLd * mParamWordsize);
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
}

void SkipLayerNormInterleavedPluginBaseLegacy::destroy() noexcept
{
    try
    {
        // This gets called when the network containing plugin is destroyed
        mGammaDev.reset(nullptr);
        mBetaDev.reset(nullptr);
        delete this;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
}

void SkipLayerNormInterleavedPluginHFaceLegacy::destroy() noexcept
{
    BERT_DEBUG_MSG("SkipLayerNormInterleavedPluginHFaceLegacy destroy");
    SkipLayerNormInterleavedPluginBaseLegacy::destroy();
}

void SkipLayerNormInterleavedPluginMTronLegacy::destroy() noexcept
{
    BERT_DEBUG_MSG("SkipLayerNormInterleavedPluginMTronLegacy destroy");
    SkipLayerNormInterleavedPluginBaseLegacy::destroy();
}

void SkipLayerNormInterleavedPluginBaseLegacy::setPluginNamespace(char const* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

char const* SkipLayerNormInterleavedPluginBaseLegacy::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

/////////////////////////////////////////////////////////

SkipLayerNormInterleavedPluginBaseLegacyCreator::SkipLayerNormInterleavedPluginBaseLegacyCreator()
{
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("beta"));
    mPluginAttributes.emplace_back(PluginField("gamma"));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

SkipLayerNormInterleavedPluginHFaceLegacyCreator::SkipLayerNormInterleavedPluginHFaceLegacyCreator()
    : SkipLayerNormInterleavedPluginBaseLegacyCreator()
{
}

SkipLayerNormInterleavedPluginMTronLegacyCreator::SkipLayerNormInterleavedPluginMTronLegacyCreator()
    : SkipLayerNormInterleavedPluginBaseLegacyCreator()
{
}

char const* SkipLayerNormInterleavedPluginBaseLegacyCreator::getPluginName() const noexcept
{
    return kSKIP_LAYER_NORM_INTERLEAVED_NAME;
}

char const* SkipLayerNormInterleavedPluginHFaceLegacyCreator::getPluginVersion() const noexcept
{
    return kSKIP_LAYER_NORM_INTERLEAVED_VERSION_HFACE_LEGACY;
}

char const* SkipLayerNormInterleavedPluginMTronLegacyCreator::getPluginVersion() const noexcept
{
    return kSKIP_LAYER_NORM_INTERLEAVED_VERSION_MTRON_LEGACY;
}

PluginFieldCollection const* SkipLayerNormInterleavedPluginBaseLegacyCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* SkipLayerNormInterleavedPluginHFaceLegacyCreator::createPlugin(
    char const* name, PluginFieldCollection const* fc) noexcept
{
    try
    {
        BERT_DEBUG_MSG("SkipLayerNormInterleavedPluginHFaceLegacyCreator createPlugin");

        Weights beta{DataType::kFLOAT, nullptr, 0};
        Weights gamma{DataType::kFLOAT, nullptr, 0};
        buildBetaAndGamma(fc, beta, gamma);

        return new SkipLayerNormInterleavedPluginHFaceLegacy(name, beta, gamma);
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* SkipLayerNormInterleavedPluginMTronLegacyCreator::createPlugin(
    char const* name, PluginFieldCollection const* fc) noexcept
{
    try
    {
        BERT_DEBUG_MSG("SkipLayerNormInterleavedPluginMTronLegacyCreator createPlugin");

        PLUGIN_VALIDATE(fc != nullptr);

        Weights beta{DataType::kFLOAT, nullptr, 0};
        Weights gamma{DataType::kFLOAT, nullptr, 0};
        buildBetaAndGamma(fc, beta, gamma);

        return new SkipLayerNormInterleavedPluginMTronLegacy(name, beta, gamma);
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* SkipLayerNormInterleavedPluginHFaceLegacyCreator::deserializePlugin(
    char const* name, void const* serialData, size_t serialLength) noexcept
{
    // This object will be deleted when the network is destroyed, which will
    // call SkipLayerNormInterleavedPlugin::destroy()
    try
    {
        BERT_DEBUG_MSG("SkipLayerNormInterleavedPluginHFaceLegacyCreator deserializePlugin");
        return new SkipLayerNormInterleavedPluginHFaceLegacy(name, serialData, serialLength);
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* SkipLayerNormInterleavedPluginMTronLegacyCreator::deserializePlugin(
    char const* name, void const* serialData, size_t serialLength) noexcept
{
    // This object will be deleted when the network is destroyed, which will
    // call SkipLayerNormInterleavedPlugin::destroy()
    try
    {
        BERT_DEBUG_MSG("SkipLayerNormInterleavedPluginMTronLegacyCreator deserializePlugin");
        return new SkipLayerNormInterleavedPluginMTronLegacy(name, serialData, serialLength);
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

void SkipLayerNormInterleavedPluginBaseLegacyCreator::setPluginNamespace(char const* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

char const* SkipLayerNormInterleavedPluginBaseLegacyCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}
