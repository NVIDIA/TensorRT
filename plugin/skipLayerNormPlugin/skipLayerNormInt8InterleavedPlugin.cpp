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
constexpr char const* kSKIP_LAYER_NORM_INTERLEAVED_VERSION_HFACE{"7"};
constexpr char const* kSKIP_LAYER_NORM_INTERLEAVED_VERSION_MTRON{"8"};
constexpr char const* kSKIP_LAYER_NORM_INTERLEAVED_NAME{"CustomSkipLayerNormPluginDynamic"};

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

void buildBetaAndGamma(PluginFieldCollection const* fc, Weights& beta, Weights& gamma)
{
    PLUGIN_VALIDATE(fc != nullptr, "SkipLayerNorm: Plugin Field collection is null");
    PLUGIN_VALIDATE(fc->fields != nullptr, "SkipLayerNorm: Plugin Fields are null");
    plugin::validateRequiredAttributesExist({"beta", "gamma"}, fc);

    for (int32_t i = 0; i < fc->nbFields; i++)
    {
        std::string fieldName(fc->fields[i].name);

        if (fieldName.compare("beta") == 0)
        {
            BERT_DEBUG_MSG("Building beta...");
            beta.values = fc->fields[i].data;
            beta.count = fc->fields[i].length;
            beta.type = fieldTypeToDataType(fc->fields[i].type);
        }

        if (fieldName.compare("gamma") == 0)
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
} // namespace

REGISTER_TENSORRT_PLUGIN(SkipLayerNormInterleavedPluginHFaceCreator);
REGISTER_TENSORRT_PLUGIN(SkipLayerNormInterleavedPluginMTronCreator);

constexpr auto kPARAM_TYPE = DataType::kHALF;

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

    mParamWordsize = getElementSize(kPARAM_TYPE);

    mBeta.convertAndCopy(beta, kPARAM_TYPE);
    mGamma.convertAndCopy(gamma, kPARAM_TYPE);
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

SkipLayerNormInterleavedPluginBase::~SkipLayerNormInterleavedPluginBase()
{
    try
    {
        mGammaDev.reset(nullptr);
        mBetaDev.reset(nullptr);
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
}

SkipLayerNormInterleavedPluginHFace::~SkipLayerNormInterleavedPluginHFace()
{
    BERT_DEBUG_MSG("SkipLayerNormInterleavedPluginHFace destructor");
}

SkipLayerNormInterleavedPluginMTron::~SkipLayerNormInterleavedPluginMTron()
{
    BERT_DEBUG_MSG("SkipLayerNormInterleavedPluginMTron destructor");
}

//////
// IPluginV3 method definitions:
// - getCapabilityInterface() (Base)
// - clone() (HFace, MTron)
//////
IPluginCapability* SkipLayerNormInterleavedPluginBase::getCapabilityInterface(PluginCapabilityType type) noexcept
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

IPluginV3* SkipLayerNormInterleavedPluginHFace::clone() noexcept
{
    try
    {
        BERT_DEBUG_MSG("SkipLayerNormInterleavedPluginHFace clone");

        auto* p = new SkipLayerNormInterleavedPluginHFace(mLayerName, mBeta, mGamma);
        p->setPluginNamespace(mNamespace.c_str());
        return p;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV3* SkipLayerNormInterleavedPluginMTron::clone() noexcept
{
    try
    {
        BERT_DEBUG_MSG("SkipLayerNormInterleavedPluginMTron clone");

        auto* p = new SkipLayerNormInterleavedPluginMTron(mLayerName, mBeta, mGamma);
        p->setPluginNamespace(mNamespace.c_str());
        return p;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

// End IPluginV3 method definitions

//////
// IPluginV3OneRuntime method definitions:
// - getFieldsToSerialize() (Base)
// - onShapeChange() (Base)
// - attachToContext() (HFace, MTron)
// - execute() (HFace, MTron)
/////
PluginFieldCollection const* SkipLayerNormInterleavedPluginBase::getFieldsToSerialize() noexcept
{
    mDataToSerialize.clear();
    mDataToSerialize.emplace_back(
        "beta", static_cast<half const*>(mBeta.values), PluginFieldType::kFLOAT16, mBeta.count);
    PLUGIN_ASSERT(mBeta.type == kPARAM_TYPE);
    mDataToSerialize.emplace_back(
        "gamma", static_cast<half const*>(mGamma.values), PluginFieldType::kFLOAT16, mGamma.count);
    PLUGIN_ASSERT(mGamma.type == kPARAM_TYPE);
    mFCToSerialize.nbFields = mDataToSerialize.size();
    mFCToSerialize.fields = mDataToSerialize.data();
    return &mFCToSerialize;
}

int32_t SkipLayerNormInterleavedPluginBase::onShapeChange(
    PluginTensorDesc const* inputs, int32_t nbInputs, PluginTensorDesc const* outputs, int32_t nbOutputs) noexcept
{
    try
    {
        // Validate input arguments
        PLUGIN_VALIDATE(inputs != nullptr);
        PLUGIN_VALIDATE(outputs != nullptr);
        PLUGIN_VALIDATE(nbOutputs == getNbOutputs());
        PLUGIN_VALIDATE(nbInputs == 2);
        PLUGIN_VALIDATE(DataType::kINT8 == inputs[0].type);
        PLUGIN_VALIDATE(DataType::kINT8 == inputs[1].type);

        auto const& inDims0 = inputs[0].dims;
        auto const& inDims1 = inputs[1].dims;
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
        return pluginStatus_t::STATUS_SUCCESS;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return pluginStatus_t::STATUS_FAILURE;
}

IPluginV3* SkipLayerNormInterleavedPluginBase::attachToContext(IPluginResourceContext* context) noexcept
{
    return clone();
}

int32_t SkipLayerNormInterleavedPluginHFace::enqueue(PluginTensorDesc const* inputDesc,
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

        const int32_t ld = iDesc.dims.d[1];
        const int32_t total = iDesc.dims.d[2];
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

int32_t SkipLayerNormInterleavedPluginMTron::enqueue(PluginTensorDesc const* inputDesc,
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

        const int32_t ld = iDesc.dims.d[1];
        const int32_t total = iDesc.dims.d[2];
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
// end IPluginV3OneRuntime method definitions

///////
// IPluginV3OneBuild method definitions
// - getNbOutputs() (MTron, HFace)
// - supportsFormatCombination() (Base)
// - getOutputShapes (Base)
// - getOutputDataType() (Base)
// - configurePlugin() (Base)
// - getWorkSpaceSize() (Base)
//////
int32_t SkipLayerNormInterleavedPluginHFace::getNbOutputs() const noexcept
{
    return 1;
}

int32_t SkipLayerNormInterleavedPluginMTron::getNbOutputs() const noexcept
{
    return 2;
}

bool SkipLayerNormInterleavedPluginBase::supportsFormatCombination(
    int32_t pos, DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    try
    {
        PLUGIN_VALIDATE(inOut != nullptr);
        PLUGIN_VALIDATE(nbInputs == 2);
        PLUGIN_VALIDATE(nbOutputs == getNbOutputs());
        PLUGIN_VALIDATE(pos >= 0 && pos < (nbInputs + nbOutputs));
        PluginTensorDesc const& desc = inOut[pos].desc;
        return desc.type == DataType::kINT8 && desc.format == TensorFormat::kCHW32;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return false;
}

int32_t SkipLayerNormInterleavedPluginBase::getOutputShapes(DimsExprs const* inputs, int32_t nbInputs,
    DimsExprs const* shapeInputs, int32_t nbShapeInputs, DimsExprs* outputs, int32_t nbOutputs,
    IExprBuilder& exprBuilder) noexcept
{
    try
    {
        PLUGIN_VALIDATE(inputs != nullptr);
        PLUGIN_VALIDATE(nbInputs == 2);
        PLUGIN_VALIDATE(nbOutputs == getNbOutputs());
        PLUGIN_VALIDATE(inputs[0].nbDims == inputs[1].nbDims);
        for (int32_t i = 0; i < nbOutputs; ++i)
        {
            outputs[i] = inputs[0];
        }
        return pluginStatus_t::STATUS_SUCCESS;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return pluginStatus_t::STATUS_FAILURE;
}

int32_t SkipLayerNormInterleavedPluginBase::getOutputDataTypes(
    DataType* outputTypes, int32_t nbOutputs, DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    try
    {
        PLUGIN_VALIDATE(inputTypes != nullptr);
        PLUGIN_VALIDATE(nbOutputs == getNbOutputs());
        PLUGIN_VALIDATE(nbInputs == 2);
        for (int32_t i = 0; i < nbOutputs; ++i)
        {
            outputTypes[i] = inputTypes[0];
        }
        return pluginStatus_t::STATUS_SUCCESS;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return pluginStatus_t::STATUS_FAILURE;
}

int32_t SkipLayerNormInterleavedPluginBase::configurePlugin(DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
    DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) noexcept
{
    return pluginStatus_t::STATUS_SUCCESS;
}

size_t SkipLayerNormInterleavedPluginBase::getWorkspaceSize(DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
    DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept
{
    return 0;
}
// End IPluginV3OneBuild method definitions

//////
// IPluginV3OneCore method definitions
// - getPluginVersion() (MTron, HFace)
// - getPluginName() (Base)
// - getPluginNamespace() (Base)
// - setPluginNamespace() (Base)
//////
char const* SkipLayerNormInterleavedPluginHFace::getPluginVersion() const noexcept
{
    return kSKIP_LAYER_NORM_INTERLEAVED_VERSION_HFACE;
}

char const* SkipLayerNormInterleavedPluginMTron::getPluginVersion() const noexcept
{
    return kSKIP_LAYER_NORM_INTERLEAVED_VERSION_MTRON;
}

char const* SkipLayerNormInterleavedPluginBase::getPluginName() const noexcept
{
    return kSKIP_LAYER_NORM_INTERLEAVED_NAME;
}

char const* SkipLayerNormInterleavedPluginBase::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

void SkipLayerNormInterleavedPluginBase::setPluginNamespace(char const* libNamespace) noexcept
{
    mNamespace = libNamespace;
}
// End IPluginV3OneCore method definitions

//////////////////////////// Plugin Creator member definitions /////////////////////////////

SkipLayerNormInterleavedPluginBaseCreator::SkipLayerNormInterleavedPluginBaseCreator()
{
    static std::mutex sMutex;
    std::lock_guard<std::mutex> lock(sMutex);
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

char const* SkipLayerNormInterleavedPluginBaseCreator::getPluginName() const noexcept
{
    return kSKIP_LAYER_NORM_INTERLEAVED_NAME;
}

char const* SkipLayerNormInterleavedPluginHFaceCreator::getPluginVersion() const noexcept
{
    return kSKIP_LAYER_NORM_INTERLEAVED_VERSION_HFACE;
}

char const* SkipLayerNormInterleavedPluginMTronCreator::getPluginVersion() const noexcept
{
    return kSKIP_LAYER_NORM_INTERLEAVED_VERSION_MTRON;
}

PluginFieldCollection const* SkipLayerNormInterleavedPluginBaseCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV3* SkipLayerNormInterleavedPluginHFaceCreator::createPlugin(
    char const* name, PluginFieldCollection const* fc, TensorRTPhase phase) noexcept
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

IPluginV3* SkipLayerNormInterleavedPluginMTronCreator::createPlugin(
    char const* name, PluginFieldCollection const* fc, TensorRTPhase phase) noexcept
{
    try
    {
        BERT_DEBUG_MSG("SkipLayerNormInterleavedPluginMTronCreator createPlugin");

        PLUGIN_VALIDATE(fc != nullptr);

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

void SkipLayerNormInterleavedPluginBaseCreator::setPluginNamespace(char const* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

char const* SkipLayerNormInterleavedPluginBaseCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}
// End Plugin Creator member definitions
