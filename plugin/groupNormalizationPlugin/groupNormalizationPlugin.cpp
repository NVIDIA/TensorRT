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

#include "groupNormalizationPlugin.h"
#include "common/dimsHelpers.h"
#include "common/serialize.hpp"

#include <numeric>
#include <stdexcept>

using namespace nvinfer1;
using namespace nvinfer1::pluginInternal;
using nvinfer1::plugin::GroupNormalizationPlugin;
using nvinfer1::plugin::GroupNormalizationPluginCreator;

namespace
{
constexpr char const* kGROUP_NORM_VERSION{"1"};
constexpr char const* kGROUP_NORM_NAME{"GroupNormalizationPlugin"};
} // namespace

// std::vector<nvinfer1::PluginField> GroupNormalizationPluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(GroupNormalizationPluginCreator);

GroupNormalizationPlugin::GroupNormalizationPlugin(float epsilon, int32_t nbGroups)
    : mEpsilon(epsilon)
    , mNbGroups(nbGroups)
{
    PLUGIN_VALIDATE(mEpsilon > 0.0F);
    PLUGIN_VALIDATE(mNbGroups > 0);
}

int32_t GroupNormalizationPlugin::initialize() noexcept
{
    return STATUS_SUCCESS;
}

GroupNormalizationPlugin::GroupNormalizationPlugin(void const* data, size_t length)
{
    // Deserialize in the same order as serialization
    deserialize_value(&data, &length, &mEpsilon);
    deserialize_value(&data, &length, &mNbGroups);
}

char const* GroupNormalizationPlugin::getPluginType() const noexcept
{
    return kGROUP_NORM_NAME;
}

char const* GroupNormalizationPlugin::getPluginVersion() const noexcept
{
    return kGROUP_NORM_VERSION;
}

int32_t GroupNormalizationPlugin::getNbOutputs() const noexcept
{
    return 1;
}

nvinfer1::DimsExprs GroupNormalizationPlugin::getOutputDimensions(
    int32_t index, nvinfer1::DimsExprs const* inputs, int32_t nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    try
    {
        // Input (from previous layer), scale and bias are the three inputs to the plugin.
        PLUGIN_VALIDATE(nbInputs == 3);
        PLUGIN_VALIDATE(index == 0);
        return inputs[0];
    }
    catch (std::exception const& e)
    {
        caughtError(e);
        return DimsExprs{};
    }
}

void GroupNormalizationPlugin::attachToContext(
    cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) noexcept
{
    try
    {
        std::string kFULL_NAME = std::string(kGROUP_NORM_NAME) + ", version: " + std::string(kGROUP_NORM_VERSION);
        mCudnnWrapper = createPluginCudnnWrapper(gpuAllocator, kFULL_NAME.c_str());
        mCudnnHandle = mCudnnWrapper->getCudnnHandle();
        PLUGIN_VALIDATE(mCudnnHandle);
        PLUGIN_CUDNNASSERT(mCudnnWrapper->cudnnCreateTensorDescriptor(&mTensorDesc));
        PLUGIN_CUDNNASSERT(mCudnnWrapper->cudnnCreateTensorDescriptor(&mBNTensorDesc));
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
}

// Detach the plugin object from its execution context.
void GroupNormalizationPlugin::detachFromContext() noexcept
{
    try
    {
        PLUGIN_CUDNNASSERT(mCudnnWrapper->cudnnDestroyTensorDescriptor(mTensorDesc));
        PLUGIN_CUDNNASSERT(mCudnnWrapper->cudnnDestroyTensorDescriptor(mBNTensorDesc));
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
}

int32_t GroupNormalizationPlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
    nvinfer1::PluginTensorDesc const* /* outputDesc */, void const* const* inputs, void* const* outputs,
    void* /* workspace */, cudaStream_t stream) noexcept
{
    try
    {
        PLUGIN_VALIDATE(inputDesc != nullptr && inputs != nullptr && outputs != nullptr);
        PLUGIN_VALIDATE(mBnScales != nullptr && mBnScales->mPtr != nullptr);
        PLUGIN_VALIDATE(mBnBias != nullptr && mBnBias->mPtr != nullptr);
        PLUGIN_VALIDATE(mCudnnHandle != nullptr);
        PLUGIN_VALIDATE(mTensorDesc != nullptr);
        PLUGIN_VALIDATE(mBNTensorDesc != nullptr);
    }
    catch (std::exception const& e)
    {
        caughtError(e);
        return STATUS_FAILURE;
    }

    PLUGIN_CHECK_CUDNN(mCudnnWrapper->cudnnSetStream(mCudnnHandle, stream));

    // The tensor descriptors were set up in configurePlugin() to make Batch Normalization actually
    // perform Group Normalization. This was done by setting the tensor descriptor shape to
    // (1, batch*num_groups, channels_per_group, volume_of_spatial_dims).
    // cudnnBatchNorm will normalize over the last two dimensions.
    float const one = 1.F;
    float const zero = 0.F;
    PLUGIN_CHECK_CUDNN(mCudnnWrapper->cudnnBatchNormalizationForwardTraining(mCudnnHandle, // handle
        CUDNN_BATCHNORM_SPATIAL, // BatchNormMode_t, try also non persistent
        &one,                    //
        &zero,                   //
        mTensorDesc,             // in/out descriptor
        inputs[0],               // input
        mTensorDesc,             // in/out descriptor
        outputs[0],              // output
        mBNTensorDesc,           //
        mBnScales->mPtr,         // 1
        mBnBias->mPtr,           // 0
        0.0,                     // exponential average factor
        nullptr,                 // resultRunningMean
        nullptr,                 // resultRunningVar
        mEpsilon,                //  eps
        nullptr,                 // resultSaveMean
        nullptr                  // resultSaveInvVar
        ));

    // Apply an additional scale and bias on each channel.
    nvinfer1::Dims inputDims = inputDesc[0].dims;
    int32_t batchSize = inputDims.d[0];
    int32_t nbChannels = inputDims.d[1];
    auto* output = static_cast<float*>(outputs[0]);
    return scaleShiftChannelsInplace(output, batchSize, nbChannels, mChannelVolume,
        static_cast<float const*>(inputs[2]), static_cast<float const*>(inputs[1]), stream); // mBetaDev, mGammaDev,
}

size_t GroupNormalizationPlugin::getSerializationSize() const noexcept
{
    return sizeof(mNbGroups) + sizeof(mEpsilon);
}

void GroupNormalizationPlugin::serialize(void* buffer) const noexcept
{
    PLUGIN_ASSERT(buffer != nullptr);
    auto* const start = reinterpret_cast<uint8_t*>(buffer);
    serialize_value(&buffer, mEpsilon);
    serialize_value(&buffer, mNbGroups);
    PLUGIN_ASSERT(start + getSerializationSize() == reinterpret_cast<uint8_t*>(buffer));
}

bool GroupNormalizationPlugin::supportsFormatCombination(
    int32_t pos, nvinfer1::PluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    try
    {
        PLUGIN_VALIDATE(inOut != nullptr);
        PLUGIN_VALIDATE(pos < nbInputs + nbOutputs);
        PLUGIN_VALIDATE(pos >= 0);
        return ((inOut[pos].type == nvinfer1::DataType::kFLOAT) && inOut[pos].format == nvinfer1::PluginFormat::kLINEAR
            && inOut[pos].type == inOut[0].type);
    }
    catch (std::exception const& e)
    {
        caughtError(e);
        return false;
    }
}

void GroupNormalizationPlugin::terminate() noexcept
{
    mBnScales.reset();
    mBnBias.reset();
}

void GroupNormalizationPlugin::destroy() noexcept
{
    // This gets called when the network containing plugin is destroyed
    delete this;
}

IPluginV2DynamicExt* GroupNormalizationPlugin::clone() const noexcept
{
    try
    {
        auto* plugin = new GroupNormalizationPlugin(mEpsilon, mNbGroups);
        plugin->setPluginNamespace(mNamespace.c_str());
        plugin->mNbScaleBias = mNbScaleBias;
        plugin->mBnScales = mBnScales;
        plugin->mBnBias = mBnBias;
        plugin->mChannelVolume = mChannelVolume;
        return plugin;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

void GroupNormalizationPlugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int32_t nbInputs,
    nvinfer1::DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
    try
    {
        PLUGIN_VALIDATE(in != nullptr);
        PLUGIN_VALIDATE(out != nullptr);
        PLUGIN_VALIDATE(nbInputs == 3);
        PLUGIN_VALIDATE(nbOutputs == getNbOutputs());

        nvinfer1::Dims inputDims = in[0].desc.dims;
        int32_t const batchSize = inputDims.d[0];
        int32_t const nbChannels = inputDims.d[1];

        if (batchSize <= 0 || nbChannels <= 0)
        {
            // Input size not yet known, nothing to configure.
            return;
        }

        if (mTensorDesc == nullptr)
        {
            // Not yet attached to context.
            return;
        }

        // Allocate scale/bias tensors needed for cudnnBatchNorm.
        mNbScaleBias = batchSize * mNbGroups;
        auto allocScaleBias = [this](std::shared_ptr<CudaBind<float>>& buf, float value) {
            PLUGIN_VALIDATE(mNbScaleBias > 0);
            if (!buf || !buf->mPtr || buf->mSize != mNbScaleBias)
            {
                // Allocate device memory.
                buf = std::make_shared<CudaBind<float>>(mNbScaleBias);

                // Initialize values.
                std::vector<float> const values(mNbScaleBias, value);
                PLUGIN_CUASSERT(
                    cudaMemcpy(buf->mPtr, values.data(), sizeof(float) * mNbScaleBias, cudaMemcpyHostToDevice));
            }
        };
        allocScaleBias(mBnScales, 1.F);
        allocScaleBias(mBnBias, 0.F);

        // Calculate size of each group
        int32_t groupSize = nbChannels / mNbGroups;
        mChannelVolume = pluginInternal::volume(inputDims, /*start*/ 2, /*stop*/ inputDims.nbDims);

        // Set tensor descriptor in a way that cudnnBatchNorm will perform Group Normalization.
        PLUGIN_CUDNNASSERT(mCudnnWrapper->cudnnSetTensor4dDescriptor(mTensorDesc, // descriptor
            CUDNN_TENSOR_NCHW,                                                    // tensor format
            CUDNN_DATA_FLOAT,                                                     // type
            1,                                                                    // Batchsize
            batchSize * mNbGroups,                                                // Channels
            groupSize,                                                            // Height
            mChannelVolume                                                        // Width
            ));
        PLUGIN_CUDNNASSERT(
            mCudnnWrapper->cudnnDeriveBNTensorDescriptor(mBNTensorDesc, mTensorDesc, CUDNN_BATCHNORM_SPATIAL));
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
}

nvinfer1::DataType GroupNormalizationPlugin::getOutputDataType(
    int32_t index, nvinfer1::DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    try
    {
        PLUGIN_VALIDATE(inputTypes != nullptr);
        PLUGIN_VALIDATE(index == 0);
        return inputTypes[0];
    }
    catch (std::exception const& e)
    {
        caughtError(e);
        return DataType{};
    }
}

size_t GroupNormalizationPlugin::getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs, int32_t nbInputs,
    nvinfer1::PluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept
{
    return 0;
}

void GroupNormalizationPlugin::setPluginNamespace(char const* libNamespace) noexcept
{
    try
    {
        PLUGIN_VALIDATE(libNamespace != nullptr);
        mNamespace = libNamespace;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
}

char const* GroupNormalizationPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

GroupNormalizationPluginCreator::GroupNormalizationPluginCreator()
{
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("eps", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("num_groups", nullptr, PluginFieldType::kINT32, 1));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

char const* GroupNormalizationPluginCreator::getPluginName() const noexcept
{
    return kGROUP_NORM_NAME;
}

char const* GroupNormalizationPluginCreator::getPluginVersion() const noexcept
{
    return kGROUP_NORM_VERSION;
}

PluginFieldCollection const* GroupNormalizationPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

char const* GroupNormalizationPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

void GroupNormalizationPluginCreator::setPluginNamespace(char const* libNamespace) noexcept
{
    try
    {
        PLUGIN_VALIDATE(libNamespace != nullptr);
        mNamespace = libNamespace;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
}

IPluginV2DynamicExt* GroupNormalizationPluginCreator::createPlugin(
    char const* name, PluginFieldCollection const* fc) noexcept
{
    try
    {
        PLUGIN_VALIDATE(fc != nullptr);

        // Set default values
        int32_t nbGroups{1};
        float epsilon{0.00001F};
        for (int32_t i = 0; i < fc->nbFields; i++)
        {
            PLUGIN_VALIDATE(fc->fields[i].name != nullptr);
            std::string fieldName(fc->fields[i].name);
            if (fieldName.compare("eps") == 0)
            {
                epsilon = *static_cast<float const*>(fc->fields[i].data);
            }
            if (fieldName.compare("num_groups") == 0)
            {
                nbGroups = *static_cast<int32_t const*>(fc->fields[i].data);
            }
        }

        GroupNormalizationPlugin* plugin = new GroupNormalizationPlugin(epsilon, nbGroups);
        plugin->setPluginNamespace(mNamespace.c_str());

        return plugin;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2DynamicExt* GroupNormalizationPluginCreator::deserializePlugin(
    char const* name, void const* serialData, size_t serialLength) noexcept
{
    try
    {
        GroupNormalizationPlugin* plugin = new GroupNormalizationPlugin(serialData, serialLength);
        plugin->setPluginNamespace(mNamespace.c_str());

        return plugin;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}
