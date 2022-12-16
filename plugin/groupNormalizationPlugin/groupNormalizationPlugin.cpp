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

#include "groupNormalizationPlugin.h"
#include "common/dimsHelpers.h"

#include <numeric>
#include <stdexcept>

using namespace nvinfer1;
using nvinfer1::plugin::GroupNormalizationPlugin;
using nvinfer1::plugin::GroupNormalizationPluginCreator;

#define PLUGIN_CHECK_CUDNN(call)                                                                                       \
    do                                                                                                                 \
    {                                                                                                                  \
        cudnnStatus_t status = call;                                                                                   \
        if (status != CUDNN_STATUS_SUCCESS)                                                                            \
        {                                                                                                              \
            return status;                                                                                             \
        }                                                                                                              \
    } while (0)

namespace
{
constexpr const char* GROUP_NORM_VERSION{"1"};
constexpr const char* GROUP_NORM_NAME{"GroupNormalizationPlugin"};
} // namespace

// // Static class fields initialization
PluginFieldCollection GroupNormalizationPluginCreator::mFC{};
std::vector<nvinfer1::PluginField> GroupNormalizationPluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(GroupNormalizationPluginCreator);

GroupNormalizationPlugin::GroupNormalizationPlugin(float epsilon, int nbGroups)
    : mEpsilon(epsilon)
    , mNbGroups(nbGroups)
{
    PLUGIN_VALIDATE(mEpsilon > 0.0F);
    // Number of groups should be positive
    PLUGIN_VALIDATE(mNbGroups > 0);
}

int GroupNormalizationPlugin::initialize() noexcept
{
    auto allocScaleBias = [this](std::shared_ptr<CudaBind<float>>& buf, float value) {
        PLUGIN_VALIDATE(mNbScaleBias > 0);
        if (!buf || !buf->mPtr || buf->mSize != mNbScaleBias)
        {
            // Allocate device memory.
            buf = std::make_shared<CudaBind<float>>(mNbScaleBias);

            // Initialize values.
            std::vector<float> const values(mNbScaleBias, value);
            PLUGIN_CUASSERT(cudaMemcpy(buf->mPtr, values.data(), sizeof(float) * mNbScaleBias, cudaMemcpyHostToDevice));
        }
    };

    allocScaleBias(mBnScales, 1.F);
    allocScaleBias(mBnBias, 0.F);
    return 0;
}

GroupNormalizationPlugin::GroupNormalizationPlugin(const void* data, size_t length)
{
    // Deserialize in the same order as serialization
    deserialize_value(&data, &length, &mEpsilon);
    deserialize_value(&data, &length, &mNbGroups);
    deserialize_value(&data, &length, &mNbScaleBias);
}

const char* GroupNormalizationPlugin::getPluginType() const noexcept
{
    return GROUP_NORM_NAME;
}

const char* GroupNormalizationPlugin::getPluginVersion() const noexcept
{
    return GROUP_NORM_VERSION;
}

int GroupNormalizationPlugin::getNbOutputs() const noexcept
{
    return 1;
}

nvinfer1::DimsExprs GroupNormalizationPlugin::getOutputDimensions(
    int index, const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    // Input (from previous layer), scale and bias are the three inputs to the plugin.
    PLUGIN_ASSERT(nbInputs == 3);
    PLUGIN_ASSERT(index == 0);
    nvinfer1::DimsExprs output(inputs[0]);
    return output;
}

void GroupNormalizationPlugin::attachToContext(
    cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) noexcept
{
    PLUGIN_ASSERT(cudnnContext);
    _cudnn_handle = cudnnContext;
    PLUGIN_CUDNNASSERT(cudnnCreateTensorDescriptor(&desc));
    PLUGIN_CUDNNASSERT(cudnnCreateTensorDescriptor(&bnDesc));
}

// Detach the plugin object from its execution context.
void GroupNormalizationPlugin::detachFromContext() noexcept
{
    PLUGIN_CUDNNASSERT(cudnnDestroyTensorDescriptor(desc));
    PLUGIN_CUDNNASSERT(cudnnDestroyTensorDescriptor(bnDesc));
}

int GroupNormalizationPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream) noexcept
{
    // Get the input dimensions
    nvinfer1::Dims input_dims = inputDesc[0].dims;
    int batchSize = input_dims.d[0];
    int nbChannels = input_dims.d[1];

    // Calculate size of each group
    int groupSize = nbChannels / mNbGroups;

    mChannelVolume = pluginInternal::volume(input_dims, /*start*/ 2, /*stop*/ inputDesc[0].dims.nbDims);

    PLUGIN_CHECK_CUDNN(cudnnSetTensor4dDescriptor(desc, // descriptor
        CUDNN_TENSOR_NCHW,                              // tensor format
        CUDNN_DATA_FLOAT,                               // type
        1,                                              // Batchsize
        batchSize * mNbGroups,                          // Channels
        groupSize,                                      // Height
        mChannelVolume                                  // Width
        ));

    PLUGIN_CHECK_CUDNN(cudnnDeriveBNTensorDescriptor(bnDesc, desc, CUDNN_BATCHNORM_SPATIAL));
    PLUGIN_CHECK_CUDNN(cudnnSetStream(_cudnn_handle, stream));

    // Reshape the data according in the cudnnSetTensor4dDescriptor.
    PLUGIN_ASSERT(mBnScales && mBnScales->mPtr);
    PLUGIN_ASSERT(mBnBias && mBnBias->mPtr);
    float a = 1.F;
    float b = 0.F;
    PLUGIN_CHECK_CUDNN(cudnnBatchNormalizationForwardTraining(_cudnn_handle, // handle
        CUDNN_BATCHNORM_SPATIAL,                                             // BatchNormMode_t, try also non persistent
        &a,                                                                  //
        &b,                                                                  //
        desc,                                                                // in/out descriptor
        inputs[0],                                                           // input
        desc,                                                                // in/out descriptor
        outputs[0],                                                          // output
        bnDesc,                                                              //
        mBnScales->mPtr,                                                     // 1
        mBnBias->mPtr,                                                       // 0
        0.0,                                                                 // exponential average factor
        nullptr,                                                             // resultRunningMean
        nullptr,                                                             // resultRunningVar
        mEpsilon,                                                            //  eps
        nullptr,                                                             // resultSaveMean
        nullptr                                                              // resultSaveInvVar
        ));

    float* output = static_cast<float*>(outputs[0]);
    return scaleShiftChannelsInplace(output, batchSize, nbChannels, mChannelVolume,
        static_cast<float const*>(inputs[2]), static_cast<float const*>(inputs[1]), stream); // mBetaDev, mGammaDev,
}

size_t GroupNormalizationPlugin::getSerializationSize() const noexcept
{
    return sizeof(mNbGroups) + sizeof(mEpsilon) + sizeof(mNbScaleBias);
}

void GroupNormalizationPlugin::serialize(void* buffer) const noexcept
{
    serialize_value(&buffer, mEpsilon);
    serialize_value(&buffer, mNbGroups);
    serialize_value(&buffer, mNbScaleBias);
}

bool GroupNormalizationPlugin::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept
{
    PLUGIN_ASSERT(inOut && pos < (nbInputs + nbOutputs));
    return ((inOut[pos].type == nvinfer1::DataType::kFLOAT) && inOut[pos].format == nvinfer1::PluginFormat::kLINEAR
        && inOut[pos].type == inOut[0].type);
}

void GroupNormalizationPlugin::terminate() noexcept {}

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
        plugin->setPluginNamespace(mPluginNamespace);
        plugin->mNbScaleBias = mNbScaleBias;
        plugin->mBnScales = mBnScales;
        plugin->mBnBias = mBnBias;
        return plugin;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

void GroupNormalizationPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept
{
    int32_t const batchSize = in[0].desc.dims.d[0] <= 0 ? in[0].max.d[0] : in[0].desc.dims.d[0];
    mNbScaleBias = batchSize * mNbGroups;
}

nvinfer1::DataType GroupNormalizationPlugin::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept
{
    PLUGIN_ASSERT(inputTypes && nbInputs > 0 && index == 0);
    return inputTypes[0];
}

size_t GroupNormalizationPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
    const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept
{
    return 0;
}

void GroupNormalizationPlugin::setPluginNamespace(const char* libNamespace) noexcept
{
    mPluginNamespace = libNamespace;
}

const char* GroupNormalizationPlugin::getPluginNamespace() const noexcept
{
    return mPluginNamespace;
}

GroupNormalizationPluginCreator::GroupNormalizationPluginCreator()
{
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("eps", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("num_groups", nullptr, PluginFieldType::kINT32, 1));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* GroupNormalizationPluginCreator::getPluginName() const noexcept
{
    return GROUP_NORM_NAME;
}

const char* GroupNormalizationPluginCreator::getPluginVersion() const noexcept
{
    return GROUP_NORM_VERSION;
}

const PluginFieldCollection* GroupNormalizationPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

const char* GroupNormalizationPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

void GroupNormalizationPluginCreator::setPluginNamespace(const char* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

IPluginV2DynamicExt* GroupNormalizationPluginCreator::createPlugin(
    const char* name, const PluginFieldCollection* fc) noexcept
{
    try
    {
        // Set default values
        int nbGroups{1};
        float epsilon{0.00001F};
        for (int i = 0; i < fc->nbFields; i++)
        {
            std::string field_name(fc->fields[i].name);
            if (field_name.compare("eps") == 0)
            {
                epsilon = *static_cast<const float*>(fc->fields[i].data);
            }
            if (field_name.compare("num_groups") == 0)
            {
                nbGroups = *static_cast<const int*>(fc->fields[i].data);
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
    const char* name, const void* serialData, size_t serialLength) noexcept
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
