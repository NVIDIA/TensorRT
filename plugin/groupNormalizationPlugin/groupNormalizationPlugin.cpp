/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

#include "groupNormalizationPlugin.h"
#include <numeric>
#include <stdexcept>

using namespace nvinfer1;
using nvinfer1::plugin::GroupNormalizationPlugin;
using nvinfer1::plugin::GroupNormalizationPluginCreator;

#define CHECK_CUDNN(call)                                                                                              \
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
    // Number of groups should be positive
    assert(nbGroups > 0);
}

int GroupNormalizationPlugin::initialize()
{
    return 0;
}

GroupNormalizationPlugin::GroupNormalizationPlugin(const void* data, size_t length)
{
    // Deserialize in the same order as serialization
    deserialize_value(&data, &length, &mEpsilon);
    deserialize_value(&data, &length, &mNbGroups);
}

const char* GroupNormalizationPlugin::getPluginType() const
{
    return GROUP_NORM_NAME;
}

const char* GroupNormalizationPlugin::getPluginVersion() const
{
    return GROUP_NORM_VERSION;
}

int GroupNormalizationPlugin::getNbOutputs() const
{
    return 1;
}

nvinfer1::DimsExprs GroupNormalizationPlugin::getOutputDimensions(
    int index, const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder)
{
    // Input (from previous layer), scale and bias are the three inputs to the plugin.
    assert(nbInputs == 3);
    assert(index == 0);
    nvinfer1::DimsExprs output(inputs[0]);
    return output;
}

void GroupNormalizationPlugin::attachToContext(
    cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator)
{
    _cudnn_handle = cudnnContext;
    cudnnCreateTensorDescriptor(&desc);
    cudnnCreateTensorDescriptor(&bnDesc);
}

// Detach the plugin object from its execution context.
void GroupNormalizationPlugin::detachFromContext()
{
    cudnnDestroyTensorDescriptor(desc);
    cudnnDestroyTensorDescriptor(bnDesc);
}

int GroupNormalizationPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream)
{
    // Get the input dimensions
    nvinfer1::Dims input_dims = inputDesc[0].dims;
    int batchSize = input_dims.d[0];
    int nbChannels = input_dims.d[1];

    // Calculate size of each group
    int groupSize = nbChannels / mNbGroups;

    mChannelVolume
        = std::accumulate(input_dims.d + 2, input_dims.d + inputDesc[0].dims.nbDims, 1, std::multiplies<int>());

    CHECK_CUDNN(cudnnSetTensor4dDescriptor(desc, // descriptor
        CUDNN_TENSOR_NCHW,                       // tensor format
        CUDNN_DATA_FLOAT,                        // type
        1,                                       // Batchsize
        batchSize * mNbGroups,                   // Channels
        groupSize,                               // Height
        mChannelVolume                           // Width
        ));

    cudnnDeriveBNTensorDescriptor(bnDesc, desc, CUDNN_BATCHNORM_SPATIAL);
    CHECK_CUDNN(cudnnSetStream(_cudnn_handle, stream));

    // Reshape the data according in the cudnnSetTensor4dDescriptor.
    float a = 1.f;
    float b = 0.f;
    CHECK_CUDNN(cudnnBatchNormalizationForwardTraining(_cudnn_handle, // handle
        CUDNN_BATCHNORM_SPATIAL,                                      // BatchNormMode_t, try also non persistent
        &a,                                                           //
        &b,                                                           //
        desc,                                                         // in/out descriptor
        inputs[0],                                                    // input
        desc,                                                         // in/out descriptor
        outputs[0],                                                   // output
        bnDesc,                                                       //
        bnScale,                                                      // 1
        bnBias,                                                       // 0
        0.0,                                                          // exponential average factor
        nullptr,                                                      // resultRunningMean
        nullptr,                                                      // resultRunningVar
        mEpsilon,                                                     //  eps
        nullptr,                                                      // resultSaveMean
        nullptr                                                       // resultSaveInvVar
        ));

    float* output = static_cast<float*>(outputs[0]);
    scaleShiftChannelsInplace(output, batchSize, nbChannels, mChannelVolume, static_cast<const float*>(inputs[2]),
        static_cast<const float*>(inputs[1]), stream); // mBetaDev, mGammaDev,
    return 0;
}

size_t GroupNormalizationPlugin::getSerializationSize() const
{
    return sizeof(mNbGroups) + sizeof(mEpsilon);
}

void GroupNormalizationPlugin::serialize(void* buffer) const
{
    serialize_value(&buffer, mEpsilon);
    serialize_value(&buffer, mNbGroups);
}

bool GroupNormalizationPlugin::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs)
{
    assert(inOut && pos < (nbInputs + nbOutputs));
    return ((inOut[pos].type == nvinfer1::DataType::kFLOAT) && inOut[pos].format == nvinfer1::PluginFormat::kNCHW
        && inOut[pos].type == inOut[0].type);
}

void GroupNormalizationPlugin::terminate()
{
    cudaFree(bnScale);
    cudaFree(bnBias);
}

void GroupNormalizationPlugin::destroy()
{
    // This gets called when the network containing plugin is destroyed
    delete this;
}

IPluginV2DynamicExt* GroupNormalizationPlugin::clone() const
{
    auto* plugin = new GroupNormalizationPlugin(mEpsilon, mNbGroups);
    plugin->setPluginNamespace(mPluginNamespace);
    return plugin;
}

void GroupNormalizationPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs)
{

    for (int i = 0; i < nbInputs; i++)
    {
        for (int j = 0; j < in[0].desc.dims.nbDims; j++)
        {
            // Do not support dynamic dimensions
            assert(in[0].desc.dims.d[j] != -1);
        }
    }

    int batchSize = in[0].desc.dims.d[0];
    int nbChannels = in[0].desc.dims.d[1];

    // Allocate device memory and initialize scale and bias values
    cudaMalloc(&bnScale, batchSize * nbChannels * sizeof(float));
    cudaMalloc(&bnBias, batchSize * nbChannels * sizeof(float));

    // allot ones and zeros to bn parameters
    std::vector<float> ones(nbChannels, 1.f);
    cudaMemcpy(bnScale, ones.data(), nbChannels * sizeof(float), cudaMemcpyHostToDevice);

    std::vector<float> zeroes(nbChannels, 0.f);
    cudaMemcpy(bnBias, zeroes.data(), nbChannels * sizeof(float), cudaMemcpyHostToDevice);
}

nvinfer1::DataType GroupNormalizationPlugin::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes, int nbInputs) const
{
    assert(inputTypes && nbInputs > 0 && index == 0);
    return inputTypes[0];
}

size_t GroupNormalizationPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
    const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const
{
    return 0;
}

void GroupNormalizationPlugin::setPluginNamespace(const char* libNamespace)
{
    mPluginNamespace = libNamespace;
}

const char* GroupNormalizationPlugin::getPluginNamespace() const
{
    return mPluginNamespace;
}

GroupNormalizationPluginCreator::GroupNormalizationPluginCreator()
{
    mPluginAttributes.emplace_back(PluginField("eps", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("num_groups", nullptr, PluginFieldType::kINT32, 1));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* GroupNormalizationPluginCreator::getPluginName() const
{
    return GROUP_NORM_NAME;
}

const char* GroupNormalizationPluginCreator::getPluginVersion() const
{
    return GROUP_NORM_VERSION;
}

const PluginFieldCollection* GroupNormalizationPluginCreator::getFieldNames()
{
    return &mFC;
}

const char* GroupNormalizationPluginCreator::getPluginNamespace() const
{
    return mNamespace.c_str();
}

void GroupNormalizationPluginCreator::setPluginNamespace(const char* libNamespace)
{
    mNamespace = libNamespace;
}

IPluginV2DynamicExt* GroupNormalizationPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
{
    // Set default values
    int nbGroups{1};
    float epsilon{0.00001f};
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

IPluginV2DynamicExt* GroupNormalizationPluginCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength)
{
    GroupNormalizationPlugin* plugin = new GroupNormalizationPlugin(serialData, serialLength);
    plugin->setPluginNamespace(mNamespace.c_str());

    return plugin;
}
