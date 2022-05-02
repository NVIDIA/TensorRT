/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "multiscaleDeformableAttnPlugin.h"
#include "multiscaleDeformableAttn.h"

using namespace nvinfer1;
using namespace plugin;

namespace nvinfer1
{
namespace plugin
{

namespace
{
static const char* DMHA_VERSION{"1"};
static const char* DMHA_NAME{"DMHA"};
} // namespace

MultiscaleDeformableAttnPlugin::MultiscaleDeformableAttnPlugin(const std::string& name)
    : mLayerName(name)
{
}

MultiscaleDeformableAttnPlugin::MultiscaleDeformableAttnPlugin(const std::string& name, const void* data, size_t length)
    : mLayerName(name)
{
}

nvinfer1::IPluginV2DynamicExt* MultiscaleDeformableAttnPlugin::clone() const PLUGIN_NOEXCEPT
{
    MultiscaleDeformableAttnPlugin* plugin = new MultiscaleDeformableAttnPlugin(mLayerName);
    plugin->setPluginNamespace(getPluginNamespace());
    return plugin;
}

nvinfer1::DimsExprs MultiscaleDeformableAttnPlugin::getOutputDimensions(int outputIndex,
    const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) PLUGIN_NOEXCEPT
{
    nvinfer1::DimsExprs ret;
    ret.nbDims = 4;
    ret.d[0] = inputs[0].d[0];
    ret.d[1] = inputs[3].d[1];
    ret.d[2] = inputs[0].d[2];
    ret.d[3] = inputs[0].d[3];

    return ret;
}

bool MultiscaleDeformableAttnPlugin::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) PLUGIN_NOEXCEPT
{
    ASSERT((nbInputs == 5));
    ASSERT((nbOutputs == 1));

    if (inOut[pos].format == nvinfer1::TensorFormat::kLINEAR)
    {
        if ((pos == 1) || (pos == 2))
        {
            return (inOut[pos].type == nvinfer1::DataType::kINT32);
        }
        else
        {
#if __CUDA_ARCH__ >= 530
            return ((inOut[pos].type == inOut[0].type)
                && ((inOut[pos].type == nvinfer1::DataType::kFLOAT) || (inOut[pos].type == nvinfer1::DataType::kHALF)));
#else
            return ((inOut[pos].type == inOut[0].type) && ((inOut[pos].type == nvinfer1::DataType::kFLOAT)));
#endif
        }
    }
    else
    {
        return false;
    }
}

void MultiscaleDeformableAttnPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* inputs, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc* outputs, int nbOutputs) PLUGIN_NOEXCEPT
{
    // Check for valid input dimensions
    ASSERT(inputs[0].desc.dims.nbDims==4);
    ASSERT(inputs[1].desc.dims.nbDims==2);
    ASSERT(inputs[2].desc.dims.nbDims==1);
    ASSERT(inputs[3].desc.dims.nbDims==6);
    ASSERT(inputs[4].desc.dims.nbDims==5);
    
    // Check M dimensions consistency
    ASSERT(inputs[0].desc.dims.d[2] == inputs[3].desc.dims.d[2]);
    ASSERT(inputs[0].desc.dims.d[2] == inputs[4].desc.dims.d[2]);

    // Check L dimensions consistency
    ASSERT(inputs[1].desc.dims.d[0] == inputs[2].desc.dims.d[0]);
    ASSERT(inputs[1].desc.dims.d[0] == inputs[3].desc.dims.d[3]);
    ASSERT(inputs[1].desc.dims.d[0] == inputs[4].desc.dims.d[3]);

    // Check P dimensions consistency
    ASSERT(inputs[3].desc.dims.d[4] == inputs[4].desc.dims.d[4]);

    // Check Lq dimensions consistency
    ASSERT(inputs[3].desc.dims.d[1] == inputs[4].desc.dims.d[1]);
}

size_t MultiscaleDeformableAttnPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
    const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const PLUGIN_NOEXCEPT
{
    return 0;
}

int MultiscaleDeformableAttnPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workSpace,
    cudaStream_t stream) PLUGIN_NOEXCEPT
{
    const int batch = inputDesc[0].dims.d[0];
    int spatial_size = inputDesc[0].dims.d[1];
    int num_heads = inputDesc[0].dims.d[2];
    int channels = inputDesc[0].dims.d[3];
    int num_levels = inputDesc[1].dims.d[0];
    int num_query = inputDesc[3].dims.d[1];
    int num_point = inputDesc[3].dims.d[4];
    int rc = 0;
    if (inputDesc[0].type == nvinfer1::DataType::kFLOAT)
    {
        const float* value = static_cast<const float*>(inputs[0]);
        const int32_t* spatialShapes = static_cast<const int32_t*>(inputs[1]);
        const int32_t* levelStartIndex = static_cast<const int32_t*>(inputs[2]);
        const float* samplingLoc = static_cast<const float*>(inputs[3]);
        const float* attnWeight = static_cast<const float*>(inputs[4]);
        float* output = static_cast<float*>(outputs[0]);

        rc = ms_deform_attn_cuda_forward(stream, value, spatialShapes, levelStartIndex, samplingLoc, attnWeight, output,
            batch, spatial_size, num_heads, channels, num_levels, num_query, num_point);
    }
#if __CUDA_ARCH__ >= 530
    else if (inputDesc[0].type == nvinfer1::DataType::kHALF)
    {
        const __half* value = static_cast<const __half*>(inputs[0]);
        const int32_t* spatialShapes = static_cast<const int32_t*>(inputs[1]);
        const int32_t* levelStartIndex = static_cast<const int32_t*>(inputs[2]);
        const __half* samplingLoc = static_cast<const __half*>(inputs[3]);
        const __half* attnWeight = static_cast<const __half*>(inputs[4]);
        __half* output = static_cast<__half*>(outputs[0]);
        
        rc = ms_deform_attn_cuda_forward(stream, value, spatialShapes, levelStartIndex, samplingLoc, attnWeight, output,
            batch, spatial_size, num_heads, channels, num_levels, num_query, num_point);
    }
#endif

    return rc;
}

void MultiscaleDeformableAttnPlugin::attachToContext(
    cudnnContext* cudnnContext, cublasContext* cublasContext, nvinfer1::IGpuAllocator* gpuAllocator) PLUGIN_NOEXCEPT
{
    mCublasHandle = cublasContext;
}

void MultiscaleDeformableAttnPlugin::detachFromContext() PLUGIN_NOEXCEPT {}

// IPluginV2Ext Methods
nvinfer1::DataType MultiscaleDeformableAttnPlugin::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes, int nbInputs) const PLUGIN_NOEXCEPT
{
    return inputTypes[0];
}

// IPluginV2 Methods
const char* MultiscaleDeformableAttnPlugin::getPluginType() const PLUGIN_NOEXCEPT
{
    return DMHA_NAME;
}

const char* MultiscaleDeformableAttnPlugin::getPluginVersion() const PLUGIN_NOEXCEPT
{
    return DMHA_VERSION;
}

int MultiscaleDeformableAttnPlugin::getNbOutputs() const PLUGIN_NOEXCEPT
{
    return 1;
}

int MultiscaleDeformableAttnPlugin::initialize() PLUGIN_NOEXCEPT
{
    return 0;
}

void MultiscaleDeformableAttnPlugin::terminate() PLUGIN_NOEXCEPT {}

size_t MultiscaleDeformableAttnPlugin::getSerializationSize() const PLUGIN_NOEXCEPT
{
    return 0;
}

void MultiscaleDeformableAttnPlugin::serialize(void* buffer) const PLUGIN_NOEXCEPT
{
}

void MultiscaleDeformableAttnPlugin::destroy() PLUGIN_NOEXCEPT
{
    delete this;
}

void MultiscaleDeformableAttnPlugin::setPluginNamespace(const char* pluginNamespace) PLUGIN_NOEXCEPT
{
    mNamespace = pluginNamespace;
}
const char* MultiscaleDeformableAttnPlugin::getPluginNamespace() const PLUGIN_NOEXCEPT
{
    return mNamespace.c_str();
}

// Pluginv1 Creator

MultiscaleDeformableAttnPluginCreator::MultiscaleDeformableAttnPluginCreator()
{
    mPluginAttributes.clear();
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* MultiscaleDeformableAttnPluginCreator::getPluginName() const PLUGIN_NOEXCEPT
{
    return DMHA_NAME;
}

const char* MultiscaleDeformableAttnPluginCreator::getPluginVersion() const PLUGIN_NOEXCEPT
{
    return DMHA_VERSION;
}

const nvinfer1::PluginFieldCollection* MultiscaleDeformableAttnPluginCreator::getFieldNames() PLUGIN_NOEXCEPT
{
    return &mFC;
}

IPluginV2* MultiscaleDeformableAttnPluginCreator::createPlugin(
    const char* name, const PluginFieldCollection* fc) PLUGIN_NOEXCEPT
{
    MultiscaleDeformableAttnPlugin* plugin = new MultiscaleDeformableAttnPlugin(name);
    return plugin;
}

IPluginV2* MultiscaleDeformableAttnPluginCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength) PLUGIN_NOEXCEPT
{
    auto plugin = new MultiscaleDeformableAttnPlugin(name, serialData, serialLength);
    plugin->setPluginNamespace(getPluginNamespace());
    return plugin;
}

void MultiscaleDeformableAttnPluginCreator::setPluginNamespace(const char* pluginNamespace) PLUGIN_NOEXCEPT
{
    mNamespace = pluginNamespace;
}

const char* MultiscaleDeformableAttnPluginCreator::getPluginNamespace() const PLUGIN_NOEXCEPT
{
    return mNamespace.c_str();
}

} // namespace plugin
} // namespace nvinfer1
