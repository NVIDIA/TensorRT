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

#include "multiscaleDeformableAttnPlugin.h"
#include "multiscaleDeformableAttn.h"

using namespace nvinfer1;
using namespace nvinfer1::plugin;

namespace
{
static char const* DMHA_VERSION{"2"};
static char const* DMHA_NAME{"MultiscaleDeformableAttnPlugin_TRT"};
} // namespace

namespace nvinfer1::plugin
{

MultiscaleDeformableAttnPlugin::MultiscaleDeformableAttnPlugin() {}

IPluginCapability* MultiscaleDeformableAttnPlugin::getCapabilityInterface(PluginCapabilityType type) noexcept
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

// IPluginV3OneCore methods
char const* MultiscaleDeformableAttnPlugin::getPluginName() const noexcept
{
    return DMHA_NAME;
}

char const* MultiscaleDeformableAttnPlugin::getPluginVersion() const noexcept
{
    return DMHA_VERSION;
}

int32_t MultiscaleDeformableAttnPlugin::getNbOutputs() const noexcept
{
    return 1;
}

void MultiscaleDeformableAttnPlugin::setPluginNamespace(char const* pluginNamespace) noexcept
{
    mNamespace = pluginNamespace;
}

char const* MultiscaleDeformableAttnPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

IPluginV3* MultiscaleDeformableAttnPlugin::clone() noexcept
{
    try
    {
        auto* plugin = new MultiscaleDeformableAttnPlugin();
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

// IPluginV3OneBuild methods
int32_t MultiscaleDeformableAttnPlugin::getOutputDataTypes(
    DataType* outputTypes, int32_t nbOutputs, DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    try
    {
        PLUGIN_VALIDATE(outputTypes != nullptr, "outputTypes pointer is null");
        PLUGIN_VALIDATE(nbOutputs > 0, "nbOutputs is not positive");
        PLUGIN_VALIDATE(inputTypes != nullptr, "inputTypes pointer is null");
        PLUGIN_VALIDATE(nbInputs > 0, "nbInputs is not positive");

        // Output type is the same as the first input type
        std::fill_n(outputTypes, nbOutputs, inputTypes[0]);

        return STATUS_SUCCESS;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return STATUS_FAILURE;
}

int32_t MultiscaleDeformableAttnPlugin::getOutputShapes(DimsExprs const* inputs, int32_t nbInputs,
    DimsExprs const* shapeInputs, int32_t nbShapeInputs, DimsExprs* outputs, int32_t nbOutputs,
    IExprBuilder& exprBuilder) noexcept
{
    try
    {
        PLUGIN_VALIDATE(outputs != nullptr, "outputs pointer is null");
        PLUGIN_VALIDATE(nbOutputs > 0, "nbOutputs is not positive");
        PLUGIN_VALIDATE(inputs != nullptr, "inputs pointer is null");
        PLUGIN_VALIDATE(nbInputs == 5, "Expected 5 inputs");

        // Output shape: [N, Lq, M, D]
        outputs[0].nbDims = 4;
        outputs[0].d[0] = inputs[0].d[0]; // Batch size
        outputs[0].d[1] = inputs[3].d[1]; // Lq (query length)
        outputs[0].d[2] = inputs[0].d[2]; // Number of heads
        outputs[0].d[3] = inputs[0].d[3]; // Hidden dimension per head

        return STATUS_SUCCESS;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return STATUS_FAILURE;
}

bool MultiscaleDeformableAttnPlugin::supportsFormatCombination(
    int32_t pos, DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    try
    {
        PLUGIN_VALIDATE(inOut != nullptr, "inOut pointer is null");
        PLUGIN_VALIDATE(nbInputs == 5, "Expected 5 inputs");
        PLUGIN_VALIDATE(nbOutputs == 1, "Expected 1 output");

        // Check format
        PluginTensorDesc const& desc = inOut[pos].desc;
        if (desc.format != TensorFormat::kLINEAR)
        {
            return false;
        }

        // Special handling for spatial_shapes and level_start_index (inputs 1 and 2)
        if (pos == 1 || pos == 2)
        {
            return desc.type == DataType::kINT32;
        }

        // Other inputs and output must have the same type, either FP32 or FP16
        if (pos == 0 || pos == 3 || pos == 4 || pos == nbInputs)
        {
            // Check that the data type matches input[0]
            bool const isFloatType = desc.type == DataType::kFLOAT || desc.type == DataType::kHALF;
            if (pos == 0) // First tensor, just check if it's a supported type
            {
                return isFloatType;
            }
            // Other tensors must match the first
            return desc.type == inOut[0].desc.type && isFloatType;
        }

        return false;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return false;
}

int32_t MultiscaleDeformableAttnPlugin::configurePlugin(
    DynamicPluginTensorDesc const* in, int32_t nbInputs, DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
    try
    {
        PLUGIN_VALIDATE(in != nullptr, "in pointer is null");
        PLUGIN_VALIDATE(out != nullptr, "out pointer is null");
        PLUGIN_VALIDATE(nbInputs == 5, "Expected 5 inputs");
        PLUGIN_VALIDATE(nbOutputs == 1, "Expected 1 output");

        // Check for valid input dimensions
        PLUGIN_VALIDATE(in[0].desc.dims.nbDims == 4, "First input must have 4 dimensions");
        PLUGIN_VALIDATE(in[1].desc.dims.nbDims == 2, "Second input must have 2 dimensions");
        PLUGIN_VALIDATE(in[2].desc.dims.nbDims == 1, "Third input must have 1 dimension");
        PLUGIN_VALIDATE(in[3].desc.dims.nbDims == 6, "Fourth input must have 6 dimensions");
        PLUGIN_VALIDATE(in[4].desc.dims.nbDims == 5, "Fifth input must have 5 dimensions");

        // Check M dimensions consistency
        PLUGIN_VALIDATE(in[0].desc.dims.d[2] == in[3].desc.dims.d[2], "Inconsistent dimensions for number of heads");
        PLUGIN_VALIDATE(in[0].desc.dims.d[2] == in[4].desc.dims.d[2], "Inconsistent dimensions for number of heads");

        // Check L dimensions consistency
        PLUGIN_VALIDATE(in[1].desc.dims.d[0] == in[2].desc.dims.d[0], "Inconsistent dimensions for number of levels");
        PLUGIN_VALIDATE(in[1].desc.dims.d[0] == in[3].desc.dims.d[3], "Inconsistent dimensions for number of levels");
        PLUGIN_VALIDATE(in[1].desc.dims.d[0] == in[4].desc.dims.d[3], "Inconsistent dimensions for number of levels");

        // Check P dimensions consistency
        PLUGIN_VALIDATE(in[3].desc.dims.d[4] == in[4].desc.dims.d[4], "Inconsistent dimensions for number of points");

        // Check Lq dimensions consistency
        PLUGIN_VALIDATE(in[3].desc.dims.d[1] == in[4].desc.dims.d[1], "Inconsistent dimensions for query length");

        return STATUS_SUCCESS;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return STATUS_FAILURE;
}

PluginFieldCollection const* MultiscaleDeformableAttnPlugin::getFieldsToSerialize() noexcept
{
    try
    {
        mDataToSerialize.clear();
        // This plugin has no fields to serialize
        mFCToSerialize.nbFields = mDataToSerialize.size();
        mFCToSerialize.fields = mDataToSerialize.data();
        return &mFCToSerialize;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

// IPluginV3OneRuntime methods
size_t MultiscaleDeformableAttnPlugin::getWorkspaceSize(DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
    DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept
{
    // No workspace needed for this plugin
    return 0;
}

int32_t MultiscaleDeformableAttnPlugin::onShapeChange(
    PluginTensorDesc const* inputs, int32_t nbInputs, PluginTensorDesc const* outputs, int32_t nbOutputs) noexcept
{
    try
    {
        PLUGIN_VALIDATE(inputs != nullptr, "inputs pointer is null");
        PLUGIN_VALIDATE(outputs != nullptr, "outputs pointer is null");
        PLUGIN_VALIDATE(nbInputs == 5, "Expected 5 inputs");
        PLUGIN_VALIDATE(nbOutputs == 1, "Expected 1 output");

        // Check for valid input dimensions
        PLUGIN_VALIDATE(inputs[0].dims.nbDims == 4, "First input must have 4 dimensions");
        PLUGIN_VALIDATE(inputs[1].dims.nbDims == 2, "Second input must have 2 dimensions");
        PLUGIN_VALIDATE(inputs[2].dims.nbDims == 1, "Third input must have 1 dimension");
        PLUGIN_VALIDATE(inputs[3].dims.nbDims == 6, "Fourth input must have 6 dimensions");
        PLUGIN_VALIDATE(inputs[4].dims.nbDims == 5, "Fifth input must have 5 dimensions");

        // Check M dimensions consistency
        PLUGIN_VALIDATE(inputs[0].dims.d[2] == inputs[3].dims.d[2], "Inconsistent dimensions for number of heads");
        PLUGIN_VALIDATE(inputs[0].dims.d[2] == inputs[4].dims.d[2], "Inconsistent dimensions for number of heads");

        // Check L dimensions consistency
        PLUGIN_VALIDATE(inputs[1].dims.d[0] == inputs[2].dims.d[0], "Inconsistent dimensions for number of levels");
        PLUGIN_VALIDATE(inputs[1].dims.d[0] == inputs[3].dims.d[3], "Inconsistent dimensions for number of levels");
        PLUGIN_VALIDATE(inputs[1].dims.d[0] == inputs[4].dims.d[3], "Inconsistent dimensions for number of levels");

        // Check P dimensions consistency
        PLUGIN_VALIDATE(inputs[3].dims.d[4] == inputs[4].dims.d[4], "Inconsistent dimensions for number of points");

        // Check Lq dimensions consistency
        PLUGIN_VALIDATE(inputs[3].dims.d[1] == inputs[4].dims.d[1], "Inconsistent dimensions for query length");

        return STATUS_SUCCESS;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return STATUS_FAILURE;
}

IPluginV3* MultiscaleDeformableAttnPlugin::attachToContext(IPluginResourceContext* context) noexcept
{
    try
    {
        // No resources need to be attached
        return clone();
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

int32_t MultiscaleDeformableAttnPlugin::enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc,
    void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    try
    {
        PLUGIN_VALIDATE(
            inputDesc != nullptr && inputs != nullptr && outputs != nullptr, "Null pointers found in enqueue");

        int32_t const batch = inputDesc[0].dims.d[0];
        int32_t spatialSize = inputDesc[0].dims.d[1];
        int32_t numHeads = inputDesc[0].dims.d[2];
        int32_t channels = inputDesc[0].dims.d[3];
        int32_t numLevels = inputDesc[1].dims.d[0];
        int32_t numQuery = inputDesc[3].dims.d[1];
        int32_t numPoint = inputDesc[3].dims.d[4];
        int32_t rc = 0;

        if (inputDesc[0].type == DataType::kFLOAT)
        {
            auto const* value = static_cast<float const*>(inputs[0]);
            auto const* spatialShapes = static_cast<int32_t const*>(inputs[1]);
            auto const* levelStartIndex = static_cast<int32_t const*>(inputs[2]);
            auto const* samplingLoc = static_cast<float const*>(inputs[3]);
            auto const* attnWeight = static_cast<float const*>(inputs[4]);
            auto* output = static_cast<float*>(outputs[0]);

            rc = ms_deform_attn_cuda_forward(stream, value, spatialShapes, levelStartIndex, samplingLoc, attnWeight,
                output, batch, spatialSize, numHeads, channels, numLevels, numQuery, numPoint);
        }
        else if (inputDesc[0].type == DataType::kHALF)
        {
            auto const* value = static_cast<__half const*>(inputs[0]);
            auto const* spatialShapes = static_cast<int32_t const*>(inputs[1]);
            auto const* levelStartIndex = static_cast<int32_t const*>(inputs[2]);
            auto const* samplingLoc = static_cast<__half const*>(inputs[3]);
            auto const* attnWeight = static_cast<__half const*>(inputs[4]);
            auto* output = static_cast<__half*>(outputs[0]);

            rc = ms_deform_attn_cuda_forward(stream, value, spatialShapes, levelStartIndex, samplingLoc, attnWeight,
                output, batch, spatialSize, numHeads, channels, numLevels, numQuery, numPoint);
        }
        else
        {
            PLUGIN_VALIDATE(false, "Unsupported data type");
        }

        return rc;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return STATUS_FAILURE;
}

// Plugin Creator Implementation
MultiscaleDeformableAttnPluginCreator::MultiscaleDeformableAttnPluginCreator()
{
    mPluginAttributes.clear();
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

char const* MultiscaleDeformableAttnPluginCreator::getPluginName() const noexcept
{
    return DMHA_NAME;
}

char const* MultiscaleDeformableAttnPluginCreator::getPluginVersion() const noexcept
{
    return DMHA_VERSION;
}

PluginFieldCollection const* MultiscaleDeformableAttnPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV3* MultiscaleDeformableAttnPluginCreator::createPlugin(
    char const* name, PluginFieldCollection const* fc, TensorRTPhase phase) noexcept
{
    try
    {
        // This plugin doesn't have any configurable parameters
        return new MultiscaleDeformableAttnPlugin();
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

void MultiscaleDeformableAttnPluginCreator::setPluginNamespace(char const* pluginNamespace) noexcept
{
    mNamespace = pluginNamespace;
}

char const* MultiscaleDeformableAttnPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

} // namespace nvinfer1::plugin
