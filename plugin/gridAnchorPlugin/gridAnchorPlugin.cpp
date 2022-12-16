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

#include "gridAnchorPlugin.h"
#include <cstring>
#include <cublas_v2.h>
#include <cudnn.h>
#include <iostream>
#include <sstream>
#include <vector>

using namespace nvinfer1;

namespace nvinfer1
{
namespace plugin
{
namespace
{
std::string GRID_ANCHOR_PLUGIN_NAMES[] = {"GridAnchor_TRT", "GridAnchorRect_TRT"};
const char* GRID_ANCHOR_PLUGIN_VERSION = "1";
} // namespace

PluginFieldCollection GridAnchorBasePluginCreator::mFC{};
std::vector<PluginField> GridAnchorBasePluginCreator::mPluginAttributes;

GridAnchorGenerator::GridAnchorGenerator(const GridAnchorParameters* paramIn, int numLayers, const char* name)
    : mPluginName(name)
    , mNumLayers(numLayers)
{
    PLUGIN_CUASSERT(cudaMallocHost((void**) &mNumPriors, mNumLayers * sizeof(int)));
    PLUGIN_CUASSERT(cudaMallocHost((void**) &mDeviceWidths, mNumLayers * sizeof(Weights)));
    PLUGIN_CUASSERT(cudaMallocHost((void**) &mDeviceHeights, mNumLayers * sizeof(Weights)));

    mParam.resize(mNumLayers);
    for (int id = 0; id < mNumLayers; id++)
    {
        mParam[id] = paramIn[id];
        PLUGIN_VALIDATE(mParam[id].numAspectRatios >= 0 && mParam[id].aspectRatios != nullptr);

        mParam[id].aspectRatios = (float*) malloc(sizeof(float) * mParam[id].numAspectRatios);

        for (int i = 0; i < paramIn[id].numAspectRatios; ++i)
        {
            mParam[id].aspectRatios[i] = paramIn[id].aspectRatios[i];
        }

        for (int i = 0; i < 4; ++i)
        {
            mParam[id].variance[i] = paramIn[id].variance[i];
        }

        std::vector<float> tmpScales(mNumLayers + 1);

        // Calculate the scales of SSD model for each layer
        for (int i = 0; i < mNumLayers; i++)
        {
            tmpScales[i] = (mParam[id].minSize + (mParam[id].maxSize - mParam[id].minSize) * id / (mNumLayers - 1));
        }
        // Add another 1.0f to tmpScales to prevent going out side of the vector in calculating the scale_next.
        tmpScales.push_back(1.0f); // has 7 entries
        // scale0 are for the first layer specifically
        std::vector<float> scale0 = {0.1f, tmpScales[0], tmpScales[0]};

        std::vector<float> aspect_ratios;
        std::vector<float> scales;

        // The first layer is different
        if (id == 0)
        {
            for (int i = 0; i < mParam[id].numAspectRatios; i++)
            {
                aspect_ratios.push_back(mParam[id].aspectRatios[i]);
                scales.push_back(scale0[i]);
            }
            mNumPriors[id] = mParam[id].numAspectRatios;
        }

        else
        {
            for (int i = 0; i < mParam[id].numAspectRatios; i++)
            {
                aspect_ratios.push_back(mParam[id].aspectRatios[i]);
            }
            // Additional aspect ratio of 1.0 as described in the paper
            aspect_ratios.push_back(1.0);

            // scales
            for (int i = 0; i < mParam[id].numAspectRatios; i++)
            {
                scales.push_back(tmpScales[id]);
            }
            auto scale_next = (id == mNumLayers - 1)
                ? 1.0
                : (mParam[id].minSize + (mParam[id].maxSize - mParam[id].minSize) * (id + 1) / (mNumLayers - 1));
            scales.push_back(sqrt(tmpScales[id] * scale_next));

            mNumPriors[id] = mParam[id].numAspectRatios + 1;
        }

        std::vector<float> tmpWidths;
        std::vector<float> tmpHeights;
        // Calculate the width and height of the prior boxes
        for (int i = 0; i < mNumPriors[id]; i++)
        {
            float sqrt_AR = sqrt(aspect_ratios[i]);
            tmpWidths.push_back(scales[i] * sqrt_AR);
            tmpHeights.push_back(scales[i] / sqrt_AR);
        }

        mDeviceWidths[id] = copyToDevice(&tmpWidths[0], tmpWidths.size());
        mDeviceHeights[id] = copyToDevice(&tmpHeights[0], tmpHeights.size());
    }
}

GridAnchorGenerator::GridAnchorGenerator(const void* data, size_t length, const char* name)
    : mPluginName(name)
{
    const char *d = reinterpret_cast<const char*>(data), *a = d;
    mNumLayers = read<int>(d);
    PLUGIN_CUASSERT(cudaMallocHost((void**) &mNumPriors, mNumLayers * sizeof(int)));
    PLUGIN_CUASSERT(cudaMallocHost((void**) &mDeviceWidths, mNumLayers * sizeof(Weights)));
    PLUGIN_CUASSERT(cudaMallocHost((void**) &mDeviceHeights, mNumLayers * sizeof(Weights)));
    mParam.resize(mNumLayers);
    for (int id = 0; id < mNumLayers; id++)
    {
        // we have to deserialize GridAnchorParameters by hand
        mParam[id].minSize = read<float>(d);
        mParam[id].maxSize = read<float>(d);
        mParam[id].numAspectRatios = read<int>(d);
        mParam[id].aspectRatios = (float*) malloc(sizeof(float) * mParam[id].numAspectRatios);
        for (int i = 0; i < mParam[id].numAspectRatios; ++i)
        {
            mParam[id].aspectRatios[i] = read<float>(d);
        }
        mParam[id].H = read<int>(d);
        mParam[id].W = read<int>(d);
        for (int i = 0; i < 4; ++i)
        {
            mParam[id].variance[i] = read<float>(d);
        }

        mNumPriors[id] = read<int>(d);
        mDeviceWidths[id] = deserializeToDevice(d, mNumPriors[id]);
        mDeviceHeights[id] = deserializeToDevice(d, mNumPriors[id]);
    }

    PLUGIN_VALIDATE(d == a + length);
}

GridAnchorGenerator::~GridAnchorGenerator()
{
    for (int id = 0; id < mNumLayers; id++)
    {
        PLUGIN_CUERROR(cudaFree(const_cast<void*>(mDeviceWidths[id].values)));
        PLUGIN_CUERROR(cudaFree(const_cast<void*>(mDeviceHeights[id].values)));
        free(mParam[id].aspectRatios);
    }
    PLUGIN_CUERROR(cudaFreeHost(mNumPriors));
    PLUGIN_CUERROR(cudaFreeHost(mDeviceWidths));
    PLUGIN_CUERROR(cudaFreeHost(mDeviceHeights));
}

int GridAnchorGenerator::getNbOutputs() const noexcept
{
    return mNumLayers;
}

Dims GridAnchorGenerator::getOutputDimensions(int index, const Dims* inputs, int nbInputDims) noexcept
{
    // Particularity of the PriorBox layer: no batchSize dimension needed
    // 2 channels. First channel stores the mean of each prior coordinate.
    // Second channel stores the variance of each prior coordinate.
    return Dims3(2, mParam[index].H * mParam[index].W * mNumPriors[index] * 4, 1);
}

int GridAnchorGenerator::initialize() noexcept
{
    return STATUS_SUCCESS;
}

void GridAnchorGenerator::terminate() noexcept {}

size_t GridAnchorGenerator::getWorkspaceSize(int maxBatchSize) const noexcept
{
    return 0;
}

int GridAnchorGenerator::enqueue(
    int batchSize, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    // Generate prior boxes for each layer
    for (int id = 0; id < mNumLayers; id++)
    {
        void* outputData = outputs[id];
        pluginStatus_t status = anchorGridInference(
            stream, mParam[id], mNumPriors[id], mDeviceWidths[id].values, mDeviceHeights[id].values, outputData);
        if (status != STATUS_SUCCESS)
        {
            return status;
        }
    }
    return STATUS_SUCCESS;
}

size_t GridAnchorGenerator::getSerializationSize() const noexcept
{
    size_t sum = sizeof(int); // mNumLayers
    for (int i = 0; i < mNumLayers; i++)
    {
        sum += 4 * sizeof(int); // mNumPriors, mParam[i].{numAspectRatios, H, W}
        sum += (6 + mParam[i].numAspectRatios)
            * sizeof(float); // mParam[i].{minSize, maxSize, aspectRatios, variance[4]}
        sum += mDeviceWidths[i].count * sizeof(float);
        sum += mDeviceHeights[i].count * sizeof(float);
    }
    return sum;
}

void GridAnchorGenerator::serialize(void* buffer) const noexcept
{
    char *d = reinterpret_cast<char*>(buffer), *a = d;
    write(d, mNumLayers);
    for (int id = 0; id < mNumLayers; id++)
    {
        // we have to serialize GridAnchorParameters by hand
        write(d, mParam[id].minSize);
        write(d, mParam[id].maxSize);
        write(d, mParam[id].numAspectRatios);
        for (int i = 0; i < mParam[id].numAspectRatios; ++i)
        {
            write(d, mParam[id].aspectRatios[i]);
        }
        write(d, mParam[id].H);
        write(d, mParam[id].W);
        for (int i = 0; i < 4; ++i)
        {
            write(d, mParam[id].variance[i]);
        }

        write(d, mNumPriors[id]);
        serializeFromDevice(d, mDeviceWidths[id]);
        serializeFromDevice(d, mDeviceHeights[id]);
    }
    PLUGIN_ASSERT(d == a + getSerializationSize());
}

Weights GridAnchorGenerator::copyToDevice(const void* hostData, size_t count) noexcept
{
    void* deviceData;
    PLUGIN_CUASSERT(cudaMalloc(&deviceData, count * sizeof(float)));
    PLUGIN_CUASSERT(cudaMemcpy(deviceData, hostData, count * sizeof(float), cudaMemcpyHostToDevice));
    return Weights{DataType::kFLOAT, deviceData, int64_t(count)};
}

void GridAnchorGenerator::serializeFromDevice(char*& hostBuffer, Weights deviceWeights) const noexcept
{
    PLUGIN_CUASSERT(
        cudaMemcpy(hostBuffer, deviceWeights.values, deviceWeights.count * sizeof(float), cudaMemcpyDeviceToHost));
    hostBuffer += deviceWeights.count * sizeof(float);
}

Weights GridAnchorGenerator::deserializeToDevice(const char*& hostBuffer, size_t count) noexcept
{
    Weights w = copyToDevice(hostBuffer, count);
    hostBuffer += count * sizeof(float);
    return w;
}
bool GridAnchorGenerator::supportsFormat(DataType type, PluginFormat format) const noexcept
{
    return (type == DataType::kFLOAT && format == PluginFormat::kLINEAR);
}

const char* GridAnchorGenerator::getPluginType() const noexcept
{
    return mPluginName.c_str();
}

const char* GridAnchorGenerator::getPluginVersion() const noexcept
{
    return GRID_ANCHOR_PLUGIN_VERSION;
}

// Set plugin namespace
void GridAnchorGenerator::setPluginNamespace(const char* pluginNamespace) noexcept
{
    mPluginNamespace = pluginNamespace;
}

const char* GridAnchorGenerator::getPluginNamespace() const noexcept
{
    return mPluginNamespace.c_str();
}

#include <iostream>
// Return the DataType of the plugin output at the requested index
DataType GridAnchorGenerator::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const
    noexcept
{
    PLUGIN_ASSERT(index < mNumLayers);
    return DataType::kFLOAT;
}

// Return true if output tensor is broadcast across a batch.
bool GridAnchorGenerator::isOutputBroadcastAcrossBatch(
    int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const noexcept
{
    return false;
}

// Return true if plugin can use input that is broadcast across batch without replication.
bool GridAnchorGenerator::canBroadcastInputAcrossBatch(int inputIndex) const noexcept
{
    return false;
}

// Configure the layer with input and output data types.
void GridAnchorGenerator::configurePlugin(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,
    const DataType* inputTypes, const DataType* outputTypes, const bool* inputIsBroadcast,
    const bool* outputIsBroadcast, PluginFormat floatFormat, int maxBatchSize) noexcept
{
    PLUGIN_ASSERT(nbOutputs == mNumLayers);
    PLUGIN_ASSERT(outputDims[0].nbDims == 3);
}

// Attach the plugin object to an execution context and grant the plugin the access to some context resource.
void GridAnchorGenerator::attachToContext(
    cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) noexcept
{
}

// Detach the plugin object from its execution context.
void GridAnchorGenerator::detachFromContext() noexcept {}

void GridAnchorGenerator::destroy() noexcept
{
    delete this;
}

IPluginV2Ext* GridAnchorGenerator::clone() const noexcept
{
    try
    {
        IPluginV2Ext* plugin = new GridAnchorGenerator(mParam.data(), mNumLayers, mPluginName.c_str());
        plugin->setPluginNamespace(mPluginNamespace.c_str());
        return plugin;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

GridAnchorBasePluginCreator::GridAnchorBasePluginCreator()
{
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("minSize", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("maxSize", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("aspectRatios", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("featureMapShapes", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("variance", nullptr, PluginFieldType::kFLOAT32, 4));
    mPluginAttributes.emplace_back(PluginField("numLayers", nullptr, PluginFieldType::kINT32, 1));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* GridAnchorBasePluginCreator::getPluginName() const noexcept
{
    return mPluginName.c_str();
}

const char* GridAnchorBasePluginCreator::getPluginVersion() const noexcept
{
    return GRID_ANCHOR_PLUGIN_VERSION;
}

const PluginFieldCollection* GridAnchorBasePluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2Ext* GridAnchorBasePluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
{
    try
    {
        float minScale = 0.2F, maxScale = 0.95F;
        int numLayers = 6;
        std::vector<float> aspectRatios;
        std::vector<int> fMapShapes;
        std::vector<float> layerVariances;
        const PluginField* fields = fc->fields;

        const bool isFMapRect = (GRID_ANCHOR_PLUGIN_NAMES[1] == mPluginName);
        for (int i = 0; i < fc->nbFields; ++i)
        {
            const char* attrName = fields[i].name;
            if (!strcmp(attrName, "numLayers"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kINT32);
                numLayers = static_cast<int>(*(static_cast<const int*>(fields[i].data)));
            }
            else if (!strcmp(attrName, "minSize"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kFLOAT32);
                minScale = static_cast<float>(*(static_cast<const float*>(fields[i].data)));
            }
            else if (!strcmp(attrName, "maxSize"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kFLOAT32);
                maxScale = static_cast<float>(*(static_cast<const float*>(fields[i].data)));
            }
            else if (!strcmp(attrName, "variance"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kFLOAT32);
                int size = fields[i].length;
                layerVariances.reserve(size);
                const auto* lVar = static_cast<const float*>(fields[i].data);
                for (int j = 0; j < size; j++)
                {
                    layerVariances.push_back(*lVar);
                    lVar++;
                }
            }
            else if (!strcmp(attrName, "aspectRatios"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kFLOAT32);
                int size = fields[i].length;
                aspectRatios.reserve(size);
                const auto* aR = static_cast<const float*>(fields[i].data);
                for (int j = 0; j < size; j++)
                {
                    aspectRatios.push_back(*aR);
                    aR++;
                }
            }
            else if (!strcmp(attrName, "featureMapShapes"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kINT32);
                int size = fields[i].length;
                PLUGIN_VALIDATE(!isFMapRect || (size % 2 == 0));
                fMapShapes.reserve(size);
                const int* fMap = static_cast<const int*>(fields[i].data);
                for (int j = 0; j < size; j++)
                {
                    fMapShapes.push_back(*fMap);
                    fMap++;
                }
            }
        }
        // Reducing the number of boxes predicted by the first layer.
        // This is in accordance with the standard implementation.
        std::vector<float> firstLayerAspectRatios;

        PLUGIN_VALIDATE(numLayers > 0);
        const int numExpectedLayers = static_cast<int>(fMapShapes.size()) >> (isFMapRect ? 1 : 0);
        PLUGIN_VALIDATE(numExpectedLayers == numLayers);

        int numFirstLayerARs = 3;
        // First layer only has the first 3 aspect ratios from aspectRatios
        firstLayerAspectRatios.reserve(numFirstLayerARs);
        for (int i = 0; i < numFirstLayerARs; ++i)
        {
            firstLayerAspectRatios.push_back(aspectRatios[i]);
        }
        // A comprehensive list of box parameters that are required by anchor generator
        std::vector<GridAnchorParameters> boxParams(numLayers);

        // One set of box parameters for one layer
        for (int i = 0; i < numLayers; i++)
        {
            int hOffset = (isFMapRect ? i * 2 : i);
            int wOffset = (isFMapRect ? i * 2 + 1 : i);
            // Only the first layer is different
            if (i == 0)
            {
                boxParams[i] = {minScale, maxScale, firstLayerAspectRatios.data(), (int) firstLayerAspectRatios.size(),
                    fMapShapes[hOffset], fMapShapes[wOffset],
                    {layerVariances[0], layerVariances[1], layerVariances[2], layerVariances[3]}};
            }
            else
            {
                boxParams[i] = {minScale, maxScale, aspectRatios.data(), (int) aspectRatios.size(), fMapShapes[hOffset],
                    fMapShapes[wOffset], {layerVariances[0], layerVariances[1], layerVariances[2], layerVariances[3]}};
            }
        }

        GridAnchorGenerator* obj = new GridAnchorGenerator(boxParams.data(), numLayers, mPluginName.c_str());
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2Ext* GridAnchorBasePluginCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength) noexcept
{
    try
    {
        // This object will be deleted when the network is destroyed, which will
        // call GridAnchor::destroy()
        GridAnchorGenerator* obj = new GridAnchorGenerator(serialData, serialLength, mPluginName.c_str());
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

GridAnchorPluginCreator::GridAnchorPluginCreator()
{
    mPluginName = GRID_ANCHOR_PLUGIN_NAMES[0];
}

GridAnchorRectPluginCreator::GridAnchorRectPluginCreator()
{
    mPluginName = GRID_ANCHOR_PLUGIN_NAMES[1];
}

} // namespace plugin
} // namespace nvinfer1
