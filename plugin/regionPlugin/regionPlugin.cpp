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
#include "regionPlugin.h"
#include <cstring>

using namespace nvinfer1;
using namespace nvinfer1::plugin;
using nvinfer1::plugin::Region;
using nvinfer1::plugin::RegionParameters; // Needed for Windows Build
using nvinfer1::plugin::RegionPluginCreator;

namespace
{
const char* REGION_PLUGIN_VERSION{"1"};
const char* REGION_PLUGIN_NAME{"Region_TRT"};

template <typename T>
void safeFree(T* ptr)
{
    if (ptr)
    {
        free(ptr);
        ptr = nullptr;
    }
}

template <typename T>
void allocateChunk(T*& ptr, int count)
{
    ptr = static_cast<T*>(malloc(count * sizeof(T)));
}

struct SoftmaxTreeDeleter
{
    void operator()(softmaxTree* smTree) const
    {
        if (smTree)
        {
            // free individual elements first
            safeFree(smTree->leaf);
            safeFree(smTree->parent);
            safeFree(smTree->child);
            safeFree(smTree->group);
            if (smTree->name)
            {
                for (int i = 0; i < smTree->n; i++)
                {
                    safeFree(smTree->name[i]);
                }
                safeFree(smTree->name);
            }
            safeFree(smTree->groupSize);
            safeFree(smTree->groupOffset);

            // free softmax tree
            safeFree(smTree);
        }
    }
};

} // namespace

PluginFieldCollection RegionPluginCreator::mFC{};
std::vector<PluginField> RegionPluginCreator::mPluginAttributes;

Region::Region(RegionParameters params)
    : num(params.num)
    , coords(params.coords)
    , classes(params.classes)
    , smTree(params.smTree, SoftmaxTreeDeleter())
{
}

Region::Region(RegionParameters params, int C, int H, int W)
    : num(params.num)
    , coords(params.coords)
    , classes(params.classes)
    , smTree(params.smTree, SoftmaxTreeDeleter())
    , C(C)
    , H(H)
    , W(W)
{
}

Region::Region(const void* buffer, size_t length)
{
    const char *d = reinterpret_cast<const char*>(buffer), *a = d;
    C = read<int>(d);
    H = read<int>(d);
    W = read<int>(d);
    num = read<int>(d);
    classes = read<int>(d);
    coords = read<int>(d);
    bool softmaxTreePresent = read<bool>(d);
    bool leafPresent = read<bool>(d);
    bool parentPresent = read<bool>(d);
    bool childPresent = read<bool>(d);
    bool groupPresent = read<bool>(d);
    bool namePresent = read<bool>(d);
    bool groupSizePresent = read<bool>(d);
    bool groupOffsetPresent = read<bool>(d);
    if (softmaxTreePresent)
    {
        softmaxTree* smTreeTemp;
        // need to read each element individually
        allocateChunk(smTreeTemp, 1);

        smTreeTemp->n = read<int>(d);

        if (leafPresent)
        {
            allocateChunk(smTreeTemp->leaf, smTreeTemp->n);
        }
        else
        {
            smTreeTemp->leaf = nullptr;
        }
        if (parentPresent)
        {
            allocateChunk(smTreeTemp->parent, smTreeTemp->n);
        }
        else
        {
            smTreeTemp->parent = nullptr;
        }
        if (childPresent)
        {
            allocateChunk(smTreeTemp->child, smTreeTemp->n);
        }
        else
        {
            smTreeTemp->child = nullptr;
        }
        if (groupPresent)
        {
            allocateChunk(smTreeTemp->group, smTreeTemp->n);
        }
        else
        {
            smTreeTemp->group = nullptr;
        }

        for (int i = 0; i < smTreeTemp->n; i++)
        {
            if (leafPresent)
            {
                smTreeTemp->leaf[i] = read<int>(d);
            }
            if (parentPresent)
            {
                smTreeTemp->parent[i] = read<int>(d);
            }
            if (childPresent)
            {
                smTreeTemp->child[i] = read<int>(d);
            }
            if (groupPresent)
            {
                smTreeTemp->group[i] = read<int>(d);
            }
        }

        if (namePresent)
        {
            allocateChunk(smTreeTemp->name, smTreeTemp->n);
        }
        else
        {
            smTreeTemp->name = nullptr;
        }

        if (namePresent)
        {
            for (int i = 0; i < smTreeTemp->n; i++)
            {
                allocateChunk(smTreeTemp->name[i], 256);
                for (int j = 0; j < 256; j++)
                {
                    smTreeTemp->name[i][j] = read<char>(d);
                }
            }
        }

        smTreeTemp->groups = read<int>(d);
        if (groupSizePresent)
        {
            allocateChunk(smTreeTemp->groupSize, smTreeTemp->groups);
        }
        else
        {
            smTreeTemp->groupSize = nullptr;
        }
        if (groupOffsetPresent)
        {
            allocateChunk(smTreeTemp->groupOffset, smTreeTemp->groups);
        }
        else
        {
            smTreeTemp->groupOffset = nullptr;
        }
        for (int i = 0; i < smTreeTemp->groups; i++)
        {
            if (groupSizePresent)
            {
                smTreeTemp->groupSize[i] = read<int>(d);
            }
            if (groupOffsetPresent)
            {
                smTreeTemp->groupOffset[i] = read<int>(d);
            }
        }
        smTree = std::shared_ptr<softmaxTree>(smTreeTemp, SoftmaxTreeDeleter());
    }
    else
    {
        smTree.reset();
    }
    PLUGIN_VALIDATE(d == a + length);
}

int Region::getNbOutputs() const noexcept
{
    return 1;
}

Dims Region::getOutputDimensions(int index, const Dims* inputs, int nbInputDims) noexcept
{
    PLUGIN_ASSERT(nbInputDims == 1);
    PLUGIN_ASSERT(index == 0);
    return inputs[0];
}

int Region::enqueue(
    int batchSize, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    const void* inputData = inputs[0];
    void* outputData = outputs[0];
    if (smTree.get())
    {
        hasSoftmaxTree = true;
    }
    else
    {
        hasSoftmaxTree = false;
    }
    pluginStatus_t status = regionInference(
        stream, batchSize, C, H, W, num, coords, classes, hasSoftmaxTree, smTree.get(), inputData, outputData);
    return status;
}

size_t Region::getSerializationSize() const noexcept
{
    // C, H, W, num, classes, coords, smTree !nullptr and other array members !nullptr, softmaxTree members
    size_t count = 6 * sizeof(int) + 8 * sizeof(bool);
    if (smTree.get())
    {
        count += 2 * sizeof(int);

        if (smTree->leaf)
        {
            count += smTree->n * sizeof(int);
        }
        if (smTree->parent)
        {
            count += smTree->n * sizeof(int);
        }
        if (smTree->child)
        {
            count += smTree->n * sizeof(int);
        }
        if (smTree->group)
        {
            count += smTree->n * sizeof(int);
        }
        if (smTree->name)
        {
            count += smTree->n * 256 * sizeof(char);
        }
        if (smTree->groupSize)
        {
            count += smTree->groups * sizeof(int);
        }
        if (smTree->groupOffset)
        {
            count += smTree->groups * sizeof(int);
        }
    }
    return count;
}

void Region::serialize(void* buffer) const noexcept
{
    char *d = reinterpret_cast<char*>(buffer), *a = d;
    write(d, C);
    write(d, H);
    write(d, W);
    write(d, num);
    write(d, classes);
    write(d, coords);
    write(d, smTree.get() != nullptr);
    write(d, smTree.get() != nullptr && smTree->leaf != nullptr);
    write(d, smTree.get() != nullptr && smTree->parent != nullptr);
    write(d, smTree.get() != nullptr && smTree->child != nullptr);
    write(d, smTree.get() != nullptr && smTree->group != nullptr);
    write(d, smTree.get() != nullptr && smTree->name != nullptr);
    write(d, smTree.get() != nullptr && smTree->groupSize != nullptr);
    write(d, smTree.get() != nullptr && smTree->groupOffset != nullptr);
    // need to do a deep copy
    if (smTree)
    {
        write(d, smTree->n);
        for (int i = 0; i < smTree->n; i++)
        {
            if (smTree->leaf)
            {
                write(d, smTree->leaf[i]);
            }
            if (smTree->parent)
            {
                write(d, smTree->parent[i]);
            }
            if (smTree->child)
            {
                write(d, smTree->child[i]);
            }
            if (smTree->group)
            {
                write(d, smTree->group[i]);
            }
        }
        if (smTree->name)
        {
            for (int i = 0; i < smTree->n; i++)
            {
                const char* str = smTree->name[i];
                for (int j = 0; j < 256; j++)
                {
                    write(d, str[j]);
                }
            }
        }
        write(d, smTree->groups);
        for (int i = 0; i < smTree->groups; i++)
        {
            if (smTree->groupSize)
            {
                write(d, smTree->groupSize[i]);
            }
            if (smTree->groupOffset)
            {
                write(d, smTree->groupOffset[i]);
            }
        }
    }
    PLUGIN_ASSERT(d == a + getSerializationSize());
}

bool Region::supportsFormat(DataType type, PluginFormat format) const noexcept
{
    return (type == DataType::kFLOAT && format == PluginFormat::kLINEAR);
}

int Region::initialize() noexcept
{
    return STATUS_SUCCESS;
}

void Region::terminate() noexcept
{
}

const char* Region::getPluginType() const noexcept
{
    return REGION_PLUGIN_NAME;
}

const char* Region::getPluginVersion() const noexcept
{
    return REGION_PLUGIN_VERSION;
}

size_t Region::getWorkspaceSize(int maxBatchSize) const noexcept
{
    return 0;
}

void Region::destroy() noexcept
{
    delete this;
}

IPluginV2Ext* Region::clone() const noexcept
{
    try
    {
        RegionParameters params{num, coords, classes, nullptr};
        Region* plugin = new Region(params, C, H, W);
        plugin->setPluginNamespace(mPluginNamespace.c_str());
        plugin->setSoftmaxTree(smTree);

        return plugin;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

// Set plugin namespace
void Region::setPluginNamespace(const char* pluginNamespace) noexcept
{
    mPluginNamespace = pluginNamespace;
}

const char* Region::getPluginNamespace() const noexcept
{
    return mPluginNamespace.c_str();
}

// Return the DataType of the plugin output at the requested index
DataType Region::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept
{
    PLUGIN_ASSERT(index == 0);
    return DataType::kFLOAT;
}

// Return true if output tensor is broadcast across a batch.
bool Region::isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const noexcept
{
    return false;
}

// Return true if plugin can use input that is broadcast across batch without replication.
bool Region::canBroadcastInputAcrossBatch(int inputIndex) const noexcept
{
    return false;
}

// Configure the layer with input and output data types.
void Region::configurePlugin(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,
    const DataType* inputTypes, const DataType* outputTypes, const bool* inputIsBroadcast,
    const bool* outputIsBroadcast, PluginFormat floatFormat, int maxBatchSize) noexcept
{
    PLUGIN_ASSERT(*inputTypes == DataType::kFLOAT && floatFormat == PluginFormat::kLINEAR);
    PLUGIN_ASSERT(nbInputs == 1);
    PLUGIN_ASSERT(nbOutputs == 1);
    C = inputDims[0].d[0];
    H = inputDims[0].d[1];
    W = inputDims[0].d[2];
    /*
     * In the below assertion, 1 stands for the objectness of the bounding box
     * We should also
     * PLUGIN_ASSERT(coords == 4);
     */
    PLUGIN_ASSERT(C == num * (coords + 1 + classes));
}

// Attach the plugin object to an execution context and grant the plugin the access to some context resource.
void Region::attachToContext(cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) noexcept {}

// Detach the plugin object from its execution context.
void Region::detachFromContext() noexcept {}

RegionPluginCreator::RegionPluginCreator()
{
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("num", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("coords", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("classes", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("smTree", nullptr, PluginFieldType::kINT32, 1));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* RegionPluginCreator::getPluginName() const noexcept
{
    return REGION_PLUGIN_NAME;
}

const char* RegionPluginCreator::getPluginVersion() const noexcept
{
    return REGION_PLUGIN_VERSION;
}

const PluginFieldCollection* RegionPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2Ext* RegionPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
{
    try
    {
        const PluginField* fields = fc->fields;
        for (int i = 0; i < fc->nbFields; ++i)
        {
            const char* attrName = fields[i].name;
            if (!strcmp(attrName, "num"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kINT32);
                params.num = *(static_cast<const int*>(fields[i].data));
            }
            if (!strcmp(attrName, "coords"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kINT32);
                params.coords = *(static_cast<const int*>(fields[i].data));
            }
            if (!strcmp(attrName, "classes"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kINT32);
                params.classes = *(static_cast<const int*>(fields[i].data));
            }
            if (!strcmp(attrName, "smTree"))
            {
                // TODO not sure if this will work
                void* tmpData = const_cast<void*>(fields[i].data);
                params.smTree = static_cast<nvinfer1::plugin::softmaxTree*>(tmpData);
            }
        }

        Region* obj = new Region(params);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2Ext* RegionPluginCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength) noexcept
{
    try
    {
        // This object will be deleted when the network is destroyed, which will
        // call Region::destroy()
        Region* obj = new Region(serialData, serialLength);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}
