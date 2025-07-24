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
#include "regionPlugin.h"
#include <cstring>

namespace nvinfer1::plugin
{
namespace
{
char const* const kREGION_PLUGIN_VERSION{"1"};
char const* const kREGION_PLUGIN_NAME{"Region_TRT"};

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
void allocateChunk(T*& ptr, int32_t count)
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
                for (int32_t i = 0; i < smTree->n; i++)
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

Region::Region(RegionParameters params)
    : num(params.num)
    , coords(params.coords)
    , classes(params.classes)
    , smTree(params.smTree, SoftmaxTreeDeleter())
{
}

Region::Region(RegionParameters params, int32_t C, int32_t H, int32_t W)
    : num(params.num)
    , coords(params.coords)
    , classes(params.classes)
    , smTree(params.smTree, SoftmaxTreeDeleter())
    , C(C)
    , H(H)
    , W(W)
{
}

Region::Region(void const* buffer, size_t length)
{
    char const *d = reinterpret_cast<char const*>(buffer), *a = d;
    C = read<int32_t>(d);
    H = read<int32_t>(d);
    W = read<int32_t>(d);
    num = read<int32_t>(d);
    classes = read<int32_t>(d);
    coords = read<int32_t>(d);
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

        smTreeTemp->n = read<int32_t>(d);

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

        for (int32_t i = 0; i < smTreeTemp->n; i++)
        {
            if (leafPresent)
            {
                smTreeTemp->leaf[i] = read<int32_t>(d);
            }
            if (parentPresent)
            {
                smTreeTemp->parent[i] = read<int32_t>(d);
            }
            if (childPresent)
            {
                smTreeTemp->child[i] = read<int32_t>(d);
            }
            if (groupPresent)
            {
                smTreeTemp->group[i] = read<int32_t>(d);
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
            for (int32_t i = 0; i < smTreeTemp->n; i++)
            {
                allocateChunk(smTreeTemp->name[i], 256);
                for (int32_t j = 0; j < 256; j++)
                {
                    smTreeTemp->name[i][j] = read<char>(d);
                }
            }
        }

        smTreeTemp->groups = read<int32_t>(d);
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
        for (int32_t i = 0; i < smTreeTemp->groups; i++)
        {
            if (groupSizePresent)
            {
                smTreeTemp->groupSize[i] = read<int32_t>(d);
            }
            if (groupOffsetPresent)
            {
                smTreeTemp->groupOffset[i] = read<int32_t>(d);
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

int32_t Region::getNbOutputs() const noexcept
{
    return 1;
}

Dims Region::getOutputDimensions(int32_t index, Dims const* inputs, int32_t nbInputDims) noexcept
{
    PLUGIN_ASSERT(nbInputDims == 1);
    PLUGIN_ASSERT(index == 0);
    return inputs[0];
}

int32_t Region::enqueue(
    int32_t batchSize, void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    void const* inputData = inputs[0];
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
    size_t count = 6 * sizeof(int32_t) + 8 * sizeof(bool);
    if (smTree.get())
    {
        count += 2 * sizeof(int32_t);

        if (smTree->leaf)
        {
            count += smTree->n * sizeof(int32_t);
        }
        if (smTree->parent)
        {
            count += smTree->n * sizeof(int32_t);
        }
        if (smTree->child)
        {
            count += smTree->n * sizeof(int32_t);
        }
        if (smTree->group)
        {
            count += smTree->n * sizeof(int32_t);
        }
        if (smTree->name)
        {
            count += smTree->n * 256 * sizeof(char);
        }
        if (smTree->groupSize)
        {
            count += smTree->groups * sizeof(int32_t);
        }
        if (smTree->groupOffset)
        {
            count += smTree->groups * sizeof(int32_t);
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
    write(d, smTree != nullptr);
    write(d, smTree != nullptr && smTree->leaf != nullptr);
    write(d, smTree != nullptr && smTree->parent != nullptr);
    write(d, smTree != nullptr && smTree->child != nullptr);
    write(d, smTree != nullptr && smTree->group != nullptr);
    write(d, smTree != nullptr && smTree->name != nullptr);
    write(d, smTree != nullptr && smTree->groupSize != nullptr);
    write(d, smTree != nullptr && smTree->groupOffset != nullptr);
    // need to do a deep copy
    if (smTree)
    {
        write(d, smTree->n);
        for (int32_t i = 0; i < smTree->n; i++)
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
            for (int32_t i = 0; i < smTree->n; i++)
            {
                char const* str = smTree->name[i];
                for (int32_t j = 0; j < 256; j++)
                {
                    write(d, str[j]);
                }
            }
        }
        write(d, smTree->groups);
        for (int32_t i = 0; i < smTree->groups; i++)
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

int32_t Region::initialize() noexcept
{
    return STATUS_SUCCESS;
}

void Region::terminate() noexcept {}

char const* Region::getPluginType() const noexcept
{
    return kREGION_PLUGIN_NAME;
}

char const* Region::getPluginVersion() const noexcept
{
    return kREGION_PLUGIN_VERSION;
}

size_t Region::getWorkspaceSize(int32_t maxBatchSize) const noexcept
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
void Region::setPluginNamespace(char const* pluginNamespace) noexcept
{
    mPluginNamespace = pluginNamespace;
}

char const* Region::getPluginNamespace() const noexcept
{
    return mPluginNamespace.c_str();
}

// Return the DataType of the plugin output at the requested index
DataType Region::getOutputDataType(int32_t index, nvinfer1::DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    PLUGIN_ASSERT(index == 0);
    return DataType::kFLOAT;
}

// Return true if output tensor is broadcast across a batch.
bool Region::isOutputBroadcastAcrossBatch(
    int32_t outputIndex, bool const* inputIsBroadcasted, int32_t nbInputs) const noexcept
{
    return false;
}

// Return true if plugin can use input that is broadcast across batch without replication.
bool Region::canBroadcastInputAcrossBatch(int32_t inputIndex) const noexcept
{
    return false;
}

// Configure the layer with input and output data types.
void Region::configurePlugin(Dims const* inputDims, int32_t nbInputs, Dims const* outputDims, int32_t nbOutputs,
    DataType const* inputTypes, DataType const* outputTypes, bool const* inputIsBroadcast,
    bool const* outputIsBroadcast, PluginFormat floatFormat, int32_t maxBatchSize) noexcept
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
void Region::attachToContext(
    cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) noexcept
{
}

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

char const* RegionPluginCreator::getPluginName() const noexcept
{
    return kREGION_PLUGIN_NAME;
}

char const* RegionPluginCreator::getPluginVersion() const noexcept
{
    return kREGION_PLUGIN_VERSION;
}

PluginFieldCollection const* RegionPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2Ext* RegionPluginCreator::createPlugin(char const* name, PluginFieldCollection const* fc) noexcept
{
    try
    {
        PluginField const* fields = fc->fields;
        for (int32_t i = 0; i < fc->nbFields; ++i)
        {
            char const* attrName = fields[i].name;
            if (!strcmp(attrName, "num"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kINT32);
                params.num = *(static_cast<int32_t const*>(fields[i].data));
            }
            if (!strcmp(attrName, "coords"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kINT32);
                params.coords = *(static_cast<int32_t const*>(fields[i].data));
            }
            if (!strcmp(attrName, "classes"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kINT32);
                params.classes = *(static_cast<int32_t const*>(fields[i].data));
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
    char const* name, void const* serialData, size_t serialLength) noexcept
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
} // namespace nvinfer1::plugin
