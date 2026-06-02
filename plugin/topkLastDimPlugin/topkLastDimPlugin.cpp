/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "topkLastDimPlugin.h"
#include "common/checkMacrosPlugin.h"
#include "common/plugin.h"
#include "transpose.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <limits>
#include <mutex>
#include <string_view>

namespace nvinfer1::plugin
{

// Kernel API implemented in topkLastDim.cu.
// The __restrict__ qualifiers must match the definition in topkLastDim.cu exactly,
// because MSVC includes __restrict in the mangled symbol name.
template <typename T>
size_t invokeComputeTopkLastDimWorkspaceSize(int32_t batchSize, int32_t inputLength, int32_t k, bool is_largest);
template <typename T>
void invokeTopkLastDim(int32_t batchSize, int32_t inputLength, int32_t k, bool is_largest,
    void const* __restrict__ input, void* __restrict__ out_val, void* __restrict__ out_idx, void* workspace,
    cudaStream_t stream);

namespace
{
char const* gKTopkLastDimPluginVersion{"1"};
char const* gKTopkLastDimPluginName{"TopkLastDim"};
} // namespace

// ========================== Plugin ==========================

TopkLastDimPlugin::TopkLastDimPlugin(int32_t typeId, int32_t k, int32_t isLargest, int32_t axis)
    : mTypeId(typeId)
    , mK(k)
    , mIsLargest(isLargest)
    , mAxis(axis)
{
    auto const type = static_cast<DataType>(mTypeId);
    PLUGIN_VALIDATE(
        type == DataType::kBF16 || type == DataType::kFLOAT || type == DataType::kHALF || type == DataType::kINT32);
    PLUGIN_VALIDATE(mK > 0);
    PLUGIN_VALIDATE(mIsLargest == 0 || mIsLargest == 1);
}

int32_t TopkLastDimPlugin::resolveAxis(int32_t nbDims) const
{
    int32_t axis = mAxis;
    if (axis < 0)
    {
        axis += nbDims;
    }
    PLUGIN_ASSERT(axis >= 0 && axis < nbDims);
    return axis;
}

int32_t TopkLastDimPlugin::elementSize() const
{
    auto const type = static_cast<DataType>(mTypeId);
    switch (type)
    {
    case DataType::kFLOAT:
    case DataType::kINT32: return 4;
    case DataType::kHALF:
    case DataType::kBF16: return 2;
    default: PLUGIN_ASSERT(false && "Unsupported data type"); return 0;
    }
}

size_t TopkLastDimPlugin::topkKernelWorkspaceSize(int32_t numRows, int32_t rowLength) const
{
    bool const isLargest = mIsLargest != 0;
    auto const type = static_cast<DataType>(mTypeId);
    if (type == DataType::kINT32)
    {
        return invokeComputeTopkLastDimWorkspaceSize<int32_t>(numRows, rowLength, mK, isLargest);
    }
    if (type == DataType::kHALF)
    {
        return invokeComputeTopkLastDimWorkspaceSize<half>(numRows, rowLength, mK, isLargest);
    }
    if (type == DataType::kFLOAT)
    {
        return invokeComputeTopkLastDimWorkspaceSize<float>(numRows, rowLength, mK, isLargest);
    }
    if (type == DataType::kBF16)
    {
        return invokeComputeTopkLastDimWorkspaceSize<__nv_bfloat16>(numRows, rowLength, mK, isLargest);
    }
    PLUGIN_ASSERT(false && "Unsupported data type");
    return 0;
}

size_t TopkLastDimPlugin::transposeWorkspaceSize(Dims const& dims) const
{
    int32_t const nbDims = dims.nbDims;
    int32_t const axis = resolveAxis(nbDims);

    // No transpose needed when axis is already the last dimension.
    if (axis == nbDims - 1)
    {
        return 0;
    }

    int64_t totalElems = 1;
    for (int32_t i = 0; i < nbDims; ++i)
    {
        totalElems *= dims.d[i];
    }

    int32_t const elemSz = elementSize();
    // Transposed input buffer (values).
    size_t bytes = totalElems * elemSz;
    // Transposed output values buffer.
    int64_t const axisLen = dims.d[axis];
    if (axisLen == 0)
    {
        return 0;
    }
    int64_t const outputElems = (totalElems / axisLen) * mK;
    bytes += outputElems * elemSz;
    // Alignment padding so transposedIndices (int32_t*) is 4-byte aligned.
    bytes = (bytes + alignof(int32_t) - 1) & ~(alignof(int32_t) - 1);
    // Transposed output indices buffer.
    bytes += outputElems * sizeof(int32_t);
    return bytes;
}

IPluginCapability* TopkLastDimPlugin::getCapabilityInterface(PluginCapabilityType type) noexcept
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

IPluginV3* TopkLastDimPlugin::clone() noexcept
{
    try
    {
        auto plugin = std::make_unique<TopkLastDimPlugin>(*this);
        return plugin.release();
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

char const* TopkLastDimPlugin::getPluginName() const noexcept
{
    return gKTopkLastDimPluginName;
}

char const* TopkLastDimPlugin::getPluginVersion() const noexcept
{
    return gKTopkLastDimPluginVersion;
}

char const* TopkLastDimPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

int32_t TopkLastDimPlugin::getNbOutputs() const noexcept
{
    return 2;
}

int32_t TopkLastDimPlugin::configurePlugin(
    DynamicPluginTensorDesc const* in, int32_t nbInputs, DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
    return 0;
}

bool TopkLastDimPlugin::supportsFormatCombination(
    int32_t pos, DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    PLUGIN_ASSERT(inOut != nullptr);
    PLUGIN_ASSERT(pos >= 0 && pos <= 2);
    PLUGIN_ASSERT(nbInputs == 1);
    PLUGIN_ASSERT(nbOutputs == 2);

    PluginTensorDesc const& desc = inOut[pos].desc;
    if (desc.format != TensorFormat::kLINEAR)
    {
        return false;
    }
    // Input (pos 0) and values output (pos 1) must match the configured type.
    if (pos < 2)
    {
        return desc.type == static_cast<DataType>(mTypeId);
    }
    // Indices output (pos 2) is always INT32.
    return desc.type == DataType::kINT32;
}

int32_t TopkLastDimPlugin::getOutputDataTypes(
    DataType* outputTypes, int32_t nbOutputs, DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    PLUGIN_ASSERT(nbInputs == 1);
    PLUGIN_ASSERT(nbOutputs == 2);
    outputTypes[0] = inputTypes[0];    // values: same type as input
    outputTypes[1] = DataType::kINT32; // indices
    return 0;
}

int32_t TopkLastDimPlugin::getOutputShapes(DimsExprs const* inputs, int32_t nbInputs, DimsExprs const* shapeInputs,
    int32_t nbShapeInputs, DimsExprs* outputs, int32_t nbOutputs, IExprBuilder& exprBuilder) noexcept
{
    PLUGIN_ASSERT(nbInputs == 1);
    PLUGIN_ASSERT(nbOutputs == 2);

    int32_t const nbDims = inputs[0].nbDims;
    int32_t const axis = resolveAxis(nbDims);

    auto const* kExpr = exprBuilder.constant(mK);
    PLUGIN_ASSERT(kExpr != nullptr);

    // Output shape is same as input but with dim[axis] replaced by k.
    for (int32_t o = 0; o < 2; ++o)
    {
        outputs[o] = inputs[0];
        outputs[o].d[axis] = kExpr;
    }
    return 0;
}

size_t TopkLastDimPlugin::getWorkspaceSize(DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
    DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept
{
    Dims const& maxDims = inputs[0].max;
    int32_t const nbDims = maxDims.nbDims;
    int32_t const axis = resolveAxis(nbDims);

    // Compute the 2D shape the kernel will see.
    int64_t numRows = 1;
    for (int32_t i = 0; i < nbDims; ++i)
    {
        if (i != axis)
        {
            numRows *= maxDims.d[i];
        }
    }
    int32_t const rowLength = maxDims.d[axis];

    PLUGIN_ASSERT(numRows <= std::numeric_limits<int32_t>::max());
    size_t bytes = topkKernelWorkspaceSize(static_cast<int32_t>(numRows), rowLength);
    bytes += transposeWorkspaceSize(maxDims);
    return bytes;
}

template <typename T>
int32_t TopkLastDimPlugin::enqueueImpl(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc,
    void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream)
{
    Dims const& dims = inputDesc[0].dims;
    int32_t const nbDims = dims.nbDims;
    int32_t const axis = resolveAxis(nbDims);

    // Compute outer, axisLen, inner for the 3D view [outer, axisLen, inner].
    int64_t outer = 1;
    for (int32_t i = 0; i < axis; ++i)
    {
        outer *= dims.d[i];
    }
    int32_t const axisLen = dims.d[axis];
    if (axisLen == 0)
    {
        return 0; // Empty tensor along axis dimension
    }
    int64_t inner = 1;
    for (int32_t i = axis + 1; i < nbDims; ++i)
    {
        inner *= dims.d[i];
    }

    PLUGIN_ASSERT(outer * inner <= std::numeric_limits<int32_t>::max());
    int32_t const numRows = static_cast<int32_t>(outer * inner);
    if (numRows == 0)
    {
        return 0;
    }

    bool const isLargest = mIsLargest != 0;

    // Fast path: axis is already the last dimension — call the kernel directly.
    if (axis == nbDims - 1)
    {
        invokeTopkLastDim<T>(numRows, axisLen, mK, isLargest, inputs[0], outputs[0], outputs[1], workspace, stream);
        PLUGIN_CUASSERT(cudaGetLastError());
        return 0;
    }

    // Multi-dimensional tensor path: transpose -> topk -> transpose back.
    int32_t const elemSz = sizeof(T);
    int64_t const totalInputElems = outer * axisLen * inner;
    int64_t const totalOutputElems = outer * inner * mK;

    // Partition workspace: [transposedInput | transposedValues | transposedIndices | topkWorkspace]
    char* wsPtr = static_cast<char*>(workspace);
    T* transposedInput = reinterpret_cast<T*>(wsPtr);
    wsPtr += totalInputElems * elemSz;
    T* transposedValues = reinterpret_cast<T*>(wsPtr);
    wsPtr += totalOutputElems * elemSz;
    // Align to 4 bytes for int32_t (needed when T is a 2-byte type and element count is odd).
    auto aligned = (reinterpret_cast<uintptr_t>(wsPtr) + alignof(int32_t) - 1) & ~(alignof(int32_t) - 1);
    int32_t* transposedIndices = reinterpret_cast<int32_t*>(aligned);
    wsPtr = reinterpret_cast<char*>(aligned);
    wsPtr += totalOutputElems * sizeof(int32_t);
    void* topkWorkspace = wsPtr;

    // Step 1: Transpose input [outer, axisLen, inner] -> [outer, inner, axisLen]
    launchBatchedTranspose2D<T>(static_cast<T const*>(inputs[0]), transposedInput, static_cast<int32_t>(outer), axisLen,
        static_cast<int32_t>(inner), stream);

    // Step 2: Run TopK on the 2D view [outer*inner, axisLen]
    invokeTopkLastDim<T>(
        numRows, axisLen, mK, isLargest, transposedInput, transposedValues, transposedIndices, topkWorkspace, stream);

    // Step 3: Transpose outputs [outer, inner, K] -> [outer, K, inner]
    launchBatchedTranspose2D<T>(transposedValues, static_cast<T*>(outputs[0]), static_cast<int32_t>(outer),
        static_cast<int32_t>(inner), mK, stream);
    launchBatchedTranspose2D<int32_t>(transposedIndices, static_cast<int32_t*>(outputs[1]), static_cast<int32_t>(outer),
        static_cast<int32_t>(inner), mK, stream);

    PLUGIN_CUASSERT(cudaGetLastError());
    return 0;
}

int32_t TopkLastDimPlugin::enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc,
    void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    auto const type = static_cast<DataType>(mTypeId);
    if (type == DataType::kINT32)
    {
        return enqueueImpl<int32_t>(inputDesc, outputDesc, inputs, outputs, workspace, stream);
    }
    if (type == DataType::kHALF)
    {
        return enqueueImpl<half>(inputDesc, outputDesc, inputs, outputs, workspace, stream);
    }
    if (type == DataType::kFLOAT)
    {
        return enqueueImpl<float>(inputDesc, outputDesc, inputs, outputs, workspace, stream);
    }
    if (type == DataType::kBF16)
    {
        return enqueueImpl<__nv_bfloat16>(inputDesc, outputDesc, inputs, outputs, workspace, stream);
    }
    PLUGIN_ASSERT(false && "Unsupported data type");
    return 0;
}

int32_t TopkLastDimPlugin::onShapeChange(
    PluginTensorDesc const* in, int32_t nbInputs, PluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
    return 0;
}

IPluginV3* TopkLastDimPlugin::attachToContext(IPluginResourceContext* context) noexcept
{
    return clone();
}

PluginFieldCollection const* TopkLastDimPlugin::getFieldsToSerialize() noexcept
{
    mDataToSerialize.clear();
    mDataToSerialize.emplace_back("type_id", &mTypeId, PluginFieldType::kINT32, 1);
    mDataToSerialize.emplace_back("k", &mK, PluginFieldType::kINT32, 1);
    mDataToSerialize.emplace_back("is_largest", &mIsLargest, PluginFieldType::kINT32, 1);
    mDataToSerialize.emplace_back("axis", &mAxis, PluginFieldType::kINT32, 1);
    mFCToSerialize.nbFields = mDataToSerialize.size();
    mFCToSerialize.fields = mDataToSerialize.data();
    return &mFCToSerialize;
}

void TopkLastDimPlugin::setPluginNamespace(char const* pluginNamespace) noexcept
{
    try
    {
        PLUGIN_ASSERT(pluginNamespace != nullptr);
        mNamespace = pluginNamespace;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
}

// ========================== Creator ==========================

TopkLastDimPluginCreator::TopkLastDimPluginCreator()
{
    static std::mutex sMutex;
    std::lock_guard<std::mutex> guard(sMutex);
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("type_id", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("k", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("is_largest", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("axis", nullptr, PluginFieldType::kINT32, 1));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

char const* TopkLastDimPluginCreator::getPluginName() const noexcept
{
    return gKTopkLastDimPluginName;
}

char const* TopkLastDimPluginCreator::getPluginVersion() const noexcept
{
    return gKTopkLastDimPluginVersion;
}

PluginFieldCollection const* TopkLastDimPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV3* TopkLastDimPluginCreator::createPlugin(
    char const* name, PluginFieldCollection const* fc, TensorRTPhase phase) noexcept
{
    try
    {
        PLUGIN_VALIDATE(fc != nullptr);
        PluginField const* fields = fc->fields;

        int32_t typeId{};
        int32_t k{};
        int32_t isLargest{};
        int32_t axis{-1}; // default: last dimension
        bool hasTypeId = false;
        bool hasK = false;
        bool hasIsLargest = false;

        using namespace std::string_view_literals;
        for (int32_t i = 0; i < fc->nbFields; ++i)
        {
            std::string_view const attrName = fields[i].name;
            if (attrName == "type_id"sv)
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kINT32);
                typeId = *static_cast<int32_t const*>(fields[i].data);
                hasTypeId = true;
            }
            else if (attrName == "k"sv)
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kINT32);
                k = *static_cast<int32_t const*>(fields[i].data);
                hasK = true;
            }
            else if (attrName == "is_largest"sv)
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kINT32);
                isLargest = *static_cast<int32_t const*>(fields[i].data);
                hasIsLargest = true;
            }
            else if (attrName == "axis"sv)
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kINT32);
                axis = *static_cast<int32_t const*>(fields[i].data);
            }
        }
        PLUGIN_VALIDATE(hasTypeId, "Missing required field 'type_id'");
        PLUGIN_VALIDATE(hasK, "Missing required field 'k'");
        PLUGIN_VALIDATE(hasIsLargest, "Missing required field 'is_largest'");
        return new TopkLastDimPlugin(typeId, k, isLargest, axis);
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

char const* TopkLastDimPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

void TopkLastDimPluginCreator::setPluginNamespace(char const* pluginNamespace) noexcept
{
    try
    {
        PLUGIN_ASSERT(pluginNamespace != nullptr);
        mNamespace = pluginNamespace;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
}

} // namespace nvinfer1::plugin
