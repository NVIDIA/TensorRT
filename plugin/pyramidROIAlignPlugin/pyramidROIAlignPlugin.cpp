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
#include "pyramidROIAlignPlugin.h"
#include "plugin.h"
#include <cuda_runtime_api.h>

using namespace nvinfer1;
using namespace plugin;
using nvinfer1::plugin::PyramidROIAlign;
using nvinfer1::plugin::PyramidROIAlignPluginCreator;

namespace
{
const char* PYRAMIDROIALGIN_PLUGIN_VERSION{"1"};
const char* PYRAMIDROIALGIN_PLUGIN_NAME{"PyramidROIAlign_TRT"};
} // namespace

PluginFieldCollection PyramidROIAlignPluginCreator::mFC{};
std::vector<PluginField> PyramidROIAlignPluginCreator::mPluginAttributes;

PyramidROIAlignPluginCreator::PyramidROIAlignPluginCreator()
{
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("pooled_size", nullptr, PluginFieldType::kINT32, 1));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* PyramidROIAlignPluginCreator::getPluginName() const noexcept
{
    return PYRAMIDROIALGIN_PLUGIN_NAME;
}

const char* PyramidROIAlignPluginCreator::getPluginVersion() const noexcept
{
    return PYRAMIDROIALGIN_PLUGIN_VERSION;
}

const PluginFieldCollection* PyramidROIAlignPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2Ext* PyramidROIAlignPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
{
    const PluginField* fields = fc->fields;
    for (int i = 0; i < fc->nbFields; ++i)
    {
        const char* attrName = fields[i].name;
        if (!strcmp(attrName, "pooled_size"))
        {
            assert(fields[i].type == PluginFieldType::kINT32);
            mPooledSize = *(static_cast<const int*>(fields[i].data));
        }
    }
    return new PyramidROIAlign(mPooledSize);
}

IPluginV2Ext* PyramidROIAlignPluginCreator::deserializePlugin(
    const char* name, const void* data, size_t length) noexcept
{
    return new PyramidROIAlign(data, length);
}

PyramidROIAlign::PyramidROIAlign(int pooled_size)
    : mPooledSize({pooled_size, pooled_size})
{

    assert(pooled_size > 0);
    // shape
    mInputSize = MaskRCNNConfig::IMAGE_SHAPE.d[1];
    mThresh = (224 * 224 * 2.0f / (mInputSize * mInputSize)) / (4.0 * 4.0f);
}

int PyramidROIAlign::getNbOutputs() const noexcept
{
    return 1;
}

int PyramidROIAlign::initialize() noexcept
{
    return 0;
}

void PyramidROIAlign::terminate() noexcept {}

void PyramidROIAlign::destroy() noexcept
{
    delete this;
}

size_t PyramidROIAlign::getWorkspaceSize(int) const noexcept
{
    return 0;
}

bool PyramidROIAlign::supportsFormat(DataType type, PluginFormat format) const noexcept
{
    return (type == DataType::kFLOAT && format == PluginFormat::kLINEAR);
}

const char* PyramidROIAlign::getPluginType() const noexcept
{
    return "PyramidROIAlign_TRT";
}

const char* PyramidROIAlign::getPluginVersion() const noexcept
{
    return "1";
}

IPluginV2Ext* PyramidROIAlign::clone() const noexcept
{
    auto plugin = new PyramidROIAlign(*this);
    plugin->setPluginNamespace(mNameSpace.c_str());
    return plugin;
}

void PyramidROIAlign::setPluginNamespace(const char* libNamespace) noexcept
{
    mNameSpace = libNamespace;
}

const char* PyramidROIAlign::getPluginNamespace() const noexcept
{
    return mNameSpace.c_str();
}

void PyramidROIAlign::check_valid_inputs(const nvinfer1::Dims* inputs, int nbInputDims)
{
    // to be compatible with tensorflow node's input:
    // roi: [N, anchors, 4],
    // feature_map list(4 maps): p2, p3, p4, p5
    assert(nbInputDims == 1 + mFeatureMapCount);

    nvinfer1::Dims rois = inputs[0];
    assert(rois.nbDims == 2);
    assert(rois.d[1] == 4);

    for (int i = 1; i < nbInputDims; ++i)
    {
        nvinfer1::Dims dims = inputs[i];

        // CHW with the same #C
        assert(dims.nbDims == 3 && dims.d[0] == inputs[i].d[0]);
    }
}

Dims PyramidROIAlign::getOutputDimensions(int index, const Dims* inputs, int nbInputDims) noexcept
{

    check_valid_inputs(inputs, nbInputDims);
    assert(index == 0);

    nvinfer1::Dims result;
    result.nbDims = 4;

    // mROICount
    result.d[0] = inputs[0].d[0];
    // mFeatureLength
    result.d[1] = inputs[1].d[0];
    // height
    result.d[2] = mPooledSize.y;
    // width
    result.d[3] = mPooledSize.x;

    return result;
}

int PyramidROIAlign::enqueue(
    int batch_size, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{

    void* pooled = outputs[0];

    cudaError_t status = roiAlign(stream, batch_size, mFeatureLength, mROICount, mThresh,

        inputs[0], &inputs[1], mFeatureSpatialSize,

        pooled, mPooledSize);

    return status;
}

size_t PyramidROIAlign::getSerializationSize() const noexcept
{
    return sizeof(int) * 2 + sizeof(int) * 3 + sizeof(float) + sizeof(int) * 2 * 4;
}

void PyramidROIAlign::serialize(void* buffer) const noexcept
{
    char *d = reinterpret_cast<char*>(buffer), *a = d;
    write(d, mPooledSize.y);
    write(d, mPooledSize.x);
    write(d, mFeatureLength);
    write(d, mROICount);
    write(d, mInputSize);
    write(d, mThresh);
    write(d, mFeatureSpatialSize[0].y);
    write(d, mFeatureSpatialSize[0].x);
    write(d, mFeatureSpatialSize[1].y);
    write(d, mFeatureSpatialSize[1].x);
    write(d, mFeatureSpatialSize[2].y);
    write(d, mFeatureSpatialSize[2].x);
    write(d, mFeatureSpatialSize[3].y);
    write(d, mFeatureSpatialSize[3].x);
    assert(d == a + getSerializationSize());
}

PyramidROIAlign::PyramidROIAlign(const void* data, size_t length)
{
    const char *d = reinterpret_cast<const char*>(data), *a = d;
    mPooledSize = {read<int>(d), read<int>(d)};
    mFeatureLength = read<int>(d);
    mROICount = read<int>(d);
    mInputSize = read<int>(d);
    mThresh = read<float>(d);
    mFeatureSpatialSize[0].y = read<int>(d);
    mFeatureSpatialSize[0].x = read<int>(d);
    mFeatureSpatialSize[1].y = read<int>(d);
    mFeatureSpatialSize[1].x = read<int>(d);
    mFeatureSpatialSize[2].y = read<int>(d);
    mFeatureSpatialSize[2].x = read<int>(d);
    mFeatureSpatialSize[3].y = read<int>(d);
    mFeatureSpatialSize[3].x = read<int>(d);

    assert(d == a + length);
}

// Return the DataType of the plugin output at the requested index
DataType PyramidROIAlign::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const
    noexcept
{
    // Only DataType::kFLOAT is acceptable by the plugin layer
    return DataType::kFLOAT;
}

// Return true if output tensor is broadcast across a batch.
bool PyramidROIAlign::isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const noexcept
{
    return false;
}

// Return true if plugin can use input that is broadcast across batch without replication.
bool PyramidROIAlign::canBroadcastInputAcrossBatch(int inputIndex) const noexcept
{
    return false;
}

// Configure the layer with input and output data types.
void PyramidROIAlign::configurePlugin(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,
    const DataType* inputTypes, const DataType* outputTypes, const bool* inputIsBroadcast,
    const bool* outputIsBroadcast, PluginFormat floatFormat, int maxBatchSize) noexcept
{
    assert(supportsFormat(inputTypes[0], floatFormat));
    check_valid_inputs(inputDims, nbInputs);

    assert(nbOutputs == 1);
    assert(nbInputs == 1 + mFeatureMapCount);

    mROICount = inputDims[0].d[0];
    mFeatureLength = inputDims[1].d[0];

    for (size_t layer = 0; layer < mFeatureMapCount; ++layer)
    {
        mFeatureSpatialSize[layer] = {inputDims[layer + 1].d[1], inputDims[layer + 1].d[2]};
    }
}

// Attach the plugin object to an execution context and grant the plugin the access to some context resource.
void PyramidROIAlign::attachToContext(
    cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) noexcept
{
}

// Detach the plugin object from its execution context.
void PyramidROIAlign::detachFromContext() noexcept {}
