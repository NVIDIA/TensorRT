/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
#include "multilevelCropAndResizePlugin.h"
#include "plugin.h"
#include <cuda_runtime_api.h>
#include <algorithm>

#include <fstream>

using namespace nvinfer1;
using namespace plugin;
using nvinfer1::plugin::MultilevelCropAndResize;
using nvinfer1::plugin::MultilevelCropAndResizePluginCreator;

namespace
{
const char* MULTILEVELCROPANDRESIZE_PLUGIN_VERSION{"1"};
const char* MULTILEVELCROPANDRESIZE_PLUGIN_NAME{"MultilevelCropAndResize_TRT"};
} // namespace

PluginFieldCollection MultilevelCropAndResizePluginCreator::mFC{};
std::vector<PluginField> MultilevelCropAndResizePluginCreator::mPluginAttributes;

MultilevelCropAndResizePluginCreator::MultilevelCropAndResizePluginCreator()
{
    mPluginAttributes.emplace_back(PluginField("pooled_size", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("image_size", nullptr, PluginFieldType::kINT32, 3));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* MultilevelCropAndResizePluginCreator::getPluginName() const
{
    return MULTILEVELCROPANDRESIZE_PLUGIN_NAME;
};

const char* MultilevelCropAndResizePluginCreator::getPluginVersion() const
{
    return MULTILEVELCROPANDRESIZE_PLUGIN_VERSION;
};

const PluginFieldCollection* MultilevelCropAndResizePluginCreator::getFieldNames()
{
    return &mFC;
};

IPluginV2Ext* MultilevelCropAndResizePluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
{
    auto image_size = TLTMaskRCNNConfig::IMAGE_SHAPE;
    const PluginField* fields = fc->fields;
    for (int i = 0; i < fc->nbFields; ++i)
    {
        const char* attrName = fields[i].name;
        if (!strcmp(attrName, "pooled_size"))
        {
            assert(fields[i].type == PluginFieldType::kINT32);
            mPooledSize = *(static_cast<const int*>(fields[i].data));
        }
        if (!strcmp(attrName, "image_size"))
        {
            assert(fields[i].type == PluginFieldType::kINT32);
            const auto dims = static_cast<const int32_t*>(fields[i].data);
            std::copy_n(dims, 3, image_size.d);
        }
    }
    return new MultilevelCropAndResize(mPooledSize, image_size);
};

IPluginV2Ext* MultilevelCropAndResizePluginCreator::deserializePlugin(const char* name, const void* data, size_t length)
{
    return new MultilevelCropAndResize(data, length);
};

MultilevelCropAndResize::MultilevelCropAndResize(int pooled_size, const nvinfer1::Dims& image_size)
    : mPooledSize({pooled_size, pooled_size})
{

    assert(pooled_size > 0);
    // shape
    mInputHeight = image_size.d[1];
    mInputWidth = image_size.d[2];
    // Threshold to P3: Smaller -> P2
    mThresh = (224 * 224) / (4.0f);
};

int MultilevelCropAndResize::getNbOutputs() const
{
    return 1;
};

int MultilevelCropAndResize::initialize()
{
    return 0;
};

void MultilevelCropAndResize::terminate(){

};

void MultilevelCropAndResize::destroy()
{
    delete this;
};

size_t MultilevelCropAndResize::getWorkspaceSize(int) const
{
    return 0;
}

bool MultilevelCropAndResize::supportsFormat(DataType type, PluginFormat format) const
{
    return (type == DataType::kFLOAT && format == PluginFormat::kNCHW);
};

const char* MultilevelCropAndResize::getPluginType() const
{
    return "MultilevelCropAndResize_TRT";
};

const char* MultilevelCropAndResize::getPluginVersion() const
{
    return "1";
};

IPluginV2Ext* MultilevelCropAndResize::clone() const
{
    return new MultilevelCropAndResize(*this);
};

void MultilevelCropAndResize::setPluginNamespace(const char* libNamespace)
{
    mNameSpace = libNamespace;
};

const char* MultilevelCropAndResize::getPluginNamespace() const
{
    return mNameSpace.c_str();
}

void MultilevelCropAndResize::check_valid_inputs(const nvinfer1::Dims* inputs, int nbInputDims)
{
    // to be compatible with tensorflow node's input:
    // roi: [N, anchors, 4],
    // feature_map list(5 maps): p2, p3, p4, p5, p6
    assert(nbInputDims == 1 + mFeatureMapCount);

    nvinfer1::Dims rois = inputs[0];
    assert(rois.nbDims == 2);
    assert(rois.d[1] == 4);

    for (int i = 1; i < nbInputDims; ++i)
    {
        nvinfer1::Dims dims = inputs[i];

        // CHW with the same #C
        assert(dims.nbDims == 3 && dims.d[0] == inputs[1].d[0]);
    }
}

Dims MultilevelCropAndResize::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
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
};

int MultilevelCropAndResize::enqueue(
    int batch_size, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream)
{

    void* pooled = outputs[0];

    cudaError_t status = roiAlignHalfCenter(stream, batch_size, mFeatureLength, mROICount, mThresh,

        mInputHeight, mInputWidth, inputs[0], &inputs[1], mFeatureSpatialSize,

        pooled, mPooledSize);

    assert(status == cudaSuccess);
    return 0;
};

size_t MultilevelCropAndResize::getSerializationSize() const
{
    return sizeof(int) * 2 + sizeof(int) * 4 + sizeof(float) + sizeof(int) * 2 * mFeatureMapCount;
};

void MultilevelCropAndResize::serialize(void* buffer) const
{
    char *d = reinterpret_cast<char*>(buffer), *a = d;
    write(d, mPooledSize.y);
    write(d, mPooledSize.x);
    write(d, mFeatureLength);
    write(d, mROICount);
    write(d, mInputHeight);
    write(d, mInputWidth);
    write(d, mThresh);
    for (int i = 0; i < mFeatureMapCount; i++)
    {
        write(d, mFeatureSpatialSize[i].y);
        write(d, mFeatureSpatialSize[i].x);
    }
    assert(d == a + getSerializationSize());
};

MultilevelCropAndResize::MultilevelCropAndResize(const void* data, size_t length)
{
    const char *d = reinterpret_cast<const char*>(data), *a = d;
    mPooledSize = {read<int>(d), read<int>(d)};
    mFeatureLength = read<int>(d);
    mROICount = read<int>(d);
    mInputHeight = read<int>(d);
    mInputWidth = read<int>(d);
    mThresh = read<float>(d);
    for (int i = 0; i < mFeatureMapCount; i++)
    {
        mFeatureSpatialSize[i].y = read<int>(d);
        mFeatureSpatialSize[i].x = read<int>(d);
    }

    assert(d == a + length);
};

// Return the DataType of the plugin output at the requested index
DataType MultilevelCropAndResize::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const
{
    // Only DataType::kFLOAT is acceptable by the plugin layer
    return DataType::kFLOAT;
}

// Return true if output tensor is broadcast across a batch.
bool MultilevelCropAndResize::isOutputBroadcastAcrossBatch(
    int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const
{
    return false;
}

// Return true if plugin can use input that is broadcast across batch without replication.
bool MultilevelCropAndResize::canBroadcastInputAcrossBatch(int inputIndex) const
{
    return false;
}

// Configure the layer with input and output data types.
void MultilevelCropAndResize::configurePlugin(const Dims* inputDims, int nbInputs, const Dims* outputDims,
    int nbOutputs, const DataType* inputTypes, const DataType* outputTypes, const bool* inputIsBroadcast,
    const bool* outputIsBroadcast, PluginFormat floatFormat, int maxBatchSize)
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
void MultilevelCropAndResize::attachToContext(
    cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator)
{
}

// Detach the plugin object from its execution context.
void MultilevelCropAndResize::detachFromContext() {}
